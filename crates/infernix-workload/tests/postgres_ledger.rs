//! PostgreSQL integration tests for the ledger contracts.
//!
//! Every test starts a throwaway cluster with `initdb`/`pg_ctl` (through
//! `infernix-workload-testing`), so these run the real SQL, not the in-memory
//! reference model. `initdb` and `pg_ctl` are in the repository dev shell.

use chrono::{Duration, Utc};
use infernix_workload::{
    BusyReleaseOutcome, EnqueueOutcome, HeartbeatOutcome, InputFingerprint, JobId, JobSpec,
    JobState, Lease, RESOURCE_BUSY, WorkerId, WorkerProfile, postgres::WorkloadClient,
};
use infernix_workload_testing::{TestCluster, fresh_database};

fn job(id: &str) -> JobSpec {
    JobSpec {
        id: JobId::new(id).unwrap(),
        workload: "animation".into(),
        queue: "asset".into(),
        input_fingerprint: InputFingerprint::new(format!("sha256:{id}")).unwrap(),
        payload: serde_json::json!({"stage": "asset"}),
        priority: 0,
        requirements: ["gpu:atlas-7900xtx".to_owned()].into_iter().collect(),
        max_attempts: 3,
    }
}

fn worker(id: &str) -> WorkerProfile {
    WorkerProfile {
        id: WorkerId::new(id).unwrap(),
        boot_id: format!("boot-{id}"),
        capabilities: ["gpu:atlas-7900xtx".to_owned()].into_iter().collect(),
        concurrency: 1,
        drained: false,
    }
}

fn pairs() -> Vec<(String, String)> {
    vec![("animation".to_owned(), "asset".to_owned())]
}

fn capabilities() -> Vec<String> {
    vec!["gpu:atlas-7900xtx".to_owned()]
}

async fn claim_one(client: &WorkloadClient, worker: &WorkerProfile) -> Lease {
    client
        .register_worker(worker)
        .await
        .expect("register worker");
    client
        .claim(
            &worker.id,
            &worker.boot_id,
            &capabilities(),
            &pairs(),
            Duration::minutes(5),
        )
        .await
        .expect("claim")
        .expect("a queued job is claimable")
        .lease
}

#[tokio::test]
async fn enqueue_is_idempotent_and_find_adopts_by_fingerprint() {
    let cluster = TestCluster::start();
    let client = fresh_database(&cluster).await.unwrap();

    let first = client.enqueue(&job("job-a")).await.unwrap();
    assert_eq!(first, EnqueueOutcome::Created(JobId::new("job-a").unwrap()));

    let mut duplicate = job("job-b");
    duplicate.input_fingerprint = InputFingerprint::new("sha256:job-a").unwrap();
    assert_eq!(
        client.enqueue(&duplicate).await.unwrap(),
        EnqueueOutcome::Existing(JobId::new("job-a").unwrap())
    );

    let fingerprint = InputFingerprint::new("sha256:job-a").unwrap();
    let found = client
        .find("animation", "asset", &fingerprint)
        .await
        .unwrap()
        .expect("adoption finds the job by fingerprint");
    assert_eq!(found.job_id.as_str(), "job-a");
    assert_eq!(found.state, JobState::Pending);
    assert_eq!(found.attempts, 0);
    assert!(found.delivered_at.is_none());

    // A different queue is a different generation identity.
    assert!(
        client
            .find("animation", "shot", &fingerprint)
            .await
            .unwrap()
            .is_none()
    );
    assert!(
        client
            .status(&JobId::new("missing").unwrap())
            .await
            .unwrap()
            .is_none()
    );
}

#[tokio::test]
async fn cancel_pending_is_immediate_and_leased_waits_for_confirmation() {
    let cluster = TestCluster::start();
    let client = fresh_database(&cluster).await.unwrap();

    client.enqueue(&job("pending")).await.unwrap();
    let outcome = client
        .cancel(&JobId::new("pending").unwrap())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(outcome.previous_state, JobState::Pending);
    assert_eq!(outcome.state, JobState::Cancelled);
    assert!(outcome.cancel_requested_at.is_some());
    assert_eq!(
        client
            .status(&JobId::new("pending").unwrap())
            .await
            .unwrap()
            .unwrap()
            .state,
        JobState::Cancelled
    );

    client.enqueue(&job("leased")).await.unwrap();
    let atlas = worker("atlas");
    let lease = claim_one(&client, &atlas).await;
    let outcome = client
        .cancel(&JobId::new("leased").unwrap())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(outcome.previous_state, JobState::Leased);
    assert_eq!(
        outcome.state,
        JobState::Leased,
        "a leased job waits for the worker to confirm"
    );

    // The flag arrives with the heartbeat rather than in a second query.
    match client
        .heartbeat(&lease, Duration::minutes(5))
        .await
        .unwrap()
    {
        HeartbeatOutcome::Renewed {
            cancel_requested_at,
        } => assert!(cancel_requested_at.is_some()),
        HeartbeatOutcome::Fenced => panic!("the lease is still current"),
    }

    assert!(client.confirm_cancelled(&lease).await.unwrap());
    let status = client
        .status(&JobId::new("leased").unwrap())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(status.state, JobState::Cancelled);
    assert_eq!(status.attempts, 1);

    // Cancelling a terminal job is a no-op that still reports its state.
    let again = client
        .cancel(&JobId::new("leased").unwrap())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(again.previous_state, JobState::Cancelled);
    assert_eq!(again.state, JobState::Cancelled);
    assert!(
        client
            .cancel(&JobId::new("missing").unwrap())
            .await
            .unwrap()
            .is_none()
    );
}

#[tokio::test]
async fn busy_release_keeps_the_attempt_and_honours_not_before() {
    let cluster = TestCluster::start();
    let client = fresh_database(&cluster).await.unwrap();
    client.enqueue(&job("job")).await.unwrap();
    let atlas = worker("atlas");
    let lease = claim_one(&client, &atlas).await;
    assert_eq!(lease.attempt, 1);

    let retry_at = Utc::now() + Duration::hours(1);
    let outcome = client
        .release_busy(&lease, retry_at, 30)
        .await
        .unwrap()
        .unwrap();
    assert!(
        matches!(
            outcome,
            BusyReleaseOutcome::Released {
                attempts: 0,
                busy_releases: 1,
                ..
            }
        ),
        "unexpected {outcome:?}"
    );

    let status = client
        .status(&JobId::new("job").unwrap())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(status.state, JobState::Pending);
    assert_eq!(status.attempts, 0, "a busy release spends no attempt");

    // The claim honours the retry delay.
    assert!(
        client
            .claim(
                &atlas.id,
                &atlas.boot_id,
                &capabilities(),
                &pairs(),
                Duration::minutes(5),
            )
            .await
            .unwrap()
            .is_none()
    );
    client
        .raw()
        .execute(
            "UPDATE infernix_workload_jobs SET not_before = now() - interval '1 second' WHERE id = $1",
            &[&"job"],
        )
        .await
        .unwrap();
    let again = client
        .claim(
            &atlas.id,
            &atlas.boot_id,
            &capabilities(),
            &pairs(),
            Duration::minutes(5),
        )
        .await
        .unwrap()
        .expect("the job is claimable once not_before passed")
        .lease;
    assert_eq!(again.attempt, 1);
}

#[tokio::test]
async fn cancel_wins_over_a_retry() {
    let cluster = TestCluster::start();
    let client = fresh_database(&cluster).await.unwrap();
    let atlas = worker("atlas");

    // A busy release on a cancelled job resolves it instead of parking it.
    client.enqueue(&job("busy")).await.unwrap();
    let lease = claim_one(&client, &atlas).await;
    client.cancel(&JobId::new("busy").unwrap()).await.unwrap();
    assert_eq!(
        client
            .release_busy(&lease, Utc::now(), 30)
            .await
            .unwrap()
            .unwrap(),
        BusyReleaseOutcome::Cancelled { busy_releases: 1 }
    );
    assert_eq!(
        client
            .status(&JobId::new("busy").unwrap())
            .await
            .unwrap()
            .unwrap()
            .state,
        JobState::Cancelled
    );

    // A normal failure on a cancelled job does the same.
    client.enqueue(&job("failed")).await.unwrap();
    let lease = claim_one(&client, &atlas).await;
    client.cancel(&JobId::new("failed").unwrap()).await.unwrap();
    assert_eq!(
        client.fail(&lease, "adapter exited with 1").await.unwrap(),
        Some(JobState::Cancelled)
    );
    let status = client
        .status(&JobId::new("failed").unwrap())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(status.state, JobState::Cancelled);
    assert!(
        client
            .claim(
                &atlas.id,
                &atlas.boot_id,
                &capabilities(),
                &pairs(),
                Duration::minutes(5),
            )
            .await
            .unwrap()
            .is_none(),
        "no cancelled job is claimable again"
    );
}

#[tokio::test]
async fn busy_release_budget_fails_with_resource_busy() {
    let cluster = TestCluster::start();
    let client = fresh_database(&cluster).await.unwrap();
    client.enqueue(&job("job")).await.unwrap();
    let atlas = worker("atlas");

    for expected in 1..=2 {
        let lease = claim_one(&client, &atlas).await;
        let outcome = client
            .release_busy(&lease, Utc::now() - Duration::seconds(1), 2)
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(
            outcome,
            BusyReleaseOutcome::Released {
                busy_releases,
                attempts: 0,
                ..
            } if busy_releases == expected
        ));
    }
    let lease = claim_one(&client, &atlas).await;
    assert_eq!(
        client
            .release_busy(&lease, Utc::now(), 2)
            .await
            .unwrap()
            .unwrap(),
        BusyReleaseOutcome::Exhausted { busy_releases: 3 }
    );
    let status = client
        .status(&JobId::new("job").unwrap())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(status.state, JobState::Failed);
    assert_eq!(status.error.as_deref(), Some(RESOURCE_BUSY));
}

#[tokio::test]
async fn delivery_requires_an_accepted_complete() {
    let cluster = TestCluster::start();
    let client = fresh_database(&cluster).await.unwrap();
    client.enqueue(&job("job")).await.unwrap();
    let atlas = worker("atlas");
    let nomad = worker("nomad");
    client.register_worker(&nomad).await.unwrap();
    let lease = claim_one(&client, &atlas).await;
    let job_id = JobId::new("job").unwrap();

    assert!(
        !client.mark_delivered(&job_id, &atlas.id).await.unwrap(),
        "a pending job is not delivered"
    );
    assert!(
        client
            .complete(&lease, &serde_json::json!({"artifacts": []}))
            .await
            .unwrap()
    );
    let status = client.status(&job_id).await.unwrap().unwrap();
    assert_eq!(status.state, JobState::Succeeded);
    assert!(
        status.delivered_at.is_none(),
        "complete does not imply delivered"
    );
    assert_eq!(
        client.undelivered_succeeded(&atlas.id).await.unwrap().len(),
        1
    );

    assert!(
        !client.mark_delivered(&job_id, &nomad.id).await.unwrap(),
        "only the completing worker may mark delivery"
    );
    assert!(client.mark_delivered(&job_id, &atlas.id).await.unwrap());
    assert!(
        client
            .status(&job_id)
            .await
            .unwrap()
            .unwrap()
            .delivered_at
            .is_some()
    );
    assert!(
        !client.mark_delivered(&job_id, &atlas.id).await.unwrap(),
        "delivery is idempotent"
    );
    assert_eq!(
        client.undelivered_succeeded(&atlas.id).await.unwrap().len(),
        0
    );
}

#[tokio::test]
async fn expired_lease_is_recovered_by_another_worker() {
    let cluster = TestCluster::start();
    let client = fresh_database(&cluster).await.unwrap();
    client.enqueue(&job("job")).await.unwrap();
    let atlas = worker("atlas");
    let nomad = worker("nomad");
    client.register_worker(&atlas).await.unwrap();
    client.register_worker(&nomad).await.unwrap();

    let stale = client
        .claim(
            &atlas.id,
            &atlas.boot_id,
            &capabilities(),
            &pairs(),
            Duration::seconds(1),
        )
        .await
        .unwrap()
        .expect("atlas claims first")
        .lease;
    tokio::time::sleep(std::time::Duration::from_millis(1200)).await;

    let recovered = client
        .claim(
            &nomad.id,
            &nomad.boot_id,
            &capabilities(),
            &pairs(),
            Duration::minutes(5),
        )
        .await
        .unwrap()
        .expect("an expired lease is claimable again")
        .lease;
    assert_eq!(recovered.job_id.as_str(), "job");
    assert_eq!(recovered.attempt, 2, "recovery consumes an attempt");
    assert_eq!(recovered.worker_id, nomad.id);

    assert!(
        !client
            .complete(&stale, &serde_json::json!({"stale": true}))
            .await
            .unwrap(),
        "a stale completion is fenced"
    );
    assert!(
        client
            .complete(&recovered, &serde_json::json!({"stale": false}))
            .await
            .unwrap(),
        "the winning attempt completes"
    );
    let status = client
        .status(&JobId::new("job").unwrap())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(status.state, JobState::Succeeded);
    assert_eq!(
        status.result,
        Some(serde_json::json!({"stale": false})),
        "the stale result never lands"
    );
    assert!(status.delivered_at.is_none());
}

#[tokio::test]
async fn migrations_are_additive_and_idempotent() {
    let cluster = TestCluster::start();
    let client = fresh_database(&cluster).await.unwrap();
    client.migrate().await.expect("migration is repeatable");

    let columns = client
        .raw()
        .query(
            "SELECT column_name FROM information_schema.columns
              WHERE table_name = 'infernix_workload_jobs'",
            &[],
        )
        .await
        .unwrap()
        .into_iter()
        .map(|row| row.get::<_, String>(0))
        .collect::<Vec<_>>();
    for expected in [
        "cancel_requested_at",
        "delivered_at",
        "not_before",
        "busy_releases",
    ] {
        assert!(columns.contains(&expected.to_owned()), "missing {expected}");
    }

    // A legacy ledger gains the columns too.
    let legacy = fresh_database(&cluster).await.unwrap();
    legacy
        .raw()
        .batch_execute(
            "ALTER TABLE infernix_workload_jobs
                 DROP COLUMN cancel_requested_at,
                 DROP COLUMN delivered_at,
                 DROP COLUMN not_before,
                 DROP COLUMN busy_releases",
        )
        .await
        .unwrap();
    legacy.migrate().await.expect("legacy ledger upgrades");
    legacy.migrate().await.expect("upgrade is repeatable");
}
