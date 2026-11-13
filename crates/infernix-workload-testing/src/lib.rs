//! Throwaway PostgreSQL clusters for the Infernix ledger integration tests.
//!
//! The cluster lives under `/tmp/infx-pg-<random>` with a short socket path,
//! because a Unix socket path must stay below 107 bytes. `initdb`/`pg_ctl`
//! come from the repository dev shell, so a test that runs outside it fails
//! loudly instead of skipping.

use std::{
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicU32, Ordering},
};

use uuid::Uuid;

use infernix_workload::postgres::{PgResult, WorkloadClient};

static DATABASE_COUNTER: AtomicU32 = AtomicU32::new(0);

/// A running throwaway cluster. Dropping it stops the server and removes the
/// data directory.
pub struct TestCluster {
    root: PathBuf,
    socket_dir: PathBuf,
}

impl TestCluster {
    /// `initdb` a fresh cluster and start it on a private socket directory.
    pub fn start() -> Self {
        let root = std::env::temp_dir().join(format!(
            "infx-pg-{}",
            &Uuid::new_v4().simple().to_string()[..8]
        ));
        let socket_dir = root.join("sock");

        run(
            Command::new("initdb")
                .arg("-D")
                .arg(&root)
                .arg("-A")
                .arg("trust")
                .arg("-U")
                .arg("postgres")
                .arg("--no-sync"),
            "initdb",
        );
        std::fs::create_dir_all(&socket_dir).expect("create test PostgreSQL socket directory");
        let log = root.join("server.log");
        run(
            Command::new("pg_ctl")
                .arg("-D")
                .arg(&root)
                .arg("-l")
                .arg(&log)
                .arg("-w")
                .arg("-o")
                .arg(format!(
                    "-k {} -c listen_addresses='' -c fsync=off -c synchronous_commit=off",
                    socket_dir.display()
                ))
                .arg("start"),
            "pg_ctl start",
        );
        Self { root, socket_dir }
    }

    /// libpq connection string for a database on this cluster.
    pub fn connection_string(&self, database: &str) -> String {
        format!(
            "host='{}' port=5432 user=postgres dbname={database}",
            self.socket_dir.display()
        )
    }

    pub fn root(&self) -> &Path {
        &self.root
    }
}

impl Drop for TestCluster {
    fn drop(&mut self) {
        let _ = Command::new("pg_ctl")
            .arg("-D")
            .arg(&self.root)
            .arg("-m")
            .arg("immediate")
            .arg("-w")
            .arg("stop")
            .output();
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

/// A migrated ledger on a private database of `cluster`.
pub async fn fresh_database(cluster: &TestCluster) -> PgResult<WorkloadClient> {
    fresh_ledger(cluster).await.map(|(_, client)| client)
}

/// A migrated ledger plus its database name, for tests that also start a
/// worker process against it.
pub async fn fresh_ledger(cluster: &TestCluster) -> PgResult<(String, WorkloadClient)> {
    let name = format!(
        "infx_test_{}",
        DATABASE_COUNTER.fetch_add(1, Ordering::SeqCst)
    );
    let admin = WorkloadClient::connect(&cluster.connection_string("postgres")).await?;
    admin
        .raw()
        .batch_execute(&format!("CREATE DATABASE {name}"))
        .await
        .expect("create test database");
    let client = WorkloadClient::connect(&cluster.connection_string(&name)).await?;
    client.migrate().await?;
    Ok((name, client))
}

fn run(command: &mut Command, what: &str) {
    let output = command
        .output()
        .unwrap_or_else(|error| panic!("{what} is required by the PostgreSQL tests: {error}"));
    if !output.status.success() {
        panic!(
            "{what} failed with {}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
