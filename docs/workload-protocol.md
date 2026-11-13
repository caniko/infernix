# Workload protocol

The durable contract between the ledger (`infernix-workload`), the worker
(`infernix-workerd`) and an adapter. Frozen by
`canix/docs/src/planning/2026-11-13-animation-interfaces.md`; this page
describes what the code actually does.

## Ledger states

`pending → leased → succeeded → (delivered)`, or `failed` / `cancelled`.

| Column | Meaning |
| --- | --- |
| `cancel_requested_at` | `CANCEL` ran; a leased job keeps its state until the worker confirms. |
| `delivered_at` | Set by `MARK_DELIVERED` after every artifact was accepted by the hub. |
| `not_before` | Retry delay applied by a busy release; `CLAIM` ignores the row until then. |
| `busy_releases` | Count of busy releases; the ledger fails the job as `resource_busy` past the worker's budget. |

`CANCEL` cancels a `pending` job immediately, flags a `leased` job, and is a
no-op on a terminal job. A job with a cancel flag is never claimed again; if
its lease expires, the ledger resolves it to `cancelled` because no adapter is
left to confirm it.

`RELEASE_BUSY` returns a leased job to `pending` with `attempts = attempts - 1`
(the device was busy, the attempt did not run) and a short `not_before` delay.
After `max_busy_releases` releases the job fails with `resource_busy`. A cancel
flag outranks a retry: a busy release or a normal failure on a cancelled job
resolves it to `cancelled`, because nothing would ever claim it again.

`COMPLETE` is refused when `cancel_requested_at` is set: the worker then runs
`CONFIRM_CANCELLED` instead of uploading, atomically. Like `HEARTBEAT`,
`COMPLETE` also requires `lease_expires_at > now()`, so an expired lease cannot
land a result even before another worker claims the job.

`MARK_DELIVERED` is fenced on the worker that completed the job. The boot id is
not part of that fence: a restart must finish delivery, and the upload is
idempotent.

Migrations are additive (`ADD COLUMN IF NOT EXISTS`), so an existing ledger
upgrades in place.

## Shared PostgreSQL client

`infernix-workload` behind the `postgres` feature exposes the async
`infernix_workload::postgres::WorkloadClient`. `infernix-workerd` and the hub
pack use the same implementation:

```rust
WorkloadClient::connect(config: &str) -> Result<Self, PgError>
client.migrate().await
client.register_worker(&WorkerProfile).await
client.drain_worker(&WorkerId, boot_id).await -> bool
client.worker_heartbeat(&WorkerId, boot_id).await -> bool
client.enqueue(&JobSpec).await -> EnqueueOutcome
client.claim(&WorkerId, boot_id, &[String], &[(String, String)], Duration).await -> Option<ClaimedJob>
client.heartbeat(&Lease, Duration).await -> HeartbeatOutcome
client.complete(&Lease, &Value).await -> bool
client.fail(&Lease, &str).await -> Option<JobState>
client.cancel(&JobId).await -> Option<CancelOutcome>
client.confirm_cancelled(&Lease).await -> bool
client.release_busy(&Lease, not_before, max_busy_releases).await -> Option<BusyReleaseOutcome>
client.mark_delivered(&JobId, &WorkerId).await -> bool
client.mark_delivery_failed(&JobId, &WorkerId, reason).await -> bool
client.status(&JobId).await -> Option<JobStatus>
client.find(workload, queue, &InputFingerprint).await -> Option<JobStatus>
client.undelivered_succeeded(&WorkerId).await -> Vec<JobStatus>
```

`InMemoryQueue` mirrors every transition for unit tests.

## Envelope v2 (worker → adapter stdin, one JSON object)

```json
{
  "protocolVersion": 2,
  "workload": "animation",
  "queue": "asset",
  "jobId": "…",
  "leaseToken": "…",
  "attempt": 1,
  "inputFingerprint": "…",
  "stagingDirectory": "/…/staging/<jobId>/<attempt>",
  "outputDirectory": "/…/staging/<jobId>/<attempt>/out",
  "cancelPath": "/…/staging/<jobId>/<attempt>/CANCEL",
  "payload": {
    "stage": "asset",
    "operation": "tts",
    "generationRequestId": "<uuid>",
    "hubUrl": "http://…",
    "profile": {},
    "inputs": [],
    "params": {},
    "seed": 0
  }
}
```

The worker also exports `INFERNIX_WORKLOAD_STAGING_DIRECTORY`,
`INFERNIX_WORKLOAD_OUTPUT_DIRECTORY` and `INFERNIX_STAGING_ROOT` (the worker's
staging root, which an adapter's persistent caches must never overlap). The hub
token is never in the envelope: adapters read their own token through
`INFERNIX_HUB_TOKEN_FILE` when they need to fetch inputs.

An adapter writes outputs only under `outputDirectory` and prints exactly one
result object on exit 0:

```json
{
  "artifacts": [
    {"source": "audio", "path": "out/line.wav", "sha256": "<hex>", "mime": "audio/wav", "size": 123}
  ],
  "contentFingerprint": "<hex>",
  "cacheHit": false,
  "telemetry": {"peakVramBytes": 0, "gpuSeconds": 0.0, "cpuSeconds": 0.0, "peakRssBytes": 0, "maxTempC": 0, "maxPowerW": 0, "device": "…", "backend": "rocm"},
  "validation": {"state": "ok", "findings": []}
}
```

`path` is relative to `stagingDirectory`. `source` matches
`[a-z0-9][a-z0-9_.-]{0,62}` and is unique. Each artifact is at most 256 MiB.
`contentFingerprint` (64 lowercase hex), `cacheHit` (bool), `telemetry` (object)
and `validation` (`state` one of `ok|warn|fail`, `findings` array) are
required; extra keys are tolerated. `telemetry` must carry `peakVramBytes`
(u64), `gpuSeconds` (number ≥ 0), `cpuSeconds` (number ≥ 0), `peakRssBytes`
(u64), `maxTempC` (number or null), `maxPowerW` (number or null), `device`
(non-empty string) and `backend` (`rocm` or `cuda`). The worker rejects a result
that is missing a field, has the wrong type, whose artifact is missing, is a
symlink, escapes staging, or mismatches its declared size or sha256, and fails
the job with `invalid_result:<code>`.

`telemetry` is stored verbatim in the job result, so the hub reads resource
numbers from the ledger manifest and not from worker logs.

## Exit codes

| Code | Meaning | Worker action |
| --- | --- | --- |
| 0 | ok | fenced `complete`, upload, `MARK_DELIVERED` |
| 10 | `resource_busy` | `RELEASE_BUSY` |
| 11 | `gpu_unavailable` | fail |
| 12 | `oom` | fail |
| 13 | `backend_mismatch` | fail |
| 14 | `invalid_input` | fail |
| 15 | `cancelled` | `CONFIRM_CANCELLED` |
| other | error | fail (retry while attempts remain) |

## Cancellation

The worker learns about a cancellation from its lease heartbeat. It writes the
attempt's `CANCEL` file, sends `SIGTERM` to the adapter's process group, waits
`worker.cancel_grace_secs` (default 30 s), then sends `SIGKILL`. It polls
`kill(-pgid, 0)` until ESRCH proves the whole group is gone — a descendant that
ignores `SIGTERM` is killed and reaped first — and only then runs
`CONFIRM_CANCELLED`. A member that survives `SIGKILL` (for example a D-state
process) keeps the lease: the worker renews it with each heartbeat cycle —
during the `SIGTERM` grace and on every iteration of the `SIGKILL` wait — and
retries `SIGKILL`; it never confirms a cancellation it cannot prove, and a lease
shorter than a wait can never expire while the worker still owns the job. A
fenced completion that is refused because a cancellation was requested runs the
same termination and ESRCH check before confirming. A job whose confirmation was
fenced is left to the worker that won the lease.

## Delivery

After the fenced `complete` is accepted, the worker uploads every artifact:

```
POST {hubUrl}/v1/generations/{generationRequestId}/artifacts
authorization: Bearer <procedure token>
x-artifact-source: <source>
content-type: <mime>
x-sha256: <hex>
<raw body>
```

`200` and an identical `409` replay both count as accepted. The hub coordinates
are copied into the stored result as `delivery: {hubUrl, generationRequestId}`,
which is what the restart pass needs.

A workload that declares no artifacts is marked delivered as soon as its
completion is accepted.

On startup, and then periodically while it runs, the worker lists its own
`succeeded` and undelivered jobs and finishes their delivery from retained
staging. The retry starts at `worker.delivery_retry_secs` (default 30 s) and
backs off exponentially to a 10 minute cap after a failed pass. Missing staging
fails the job as `staging_lost`. Staging is removed only after delivery (or, for
a cancelled job, kept for inspection: it will never be delivered).

## CLI

```
infernix-workerd --config <file> migrate
infernix-workerd --config <file> worker [--once]
infernix-workerd --config <file> drain
infernix-workerd --config <file> enqueue --workload … --queue … --input-fingerprint … --payload …
infernix-workerd --config <file> status <job-id>     # JobStatus JSON
infernix-workerd --config <file> cancel <job-id>     # {jobId, previousState, state, cancelRequestedAt, deferred}
```

`cancel` reports `deferred: true` while a leased job still waits for its worker
to confirm termination.

## Worker configuration

```toml
[worker]
id = "atlas"
capabilities = ["gpu:atlas-7900xtx"]
gpu_device = "atlas-7900xtx"    # also advertised as gpu:<device>
concurrency = 1
lease_duration_secs = 900
heartbeat_interval_secs = 30
poll_interval_secs = 5
delivery_retry_secs = 30        # base delay for retained-delivery retries
staging_root = "/var/lib/infernix-workload/staging"
cancel_grace_secs = 30          # SIGTERM → SIGKILL grace
max_busy_releases = 30
busy_retry_delay_secs = 30
# hub_url = "http://…"          # fallback when the payload carries no hubUrl

[[adapters]]
workload = "animation"
queues = ["asset"]
profiles = ["sam2-1-hiera-small-rocm"]   # job `profile:<id>` requirements
command = "/run/current-system/sw/bin/animation-adapter"
artifacts = true               # validate result.artifacts and deliver them
```

The worker's claim capabilities are the configured labels plus `gpu:<gpu_device>`
plus one `profile:<id>` per adapter profile, so a job is claimed only by a
worker that has an adapter for it. The job's single `profile:<id>` requirement
selects the adapter (never adapter order); a claimed job with no matching
adapter fails as `invalid_input`. Two adapters serving the same
`(workload, queue, profile)` are a configuration error.

`services.infernix.workloadFabric` renders this file and runs the worker as
`infernix-worker` with `MemoryMax`, `CPUQuota`, `TasksMax`, read-write access
limited to staging, cache and model directories, the hub token as a systemd
credential, and the device-gate lock (`/run/infernix/gpu/<device>.lock`, group
`render`, mode 0660).

## GPU admission

A worker on a shared desktop can refuse GPU work while the machine's owner uses
it, and hand a running job back when the owner returns. The probe is
host-specific; the worker only reads its exit status.

```toml
[worker.admission]
command = ["/nix/store/…/bin/canix", "host", "gpu-admission", "check", "--device", "murph-a770"]
recheck_secs = 15      # reuse one verdict for at most this long (default 15)
timeout_secs = 10      # a slower probe is a deny (default 10)
preempt = true         # stop a running adapter when the verdict flips (default false)
```

Exit 0 admits. Any other exit code, a signal, a timeout (the probe is killed) or
a spawn failure denies: the worker fails closed. The logged reason is the first
non-empty line of the probe's stdout, else its stderr, capped at 300 characters,
and a verdict is logged only on an admit↔deny transition (and the first one).

While the verdict is deny the worker claims nothing: delivery recovery and
joining finished jobs keep running, and the poll interval still bounds the loop.
A verdict is cached for `recheck_secs`, so a deny does not re-probe every poll.

With `preempt = true` a running adapter is stopped when the verdict flips to
deny: the attempt's `CANCEL` file is written, the process group gets `SIGTERM`,
and `SIGKILL` follows after `worker.cancel_grace_secs`. The job returns to
`pending` without spending an attempt (the same `RELEASE_BUSY` path as exit 10,
so `max_busy_releases` and `busy_retry_delay_secs` bound the retries). An adapter
that exits 0 before the signal lands still completes normally, and a ledger
cancellation the worker observes first keeps the usual cancellation path.

Without `[worker.admission]` the worker behaves exactly as before.
