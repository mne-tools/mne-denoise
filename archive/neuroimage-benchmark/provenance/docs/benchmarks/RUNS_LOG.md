# Runs Log (append-only)

Written **only by the submission controller**, never concurrently by array tasks. One row per submitted job
(or job array). Keep newest at the bottom.

Columns: date (UTC) · arm · cluster · account · sbatch job id · array spec · `run_fingerprint` ·
`execution_id` · results path · outcome (`submitted/running/completed/failed/partial`) · notes.

| date | arm | cluster | account | job id | array | run_fingerprint | execution_id | results path | outcome | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — | — | — | (no runs yet — P0) |

## Conventions
- `run_fingerprint = hash(git_sha, config_hash, dataset_manifest_hash, environment_hash)` — same fingerprint
  ⇒ identical scientific specification (retries share it).
- `execution_id = <timestamp>_<slurm_job_id>` — unique per execution.
- Failed/timeout runs are recorded, not deleted (intention-to-benchmark). Per-method failure-rate tables are
  produced at aggregation.
