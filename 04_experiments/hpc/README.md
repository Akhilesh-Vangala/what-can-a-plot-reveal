# Cluster jobs (Slurm, A100)

## Workflow
1. `bash 04_experiments/make_bundle.sh` → `04_experiments/visml_bundle.zip`: all code, all job scripts (generated from
   one template by `hpc/make_jobs.py`) and a VERSION stamp.
2. Upload `visml_bundle.zip` to `~/visml_hpc` on the cluster.
3. `unzip -oq ~/visml_hpc/visml_bundle.zip -d $SCRATCH/visml/code && bash $SCRATCH/visml/code/hpc/run_stage.sh <stage>`

## Stages

| Stage | What it runs |
|---|---|
| `check` | inputs and queue |
| `confirm` | the confirmatory study: both data sources, benchmarks, all readers, controls, interventions (chained jobs) |
| `interventions [RUN]` | evidence badges and companion views, benchmark and readers |
| `positive` | positive control (blatant failures) and value-reading probe |
| `realworld` | real-world check: transfer of the evidential map, five readers |
| `rtr` | read-then-reason: value extraction by the readers, fixed-rule diagnosis |
| `gaps` | values-printed designs for the readers; multi-plot benchmark |
| `review` | kernel two-sample tests, stronger image benchmark B3 (both runs), binary and 2AFC reader tasks (synthetic) |
| `review_real` | binary and 2AFC reader tasks on the real-data run |

## Job design
- Every job runs code only from the unpacked bundle and prints its VERSION.
- Every job checks its inputs before starting; the launcher refuses to submit when an input is missing or a result
  would be overwritten.
- `set -o pipefail` and an explicit exit code, so a failed step is never reported as completed; dependent jobs are
  cancelled on failure (`--kill-on-invalid-dep=yes`).
- Jobs requeue if a node fails and resume from the answers already written.
