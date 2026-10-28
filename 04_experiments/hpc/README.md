# HPC (NYU Cloud Bursting) — how jobs are run (rebuilt 2026-10-10 after three days of job failures)

## The one workflow
1. On the Mac: `bash 04_experiments/make_bundle.sh` → `04_experiments/visml_bundle.zip` (all code + all job scripts +
   a VERSION stamp; job scripts are generated from one template by `hpc/make_jobs.py`; syntax-checked on build).
2. Upload **only** `visml_bundle.zip` to `~/visml_hpc` (OOD Files).
3. In an OOD terminal: `unzip -oq ~/visml_hpc/visml_bundle.zip -d $SCRATCH/visml/code && bash $SCRATCH/visml/code/hpc/run_stage.sh <stage>`
   Stages: `check` (inputs + queue) · `interventions [RUN]` · `confirm` (the whole preregistered study, 10 chained jobs).
   Jobs are 4 generic templates (`cpu`, `b2`, `readers`, `iv_cpu`) taking the run name as first argument.

## Safety features (each fixes a failure we hit)
| Failure seen | Guard now |
|---|---|
| Jobs ran an older code version than intended (no features.npz) | single bundle; jobs run code only from `$SCRATCH/visml/code`; VERSION printed by every job |
| A crashed step reported COMPLETED (log filter hid the exit code) | `set -o pipefail` + FAIL flag + `exit $FAIL` in every job |
| Readers started with no input file | every job checks its inputs first (`need`), launcher preflight refuses to submit |
| Dependents waited forever after a failure | `--kill-on-invalid-dep=yes` |
| Cloud node died (munge failure, CANCELLED by 0) | `#SBATCH --requeue` + `--open-mode=append`; scripts resume/skip finished work |
| Untested model settings crashed 38B twice | settings changed only after a test; InternVL3-38B fixed at 8k context (what worked) |
| interventions needed features.npz | rebuilds the needed runs from the manifest if missing (seeded → identical; verified) |

Tested end to end on 2026-10-10 in a local simulation of the cluster (bundle unzipped into a fake $SCRATCH, real job
scripts, stand-ins for singularity/vLLM/sbatch): CPU stage with features missing ✓ (rebuilt runs reproduce the original
benchmarks exactly), reader stage ✓ (240/240 parsed), missing input → exit 3 ✓, crash → exit 1 ✓, launcher refuses on
missing input ✓ and submits with afterok + kill-on-invalid-dep ✓.

Old scripts and zips: `hpc/archive/` (do not upload).

## Job log
- 2026-10-07: GPU test 5493 (A100 OK) · prototype B2+Qwen 5577 · 4 readers 5584 · multi_v3 CPU 5637 → GPU 5638
- 2026-10-08: v4 submitted — CPU synthetic 5850, CPU magic 5851 → B2 5852 (syn) / 5853 (magic); readers fast 5854,
  big 5855 (syn), magic 5856. Results in $SCRATCH/visml/code/runs/v4_{synthetic,magic}/.
- 2026-10-08 (afternoon): 5850 CPU synthetic ✅ · 5852 B2 synthetic ✅ · 5855 Qwen-32B ✅ but InternVL3-38B crashed
  (16k context did not fit on A100-40GB → fixed: 8k + max 6 image tiles) · 5851 CPU magic killed by a cluster node
  failure (munge auth daemon died, "CANCELLED by 0"), so 5853/5856 never ran.
  Resubmitted: 5884 CPU magic → 5885 B2 magic, 5886 readers magic; 5887 InternVL3-38B rerun. 5854 fast readers running.
- 2026-10-09: v4 all done except InternVL3-38B (TypeError: max_dynamic_patch unsupported → removed; 8k context).
  Submitted interventions 6177 (iv-cpu, killed by cluster after 12 min, no log) → 6178 cancelled; resubmitted
  6231 iv-cpu → 6232 iv-readers. InternVL3-38B rerun 6179 running.
- 2026-10-10 (evening): new bundle workflow, version 2026-10-10_1535-89b0ddf2 — preflight 13/13 ok; submitted interventions iv-cpu 6373 → iv-readers 6374.

- 2026-10-11: bundle 2026-10-11_1517-50f6a71c (generic job templates; confirm stage; few-shot + numbers-only controls;
  seed tag "confirm"; confirm sizes 3,000 healthy / 900 per failure level). Simulated end to end: cpu (smoke, seed tag) ✓,
  b2 wrapper ✓ (script unchanged from v4), readers main/controls/core ✓ (96/96 controls parsed), iv_cpu ✓, iv readers ✓,
  launcher confirm stage: 10 jobs, correct afterok chains + kill-on-invalid-dep ✓, refuses to overwrite ✓.
- 2026-10-12: CONFIRMATORY STUDY submitted, bundle 2026-10-11_1517-50f6a71c — synthetic: cpu 6873 → b2 6875, readers 6877 (fast+controls), 6878 (32B+controls+38B), iv 6880 → 6881; real: cpu 6874 → b2 6876, readers 6879.
- 2026-10-12 22:00: all of 6873–6881 CANCELLED by root (uid 0) at the same second, incl. a still-pending job; partition nodes draining, scheduler job IDs reused (controller reset). System-side, not code; no outputs produced. Infrastructure re-run with identical seeds (not a deviation).
- 2026-10-12: CONFIRMATORY STUDY resubmitted, same bundle — synthetic: cpu 6913 → b2 6915, readers 6917 (fast+controls), 6918 (32B+controls+38B), iv 6920 → 6921; real: cpu 6914 → b2 6916, readers 6919.
- 2026-10-13: cpu 6913/6914 COMPLETED (FAIL=0). Readers 6917 CANCELLED by root while pending (0 s; cloud node start failure, same pattern as 6873). Resubmitted identically as 6999 (no dependency needed).
- 2026-10-13: 6999 FAILED on its last step only (InternVL3-8B on controls): few-shot prompt 27,062 tokens > max_model_len 16384. All other 6999 outputs complete. Fix: InternVL3-8B max_model_len 32768 (model maximum; image tiling unchanged). Re-run of that one model on controls (resumes; completed items kept).
- 2026-10-13: bundle 2026-10-13_2324-3353a935 adds make_positive.py + pos_cpu job + `run_stage.sh positive`
  (positive control: blatant failures on textbook plots, 50 per group × 2 sources; value probe: 250 confirmatory runs
  re-trained from the manifest, verified identical on Mac: max |Δ deploy_acc| = 0). Exploratory credibility check;
  severities + pass rule fixed in make_positive.py before any reader saw the items.
- 2026-10-13: positive control submitted (bundle 2026-10-13_2324-3353a935): pos-cpu 7289 → readers 7290 (positive, 5 models), 7291 (value probe, 5 models).
- 2026-10-14: bundle 2026-10-14_0222-07920578 adds rw_transfer.py, score_confirm.py, rw_cpu job, `run_stage.sh realworld` (Step 3). Data: realworld_data.zip (440 real models, 11 cases, 10 verified; generated locally by realworld.py).
- 2026-10-14: real-world check submitted: transfer rw-cpu 7315, readers 7316 (7B/8B/LLaVA), 7317 (32B/38B). All 3,960 images present.
- 2026-10-14: read-then-reason batch (bundle 2026-10-14_1827-d5d9e085): prep rtr-cpu 7428 -> extraction readers confirm_synthetic 7429 (7B/8B/LLaVA), 7430 (32B/38B); positive 7431, 7432; realworld 7433, 7434. Prep also runs the proportion-based transfer.
- 2026-10-14: 7428 held with 'user env retrieval failed' (node could not load the login environment; scontrol release not permitted). Batch 7428–7434 cancelled before anything ran; resubmitted unchanged: prep 7437 -> readers confirm_synthetic 7438,7439; positive 7440,7441; realworld 7442,7443.
- 2026-10-15: gaps batch (bundle 2026-10-15_0417-53642d6f): annotated designs ann-cpu 7495 -> readers 7496 (7B/8B/LLaVA), 7497 (32B/38B); multi-plot benchmark sets-cpu 7498 (both runs).
- 2026-10-28: instructor-review checks (bundle 2026-10-28_1349-88473a99): prep rev-cpu 13961 (2AFC/binary items + kernel two-sample tests) -> readers 13962 (7B/8B/LLaVA), 13963 (32B/38B); stronger image benchmark B3 13964 (confirm_synthetic), 13965 (confirm_magic).
