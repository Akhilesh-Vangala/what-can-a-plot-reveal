# What Can a Plot Reveal?

Detection vs. diagnosis in ML diagnostic plots, for human and AI readers.

Practitioners and AI agents check classifiers by looking at standard plots: loss curves, reliability diagrams,
confusion matrices, ROC curves. This project measures which model failures each plot can actually reveal, and whether
readers find what is there. We plant known failures in thousands of trained classifiers, render nine standard
diagnostic plots, bound the evidence each plot carries with statistical benchmarks, and compare AI readers against
those bounds.

## Study design

| Component | What it is |
|---|---|
| Models | Two-layer MLP binary classifiers on a synthetic tabular task and on MagicTelescope (OpenML); healthy models vary in data difficulty, size, width, regularization and learning rate |
| Failures | Overfitting, miscalibration (scaled logits), label noise, class imbalance, data leakage (label-encoding feature + clean deployment set), distribution shift (shifted test features); one severity knob each, three levels calibrated so a full-state detector reaches CV AUC 0.65 / 0.80 / 0.95 |
| Plots (9) | Loss curves, accuracy curves, reliability diagram, reliability diagram with bin sizes, confusion matrix (counts / row-normalized), ROC, precision–recall, validation-vs-test score histogram; rendered from exact mark tables |
| Benchmarks | B0 = full model state; B1 = exact plotted values (mark table); B2 = ResNet-18 on the rendered image. Detection AUC and 7-way balanced diagnosis accuracy (prior-corrected), bootstrap CIs |
| Evidential map | Each failure × plot cell labelled identifiable (AUC ≥ .70, recall ≥ .50), detect-only (AUC ≥ .70, recall < .50) or blind |
| Readers | Qwen2.5-VL 7B/32B, InternVL3 8B/38B, LLaVA-OneVision 7B (vLLM, temperature 0) |
| Controls | Prompt paraphrase, describe-first, few-shot (one labelled example per state), values-as-text, value-reading probe, positive control with blatant failures, evidence badges and companion views |
| Real-world check | 10 verified cases on 8 real datasets (documented practices: temporal shift, natural imbalance, oversampling, oversample-before-split leakage, overparameterized nets), each with a healthy control on the same data |

## Main results (confirmatory run, 38,400 models)

- 6 of 8 hypotheses supported; all directional tests survive Holm correction.
- Many plots detect a failure without diagnosing it (10 / 9 detect-only cells on synthetic / real data); every single
  plot diagnoses 0.14–0.45 worse (balanced accuracy) than the full model state.
- Drawing bin sizes on the reliability diagram raises class-imbalance detection by 0.25–0.33; row-normalizing the
  confusion matrix lowers diagnosis.
- AI readers diagnose at chance (0.14–0.17) against 0.44–0.46 for the image benchmark, also with few-shot examples,
  with the exact values as text, and on blatant failures; answers follow plot type (every loss curve → "overfitting").
- Four of five readers read curve values almost exactly (r = 0.95–1.00); a read-then-reason pipeline (reader extracts values, a
  simple statistical rule diagnoses) roughly doubles their accuracy (0.28–0.36).
- On real models, the evidential map's detectability predictions hold in 72% of case–plot pairs.

Full, dated results: [`04_experiments/FINDINGS_LOG.md`](04_experiments/FINDINGS_LOG.md) and the per-run reports in
`04_experiments/code/runs/*/` (e.g. `confirm_synthetic/CONFIRM_RESULTS.md`, `positive/POSITIVE_RESULTS.md`,
`realworld/REALWORLD_RESULTS.md`).

## Repository layout

```
04_experiments/
  code/
    plotreveal/          world (models + failures), plots (mark tables, rendering), benchmarks, seeds
    run_v4.py            main study: calibrate severities, train, render, benchmarks, reader item sets
    interventions.py     evidence badges and companion views
    make_positive.py     positive control + value probe
    realworld.py         real-world cases;  rw_transfer.py  map transfer to real models
    score_*.py           scoring (confirmatory tests, positive control, real world, interventions)
    holm_confirm.py      Holm correction within hypothesis families
    export_explorer.py   data for the D3 explorer
    runs/                results (reader responses, benchmark results, reports); images are not stored
  hpc/                   Slurm job system (generated job scripts, launcher, vLLM reader, image benchmark)
  explorer/              interactive D3 explorer of the evidential map
  make_bundle.sh         builds the cluster upload (code + jobs + version stamp)
```

## Reproducing

Every model and plot is regenerated exactly from seeds (`plotreveal/seeds.py`); plot images, feature arrays and
manifests are therefore not stored here.

```bash
pip install -r 04_experiments/code/requirements.txt
cd 04_experiments/code
python run_v4.py --source synthetic --name confirm_synthetic --size confirm --seed-tag confirm   # CPU, hours
python realworld.py                                                                              # ~2 min
```

Readers and the image benchmark need a GPU; on a Slurm cluster, build the bundle with `bash make_bundle.sh`, upload
it, and run `hpc/run_stage.sh <stage>` (stages: `confirm`, `interventions`, `positive`, `realworld`; see
`04_experiments/hpc/README.md`). Scoring runs locally from the downloaded results, e.g.
`python score_confirm.py runs/confirm_synthetic runs/confirm_magic`.
