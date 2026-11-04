# What Can a Plot Reveal?

Detection vs. diagnosis in ML diagnostic plots, for human and AI readers.

Practitioners and AI agents check classifiers by looking at standard plots: loss curves, reliability diagrams,
confusion matrices, ROC curves. This project measures which model failures each plot can actually reveal, and whether
readers find what is there. We plant known failures in thousands of trained classifiers, render nine standard
diagnostic plots, bound the evidence each plot carries with statistical benchmarks, and compare AI readers against
those bounds. The results feed an interactive tool, the Plot Evidence Explorer.

## Study design

| Component | What it is |
|---|---|
| Models | Two-layer MLP binary classifiers on a synthetic tabular task and on MagicTelescope (OpenML); healthy models vary in data difficulty, size, width, regularization and learning rate |
| Failures | Overfitting, miscalibration (scaled logits), label noise, class imbalance, data leakage (label-encoding feature + clean deployment set), distribution shift (shifted test features); one severity knob each, three levels calibrated so a full-state detector reaches CV AUC 0.65 / 0.80 / 0.95 |
| Plots (9) | Loss curves, accuracy curves, reliability diagram, reliability diagram with bin sizes, confusion matrix (counts / row-normalized), ROC, precision–recall, validation-vs-test score histogram; rendered from exact mark tables |
| Benchmarks | B0 = full model state; B1 = exact plotted values (mark table); B2 = ResNet-18 on the rendered image; B3 = ConvNeXt-Tiny at native resolution (robustness check). Detection AUC and 7-way balanced diagnosis accuracy (prior-corrected), bootstrap CIs |
| Evidential map | Each failure × plot cell labelled identifiable (AUC ≥ .70, recall ≥ .50), detect-only (AUC ≥ .70, recall < .50) or blind; blind cells graded invisible (proven: plot identical to a healthy twin's), invisible (no difference found: kernel two-sample test) or weak |
| Readers | Qwen2.5-VL 7B/32B, InternVL3 8B/38B, LLaVA-OneVision 7B (vLLM, temperature 0) |
| Reader tasks | 7-way diagnosis; healthy vs. failing (detection AUC); two-plot forced choice (which model fails) |
| Controls | Prompt paraphrase, describe-first, few-shot (one labelled example per state), values-as-text, value-reading probe, positive control with blatant failures, evidence badges and companion views |
| Expert sessions | 4–6 formative think-aloud sessions on 12 plots and the Explorer, reported qualitatively (`human_study/PROTOCOL.md`) |
| Real-world check | 10 verified cases on 8 real datasets (documented practices: temporal shift, natural imbalance, oversampling, oversample-before-split leakage, overparameterized nets), each with a healthy control on the same data |

## Main results (confirmatory run, 38,400 models)

- 6 of 8 hypotheses supported; all directional tests survive Holm correction.
- Many plots detect a failure without diagnosing it (10 / 9 detect-only cells on synthetic / real data); every single
  plot diagnoses 0.14–0.45 worse (balanced accuracy) than the full model state.
- Drawing bin sizes on the reliability diagram raises class-imbalance detection by 0.25–0.33; row-normalizing the
  confusion matrix lowers diagnosis.
- AI readers diagnose at chance (0.14–0.17) against 0.44–0.46 for the image benchmark, also with few-shot examples,
  with the exact values as text, and on blatant failures; answers follow plot type (every loss curve → "overfitting").
- Four of five readers read curve values almost exactly (r = 0.95–1.00). Separating reading from reasoning (readers
  only read drawn values, a fixed rule diagnoses) raises balanced accuracy from 0.11–0.18 to 0.21–0.29 on the same
  models (ceiling 0.39): it reaches the ceiling on loss curves and confusion matrices, while reading positions and
  areas (reliability points, ROC, histograms) becomes the bottleneck.
- No single plot is enough, but the nine standard plots together carry almost everything the full model state
  does (0.82 / 0.79 vs 0.81 / 0.79); the best pair (reliability with bin sizes + a learning curve) reaches 0.77–0.78.
- Readers apply textbook cues ("gap between curves", "off the diagonal") without a sense of magnitude; with values
  printed on the plot and a rule trained on each reader's own readings, the best readers reach 0.68–0.69 on five plots
  (ceiling 0.76) — while their direct diagnosis of the same annotated plots stays at chance.
- On real models, the evidential map's detectability predictions hold in 72% of case–plot pairs (73% with counts as
  proportions).

### Robustness checks
- **What "blind" means.** Of 15 blind cells (synthetic) / 17 (real), 5 / 7 are provably invisible (the plotted values
  are identical to those of the failing model's healthy twin), 4 / 5 show no difference in a kernel two-sample test, and
  6 / 4 are weak (a difference exists, but no estimator reaches AUC 0.70). The stronger image benchmark B3 (mean AUC
  0.78 / 0.76) never exceeds the earlier benchmarks by more than 0.05; one cell becomes detectable (distribution shift
  on precision–recall, real data, AUC 0.72). Labels are stable over cutoffs AUC 0.60–0.80 × recall 0.30–0.70 for all
  synthetic blind cells and 14 of 17 real ones.
- **Is the 7-way task too hard?** No. On simpler tasks the readers stay near chance while the benchmark does not:
  healthy vs. failing AUC 0.50–0.54 (benchmark 0.80 on the same items); which of two plots fails 0.51–0.58 (benchmark
  0.79). The best reader succeeds mainly on textbook cases (overfitting on loss curves, AUC 0.86).
- **What a redesign changes.** Redesigns add evidence, the readers do not use it: drawing bin sizes turns class
  imbalance from blind into identifiable (AUC 0.64 → 0.97), readers 0.50 → 0.50; a companion plot adds 0.10–0.25
  balanced accuracy for the benchmark, none for the readers.

Full, dated results: [`04_experiments/FINDINGS_LOG.md`](04_experiments/FINDINGS_LOG.md); deviations from the analysis
plan: [`04_experiments/DEVIATIONS.md`](04_experiments/DEVIATIONS.md); per-run reports in `04_experiments/code/runs/`
(e.g. `confirm_synthetic/CONFIRM_RESULTS.md`, `REVIEW_RESULTS.md`, `REDESIGNS.md`, `realworld/REALWORLD_RESULTS.md`).

## Plot Evidence Explorer

Open `04_experiments/explorer/index.html` in a browser (no server needed): the evidential map, a cell view with example
plots, a "my plots" view (which failures your plots can catch, and which plot to add), a redesigns view, and the AI
readers' answers next to the benchmark.

## Repository layout

```
04_experiments/
  code/
    plotreveal/          world (models + failures), plots (mark tables, rendering), benchmarks, seeds
    run_v4.py            main study: calibrate severities, train, render, benchmarks, reader item sets
    interventions.py     evidence badges and companion views
    make_positive.py     positive control + value probe
    realworld.py         real-world cases;  rw_transfer.py  map transfer to real models
    score_*.py           scoring (confirmatory tests, positive control, real world, interventions, robustness)
    holm_confirm.py      Holm correction within hypothesis families
    certify_blind.py     healthy-twin test for blind cells;  blind_sensitivity.py  cutoff sensitivity
    review_prepare.py    kernel two-sample tests, binary and 2AFC reader items;  redesigns.py  redesign summary
    export_explorer.py   data for the D3 explorer
    runs/                results (reader responses, benchmark results, reports); images are not stored
  hpc/                   Slurm job system (generated job scripts, launcher, vLLM reader, image benchmark)
  explorer/              Plot Evidence Explorer (D3)
  figures/               paper figures (make_figures.py)
  human_study/           expert-session protocol and plots
  FINDINGS_LOG.md        dated research log;  DEVIATIONS.md  deviations from the analysis plan
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
it, and run `hpc/run_stage.sh <stage>` (stages in `04_experiments/hpc/README.md`). Scoring runs locally from the downloaded results, e.g.
`python score_confirm.py runs/confirm_synthetic runs/confirm_magic`.
