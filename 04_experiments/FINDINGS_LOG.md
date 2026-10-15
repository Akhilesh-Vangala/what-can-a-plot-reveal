# Findings log (dated; prototype numbers — not final results)

Each entry: what ran, the numbers, how sure we are, and what it does NOT show yet.
Run folders: `04_experiments/code/runs/<name>/` (report.md, score_*.md, readers_summary.md).

## 2026-10-07 — Overfitting × learning curves, prototype v2 (`runs/overfit_proto_v2`)
**Setup.** 1,500 healthy + 3×600 overfit small MLPs (synthetic tabular data); 60/20/20 splits; severities calibrated by
B0 detection AUC (targets 0.65 / 0.80 / 0.95 → s = 0.070 / 0.117 / 0.270). Plots: loss curves, accuracy curves.

**Benchmarks (held-out detection AUC).**
| level | B0 | B1 loss | B2 loss | B1 acc | B2 acc |
|---|---|---|---|---|---|
| 0.65 | 0.691 | 0.690 | 0.693 | 0.646 | 0.549* |
| 0.80 | 0.810 | 0.820 | 0.763 | 0.781 | 0.713 |
| 0.95 | 0.956 | 0.952 | 0.956 | 0.907 | 0.919 |
B2 = ImageNet-pretrained ResNet-18 at 224 px (GPU). *B2 acc at 0.65 is unstable (gets worse with more data) → not trusted.
- Calibration hits targets; ordering B0 ≥ B1 ≥ B2 holds within noise; B0/B1 saturate.
- **Encoding loss (replicated at all 3 levels):** accuracy curves carry less overfitting evidence than loss curves.
- **Possible rendering loss** for accuracy curves at low/medium severity — needs a more robust B2 (seeds / best-of-K).

**Reader: Qwen2.5-VL-7B-Instruct** (vLLM, temperature 0, 1,386 items, 0 unparseable).
- Diagnosis AUC 0.51 / 0.54 / 0.63 (loss) and 0.54 / 0.57 / 0.60 (acc) vs B2 0.69–0.96 → large reader loss.
- **Over-claiming:** P(overfitting) on *healthy* plots 0.81 (loss) / 0.78 (acc), ≈ same as on overfit plots;
  "cannot tell" used 0/1,320 times; prior-only (no image) P(overfitting) = 0.20 → the plot, not the prompt, drives it.
- Lineups: 7/48 hits (chance 5%; benchmark 18–74%); null-lineup false alarms 16/16; position 17 chosen 24/64 times.
- **Not yet shown:** whether larger/other models behave the same; formal over-claiming vs B2 posterior; whether a
  human would also call normal train/val gaps "overfitting"; anything beyond one failure type.

## 2026-10-07 (later) — Reader controls + 2nd model (job 5584, partial: Qwen-7B, LLaVA-OV-7B)
**Value probe** (100 single plots per spec: 50 healthy + 50 strongest-level overfit; "read these numbers off the plot").
| reader | spec | median abs error (val last) | corr(read val-rise, true) | AUC from its OWN read numbers | AUC from true numbers | AUC of its diagnosis |
|---|---|---|---|---|---|---|
| Qwen2.5-VL-7B | loss | 0.010 | 0.96 | **0.83** | 0.85 | **0.62** |
| Qwen2.5-VL-7B | acc | 0.006 | 0.49 | **0.80** | 0.83 | **0.56** |
| LLaVA-OV-7B | loss | 0.059 | −0.18 | 0.45 | 0.85 | 0.50 |
| LLaVA-OV-7B | acc | 0.031 | 0.22 | 0.49 | 0.83 | 0.60 |
(score from read numbers = signed val-rise + |val − train| gap; chosen post hoc → must be pre-registered before the full study.)
- **Qwen-7B "sees but does not use":** it reads the evidence almost perfectly (its own numbers separate overfit from
  healthy at AUC 0.83 ≈ truth 0.85) but its diagnosis is near chance (0.62 / 0.56) → reasoning loss, not perception loss.
- **LLaVA-OV fails earlier, at perception:** cannot read the values (corr ≈ 0) → perception loss.
- Two readers, two different failure points → the B2 → reader gap splits into perception vs reasoning (gap 4.1 works).

**Shuffled-lineup control** (same 20 plots, new positions; 64 pairs).
- Qwen-7B: same plot 10/64, same position 10/64 (chance ≈ 3) — weak either way; still panel 17 in 45/128 picks.
- LLaVA-OV: same plot 6/64, **same position 34/64** → its lineup answers are mostly position preference (16 or 1).
- Both: null-lineup false alarms 16/16; "cannot tell" 0% on all 1,320 diagnosis items.
- LLaVA-OV P(overfitting | healthy) 0.30–0.54 (less over-claiming than Qwen's 0.78–0.81) but diagnosis AUC ≈ 0.50.

## 2026-10-07 (evening) — Qwen2.5-VL-32B-AWQ added (job 5584 complete; InternVL3-8B finished but not yet scored)
| metric | Qwen-32B | Qwen-7B | LLaVA-OV-7B |
|---|---|---|---|
| diagnosis AUC loss L0/L1/L2 | 0.52 / 0.55 / 0.63 | 0.51 / 0.54 / 0.63 | 0.50 / 0.51 / 0.48 |
| diagnosis AUC acc L0/L1/L2 | 0.54 / 0.58 / 0.68 | 0.54 / 0.57 / 0.60 | 0.55 / 0.59 / 0.59 |
| AUC from its own read numbers (L2, loss / acc) | 0.78 / 0.77 | 0.83 / 0.80 | 0.45 / 0.49 |
| probe corr(read rise, true) loss | 0.96 | 0.96 | −0.18 |
| P(overfitting) on healthy plots (loss / acc) | 0.56 / 0.38 | 0.81 / 0.78 | 0.30 / 0.54 |
| "cannot tell" rate (loss / acc) | 0.93 / 1.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| lineup hits / null false alarms | 2/48 · 16/16 | 7/48 · 16/16 | 2/48 · 16/16 |
| shuffled control: same plot / same position | 13/64 / 5/64 | 10/64 / 10/64 | 6/64 / 34/64 |
- **Scale does not fix it:** 32B reads the plot as well as 7B (corr 0.96) and its own numbers still separate the classes
  (0.78) but its diagnosis is again near chance (0.63) → "sees but does not use" replicates across model size.
- **Opposite calibration failure:** 32B says "cannot tell" on ~all plots (0.98 healthy, 0.81 strongest overfit — barely
  tracks evidence) while still naming overfitting as most likely; boilerplate reason ("other issues cannot be ruled out").
  7B never hedges; 32B always hedges — neither hedge tracks how much evidence the plot actually contains.
- **No reader believes a healthy plot is healthy:** P(healthy | healthy plot) ≈ 0.12 (32B), < 0.2 (7B).
- Null-lineup false alarms 16/16 for all three readers.

## 2026-10-07 (evening) — InternVL3-8B added (all 4 readers of job 5584 scored)
- Best perception of the four: probe corr 0.98 (loss) / 0.72 (acc), median error 0.003; AUC from its own read numbers
  0.85 / 0.82 (= the true numbers' 0.85 / 0.83).
- Diagnosis: names **overfitting as the top cause for 420/420 loss plots** (300 healthy + 120 strongest overfit);
  P(overfit) 0.74 on healthy vs 0.68 on overfit → AUC 0.29–0.41 (loss), ~0.48 (acc). The below-chance AUC comes from
  small variation around a constant answer, not a real inversion — report it as "constant answer", not "inverted".
- Reason text is generic ("training loss decreasing faster than validation loss, indicating overfitting") for every plot.
- **Across 4 readers:** perception ranges from none (LLaVA) to near-perfect (InternVL, Qwen-7B/32B), but no reader's
  diagnosis uses what it perceives (best diagnosis AUC 0.68 vs own-read 0.77–0.85 vs B2 0.92–0.96 at the strongest level).
  All four: 16/16 null-lineup false alarms.

## 2026-10-08 — Multi-failure run multi_v3 (4 failures × 6 plots; B0/B1 CPU job 5637, B2 + 5 readers GPU job 5638)
Calibrated severities (B0 AUC 0.65/0.80/0.95): overfitting 0.055/0.089/0.315 · miscalibration 0.024/0.034/0.080 ·
label noise 0.010/0.031/0.057 · class imbalance 0.068/0.146/0.233. Full tables: runs/multi_v3/{report,evidential_map,readers_multi}.md.

**Evidential map, strongest level (best of B1/B2; det = AUC, diag = 5-way recall, chance 0.20).**
| failure | lc_loss | lc_acc | rel | cm | roc | pr |
|---|---|---|---|---|---|---|
| overfitting | 0.97/0.86 ID | 0.96/0.76 ID | 0.93/0.73 ID | 0.68/0.09 – | 0.73/0.19 detect-only | 0.70/0.16 detect-only |
| miscalibration | 0.74/0.22 detect-only | 0.53/0.11 – | 0.83/0.19 **detect-only** | 0.53/0.00 – | 0.50/0.04 – | 0.54/0.00 – |
| label noise | 0.85/0.51 ID | 0.79/0.13 detect-only | 0.91/0.74 ID | 0.68/0.03 – | 0.92/0.67 ID | 0.89/0.76 ID |
| class imbalance | 0.53/0.00 – | 0.59/0.00 – | 0.90/0.55 ID* | 0.91/0.58 ID | 0.83/0.42 detect-only | 0.91/0.63 ID |
- 5-way diagnosis accuracy, strongest level: B0 0.76; single plots 0.49–0.63 (B1), 0.49–0.56 (B2). **No single plot
  diagnoses well even where it detects at ≥ 0.9 → central claim (detection ≠ diagnosis) supported.**
- **Miscalibration is detect-only even on the reliability diagram** (det 0.83, recall 0.19; 47% read as healthy) and
  invisible on CM/ROC/PR/accuracy curves (≈ 0.5, as theory says). Large encoding loss: B0 0.99 vs reliability 0.83.
- Class imbalance invisible on learning curves (0.48–0.59); overfitting detect-only on ROC/PR.
- **Rendering loss (B1 → B2)** small for most cells; large for imbalance on ROC (0.83 → 0.51: B1 uses fine step
  granularity of the curve that is invisible at image scale).
- ⚠️ **Benchmark bug found:** the reliability-diagram mark table included per-bin COUNTS, which are not drawn in the
  default design → B1(rel) had information the image lacks (explains imbalance on rel: B1 0.90 vs B2 0.51). Fixed in
  plots.py (counts only enter B1 if a "show_counts" design draws them); B1(rel) must be recomputed in the next run.
  Cells marked * depend on it. All other plots' mark tables were checked: they contain only drawn values.

**Five readers (levels 1–2, 6 plots, ~6,600 items each; argmax over 7 causes; chance 0.20).**
| reader | 5-way acc | P(true) | mean top prob | P(healthy\|healthy) | cannot-tell | most common answer |
|---|---|---|---|---|---|---|
| InternVL3-38B-AWQ | 0.27 | 0.22 | 0.63 | 0.50 | 0.33 | healthy (80%) |
| Qwen2.5-VL-7B | 0.24 | 0.16 | 0.41 | 0.12 | 0.03 | healthy / overfitting |
| Qwen2.5-VL-32B-AWQ | 0.21 | 0.19 | 0.56 | 0.30 | 0.67 | overfitting / healthy |
| LLaVA-OV-7B | 0.20 | 0.17 | 0.37 | 0.16 | 0.00 | overfitting / label noise |
| InternVL3-8B | 0.18 | 0.15 | 0.53 | 0.08 | 0.00 | overfitting / dist. shift |
Benchmark 5-way accuracy (B2) on the same plots: 0.39–0.50 (L1), 0.49–0.56 (L2). **All readers ≈ chance.**
- **Readers answer the PLOT TYPE, not the plot:** on HEALTHY plots, 4/5 readers say "overfitting" for 100% of learning
  curves; 3/5 say "miscalibration" for 58–100% of reliability diagrams; InternVL3-8B says "class imbalance" for 99%
  of confusion matrices. Apparent hits (e.g. 100% "miscalibration" recall on reliability diagrams) are this stereotype.
- Scale: InternVL3-38B is the only reader that defaults to "healthy" (under-claiming) and the best (0.27) — still far
  below the benchmark.

**Resolution experiment (v2 overfitting items, same plots at 200 vs 100 dpi).**
- LLaVA perception improves (probe corr 0.00 → 0.50 on loss; own-read AUC 0.45 → 0.59) but its diagnosis does not
  (0.50 → 0.50). Qwen-7B already reads perfectly (0.99) — no change. → Image resolution is a rendering-level factor for
  weak readers' perception, but not the bottleneck for diagnosis (reasoning is).

## 2026-10-08 — CORRECTION to the multi_v3 numbers above (audit before v4)
- 5-way accuracy was reported against chance 0.20, but classes are imbalanced (healthy 3× each failure; reader set
  27% healthy), so "always healthy" scores 0.43 (benchmark) / 0.27 (readers). **Use balanced accuracy (mean recall).**
- B1 is now the validation-best of {mark-table model, image model} (rendering is a function of the marks), mirroring
  B0 = best of {state, plots}. Ordering B0 ≥ B1* holds in 72/72 detection cells (0 violations > 0.03).
- Corrected balanced 5-way diagnosis (chance 0.20): B0 0.26 / 0.43 / **0.73**; single plots L2: lc_loss 0.47,
  lc_acc 0.37, rel 0.55, cm 0.31, roc 0.41, pr 0.44 → detection ≠ diagnosis holds more strongly.
- Corrected reader balanced accuracy (levels 1–2): InternVL3-38B 0.22, Qwen-32B 0.21, Qwen-7B 0.20, LLaVA 0.20,
  InternVL3-8B 0.16 — **all at chance**. Recall profile shows each reader's default answer (38B: healthy 0.84).
- The earlier "acc 0.18–0.27" and "best single plot 0.63" figures are superseded by these.

## 2026-10-08 — Engineering notes (v4 prep)
- interventions.py: evidence badges + companion views generated from the measured evidential map; tested on a smoke run
  (companion choice is data-driven, e.g. ROC → reliability diagram to resolve miscalibration vs healthy).
- Fixed a macOS crash: PyTorch and scikit-learn each loaded an OpenMP runtime and gradient boosting segfaulted; torch is
  now imported lazily (only for the CNN). The cluster run of v4 uses the earlier eager import (Linux, no crash seen in v3).

## 2026-10-09 — v4 (6 failures × 9 plots; synthetic + real MagicTelescope; all audit fixes)
Jobs: 5850/5884 CPU, 5852/5885 B2, 5854/5855/5886 readers (InternVL3-38B failed twice on config → rerun pending).
Full tables: runs/v4_{synthetic,magic}/{report,evidential_map,readers}.md, readers_summary.json.

**Diagnosis, strongest level, balanced accuracy (chance 1/7 = 0.14).** B0: 0.77 (syn) / 0.79 (real).
Single plots (B1*): syn 0.32–0.59, real 0.38–0.68 → **detection ≠ diagnosis replicates on real data.**

**Evidential map highlights (detection AUC / diagnosis recall, strongest level; syn | real).**
- Miscalibration: detect-only even on the reliability diagram (0.82/0.25 | 0.78/0.32); invisible on CM, ROC, PR,
  accuracy curves (0.44–0.52 | 0.46–0.52) — as theory predicts.
- Learning curves blind to class imbalance (0.50 | 0.52) and distribution shift (0.51 | 0.53).
- Overfitting invisible on CM/ROC/PR (0.50–0.66 | 0.50–0.58); identifiable on learning curves and reliability.
- **Design variant, reliability with bin sizes (rel_n) vs standard (rel):** imbalance detection 0.59 → 0.96 (syn),
  0.70 → 0.96 (real); balanced diagnosis 0.51 → 0.59 (syn), 0.56 → 0.68 (real). Drawing bin sizes is the single
  most informative design change measured. (B1 for rel no longer contains undrawn counts.)
- **Design variant, row-normalised CM (cm_norm) vs counts (cm):** diagnosis 0.42 → 0.33 (syn), 0.47 → 0.43 (real);
  imbalance recall 0.87 → 0.62 (syn). Normalising hides the base rate → less evidence.
- Rendering loss (B1 → B2) < 0.05 in every cell: the drawn image keeps what the marks carry at this resolution.

**Readers (balanced accuracy, chance 0.14).** Synthetic (13,509 items each): Qwen-7B 0.14, Qwen-32B 0.15,
InternVL3-8B 0.15, LLaVA 0.14. Real data (4,500 core items): Qwen-7B 0.14, Qwen-32B 0.14, InternVL3-8B 0.14.
- **All readers at chance on both datasets.**
- Stereotype: on healthy loss curves every reader answers "overfitting" 100% (91% for Qwen-7B with prompt variants).
- Over-claiming index (reader top-prob − benchmark posterior on that state, same item): 0.17–0.46; confident
  (≥ 0.7) where the benchmark is ambiguous (< 0.5) on 3–34% of items.
- "Cannot tell" does not track ambiguity: hedging AUC 0.32–0.65 (≈ 0.5); Qwen-32B hedges on 56–59% of items.
- **Prompt controls do not help:** paraphrase + describe-first: Qwen-7B 0.15, InternVL3-8B 0.14 (over-claiming
  rises to 0.42–0.57) → the failure is not a prompting artefact.

## 2026-10-10 — Design interventions, benchmark half (iv-cpu 6373; v4 synthetic, strongest level, 7-way balanced acc)
Companion chosen per plot from the measured confusions (the plot that best separates the plot's most-confused pair).
| plot | alone | with companion | companion |
|---|---|---|---|
| loss curves | 0.437 | 0.626 | confusion matrix |
| reliability | 0.507 | 0.586 | val-vs-test score histogram |
| confusion matrix | 0.413 | 0.555 | val-vs-test score histogram |
| ROC | 0.404 | 0.570 | val-vs-test score histogram |
| val-vs-test scores | 0.490 | 0.658 | accuracy curves |
- Companion views recover +0.08 to +0.19 balanced accuracy (encoding loss partly recovered by an informational fix).
- The validation-vs-test score histogram is the best partner for 3 of 5 plots → design recommendation.
- Single-plot values reproduce the v4 report (features rebuilt from the seeded manifest; exact reproduction verified
  locally on a smoke run). Reader half (badge / companion / both, 6,000 items × 4 readers) running as job 6374.

## 2026-10-11 — Design interventions, reader half (iv-readers 6374; 4 readers × 6,000 items; v4 synthetic)
Same held-out runs (100 healthy + 50 per failure, strongest level) × 5 core plots; plain = main v4 answers.
Balanced accuracy (chance 0.14) — Δ vs plain with paired bootstrap 95% CI over runs:
| reader | plain | badge | companion | badge+companion |
|---|---|---|---|---|
| InternVL3-8B | 0.156 | 0.147 (−0.008, ns) | 0.141 (−0.015) | 0.141 (−0.015) |
| Qwen2.5-VL-32B | 0.146 | 0.145 (ns) | 0.143 (ns) | 0.135 (−0.011) |
| Qwen2.5-VL-7B | 0.144 | 0.154 (+0.010) | 0.135 (−0.009) | 0.148 (ns) |
| LLaVA-OV-7B | 0.139 | 0.136 (ns) | 0.142 (ns) | 0.127 (−0.013) |
- **No intervention makes readers more accurate** (all ≈ chance, |Δ| ≤ 0.015), although the same companions raise the
  BENCHMARK by +0.08 to +0.19 → the added evidence is in the image; readers do not use it.
- Over-claiming index: badge+companion lowers it for InternVL3-8B (−0.11) and Qwen-7B (−0.05); companion lowers it for
  Qwen-32B (−0.28, but cannot-tell rises to 80%); LLaVA's rises (+0.17 to +0.19). Not a uniform improvement.
- Badges raise "cannot tell" (0–43% → 19–63%): they induce hedging, not better diagnosis.
- **Cue-following, not evidence-following:** badges remove the "healthy loss curve → overfitting" answer for Qwen-32B
  and LLaVA (100% → 0%), but the answer moves to whatever the cue suggests — Qwen-32B + badge calls healthy loss curves
  "label noise" 56% (a state the badge lists as identifiable from loss curves); with a confusion-matrix companion it
  says "class imbalance" 96% (the confusion-matrix stereotype); LLaVA + badge abstains (all-zero probabilities) on 68%.
- H7 (preregistration draft) as stated is NOT supported for AI readers: badges do not reliably reduce over-claiming, and
  companions do not raise reader accuracy even where the benchmark improves. Reported as a negative result.
- Data note: LLaVA returned all-zero probability vectors for many badge items (counted as no answer; n reported).
- InternVL3-38B (v4 core set, 4,500 items; job 6179): balanced acc 0.16 (chance 0.14); over-claim 0.43; confident where
  the benchmark is ambiguous 40%; cannot-tell 22% (hedging AUC 0.60). Stereotypes: reliability → "miscalibration" 100%
  and val-vs-test histogram → "miscalibration" 99% on healthy plots; healthy loss curves → "healthy" 78%. The largest
  reader is no better than chance either.

## 2026-10-11 — Pre-confirmation optimisation check (v4 data)
- Calibration accuracy: achieved B0 0.66 / 0.81 / 0.96 (synthetic), 0.67 / 0.84 / 0.97 (real) vs targets 0.65 / 0.80 / 0.95.
- Benchmark saturation: median gain 50% → 100% training data +0.008; 20/180 (syn) and 22/180 (real) cells gain > 0.03
  (largest: cm_norm imbalance L1 real +0.31). → Confirmatory run uses 1.5× data (3,000 healthy / 900 per failure level)
  and flags unsaturated cells. B2 diagnosis saturation: median +0.009.
- Added reader controls for the confirmatory run: few-shot (one labelled example per state) and numbers-only (exact
  plotted values as text) — fairest chance for readers and a perception-vs-reasoning test across all failures.

## 2026-10-13 — CONFIRMATORY STUDY scored (confirm_synthetic + confirm_magic; bundle 2026-10-11_1517-50f6a71c)
Full output: code/runs/confirm_synthetic/CONFIRM_RESULTS.md (score_confirm.py). Fresh seed namespace verified
(deploy_acc differs on 1,981/2,000 runs vs pilot; all reader replies dated 2026-10-13). InternVL3-8B controls added
(job 7114 COMPLETED, 2,400/2,400, after max_model_len fix) — H7 now scored on all 3 control readers.
- H1 blind spots: NOT SUPPORTED as stated, 6/8 (syn) and 5/8 (real) cells blind. Replicated deviation: accuracy curves
  reveal class imbalance (lc_acc AUC 0.59 syn, 0.73 real; real lc_loss 0.58). Miscalibration invisible on ROC/PR/CM/acc
  and shift invisible on curves — confirmed on both sources.
- H2 detection ≠ diagnosis: SUPPORTED, 10 (syn) / 9 (real) detect-only cells; B0* beats every plot by 0.13–0.45.
- H3 reliability encoding loss: SUPPORTED (+0.17 syn, +0.16 real).
- H4 design variants: SUPPORTED both sources (rel_n imbalance detection +0.33 / +0.25; diagnosis rel_n +0.10 / +0.14;
  cm_norm −0.07 / −0.05).
- H5 readers below B2: SUPPORTED, all 5 readers (syn) and 3 (real): readers 0.14–0.17 vs B2 0.44–0.46.
  Abstention ("cannot_tell"): Qwen-7B 22%, LLaVA 13%, InternVL-8B 8%, 38B 6%, Qwen-32B 0%; conclusions unchanged if
  abstentions count as wrong.
- H6 stereotype: SUPPORTED, all readers both sources (healthy loss curve → "overfitting" 100%; 38B 22%).
- H7 controls: SUPPORTED, all 3 readers (few-shot 0.14–0.18, numbers-only 0.14–0.17 vs B2 0.44–0.50) → gap is reasoning, not
  perception. Qwen-32B abstains 30% under few-shot; InternVL-8B abstains 66% on
  numbers-only (0.05 if abstentions count as wrong) — given exact values, it mostly declines to answer.
- H8 interventions: NOT SUPPORTED as a whole. (a) companions raise benchmark for all 5 plots (+0.10 to +0.25) ✓;
  (b) readers unchanged for 3/4 readers, Qwen-32B slightly WORSE with companions (90% CI [−0.030, −0.010]);
  (c) badge cue-following in InternVL-8B and Qwen-7B only; Qwen-32B and LLaVA name badge-listed states LESS often.
- Readers are template-like: Qwen-32B gave 384 distinct replies across 13,509 items.
- EXPLORATORY (2026-10-13): reader DETECTION from existing answers (1 − P(healthy); strongest level; core plots):
  reader AUC 0.47–0.61 (best: 38B on loss curves 0.61) vs benchmark B1 0.72–0.89 (mean over failures), both sources.
  Readers do not even detect that something is wrong — the 7-way chance result does not hide a detection ability.
  (Pilot value probe, overfit_proto_v2: readers READ curve values accurately, corr 0.96–0.98 → perception is fine.)

## 2026-10-14 — Positive control + value probe (exploratory; jobs 7289–7291; pass rule fixed in make_positive.py before reading)
Positive control (blatant failures on their textbook plot, 50 per group × 2 sources, 5 readers):
- Pass rule (AUC ≥ 0.80 AND named ≥ 50% on ≥ 4 of 5 primary failures): NO reader passes on either source (best 1/5).
- Answers follow plot type, not content: every loss curve → "overfitting" (healthy false alarm 100% for 4/5 readers;
  38B 8–22%), every reliability diagram → "miscalibration", perfect ROC (leakage, AUC 1.00) → "miscalibration" or
  abstain; class-imbalance confusion matrices (874/15/43/68) → "healthy". 38B detects overfitting (AUC 0.95/0.89) and
  Qwen-32B miscalibration (0.78/0.88) from probabilities, but the argmax answer is the stereotype.
- Raw replies show perception of the signature without the diagnosis, e.g. InternVL-8B on leakage ROC: "The plot shows
  a perfect classifier, which is unlikely in practice" → cannot_tell. Not a setup bug (images seen, items correct).
Value probe (250 confirmatory models, loss/acc curves):
- 4/5 readers read values almost exactly: corr 0.95–1.00 for last/best values on loss curves, median error ≈ 0.003–0.02.
  LLaVA is the exception (corr 0.24–0.87).
- Overfitting detected from the readers' OWN readings (val_last − val_best): AUC 0.83–0.92 vs 0.94 from true values —
  yet the same readers call every healthy loss curve "overfitting" when asked to diagnose.
→ Strongest evidence so far that the reader gap is in diagnosis/reasoning, not perception or subtle signals.

## 2026-10-14 — Holm correction + Step 3 (real-world check) set up
- Holm (holm_confirm.py → runs/confirm_synthetic/HOLM.md): every supported directional test survives within its
  family on both sources (H2 9/9, H3 1/1, H4 3/3, H5 5/5 & 3/3, H7a 3/3, H7b 3/3; largest p = 4e-6).
- Real-world set (realworld.py → runs/realworld): 11 cases on 8 real OpenML datasets, each a documented real practice
  with a healthy control on the same data, 20 seeds each (440 models). 10/11 verified objectively; feature-selection
  leakage on madelon NOT verified (AUC inflation 0.012 < 0.03) → excluded. Real models carry co-occurring conditions
  (time shift and overfitting also miscalibrate; bank/kc1 data are naturally imbalanced), so each model's present
  conditions are measured with fixed thresholds (debiased ECE for small test sets) and readers are credited for
  naming any present condition. Visual check: imbalance, oversampling miscalibration, leakage ROC and overfitting
  signatures are textbook; bank time shift drops accuracy 22 points but barely changes the score histogram.
- HPC: transfer 7315, readers 7316/7317 submitted. Scorers tested with oracle (100%) and random (chance) readers.

## 2026-10-14 — Step 3 results: real-world check (10 verified cases, 400 real models, 5 readers × 3,600 plots)
Transfer of the evidential map (rw_transfer.py; confirmatory B1 rebuilt, applied unchanged to real plots):
- Map's "detectable vs blind" agrees with real detectability (AUC ≥ 0.70) in 72% of 90 case-plot pairs
  (chance agreement given the margins ≈ 53%): 43 detectable/detectable, 22 blind/blind, 14 map-detectable but real
  not, 11 map-blind but real detectable.
- Most disagreements are explainable: real mechanisms are compound (oversampling for miscalibration also changes the
  loss curves, AUC 1.00 on kc1; real imbalance shows on loss curves, consistent with the H1 deviation), and the
  score-histogram detector inverts on real shift/miscal cases (AUC 0.01–0.09: separates, wrong direction).
  Caveat: count-based marks scale with test size, which differs between real cases and their balanced controls.
Readers on real failures (score_realworld.py; credit for naming any measured condition):
- Readers detect the planted failure (AUC ≥ 0.70) in 8–19% of case-plot pairs vs 59–64% for the benchmark.
- Stereotypes replicate on real data: every reliability diagram → "miscalibration" (failure 100% / control 100%),
  every loss curve → "overfitting" (control 100%) for 4/5 readers; imbalanced confusion matrices and shifted score
  histograms are almost never named, though InternVL-8B ranks imbalance correctly in probability (AUC 0.88–0.93).
- Exceptions: InternVL-38B discriminates real overfitting (named 100% vs controls 5–20%, AUC 1.00) and partly real
  leakage (named 50% vs controls 5%) — the same reader that was least stereotyped in the confirmatory run (22%).

## 2026-10-14 — Exploratory: can readers be fixed? (existing data, no new runs)
- Contextual calibration (divide reader probabilities by its no-plot answer, Zhao et al. 2021) or by its mean
  answer per plot type: balanced accuracy 0.142–0.165 → 0.147–0.165. Barely helps: the answers carry little signal.
- READ-THEN-REASON (value probe, 250 confirmatory models, loss+accuracy curves): a logistic rule trained on TRUE
  curve values (train/val last, best, best epoch, gaps; 5-fold CV, repeated 20×) applied to the values each reader
  READ: Qwen-7B 0.337 [0.277, 0.396], InternVL-38B 0.355 [0.293, 0.417], InternVL-8B 0.30, Qwen-32B 0.28,
  LLaVA 0.21 — vs the same readers' DIRECT diagnosis of those plots 0.14–0.23, and a true-value ceiling of 0.377.
  → Readers' perception recovers 75–94% of what the curves carry; their own reasoning throws it away.
  Candidate positive intervention (RQ3): VLM reads, statistics reasons. Exploratory; to be generalized to all plots.

## 2026-10-14 — Secondary analyses named in the analysis plan (score_secondary.py → SECONDARY_RESULTS.md)
- No-plot baseline (synthetic main set): readers give the SAME answer as with the plot hidden in 23–52% of items
  (Qwen-32B 52%, Qwen-7B 42%, InternVL-8B 26%, LLaVA 23%); with-plot balanced accuracy 0.144–0.147 vs 0.143 without.
- Over-claiming (reader top probability − benchmark posterior on that answer): +0.15 to +0.46; confident (≥ 0.7)
  where the benchmark is uncertain in 4–35% of answers. Hedging AUC ("cannot tell" vs benchmark entropy) 0.26–0.71:
  abstention is not consistently placed where the evidence is weak.
- Regression of correctness (deviation from plan, stated): the planned mixed-effects GLMM with ~1,500 run levels did
  not finish (variational fit > 2 h CPU); replaced by a logistic GEE clustered by model run, readers as fixed
  effects. Plot type and failure both matter; leakage is never diagnosed by any reader (separation, no estimate),
  likewise label noise on real data; within-run correlation ≈ 0.

## 2026-10-15 — Read-then-reason on all plot types + proportion-based transfer (jobs 7437–7443; exploratory)
Readers only READ drawn values (no diagnosis); a fixed logistic rule trained on exact values of synthetic training
models diagnoses (rtr.py, score_rtr.py → runs/confirm_synthetic/RTR_RESULTS.md).
- Synthetic core set, same models and plots, balanced accuracy (7 states):
  direct diagnosis 0.11–0.18 → read-then-reason 0.21–0.29 (ceiling with exact values 0.39).
  Loss curves 0.14–0.23 → 0.26–0.35 (ceiling 0.35); confusion matrix 0.14–0.33 → 0.35–0.41 (ceiling 0.42):
  where values are printed or are curve endpoints, readers reach the ceiling.
  Reliability (0.14–0.30 vs 0.46), ROC (0.14–0.22 vs 0.34), histogram (0.12–0.21 vs 0.41) stay well below:
  reading positions/areas is the new bottleneck (e.g. ROC TPR at FPR 0.5 read with r = −0.32 to 0.54;
  confusion-matrix cells r = 1.00 for InternVL-38B and Qwen-32B; LLaVA reads poorly throughout).
- Positive control: 2/5 primary failures pass for 4 readers on synthetic (1/5 on real) vs ≤ 1/5 directly; e.g.
  InternVL-38B now names blatant class imbalance in 100% of models (direct 0%) and overfitting 94% at AUC 0.98.
  Pass rule (≥ 4/5) still not met: miscalibration, leakage and shift live on plots that are hard to read.
- Real world: detection AUC ≥ 0.70 on the textbook plot in 60% of cases (4/5 readers; direct 25–50%), failure named
  in 49–60% of failure models, but control false alarms 74–80%: a rule trained on synthetic models over-flags real
  healthy controls (domain shift), so it is not yet a deployable detector.
- Proportion-based transfer (counts → shares): map agreement 73% (was 72%) — the real-world transfer result was not
  driven by test-set size.

## 2026-10-15 — Improvement audit (what could still change a result?)
- Read-then-reason weak plots are NOT a parsing problem (900/900 answers parse). Readers read reliability point
  positions well (r 0.82–0.91) but snap x to round values (losing bin locations, where imbalance shows) and cannot
  estimate histogram area shares (r ≈ 0–0.4); ROC read at r ≈ 0.8. Matches position > area (Cleveland & McGill).
- Read-then-reason gain over direct diagnosis, same models (robustness_checks.py): +0.12 to +0.15 for all 5 readers,
  95% bootstrap CIs all ≥ +0.10.
- Evidential map is robust to cutoffs: detect-only cells ≥ 5 and blind cells 14–20 on both sources under AUC
  0.65/0.70/0.75 and recall 0.4/0.5/0.6 (H2a needs ≥ 3).
- Optional GPU batches judged NOT needed: prompt-wording controls (pilot null; confirmatory few-shot, values-as-text
  and positive control already rule out prompting), interventions on real data (H8 reader effect already null;
  benchmark part not central). No further runs.
