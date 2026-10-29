# Deviations from the analysis plan, and infrastructure re-runs

Everything below is also recorded, with dates and job numbers, in `FINDINGS_LOG.md` and `hpc/README.md`.

## Analysis deviations

| Planned | What was done | Why | Effect on conclusions |
|---|---|---|---|
| Mixed-effects logistic regression, random intercepts for run and reader | Logistic GEE clustered by model run (exchangeable), readers as fixed effects (`score_secondary.py`) | The variational GLMM with ~1,500 run levels did not finish (> 2 h CPU) | None expected: same question, population-averaged effects; within-run correlation ≈ 0 |
| H8c tested with a McNemar-style paired comparison | Paired shares of answers naming a badge-listed state on the same items | Same quantity, reported descriptively per reader | H8 reported as not supported in either form |
| Saturation flags reported with every benchmark | Reported after the fact (`saturation_report.py` → `runs/confirm_synthetic/SATURATION.md`) | Computed in the run but omitted from the first report | Two borderline cells: class imbalance on loss curves (synthetic, H1 cell, AUC 0.56, still gaining); label noise on accuracy curves (real, AUC 0.698, still gaining) |
| H1 blind-spot test: "CI includes 0.5" | As planned; cells with CI entirely below 0.5 are annotated as selection artefacts | Below-chance AUC means no usable signal, not visibility | H1 remains not supported (accuracy curves reveal class imbalance) |

## Settings changed during the confirmatory run

| Change | Why | Effect |
|---|---|---|
| InternVL3-8B `max_model_len` 16,384 → 32,768 for the few-shot controls (job 7114) | Few-shot prompts with 8 images reached 27,062 tokens | Image tiling unchanged; only the context limit grew |

## Infrastructure re-runs (identical seeds, not deviations)

- 6873–6881: all cancelled by the system (uid 0) at 22:00 on 2026-10-12 before producing output → resubmitted as 6913–6921.
- 6917: cancelled by the system while pending → resubmitted identically as 6999.
- 7428–7434: held with "user env retrieval failed" before running → cancelled and resubmitted as 7437–7443.

## Exploratory analyses (not in the analysis plan; labelled as exploratory everywhere)

| Analysis | Script | Report |
|---|---|---|
| Positive control (blatant failures) and value probe | `make_positive.py`, `score_positive.py` | `runs/positive/POSITIVE_RESULTS.md` |
| Real-world check (10 verified cases; madelon feature-selection leakage failed verification and is excluded; no real tabular label-noise source) | `realworld.py`, `rw_transfer.py`, `score_realworld.py` | `runs/realworld/REALWORLD_RESULTS.md`, `TRANSFER.md`, `TRANSFER_PROP.md` |
| Contextual calibration of reader answers | inline analysis, logged | `FINDINGS_LOG.md` (2026-10-14) |
| Read-then-reason (readers read values, a fixed rule diagnoses) | `rtr.py`, `rtr_prepare.py`, `score_rtr.py` | `runs/confirm_synthetic/RTR_RESULTS.md` |
| Robustness: read-then-reason CIs, evidential-map cutoffs | `robustness_checks.py` | `runs/confirm_synthetic/ROBUSTNESS.md` |
| Why readers fail (reason texts vs exact plot values) | inline analysis, logged | `FINDINGS_LOG.md` (gap 4) |
| Why accuracy curves reveal imbalance (class-mix effect) | inline analysis, logged | `FINDINGS_LOG.md` (gap 7) |
| Real-world false alarms (domain shift) | inline analysis, logged | `FINDINGS_LOG.md` (gap 6) |
| Multi-plot read-then-reason; rule trained on each reader's own readings | inline analysis, logged; recomputed in `figures/make_figures.py` (fig. 9) | `FINDINGS_LOG.md` (gap 8) |
| Machine-reader designs (values printed on the plot) | `annotate.py`, `gaps_prepare.py ann`, `score_gaps.py` | `runs/confirm_synthetic/GAPS_RESULTS.md` |
| Multi-plot diagnosis benchmark (sets of plots) | `gaps_prepare.py sets`, `score_gaps.py` | `runs/confirm_*/plotsets.json`, `GAPS_RESULTS.md` |

## Analysis-code fix found by the end-to-end re-run (2026-10-15)

Scorers that collect a reader's diagnosis answers by file pattern began to also pick up answer files created later
for other tasks (`_extract`, `_ann`, `_probe`). The committed reports were produced before those files existed and
are correct; a re-run would have mixed them in. Fixed with one shared rule (`score_confirm.is_diagnosis_file`) used by
every scorer and the figure script; the full re-run then reproduced the committed reports.

## Changes after the instructor's review of the proposal (2026-10-28)

| Point in the review | Change | Status |
|---|---|---|
| A controlled human study needs IRB approval, not feasible in the course | Controlled human-reader study dropped (materials archived in `human_study/archive/`). Replaced by 4–6 formative expert sessions (think-aloud on 12 plots, Explorer task, interview), reported qualitatively only. The Explorer task study with practitioners becomes part of the same sessions. Quantitative reader evidence comes from the five AI readers. | protocol `human_study/PROTOCOL.md` |
| "Blind" may reflect the estimator, not the plot | (a) Cells blind by construction: failing model re-trained next to its healthy twin, plotted values compared exactly (`certify_blind.py`). (b) Full cutoff sensitivity (AUC 0.60–0.80 × recall 0.30–0.70) with a grade per blind cell (`blind_sensitivity.py`). (c) Classifier-free kernel two-sample test per cell and a stronger, different image benchmark B3 (ConvNeXt-Tiny at native resolution) — cluster stage `review`, scored by `score_review.py`. Terminology: "blind" is split into **invisible** (no difference in the plot) and **weak** (a difference exists, but no estimator reaches AUC 0.70). | done (2026-10-29) |
| Near-chance 7-way performance may be a task-design effect | Two simpler reader tasks on the same held-out core items, five readers: binary (healthy vs problem; AUC per failure × plot, compared with the benchmark on the same items) and two-alternative forced choice (which of two plots is the failing model; chance 0.5; benchmark 2AFC accuracy on the same pairs). | done (2026-10-29) |
| Core vs stretch scope | Marked in the mid-term update. | — |
