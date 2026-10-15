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

Positive control and value probe; real-world check (one case, madelon feature-selection leakage, failed objective
verification and is excluded; label noise has no real tabular source and is not included); contextual calibration;
read-then-reason; proportion-based transfer.
