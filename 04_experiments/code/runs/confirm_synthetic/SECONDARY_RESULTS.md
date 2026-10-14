# Secondary analyses (analysis plan §4–5)

### No-plot baseline — confirm_synthetic

| reader | answer unchanged by seeing the plot | no-plot balanced acc | with-plot balanced acc |
|---|---|---|---|
| InternVL3-38B-AWQ_core | (no no-plot items in this set) | | |
| InternVL3-8B | 26% | 0.143 | 0.147 |
| Qwen2.5-VL-32B-Instruct-AWQ | 52% | 0.143 | 0.145 |
| Qwen2.5-VL-7B-Instruct | 42% | 0.143 | 0.144 |
| llava-onevision-qwen2-7b-ov-hf | 23% | 0.143 | 0.145 |

### Over-claiming and hedging — confirm_synthetic

| reader | items | over-claim index | confident where benchmark uncertain | cannot-tell rate | hedging AUC |
|---|---|---|---|---|---|
| InternVL3-38B-AWQ_core | 4249 | +0.43 | 35% | 20% | 0.71 |
| InternVL3-8B | 12396 | +0.46 | 27% | 9% | 0.36 |
| Qwen2.5-VL-32B-Instruct-AWQ | 13497 | +0.44 | 29% | 60% | 0.63 |
| Qwen2.5-VL-7B-Instruct | 10523 | +0.23 | 23% | 24% | 0.65 |
| llava-onevision-qwen2-7b-ov-hf | 11697 | +0.24 | 4% | 13% | 0.39 |

### Regression of reader correctness (logistic GEE, clustered by model run) — confirm_synthetic

52,362 reader answers, 1,765 model runs, 5 readers; within-run correlation -0.02. Log-odds of a correct answer; reference = loss curves, healthy model, first reader.

| term | log-odds | SE | 95% CI excludes 0 |
|---|---|---|---|
| Intercept | -0.50 | 0.05 | yes |
| C(plot)[T.cm] | +0.33 | 0.09 | yes |
| C(plot)[T.cm_norm] | +0.40 | 0.10 | yes |
| C(plot)[T.dist] | -0.34 | 0.07 | yes |
| C(plot)[T.lc_acc] | -0.13 | 0.02 | yes |
| C(plot)[T.pr] | +0.08 | 0.08 | no |
| C(plot)[T.rel] | -0.09 | 0.09 | no |
| C(plot)[T.rel_n] | +0.02 | 0.09 | no |
| C(plot)[T.roc] | +0.28 | 0.08 | yes |
| C(failure)[T.class_imbalance] | -2.28 | 0.04 | yes |
| C(failure)[T.data_leakage] | never correct (no estimate) | — | — |
| C(failure)[T.distribution_shift] | -1.89 | 0.03 | yes |
| C(failure)[T.label_noise] | -1.48 | 0.02 | yes |
| C(failure)[T.miscalibration] | -0.26 | 0.02 | yes |
| C(failure)[T.overfitting] | +0.09 | 0.01 | yes |
| C(reader)[T.InternVL3-8B] | -0.57 | 0.07 | yes |
| C(reader)[T.Qwen2.5-VL-32B-Instruct-AWQ] | -0.45 | 0.04 | yes |
| C(reader)[T.Qwen2.5-VL-7B-Instruct] | -0.21 | 0.04 | yes |
| C(reader)[T.llava-onevision-qwen2-7b-ov-hf] | -0.53 | 0.06 | yes |

### No-plot baseline — confirm_magic

| reader | answer unchanged by seeing the plot | no-plot balanced acc | with-plot balanced acc |
|---|---|---|---|
| InternVL3-8B_core | (no no-plot items in this set) | | |
| Qwen2.5-VL-32B-Instruct-AWQ_core | (no no-plot items in this set) | | |
| Qwen2.5-VL-7B-Instruct_core | (no no-plot items in this set) | | |

### Over-claiming and hedging — confirm_magic

| reader | items | over-claim index | confident where benchmark uncertain | cannot-tell rate | hedging AUC |
|---|---|---|---|---|---|
| InternVL3-8B_core | 4214 | +0.38 | 17% | 6% | 0.26 |
| Qwen2.5-VL-32B-Instruct-AWQ_core | 4500 | +0.44 | 27% | 58% | 0.62 |
| Qwen2.5-VL-7B-Instruct_core | 3565 | +0.15 | 13% | 22% | 0.61 |

### Regression of reader correctness (logistic GEE, clustered by model run) — confirm_magic

12,279 reader answers, 900 model runs, 3 readers; within-run correlation -0.06. Log-odds of a correct answer; reference = loss curves, healthy model, first reader.

| term | log-odds | SE | 95% CI excludes 0 |
|---|---|---|---|
| Intercept | -2.30 | 0.13 | yes |
| C(plot)[T.cm] | +0.58 | 0.15 | yes |
| C(plot)[T.dist] | +0.01 | 0.11 | no |
| C(plot)[T.rel] | +0.74 | 0.15 | yes |
| C(plot)[T.roc] | +1.27 | 0.16 | yes |
| C(failure)[T.class_imbalance] | -1.78 | 0.10 | yes |
| C(failure)[T.data_leakage] | never correct (no estimate) | — | — |
| C(failure)[T.distribution_shift] | -1.06 | 0.08 | yes |
| C(failure)[T.label_noise] | never correct (no estimate) | — | — |
| C(failure)[T.miscalibration] | +0.68 | 0.03 | yes |
| C(failure)[T.overfitting] | +0.49 | 0.02 | yes |
| C(reader)[T.Qwen2.5-VL-32B-Instruct-AWQ] | +0.51 | 0.07 | yes |
| C(reader)[T.Qwen2.5-VL-7B-Instruct] | +0.92 | 0.11 | yes |

