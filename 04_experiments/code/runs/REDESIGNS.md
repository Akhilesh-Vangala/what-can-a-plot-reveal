# What each redesign changes (point 1; existing results only)

Benchmark = how much evidence the plot carries (B1*, strongest severity). Readers = the open AI readers on the same models (synthetic run).

## 1. Bin sizes drawn on the reliability diagram (rel -> rel_n)

**confirm_synthetic, benchmark**

| failure | AUC rel -> +sizes | recall rel -> +sizes | label rel -> +sizes |
|---|---|---|---|
| overfitting | 0.93 -> 0.93 | 0.58 -> 0.55 | identifiable -> identifiable |
| miscalibration | 0.80 -> 0.81 | 0.23 -> 0.32 | detect-only -> detect-only |
| label_noise | 0.92 -> 0.93 | 0.82 -> 0.81 | identifiable -> identifiable |
| class_imbalance | 0.64 -> 0.97 | 0.33 -> 0.87 | blind -> identifiable **changed** |
| data_leakage | 0.87 -> 0.86 | 0.65 -> 0.58 | identifiable -> identifiable |
| distribution_shift | 0.97 -> 0.97 | 0.81 -> 0.83 | identifiable -> identifiable |

**confirm_synthetic, AI readers** (detection AUC for class imbalance; balanced 7-way accuracy, chance 0.14)

| reader | n models | imbalance AUC rel -> +sizes | imbalance recall rel -> +sizes | balanced acc rel -> +sizes |
|---|---|---|---|---|
| InternVL3-8B | 1500 | 0.50 -> 0.50 | 0.00 -> 0.00 | 0.14 -> 0.14 |
| Qwen2.5-VL-32B | 1500 | 0.51 -> 0.51 | 0.00 -> 0.00 | 0.14 -> 0.14 |
| Qwen2.5-VL-7B | 1389 | 0.51 -> 0.50 | 0.00 -> 0.00 | 0.14 -> 0.14 |
| LLaVA-OV-7B | 1500 | 0.50 -> 0.50 | 0.00 -> 0.00 | 0.14 -> 0.14 |

**confirm_magic, benchmark**

| failure | AUC rel -> +sizes | recall rel -> +sizes | label rel -> +sizes |
|---|---|---|---|
| overfitting | 0.93 -> 0.93 | 0.63 -> 0.58 | identifiable -> identifiable |
| miscalibration | 0.81 -> 0.83 | 0.41 -> 0.39 | detect-only -> detect-only |
| label_noise | 0.89 -> 0.89 | 0.76 -> 0.76 | identifiable -> identifiable |
| class_imbalance | 0.72 -> 0.97 | 0.24 -> 0.86 | detect-only -> identifiable **changed** |
| data_leakage | 0.83 -> 0.94 | 0.46 -> 0.77 | detect-only -> identifiable **changed** |
| distribution_shift | 0.94 -> 0.95 | 0.70 -> 0.76 | identifiable -> identifiable |

## 2. Companion plot added (synthetic)

| plot | companion | benchmark alone -> with companion |
|---|---|---|
| lc_loss | roc | 0.42 -> 0.67 |
| rel | lc_loss | 0.53 -> 0.63 |
| cm | lc_loss | 0.45 -> 0.65 |
| roc | lc_loss | 0.50 -> 0.67 |
| dist | lc_loss | 0.54 -> 0.74 |

| reader | items | balanced acc plain -> with companion |
|---|---|---|
| InternVL3-8B | 1079 | 0.15 -> 0.15 |
| Qwen2.5-VL-32B | 1270 | 0.15 -> 0.13 |
| Qwen2.5-VL-7B | 992 | 0.14 -> 0.14 |
| LLaVA-OV-7B | 954 | 0.14 -> 0.14 |

## 3. Values printed on the plot (synthetic; direct 7-way diagnosis, chance 0.14)

| reader | reliability | ROC | histogram |
|---|---|---|---|
| InternVL3-38B | 0.11 -> 0.13 | 0.14 -> 0.14 | 0.15 -> 0.14 |
| InternVL3-8B | 0.14 -> 0.14 | 0.07 -> 0.12 | 0.14 -> 0.14 |
| Qwen2.5-VL-32B | 0.14 -> 0.14 | 0.14 -> 0.14 | 0.14 -> 0.14 |
| Qwen2.5-VL-7B | 0.14 -> 0.13 | 0.14 -> 0.14 | 0.14 -> 0.12 |
| LLaVA-OV-7B | 0.14 -> 0.14 | 0.12 -> 0.14 | 0.00 -> 0.01 |

Read-then-reason (reader transcribes values, fixed rule diagnoses), ceiling = exact values: rel 0.46, roc 0.34, dist 0.41

| reader | reliability | ROC | histogram |
|---|---|---|---|
| InternVL3-38B | 0.26 -> 0.29 | 0.21 -> 0.30 | 0.16 -> 0.39 |
| InternVL3-8B | 0.16 -> 0.19 | 0.20 -> 0.30 | 0.12 -> 0.38 |
| Qwen2.5-VL-32B | 0.30 -> 0.25 | 0.22 -> 0.24 | 0.21 -> 0.32 |
| Qwen2.5-VL-7B | 0.18 -> 0.17 | 0.14 -> 0.24 | 0.15 -> 0.23 |
| LLaVA-OV-7B | 0.14 -> 0.17 | 0.14 -> 0.14 | 0.14 -> 0.30 |

## 4. Several plots together (benchmark, balanced diagnosis accuracy)

- confirm_synthetic: best single plot (rel_n) 0.63 -> all nine 0.82
- confirm_magic: best single plot (rel_n) 0.65 -> all nine 0.79
