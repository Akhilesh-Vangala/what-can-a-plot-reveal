# Instructor-review checks (points 4 and 5)

## confirm_synthetic

### Point 4: what the blind cells are

Two-sample test detects a difference (Holm p < 0.05) in 44/54 cells.

B3 (ConvNeXt-Tiny, native resolution) over all 54 cells: mean AUC 0.781 (B2 ResNet-18 0.778; B1* 0.789). B3 exceeds B1* by more than 0.05 in 0 cells.

| failure | plot | B1* AUC | B3 AUC [95% CI] | two-sample p (Holm) | earlier grade | final |
|---|---|---|---|---|---|---|
| overfitting | cm | 0.60 | 0.59 [0.54, 0.63] | 0.027 | robust | **weak** |
| overfitting | cm_norm | 0.60 | 0.60 [0.55, 0.65] | 0.027 | robust | **weak** |
| overfitting | roc | 0.64 | 0.62 [0.58, 0.67] | 0.027 | robust | **weak** |
| overfitting | pr | 0.62 | 0.65 [0.60, 0.69] | 0.027 | robust | **weak** |
| miscalibration | lc_loss | 0.65 | 0.69 [0.64, 0.74] | 0.027 | robust | **weak** |
| miscalibration | lc_acc | 0.49 | 0.50 [0.45, 0.55] | 1 | certified | **invisible (proven)** |
| miscalibration | cm | 0.48 | 0.52 [0.48, 0.55] | 0.83 | certified | **invisible (proven)** |
| miscalibration | cm_norm | 0.53 | 0.53 [0.48, 0.58] | 0.83 | certified | **invisible (proven)** |
| miscalibration | roc | 0.52 | 0.50 [0.45, 0.54] | 1 | robust | **invisible (no difference found)** |
| miscalibration | pr | 0.52 | 0.49 [0.44, 0.54] | 1 | robust | **invisible (no difference found)** |
| class_imbalance | lc_loss | 0.50 | 0.53 [0.48, 0.57] | 1 | robust | **invisible (no difference found)** |
| class_imbalance | lc_acc | 0.59 | 0.58 [0.53, 0.63] | 1 | robust | **invisible (no difference found)** |
| class_imbalance | rel | 0.64 | 0.59 [0.55, 0.64] | 0.027 | robust | **weak** |
| distribution_shift | lc_loss | 0.51 | 0.49 [0.44, 0.54] | 1 | certified | **invisible (proven)** |
| distribution_shift | lc_acc | 0.51 | 0.52 [0.47, 0.57] | 1 | certified | **invisible (proven)** |

Final: 4 invisible (no difference found), 5 invisible (proven), 6 weak

### Point 5: is the 7-way task too hard?

### Binary task: healthy or problem? (detection AUC per failure x plot cell, 30 cells; chance 0.5)

Reader and benchmark (B1*) are scored on the same held-out items of each cell. Cells above chance = 95% bootstrap CI above 0.5.

| reader | answered | mean AUC | cells above chance | best cell | benchmark mean AUC | overfitting | miscalibration | label_noise | class_imbalance | data_leakage | distribution_shift |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-VL-7B-Instruct | 4500/4500 | 0.54 | 11/30 | distribution_shift on rel 0.71 | 0.80 | 0.56 | 0.51 | 0.55 | 0.51 | 0.53 | 0.57 |
| InternVL3-8B | 4196/4500 | 0.50 | 4/30 | overfitting on lc_loss 0.62 | 0.80 | 0.51 | 0.51 | 0.48 | 0.49 | 0.51 | 0.49 |
| llava-onevision-qwen2-7b-ov-hf | 4500/4500 | 0.52 | 7/30 | overfitting on rel 0.71 | 0.80 | 0.52 | 0.51 | 0.53 | 0.49 | 0.53 | 0.53 |
| Qwen2.5-VL-32B-Instruct-AWQ | 4500/4500 | 0.53 | 11/30 | class_imbalance on cm 0.80 | 0.80 | 0.54 | 0.51 | 0.48 | 0.55 | 0.57 | 0.55 |
| InternVL3-38B-AWQ | 4500/4500 | 0.52 | 10/30 | overfitting on lc_loss 0.86 | 0.80 | 0.55 | 0.50 | 0.54 | 0.52 | 0.47 | 0.52 |

### Two-alternative forced choice: which of two plots is the failing model? (chance 0.50)

Benchmark (B1*) on the same pairs: **0.79** (900 pairs).

| reader | answered | accuracy [95% CI] | picks A | best plot | per failure (accuracy) |
|---|---|---|---|---|---|
| Qwen2.5-VL-7B-Instruct | 900/900 | 0.51 [0.48, 0.55] | 9% | rel 0.55 | overfitting 0.57, miscalibration 0.47, label_noise 0.52, class_imbalance 0.47, data_leakage 0.51, distribution_shift 0.54 |
| InternVL3-8B | 898/900 | 0.53 [0.49, 0.56] | 16% | rel 0.58 | overfitting 0.53, miscalibration 0.51, label_noise 0.49, class_imbalance 0.52, data_leakage 0.62, distribution_shift 0.49 |
| llava-onevision-qwen2-7b-ov-hf | 900/900 | 0.53 [0.49, 0.56] | 49% | rel 0.56 | overfitting 0.54, miscalibration 0.52, label_noise 0.48, class_imbalance 0.52, data_leakage 0.58, distribution_shift 0.51 |
| Qwen2.5-VL-32B-Instruct-AWQ | 900/900 | 0.52 [0.49, 0.55] | 23% | rel 0.55 | overfitting 0.60, miscalibration 0.49, label_noise 0.49, class_imbalance 0.44, data_leakage 0.49, distribution_shift 0.61 |
| InternVL3-38B-AWQ | 900/900 | 0.58 [0.55, 0.61] | 48% | lc_loss 0.62 | overfitting 0.65, miscalibration 0.61, label_noise 0.53, class_imbalance 0.50, data_leakage 0.55, distribution_shift 0.62 |

## confirm_magic

### Point 4: what the blind cells are

Two-sample test detects a difference (Holm p < 0.05) in 41/54 cells.

B3 (ConvNeXt-Tiny, native resolution) over all 54 cells: mean AUC 0.758 (B2 ResNet-18 0.760; B1* 0.783). B3 exceeds B1* by more than 0.05 in 0 cells.

| failure | plot | B1* AUC | B3 AUC [95% CI] | two-sample p (Holm) | earlier grade | final |
|---|---|---|---|---|---|---|
| overfitting | cm | 0.55 | 0.54 [0.49, 0.58] | 0.052 | robust | **invisible (no difference found)** |
| overfitting | cm_norm | 0.56 | 0.54 [0.49, 0.59] | 0.066 | robust | **invisible (no difference found)** |
| overfitting | roc | 0.53 | 0.55 [0.50, 0.60] | 0.429 | robust | **invisible (no difference found)** |
| overfitting | pr | 0.57 | 0.51 [0.46, 0.55] | 0.882 | robust | **invisible (no difference found)** |
| miscalibration | lc_loss | 0.58 | 0.59 [0.55, 0.64] | 1 | robust | **invisible (no difference found)** |
| miscalibration | lc_acc | 0.50 | 0.47 [0.42, 0.51] | 1 | certified | **invisible (proven)** |
| miscalibration | cm | 0.54 | 0.50 [0.45, 0.55] | 0.605 | certified | **invisible (proven)** |
| miscalibration | cm_norm | 0.51 | 0.50 [0.45, 0.55] | 1 | certified | **invisible (proven)** |
| miscalibration | roc | 0.53 | 0.48 [0.42, 0.53] | 1 | certified | **invisible (proven)** |
| miscalibration | pr | 0.53 | 0.48 [0.43, 0.52] | 0.912 | certified | **invisible (proven)** |
| label_noise | lc_acc | 0.70 | 0.50 [0.45, 0.55] | 0.027 | borderline | **weak** |
| label_noise | cm | 0.64 | 0.64 [0.60, 0.68] | 0.027 | robust | **weak** |
| label_noise | cm_norm | 0.67 | 0.63 [0.59, 0.67] | 0.027 | borderline | **weak** |
| class_imbalance | lc_loss | 0.58 | 0.54 [0.49, 0.58] | 0.027 | robust | **weak** |
| distribution_shift | lc_loss | 0.50 | 0.52 [0.47, 0.57] | 1 | certified | **invisible (proven)** |
| distribution_shift | lc_acc | 0.49 | 0.48 [0.44, 0.53] | 1 | certified | **invisible (proven)** |
| distribution_shift | pr | 0.67 | 0.72 [0.68, 0.76] | 0.027 | borderline | **revised (B3 detects)** |

Final: 5 invisible (no difference found), 7 invisible (proven), 1 revised (B3 detects), 4 weak

