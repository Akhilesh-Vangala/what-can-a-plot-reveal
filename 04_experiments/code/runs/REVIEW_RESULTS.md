# Instructor-review checks (points 4 and 5)

## confirm_synthetic

### Point 4: what the blind cells are

Two-sample test detects a difference (Holm p < 0.05) in 44/54 cells.

| failure | plot | B1* AUC | B3 AUC [95% CI] | two-sample p (Holm) | earlier grade | final |
|---|---|---|---|---|---|---|
| overfitting | cm | 0.60 | – | 0.027 | robust | **weak, B3 pending** |
| overfitting | cm_norm | 0.60 | – | 0.027 | robust | **weak, B3 pending** |
| overfitting | roc | 0.64 | – | 0.027 | robust | **weak, B3 pending** |
| overfitting | pr | 0.62 | – | 0.027 | robust | **weak, B3 pending** |
| miscalibration | lc_loss | 0.65 | – | 0.027 | robust | **weak, B3 pending** |
| miscalibration | lc_acc | 0.49 | – | 1 | certified | **invisible (proven)** |
| miscalibration | cm | 0.48 | – | 0.83 | certified | **invisible (proven)** |
| miscalibration | cm_norm | 0.53 | – | 0.83 | certified | **invisible (proven)** |
| miscalibration | roc | 0.52 | – | 1 | robust | **invisible (no difference found), B3 pending** |
| miscalibration | pr | 0.52 | – | 1 | robust | **invisible (no difference found), B3 pending** |
| class_imbalance | lc_loss | 0.50 | – | 1 | robust | **invisible (no difference found), B3 pending** |
| class_imbalance | lc_acc | 0.59 | – | 1 | robust | **invisible (no difference found), B3 pending** |
| class_imbalance | rel | 0.64 | – | 0.027 | robust | **weak, B3 pending** |
| distribution_shift | lc_loss | 0.51 | – | 1 | certified | **invisible (proven)** |
| distribution_shift | lc_acc | 0.51 | – | 1 | certified | **invisible (proven)** |

Final: 4 invisible (no difference found), B3 pending, 5 invisible (proven), 6 weak, B3 pending

## confirm_magic

### Point 4: what the blind cells are

Two-sample test detects a difference (Holm p < 0.05) in 41/54 cells.

| failure | plot | B1* AUC | B3 AUC [95% CI] | two-sample p (Holm) | earlier grade | final |
|---|---|---|---|---|---|---|
| overfitting | cm | 0.55 | – | 0.052 | robust | **invisible (no difference found), B3 pending** |
| overfitting | cm_norm | 0.56 | – | 0.066 | robust | **invisible (no difference found), B3 pending** |
| overfitting | roc | 0.53 | – | 0.429 | robust | **invisible (no difference found), B3 pending** |
| overfitting | pr | 0.57 | – | 0.882 | robust | **invisible (no difference found), B3 pending** |
| miscalibration | lc_loss | 0.58 | – | 1 | robust | **invisible (no difference found), B3 pending** |
| miscalibration | lc_acc | 0.50 | – | 1 | certified | **invisible (proven)** |
| miscalibration | cm | 0.54 | – | 0.605 | certified | **invisible (proven)** |
| miscalibration | cm_norm | 0.51 | – | 1 | certified | **invisible (proven)** |
| miscalibration | roc | 0.53 | – | 1 | certified | **invisible (proven)** |
| miscalibration | pr | 0.53 | – | 0.912 | certified | **invisible (proven)** |
| label_noise | lc_acc | 0.70 | – | 0.027 | borderline | **weak, B3 pending** |
| label_noise | cm | 0.64 | – | 0.027 | robust | **weak, B3 pending** |
| label_noise | cm_norm | 0.67 | – | 0.027 | borderline | **weak, B3 pending** |
| class_imbalance | lc_loss | 0.58 | – | 0.027 | robust | **weak, B3 pending** |
| distribution_shift | lc_loss | 0.50 | – | 1 | certified | **invisible (proven)** |
| distribution_shift | lc_acc | 0.49 | – | 1 | certified | **invisible (proven)** |
| distribution_shift | pr | 0.67 | – | 0.027 | borderline | **weak, B3 pending** |

Final: 5 invisible (no difference found), B3 pending, 7 invisible (proven), 5 weak, B3 pending

