# Sensitivity of the evidential map (instructor review, point 4)

Strongest severity. B1* = validation-best of the plotted-value benchmark (B1) and the image benchmark (B2).

## confirm_synthetic

### (a) Label counts under alternative cutoffs (identifiable / detect-only / blind)

| AUC cutoff \ recall cutoff | 0.30 | 0.40 | 0.50 | 0.60 | 0.70 |
|---|---|---|---|---|---|
| 0.60 | 40/4/10 | 35/9/10 | 29/15/10 | 24/20/10 | 20/24/10 |
| 0.65 | 37/2/15 | 34/5/15 | 29/10/15 | 24/15/15 | 20/19/15 |
| 0.70 | 37/2/15 | 34/5/15 | **29/10/15** | 24/15/15 | 20/19/15 |
| 0.75 | 33/2/19 | 30/5/19 | 27/8/19 | 22/13/19 | 19/16/19 |
| 0.80 | 30/1/23 | 28/3/23 | 27/4/23 | 22/9/23 | 19/12/23 |

Cells whose label is the same under all 25 cutoff pairs: 30/54; median share of cutoff pairs that keep a cell's label: 1.00.

### (b) Every blind cell (AUC < 0.70 at the default cutoffs) and the evidence behind it

| failure | plot | B1* AUC [95% CI] | best estimator (val AUC) | blind under cutoffs | grade |
|---|---|---|---|---|---|
| overfitting | cm | 0.60 [0.56, 0.65] | logreg (0.63) | 80% | robust |
| overfitting | cm_norm | 0.60 [0.55, 0.65] | logreg (0.63) | 100% | robust |
| overfitting | roc | 0.64 [0.60, 0.68] | resnet18_image (0.68) | 80% | robust |
| overfitting | pr | 0.62 [0.58, 0.67] | resnet18_image (0.66) | 80% | robust |
| miscalibration | lc_loss | 0.65 [0.61, 0.70] | resnet18_image (0.66) | 80% | robust |
| miscalibration | lc_acc | 0.49 [0.44, 0.54] | resnet18_image (0.53) | 100% | certified |
| miscalibration | cm | 0.48 [0.43, 0.53] | rf (0.53) | 100% | certified |
| miscalibration | cm_norm | 0.53 [0.48, 0.58] | rf (0.58) | 100% | certified |
| miscalibration | roc | 0.52 [0.48, 0.57] | logreg (0.55) | 100% | robust |
| miscalibration | pr | 0.52 [0.47, 0.56] | logreg (0.52) | 100% | robust |
| class_imbalance | lc_loss | 0.50 [0.46, 0.55] | resnet18_image (0.56) | 100% | robust |
| class_imbalance | lc_acc | 0.59 [0.54, 0.64] | hgb (0.59) | 100% | robust |
| class_imbalance | rel | 0.64 [0.60, 0.68] | rf (0.67) | 80% | robust |
| distribution_shift | lc_loss | 0.51 [0.46, 0.56] | resnet18_image (0.51) | 100% | certified |
| distribution_shift | lc_acc | 0.51 [0.46, 0.56] | rf (0.55) | 100% | certified |

Blind cells: 15 = 5 certified + 10 robust + 0 borderline.

## confirm_magic

### (a) Label counts under alternative cutoffs (identifiable / detect-only / blind)

| AUC cutoff \ recall cutoff | 0.30 | 0.40 | 0.50 | 0.60 | 0.70 |
|---|---|---|---|---|---|
| 0.60 | 38/3/13 | 36/5/13 | 29/12/13 | 25/16/13 | 18/23/13 |
| 0.65 | 37/3/14 | 35/5/14 | 29/11/14 | 25/15/14 | 18/22/14 |
| 0.70 | 34/3/17 | 32/5/17 | **28/9/17** | 25/12/17 | 18/19/17 |
| 0.75 | 33/1/20 | 32/2/20 | 28/6/20 | 25/9/20 | 18/16/20 |
| 0.80 | 31/1/22 | 30/2/22 | 26/6/22 | 24/8/22 | 17/15/22 |

Cells whose label is the same under all 25 cutoff pairs: 31/54; median share of cutoff pairs that keep a cell's label: 1.00.

### (b) Every blind cell (AUC < 0.70 at the default cutoffs) and the evidence behind it

| failure | plot | B1* AUC [95% CI] | best estimator (val AUC) | blind under cutoffs | grade |
|---|---|---|---|---|---|
| overfitting | cm | 0.55 [0.50, 0.60] | logreg (0.61) | 100% | robust |
| overfitting | cm_norm | 0.56 [0.52, 0.61] | logreg (0.60) | 100% | robust |
| overfitting | roc | 0.53 [0.48, 0.58] | mlp (0.59) | 100% | robust |
| overfitting | pr | 0.57 [0.53, 0.62] | logreg (0.60) | 100% | robust |
| miscalibration | lc_loss | 0.58 [0.53, 0.64] | resnet18_image (0.59) | 100% | robust |
| miscalibration | lc_acc | 0.50 [0.45, 0.54] | resnet18_image (0.52) | 100% | certified |
| miscalibration | cm | 0.54 [0.49, 0.58] | resnet18_image (0.52) | 100% | certified |
| miscalibration | cm_norm | 0.51 [0.46, 0.55] | mlp (0.49) | 100% | certified |
| miscalibration | roc | 0.53 [0.48, 0.57] | logreg (0.52) | 100% | certified |
| miscalibration | pr | 0.53 [0.47, 0.57] | rf (0.53) | 100% | certified |
| label_noise | lc_acc | 0.70 [0.66, 0.74] | mlp (0.70) | 60% | borderline |
| label_noise | cm | 0.64 [0.60, 0.68] | mlp (0.65) | 80% | robust |
| label_noise | cm_norm | 0.67 [0.62, 0.72] | resnet18_image (0.65) | 60% | borderline |
| class_imbalance | lc_loss | 0.58 [0.53, 0.63] | rf (0.61) | 100% | robust |
| distribution_shift | lc_loss | 0.50 [0.45, 0.55] | logreg (0.57) | 100% | certified |
| distribution_shift | lc_acc | 0.49 [0.44, 0.54] | resnet18_image (0.56) | 100% | certified |
| distribution_shift | pr | 0.67 [0.63, 0.72] | resnet18_image (0.70) | 60% | borderline |

Blind cells: 17 = 7 certified + 7 robust + 3 borderline.

