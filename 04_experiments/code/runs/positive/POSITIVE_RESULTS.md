# Positive control and value probe (exploratory credibility checks)

## Positive control: blatant failures on their textbook plot

Detection AUC = healthy vs the failure on the same plot type (score 1 − P(healthy)). Named = share of failure items whose top answer is the failure. Healthy false alarm = share of healthy items on that plot whose top answer is not healthy. Pass rule (fixed before reading): AUC ≥ 0.80 and named ≥ 50% on ≥ 4 of the 5 primary failures (overfitting, miscalibration, class_imbalance, data_leakage, distribution_shift).

### InternVL3-38B-AWQ — synthetic

| failure (plot) | detection AUC [95% CI] | named | healthy false alarm | top answer on failure items | no answer |
|---|---|---|---|---|---|
| overfitting (lc_loss) | 0.95 [0.90, 0.99] | 94% | 22% | overfitting 47, healthy 3 | 0 |
| miscalibration (rel) | 0.50 [0.50, 0.50] | 100% | 100% | miscalibration 50 | 21 |
| label_noise (lc_acc) † | 0.88 [0.81, 0.94] | 0% | 0% | overfitting 35, healthy 15 | 0 |
| class_imbalance (cm) | 0.54 [0.51, 0.58] | 0% | 0% | healthy 50 | 0 |
| data_leakage (roc) | 1.00 [1.00, 1.00] | 0% | 0% | miscalibration 50 | 0 |
| distribution_shift (dist) | 0.50 [0.50, 0.50] | 0% | 100% | miscalibration 50 | 0 |

**1/5 primary failures passed → CANNOT do the task even when the signal is blatant** († label noise: no unique single-plot signature, reported only)

### InternVL3-38B-AWQ — magic

| failure (plot) | detection AUC [95% CI] | named | healthy false alarm | top answer on failure items | no answer |
|---|---|---|---|---|---|
| overfitting (lc_loss) | 0.89 [0.83, 0.95] | 84% | 8% | overfitting 42, healthy 8 | 0 |
| miscalibration (rel) | 0.50 [0.50, 0.50] | 100% | 100% | miscalibration 50 | 37 |
| label_noise (lc_acc) † | 0.58 [0.47, 0.69] | 0% | 8% | healthy 31, overfitting 19 | 0 |
| class_imbalance (cm) | 0.51 [0.50, 0.53] | 0% | 0% | healthy 50 | 0 |
| data_leakage (roc) | 1.00 [1.00, 1.00] | 0% | 0% | miscalibration 50 | 0 |
| distribution_shift (dist) | 0.51 [0.50, 0.53] | 0% | 100% | miscalibration 49, overfitting 1 | 0 |

**1/5 primary failures passed → CANNOT do the task even when the signal is blatant** († label noise: no unique single-plot signature, reported only)

### InternVL3-8B — synthetic

| failure (plot) | detection AUC [95% CI] | named | healthy false alarm | top answer on failure items | no answer |
|---|---|---|---|---|---|
| overfitting (lc_loss) | 0.51 [0.40, 0.62] | 100% | 100% | overfitting 50 | 0 |
| miscalibration (rel) | 0.67 [0.60, 0.74] | 100% | 100% | miscalibration 50 | 0 |
| label_noise (lc_acc) † | 0.75 [0.66, 0.83] | 0% | 100% | overfitting 50 | 0 |
| class_imbalance (cm) | 0.47 [0.43, 0.50] | 48% | 100% | class_imbalance 24, miscalibration 24, distribution_shift 2 | 0 |
| data_leakage (roc) | nan [nan, nan] | 0% | 100% |  | 88 |
| distribution_shift (dist) | 0.46 [0.36, 0.57] | 30% | 100% | miscalibration 30, distribution_shift 15, overfitting 5 | 0 |

**0/5 primary failures passed → CANNOT do the task even when the signal is blatant** († label noise: no unique single-plot signature, reported only)

### InternVL3-8B — magic

| failure (plot) | detection AUC [95% CI] | named | healthy false alarm | top answer on failure items | no answer |
|---|---|---|---|---|---|
| overfitting (lc_loss) | 0.79 [0.70, 0.87] | 100% | 100% | overfitting 50 | 0 |
| miscalibration (rel) | 0.82 [0.74, 0.88] | 100% | 100% | miscalibration 50 | 0 |
| label_noise (lc_acc) † | 0.63 [0.54, 0.72] | 0% | 100% | overfitting 50 | 0 |
| class_imbalance (cm) | 0.49 [0.47, 0.50] | 48% | 100% | miscalibration 26, class_imbalance 24 | 0 |
| data_leakage (roc) | nan [nan, nan] | 0% | 100% |  | 66 |
| distribution_shift (dist) | 0.53 [0.42, 0.63] | 16% | 100% | miscalibration 40, distribution_shift 8, overfitting 2 | 0 |

**1/5 primary failures passed → CANNOT do the task even when the signal is blatant** († label noise: no unique single-plot signature, reported only)

### Qwen2.5-VL-32B-Instruct-AWQ — synthetic

| failure (plot) | detection AUC [95% CI] | named | healthy false alarm | top answer on failure items | no answer |
|---|---|---|---|---|---|
| overfitting (lc_loss) | 0.79 [0.71, 0.86] | 100% | 100% | overfitting 50 | 0 |
| miscalibration (rel) | 0.78 [0.71, 0.85] | 100% | 100% | miscalibration 50 | 0 |
| label_noise (lc_acc) † | 0.84 [0.75, 0.91] | 0% | 100% | overfitting 50 | 0 |
| class_imbalance (cm) | 0.47 [0.35, 0.60] | 14% | 56% | healthy 43, class_imbalance 7 | 0 |
| data_leakage (roc) | 1.00 [1.00, 1.00] | 0% | 0% | miscalibration 50 | 1 |
| distribution_shift (dist) | 0.55 [0.49, 0.61] | 0% | 100% | miscalibration 50 | 0 |

**0/5 primary failures passed → CANNOT do the task even when the signal is blatant** († label noise: no unique single-plot signature, reported only)

### Qwen2.5-VL-32B-Instruct-AWQ — magic

| failure (plot) | detection AUC [95% CI] | named | healthy false alarm | top answer on failure items | no answer |
|---|---|---|---|---|---|
| overfitting (lc_loss) | 0.58 [0.51, 0.66] | 100% | 100% | overfitting 50 | 0 |
| miscalibration (rel) | 0.88 [0.82, 0.94] | 100% | 94% | miscalibration 50 | 0 |
| label_noise (lc_acc) † | 0.68 [0.57, 0.77] | 0% | 100% | overfitting 50 | 0 |
| class_imbalance (cm) | 0.29 [0.18, 0.41] | 4% | 68% | healthy 48, class_imbalance 2 | 0 |
| data_leakage (roc) | 1.00 [1.00, 1.00] | 0% | 0% | miscalibration 50 | 0 |
| distribution_shift (dist) | 0.70 [0.62, 0.78] | 0% | 100% | miscalibration 50 | 0 |

**1/5 primary failures passed → CANNOT do the task even when the signal is blatant** († label noise: no unique single-plot signature, reported only)

### Qwen2.5-VL-7B-Instruct — synthetic

| failure (plot) | detection AUC [95% CI] | named | healthy false alarm | top answer on failure items | no answer |
|---|---|---|---|---|---|
| overfitting (lc_loss) | 0.72 [0.62, 0.81] | 100% | 100% | overfitting 50 | 0 |
| miscalibration (rel) | 0.70 [0.59, 0.82] | 40% | 0% | healthy 12, miscalibration 8 | 32 |
| label_noise (lc_acc) † | 0.47 [0.39, 0.56] | 0% | 100% | overfitting 50 | 0 |
| class_imbalance (cm) | nan [nan, nan] | 0% | nan% |  | 100 |
| data_leakage (roc) | 0.50 [0.50, 0.50] | 0% | 0% | healthy 50 | 0 |
| distribution_shift (dist) | 0.49 [0.42, 0.56] | 0% | 90% | overfitting 48, healthy 2 | 0 |

**0/5 primary failures passed → CANNOT do the task even when the signal is blatant** († label noise: no unique single-plot signature, reported only)

### Qwen2.5-VL-7B-Instruct — magic

| failure (plot) | detection AUC [95% CI] | named | healthy false alarm | top answer on failure items | no answer |
|---|---|---|---|---|---|
| overfitting (lc_loss) | 0.46 [0.38, 0.55] | 100% | 100% | overfitting 50 | 0 |
| miscalibration (rel) | 0.81 [0.74, 0.89] | 62% | 0% | miscalibration 25, healthy 15 | 10 |
| label_noise (lc_acc) † | 0.64 [0.54, 0.74] | 0% | 100% | overfitting 50 | 0 |
| class_imbalance (cm) | nan [nan, nan] | 0% | nan% | healthy 3 | 97 |
| data_leakage (roc) | 0.50 [0.50, 0.50] | 0% | 0% | healthy 50 | 2 |
| distribution_shift (dist) | 0.51 [0.45, 0.57] | 0% | 98% | overfitting 47, healthy 2, miscalibration 1 | 0 |

**1/5 primary failures passed → CANNOT do the task even when the signal is blatant** († label noise: no unique single-plot signature, reported only)

### llava-onevision-qwen2-7b-ov-hf — synthetic

| failure (plot) | detection AUC [95% CI] | named | healthy false alarm | top answer on failure items | no answer |
|---|---|---|---|---|---|
| overfitting (lc_loss) | 0.54 [0.51, 0.58] | 100% | 100% | overfitting 50 | 0 |
| miscalibration (rel) | 0.50 [0.50, 0.50] | 0% | 100% | label_noise 50 | 0 |
| label_noise (lc_acc) † | 0.51 [0.40, 0.61] | 0% | 100% | overfitting 50 | 0 |
| class_imbalance (cm) | 0.41 [0.33, 0.51] | 0% | 6% | healthy 49, overfitting 1 | 1 |
| data_leakage (roc) | nan [nan, nan] | 0% | 100% |  | 60 |
| distribution_shift (dist) | nan [nan, nan] | 0% | nan% | overfitting 2 | 98 |

**0/5 primary failures passed → CANNOT do the task even when the signal is blatant** († label noise: no unique single-plot signature, reported only)

### llava-onevision-qwen2-7b-ov-hf — magic

| failure (plot) | detection AUC [95% CI] | named | healthy false alarm | top answer on failure items | no answer |
|---|---|---|---|---|---|
| overfitting (lc_loss) | 0.55 [0.50, 0.61] | 100% | 100% | overfitting 50 | 0 |
| miscalibration (rel) | 0.50 [0.50, 0.50] | 0% | 100% | label_noise 50 | 0 |
| label_noise (lc_acc) † | 0.56 [0.46, 0.66] | 0% | 100% | overfitting 50 | 1 |
| class_imbalance (cm) | 0.41 [0.34, 0.49] | 0% | 0% | healthy 50 | 0 |
| data_leakage (roc) | nan [nan, nan] | 0% | 100% |  | 50 |
| distribution_shift (dist) | 0.50 [0.50, 0.50] | 0% | 100% | overfitting 3 | 78 |

**0/5 primary failures passed → CANNOT do the task even when the signal is blatant** († label noise: no unique single-plot signature, reported only)

## Value probe (confirmatory models): read numbers off the curves

Correlation and median absolute error between read and true values; 'rise' = val_last − val_best (the overfitting signature). AUC = overfitting vs healthy from the reader's OWN read numbers vs from the true numbers.

### InternVL3-38B-AWQ

| plot | quantity | corr | median abs error | parsed |
|---|---|---|---|---|
| lc_loss | train_last | 0.99 | 0.00459 | 250/250 |
| lc_loss | val_last | 1.00 | 0.00333 | 250/250 |
| lc_loss | val_best | 0.96 | 0.0048 | 250/250 |
| lc_loss | val_best_epoch | 0.68 | 7 | 250/250 |
| lc_loss | overfitting AUC from rise: own reading 0.91 vs true values 0.94 | | | |
| lc_acc | train_last | 0.97 | 0.00448 | 250/250 |
| lc_acc | val_last | 0.99 | 0.006 | 250/250 |
| lc_acc | val_best | 0.98 | 0.006 | 250/250 |
| lc_acc | val_best_epoch | 0.71 | 9 | 250/250 |
| lc_acc | overfitting AUC from rise: own reading 0.59 vs true values 0.80 | | | |

### InternVL3-8B

| plot | quantity | corr | median abs error | parsed |
|---|---|---|---|---|
| lc_loss | train_last | 0.99 | 0.00625 | 250/250 |
| lc_loss | val_last | 1.00 | 0.00379 | 250/250 |
| lc_loss | val_best | 0.95 | 0.0107 | 250/250 |
| lc_loss | val_best_epoch | 0.34 | 40 | 250/250 |
| lc_loss | overfitting AUC from rise: own reading 0.92 vs true values 0.94 | | | |
| lc_acc | train_last | 0.96 | 0.00395 | 250/250 |
| lc_acc | val_last | 0.99 | 0.004 | 250/250 |
| lc_acc | val_best | 0.98 | 0.004 | 250/250 |
| lc_acc | val_best_epoch | 0.44 | 17 | 250/250 |
| lc_acc | overfitting AUC from rise: own reading 0.68 vs true values 0.80 | | | |

### Qwen2.5-VL-32B-Instruct-AWQ

| plot | quantity | corr | median abs error | parsed |
|---|---|---|---|---|
| lc_loss | train_last | 0.95 | 0.019 | 250/250 |
| lc_loss | val_last | 0.98 | 0.012 | 250/250 |
| lc_loss | val_best | 0.96 | 0.0134 | 250/250 |
| lc_loss | val_best_epoch | 0.60 | 20 | 250/250 |
| lc_loss | overfitting AUC from rise: own reading 0.83 vs true values 0.94 | | | |
| lc_acc | train_last | 0.91 | 0.00951 | 250/250 |
| lc_acc | val_last | 0.91 | 0.014 | 250/250 |
| lc_acc | val_best | 0.88 | 0.016 | 250/250 |
| lc_acc | val_best_epoch | 0.46 | 29 | 250/250 |
| lc_acc | overfitting AUC from rise: own reading 0.62 vs true values 0.80 | | | |

### Qwen2.5-VL-7B-Instruct

| plot | quantity | corr | median abs error | parsed |
|---|---|---|---|---|
| lc_loss | train_last | 0.96 | 0.0182 | 250/250 |
| lc_loss | val_last | 0.99 | 0.01 | 250/250 |
| lc_loss | val_best | 0.97 | 0.0115 | 250/250 |
| lc_loss | val_best_epoch | 0.71 | 7 | 250/250 |
| lc_loss | overfitting AUC from rise: own reading 0.88 vs true values 0.94 | | | |
| lc_acc | train_last | 0.90 | 0.0178 | 250/250 |
| lc_acc | val_last | 0.95 | 0.008 | 250/250 |
| lc_acc | val_best | 0.94 | 0.008 | 250/250 |
| lc_acc | val_best_epoch | 0.57 | 12 | 250/250 |
| lc_acc | overfitting AUC from rise: own reading 0.60 vs true values 0.80 | | | |

### llava-onevision-qwen2-7b-ov-hf

| plot | quantity | corr | median abs error | parsed |
|---|---|---|---|---|
| lc_loss | train_last | 0.87 | 0.0359 | 250/250 |
| lc_loss | val_last | 0.55 | 0.0357 | 250/250 |
| lc_loss | val_best | 0.71 | 0.0586 | 250/250 |
| lc_loss | val_best_epoch | 0.38 | 20 | 250/250 |
| lc_loss | overfitting AUC from rise: own reading 0.40 vs true values 0.94 | | | |
| lc_acc | train_last | 0.50 | 0.0304 | 250/250 |
| lc_acc | val_last | 0.41 | 0.032 | 250/250 |
| lc_acc | val_best | 0.24 | 0.024 | 250/250 |
| lc_acc | val_best_epoch | 0.30 | 16.5 | 250/250 |
| lc_acc | overfitting AUC from rise: own reading 0.38 vs true values 0.80 | | | |

