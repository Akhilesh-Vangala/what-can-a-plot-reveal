# Evidential map — synthetic data

B1* = best on validation of mark-table / image model. Detected: AUC ≥ 0.7; diagnosed: recall ≥ 0.5 (chance 0.14).

## Level 0 (B0 target 0.65)

| failure | B0* | lc_loss | lc_acc | rel | rel_n | cm | cm_norm | roc | pr | dist |
|---|---|---|---|---|---|---|---|---|---|---|
| overfitting | 0.70 | 0.70 / 0.42 detect-only | 0.67 / 0.46 – | 0.63 / 0.16 – | 0.63 / 0.21 – | 0.48 / 0.15 – | 0.58 / 0.42 – | 0.52 / 0.12 – | 0.50 / 0.03 – | 0.62 / 0.05 – |
| miscalibration | 0.69 | 0.52 / 0.02 – | 0.52 / 0.08 – | 0.58 / 0.15 – | 0.62 / 0.11 – | 0.46 / 0.13 – | 0.49 / 0.12 – | 0.48 / 0.09 – | 0.50 / 0.04 – | 0.45 / 0.00 – |
| label_noise | 0.63 | 0.59 / 0.43 – | 0.52 / 0.33 – | 0.59 / 0.28 – | 0.63 / 0.38 – | 0.54 / 0.21 – | 0.55 / 0.01 – | 0.61 / 0.19 – | 0.61 / 0.70 – | 0.59 / 0.13 – |
| class_imbalance | 0.67 | 0.55 / 0.13 – | 0.52 / 0.07 – | 0.50 / 0.20 – | 0.64 / 0.28 – | 0.59 / 0.29 – | 0.55 / 0.33 – | 0.59 / 0.27 – | 0.60 / 0.33 – | 0.66 / 0.07 – |
| data_leakage | 0.62 | 0.56 / 0.21 – | 0.55 / 0.23 – | 0.56 / 0.12 – | 0.53 / 0.14 – | 0.47 / 0.12 – | 0.50 / 0.00 – | 0.59 / 0.11 – | 0.56 / 0.13 – | 0.52 / 0.62 – |
| distribution_shift | 0.68 | 0.50 / 0.09 – | 0.47 / 0.10 – | 0.68 / 0.48 – | 0.68 / 0.51 – | 0.66 / 0.41 – | 0.63 / 0.41 – | 0.48 / 0.12 – | 0.53 / 0.00 – | 0.66 / 0.33 – |

Balanced diagnosis accuracy (B1*): lc_loss 0.20, lc_acc 0.20, rel 0.21, rel_n 0.25, cm 0.21, cm_norm 0.19, roc 0.17, pr 0.20, dist 0.19; B0 0.27

## Level 1 (B0 target 0.8)

| failure | B0* | lc_loss | lc_acc | rel | rel_n | cm | cm_norm | roc | pr | dist |
|---|---|---|---|---|---|---|---|---|---|---|
| overfitting | 0.84 | 0.84 / 0.56 ID | 0.81 / 0.60 ID | 0.82 / 0.42 detect-only | 0.79 / 0.43 detect-only | 0.56 / 0.07 – | 0.55 / 0.00 – | 0.57 / 0.07 – | 0.55 / 0.06 – | 0.72 / 0.03 detect-only |
| miscalibration | 0.84 | 0.58 / 0.10 – | 0.49 / 0.11 – | 0.68 / 0.15 – | 0.68 / 0.17 – | 0.51 / 0.03 – | 0.50 / 0.00 – | 0.50 / 0.08 – | 0.47 / 0.03 – | 0.57 / 0.66 – |
| label_noise | 0.82 | 0.71 / 0.41 detect-only | 0.62 / 0.35 – | 0.77 / 0.63 ID | 0.76 / 0.61 ID | 0.62 / 0.31 – | 0.64 / 0.39 – | 0.79 / 0.36 detect-only | 0.77 / 0.53 ID | 0.67 / 0.32 – |
| class_imbalance | 0.79 | 0.49 / 0.14 – | 0.48 / 0.02 – | 0.51 / 0.21 – | 0.78 / 0.42 detect-only | 0.79 / 0.53 ID | 0.70 / 0.28 detect-only | 0.74 / 0.33 detect-only | 0.78 / 0.56 ID | 0.78 / 0.45 detect-only |
| data_leakage | 0.75 | 0.62 / 0.47 – | 0.61 / 0.36 – | 0.57 / 0.22 – | 0.55 / 0.23 – | 0.53 / 0.21 – | 0.49 / 0.53 – | 0.66 / 0.23 – | 0.65 / 0.38 – | 0.60 / 0.17 – |
| distribution_shift | 0.82 | 0.50 / 0.04 – | 0.48 / 0.07 – | 0.86 / 0.58 ID | 0.85 / 0.55 ID | 0.75 / 0.53 ID | 0.73 / 0.45 detect-only | 0.58 / 0.14 – | 0.51 / 0.21 – | 0.81 / 0.45 detect-only |

Balanced diagnosis accuracy (B1*): lc_loss 0.26, lc_acc 0.22, rel 0.34, rel_n 0.37, cm 0.26, cm_norm 0.24, roc 0.24, pr 0.27, dist 0.31; B0 0.43

## Level 2 (B0 target 0.95)

| failure | B0* | lc_loss | lc_acc | rel | rel_n | cm | cm_norm | roc | pr | dist |
|---|---|---|---|---|---|---|---|---|---|---|
| overfitting | 0.97 | 0.96 / 0.85 ID | 0.95 / 0.67 ID | 0.93 / 0.65 ID | 0.93 / 0.65 ID | 0.63 / 0.03 – | 0.63 / 0.05 – | 0.66 / 0.15 – | 0.66 / 0.41 – | 0.87 / 0.38 detect-only |
| miscalibration | 0.98 | 0.66 / 0.21 – | 0.45 / 0.27 – | 0.82 / 0.25 detect-only | 0.80 / 0.19 detect-only | 0.50 / 0.05 – | 0.45 / 0.03 – | 0.44 / 0.15 – | 0.44 / 0.11 – | 0.80 / 0.35 detect-only |
| label_noise | 0.95 | 0.84 / 0.69 ID | 0.76 / 0.41 detect-only | 0.91 / 0.80 ID | 0.93 / 0.81 ID | 0.68 / 0.47 – | 0.68 / 0.44 – | 0.94 / 0.69 ID | 0.94 / 0.76 ID | 0.80 / 0.63 ID |
| class_imbalance | 0.95 | 0.50 / 0.19 – | 0.55 / 0.25 – | 0.59 / 0.29 – | 0.96 / 0.72 ID | 0.95 / 0.87 ID | 0.81 / 0.62 ID | 0.89 / 0.55 ID | 0.95 / 0.82 ID | 0.95 / 0.66 ID |
| data_leakage | 0.98 | 0.88 / 0.82 ID | 0.88 / 0.36 detect-only | 0.89 / 0.66 ID | 0.87 / 0.72 ID | 0.76 / 0.78 ID | 0.76 / 0.60 ID | 0.91 / 0.68 ID | 0.91 / 0.64 ID | 0.88 / 0.32 detect-only |
| distribution_shift | 0.92 | 0.51 / 0.19 – | 0.49 / 0.19 – | 0.90 / 0.64 ID | 0.90 / 0.67 ID | 0.86 / 0.64 ID | 0.83 / 0.57 ID | 0.69 / 0.28 – | 0.68 / 0.33 – | 0.94 / 0.65 ID |

Balanced diagnosis accuracy (B1*): lc_loss 0.44, lc_acc 0.32, rel 0.51, rel_n 0.59, cm 0.42, cm_norm 0.33, roc 0.40, pr 0.47, dist 0.47; B0 0.77

## Rendering loss at the strongest level (B1 mark-table AUC → B2 image AUC; only differences > 0.05)

- miscalibration on lc_acc: 0.50 → 0.45
- miscalibration on rel: 0.82 → 0.76
- label_noise on lc_acc: 0.76 → 0.67
- class_imbalance on rel_n: 0.96 → 0.86
- class_imbalance on roc: 0.89 → 0.60
