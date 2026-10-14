# Evidential map — magic data

B1* = best on validation of mark-table / image model. Detected: AUC ≥ 0.7; diagnosed: recall ≥ 0.5 (chance 0.14).

## Level 0 (B0 target 0.65)

| failure | B0* | lc_loss | lc_acc | rel | rel_n | cm | cm_norm | roc | pr | dist |
|---|---|---|---|---|---|---|---|---|---|---|
| overfitting | 0.63 | 0.64 / 0.32 – | 0.55 / 0.28 – | 0.64 / 0.34 – | 0.66 / 0.29 – | 0.49 / 0.07 – | 0.46 / 0.26 – | 0.52 / 0.17 – | 0.54 / 0.06 – | 0.58 / 0.28 – |
| miscalibration | 0.65 | 0.53 / 0.13 – | 0.48 / 0.09 – | 0.60 / 0.23 – | 0.58 / 0.11 – | 0.51 / 0.00 – | 0.50 / 0.01 – | 0.48 / 0.16 – | 0.47 / 0.02 – | 0.54 / 0.11 – |
| label_noise | 0.72 | 0.54 / 0.20 – | 0.56 / 0.14 – | 0.65 / 0.38 – | 0.67 / 0.39 – | 0.54 / 0.15 – | 0.54 / 0.05 – | 0.65 / 0.33 – | 0.66 / 0.47 – | 0.57 / 0.30 – |
| class_imbalance | 0.65 | 0.53 / 0.12 – | 0.58 / 0.14 – | 0.55 / 0.15 – | 0.59 / 0.26 – | 0.64 / 0.42 – | 0.59 / 0.46 – | 0.50 / 0.12 – | 0.65 / 0.58 – | 0.59 / 0.27 – |
| data_leakage | 0.62 | 0.56 / 0.17 – | 0.59 / 0.30 – | 0.55 / 0.12 – | 0.54 / 0.18 – | 0.53 / 0.34 – | 0.54 / 0.25 – | 0.50 / 0.21 – | 0.50 / 0.24 – | 0.56 / 0.26 – |
| distribution_shift | 0.72 | 0.52 / 0.07 – | 0.45 / 0.05 – | 0.72 / 0.37 detect-only | 0.76 / 0.38 detect-only | 0.75 / 0.46 detect-only | 0.65 / 0.17 – | 0.48 / 0.10 – | 0.50 / 0.04 – | 0.62 / 0.22 – |

Balanced diagnosis accuracy (B1*): lc_loss 0.17, lc_acc 0.16, rel 0.26, rel_n 0.26, cm 0.21, cm_norm 0.17, roc 0.17, pr 0.21, dist 0.22; B0 0.27

## Level 1 (B0 target 0.8)

| failure | B0* | lc_loss | lc_acc | rel | rel_n | cm | cm_norm | roc | pr | dist |
|---|---|---|---|---|---|---|---|---|---|---|
| overfitting | 0.83 | 0.83 / 0.60 ID | 0.72 / 0.30 detect-only | 0.80 / 0.52 ID | 0.81 / 0.53 ID | 0.57 / 0.06 – | 0.53 / 0.07 – | 0.55 / 0.14 – | 0.49 / 0.07 – | 0.80 / 0.53 ID |
| miscalibration | 0.81 | 0.51 / 0.11 – | 0.51 / 0.17 – | 0.67 / 0.23 – | 0.69 / 0.21 – | 0.52 / 0.03 – | 0.50 / 0.24 – | 0.54 / 0.17 – | 0.49 / 0.06 – | 0.70 / 0.14 – |
| label_noise | 0.87 | 0.69 / 0.34 – | 0.63 / 0.41 – | 0.79 / 0.55 ID | 0.78 / 0.47 detect-only | 0.61 / 0.27 – | 0.59 / 0.34 – | 0.78 / 0.37 detect-only | 0.76 / 0.59 ID | 0.68 / 0.42 – |
| class_imbalance | 0.86 | 0.52 / 0.13 – | 0.58 / 0.23 – | 0.63 / 0.17 – | 0.85 / 0.57 ID | 0.86 / 0.70 ID | 0.76 / 0.65 ID | 0.82 / 0.41 detect-only | 0.85 / 0.57 ID | 0.84 / 0.60 ID |
| data_leakage | 0.87 | 0.70 / 0.37 detect-only | 0.75 / 0.36 detect-only | 0.65 / 0.28 – | 0.82 / 0.50 ID | 0.70 / 0.53 – | 0.68 / 0.36 – | 0.71 / 0.32 detect-only | 0.63 / 0.46 – | 0.81 / 0.53 ID |
| distribution_shift | 0.80 | 0.45 / 0.15 – | 0.48 / 0.15 – | 0.76 / 0.46 detect-only | 0.76 / 0.46 detect-only | 0.76 / 0.53 ID | 0.72 / 0.32 detect-only | 0.54 / 0.09 – | 0.52 / 0.11 – | 0.78 / 0.27 detect-only |

Balanced diagnosis accuracy (B1*): lc_loss 0.26, lc_acc 0.25, rel 0.35, rel_n 0.44, cm 0.32, cm_norm 0.28, roc 0.25, pr 0.29, dist 0.39; B0 0.47

## Level 2 (B0 target 0.95)

| failure | B0* | lc_loss | lc_acc | rel | rel_n | cm | cm_norm | roc | pr | dist |
|---|---|---|---|---|---|---|---|---|---|---|
| overfitting | 0.99 | 0.99 / 0.86 ID | 0.92 / 0.66 ID | 0.93 / 0.72 ID | 0.93 / 0.62 ID | 0.50 / 0.26 – | 0.58 / 0.28 – | 0.50 / 0.17 – | 0.56 / 0.07 – | 0.92 / 0.62 ID |
| miscalibration | 0.96 | 0.56 / 0.11 – | 0.51 / 0.10 – | 0.78 / 0.32 detect-only | 0.82 / 0.35 detect-only | 0.52 / 0.33 – | 0.50 / 0.00 – | 0.46 / 0.13 – | 0.46 / 0.24 – | 0.80 / 0.22 detect-only |
| label_noise | 0.98 | 0.89 / 0.61 ID | 0.82 / 0.44 detect-only | 0.96 / 0.90 ID | 0.96 / 0.85 ID | 0.72 / 0.67 ID | 0.72 / 0.63 ID | 0.94 / 0.72 ID | 0.95 / 0.80 ID | 0.90 / 0.82 ID |
| class_imbalance | 0.95 | 0.52 / 0.08 – | 0.72 / 0.23 detect-only | 0.70 / 0.22 – | 0.96 / 0.68 ID | 0.95 / 0.72 ID | 0.88 / 0.71 ID | 0.92 / 0.65 ID | 0.97 / 0.75 ID | 0.95 / 0.76 ID |
| data_leakage | 0.99 | 0.91 / 0.70 ID | 0.92 / 0.62 ID | 0.80 / 0.47 detect-only | 0.96 / 0.73 ID | 0.80 / 0.56 ID | 0.81 / 0.67 ID | 0.91 / 0.65 ID | 0.91 / 0.74 ID | 0.94 / 0.60 ID |
| distribution_shift | 0.97 | 0.53 / 0.44 – | 0.49 / 0.08 – | 0.94 / 0.82 ID | 0.94 / 0.82 ID | 0.94 / 0.75 ID | 0.92 / 0.57 ID | 0.69 / 0.38 – | 0.67 / 0.68 – | 0.99 / 0.71 ID |

Balanced diagnosis accuracy (B1*): lc_loss 0.40, lc_acc 0.38, rel 0.56, rel_n 0.68, cm 0.47, cm_norm 0.43, roc 0.44, pr 0.48, dist 0.60; B0 0.79

## Rendering loss at the strongest level (B1 mark-table AUC → B2 image AUC; only differences > 0.05)

- miscalibration on cm: 0.52 → 0.46
- miscalibration on roc: 0.51 → 0.46
- miscalibration on dist: 0.80 → 0.74
- label_noise on lc_loss: 0.89 → 0.80
- label_noise on lc_acc: 0.82 → 0.58
- class_imbalance on lc_acc: 0.72 → 0.58
- class_imbalance on roc: 0.92 → 0.54
- data_leakage on lc_acc: 0.92 → 0.73
- data_leakage on rel: 0.80 → 0.73
