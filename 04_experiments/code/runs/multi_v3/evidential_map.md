# Evidential map (multi-failure run)

Provisional thresholds: detected if AUC ≥ 0.7; diagnosed if 5-way recall ≥ 0.5 (chance 0.20).

## Level 0 (B0 detection target 0.65)

| failure | lc_loss | lc_acc | rel | cm | roc | pr |
|---|---|---|---|---|---|---|
| overfitting | det 0.66 · diag 0.17 · **neither** | det 0.60 · diag 0.16 · **neither** | det 0.65 · diag 0.07 · **neither** | det 0.51 · diag 0.00 · **neither** | det 0.48 · diag 0.06 · **neither** | det 0.53 · diag 0.13 · **neither** |
| miscalibration | det 0.57 · diag 0.36 · **neither** | det 0.56 · diag 0.04 · **neither** | det 0.62 · diag 0.54 · **neither** | det 0.52 · diag 0.00 · **neither** | det 0.48 · diag 0.07 · **neither** | det 0.50 · diag 0.55 · **neither** |
| label_noise | det 0.52 · diag 0.23 · **neither** | det 0.55 · diag 0.07 · **neither** | det 0.56 · diag 0.02 · **neither** | det 0.51 · diag 0.00 · **neither** | det 0.60 · diag 0.72 · **neither** | det 0.58 · diag 0.08 · **neither** |
| class_imbalance | det 0.49 · diag 0.20 · **neither** | det 0.51 · diag 0.03 · **neither** | det 0.68 · diag 0.34 · **neither** | det 0.70 · diag 0.19 · **detect-only** | det 0.54 · diag 0.12 · **neither** | det 0.65 · diag 0.30 · **neither** |

5-way diagnosis accuracy (B1 / B2): lc_loss 0.43/0.27, lc_acc 0.43/0.41, rel 0.43/0.17, cm 0.43/0.44, roc 0.38/0.29, pr 0.40/0.22; B0 0.45

## Level 1 (B0 detection target 0.8)

| failure | lc_loss | lc_acc | rel | cm | roc | pr |
|---|---|---|---|---|---|---|
| overfitting | det 0.75 · diag 0.30 · **detect-only** | det 0.71 · diag 0.33 · **detect-only** | det 0.72 · diag 0.19 · **detect-only** | det 0.50 · diag 0.00 · **neither** | det 0.53 · diag 0.05 · **neither** | det 0.53 · diag 0.00 · **neither** |
| miscalibration | det 0.55 · diag 0.02 · **neither** | det 0.55 · diag 0.01 · **neither** | det 0.64 · diag 0.02 · **neither** | det 0.45 · diag 0.00 · **neither** | det 0.51 · diag 0.02 · **neither** | det 0.48 · diag 0.00 · **neither** |
| label_noise | det 0.66 · diag 0.20 · **neither** | det 0.62 · diag 0.15 · **neither** | det 0.77 · diag 0.37 · **detect-only** | det 0.56 · diag 0.00 · **neither** | det 0.78 · diag 0.29 · **detect-only** | det 0.79 · diag 0.43 · **detect-only** |
| class_imbalance | det 0.49 · diag 0.06 · **neither** | det 0.47 · diag 0.00 · **neither** | det 0.78 · diag 0.27 · **detect-only** | det 0.80 · diag 0.69 · **identifiable** | det 0.72 · diag 0.29 · **detect-only** | det 0.81 · diag 0.40 · **detect-only** |

5-way diagnosis accuracy (B1 / B2): lc_loss 0.44/0.39, lc_acc 0.44/0.40, rel 0.48/0.44, cm 0.46/0.39, roc 0.39/0.44, pr 0.47/0.50; B0 0.53

## Level 2 (B0 detection target 0.95)

| failure | lc_loss | lc_acc | rel | cm | roc | pr |
|---|---|---|---|---|---|---|
| overfitting | det 0.97 · diag 0.86 · **identifiable** | det 0.96 · diag 0.76 · **identifiable** | det 0.93 · diag 0.73 · **identifiable** | det 0.68 · diag 0.09 · **neither** | det 0.73 · diag 0.19 · **detect-only** | det 0.70 · diag 0.16 · **detect-only** |
| miscalibration | det 0.74 · diag 0.22 · **detect-only** | det 0.53 · diag 0.11 · **neither** | det 0.83 · diag 0.19 · **detect-only** | det 0.53 · diag 0.00 · **neither** | det 0.50 · diag 0.04 · **neither** | det 0.54 · diag 0.00 · **neither** |
| label_noise | det 0.85 · diag 0.51 · **identifiable** | det 0.79 · diag 0.13 · **detect-only** | det 0.91 · diag 0.74 · **identifiable** | det 0.68 · diag 0.03 · **neither** | det 0.92 · diag 0.67 · **identifiable** | det 0.89 · diag 0.76 · **identifiable** |
| class_imbalance | det 0.53 · diag 0.00 · **neither** | det 0.59 · diag 0.00 · **neither** | det 0.90 · diag 0.55 · **identifiable** | det 0.91 · diag 0.58 · **identifiable** | det 0.83 · diag 0.42 · **detect-only** | det 0.91 · diag 0.63 · **identifiable** |

5-way diagnosis accuracy (B1 / B2): lc_loss 0.56/0.56, lc_acc 0.53/0.51, rel 0.63/0.55, cm 0.49/0.49, roc 0.51/0.50, pr 0.56/0.53; B0 0.76

## Most-confused pairs at the strongest level (benchmark B1 confusion; share of the row)

- lc_loss: class_imbalance → predicted healthy (87%)
- lc_acc: class_imbalance → predicted healthy (93%)
- rel: miscalibration → predicted healthy (47%)
- cm: miscalibration → predicted healthy (95%)
- roc: miscalibration → predicted healthy (82%)
- pr: miscalibration → predicted healthy (86%)
