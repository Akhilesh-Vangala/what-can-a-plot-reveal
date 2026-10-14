# Overfitting × learning curve prototype

Sizes: {"cal_healthy": 300, "cal_fail": 150, "main_healthy": 900, "main_fail": 300, "epochs": 100, "lineups": 4, "null_lineups": 4}

## 1. Severity calibration (B0 detection AUC)

| s | B0 AUC | 95% CI | best |
|---|---|---|---|
| 0.02 | 0.436 | 0.323–0.563 | mlp |
| 0.04 | 0.423 | 0.297–0.548 | logreg |
| 0.06 | 0.552 | 0.426–0.683 | logreg |
| 0.08 | 0.685 | 0.560–0.789 | logreg |
| 0.10 | 0.784 | 0.671–0.886 | logreg |
| 0.15 | 0.804 | 0.691–0.903 | logreg |
| 0.20 | 0.893 | 0.823–0.949 | logreg |
| 0.30 | 0.981 | 0.959–0.997 | logreg |
| 0.50 | 0.981 | 0.945–1.000 | hgb |
| 0.70 | 0.980 | 0.943–1.000 | logreg |
| 1.00 | 0.999 | 0.995–1.000 | hgb |

Chosen: target 0.65 → s=0.075; target 0.8 → s=0.140; target 0.95 → s=0.265

## 2. Detection by level (held-out AUC [95% CI]; lineup acc = P(real panel ranks first of 20))

| target | s | B0 | B1 loss | B2 loss | B1 acc | B2 acc |
|---|---|---|---|---|---|---|
| 0.65 | 0.075 | 0.729 [0.65–0.80] · lineup 0.21 | 0.738 [0.66–0.81] · lineup 0.20 | 0.716 [0.64–0.78] · lineup 0.18 | 0.706 [0.63–0.77] · lineup 0.16 | 0.541 [0.46–0.62] · lineup 0.05 |
| 0.8 | 0.140 | 0.865 [0.81–0.91] · lineup 0.35 | 0.871 [0.82–0.92] · lineup 0.36 | 0.808 [0.74–0.87] · lineup 0.29 | 0.835 [0.78–0.89] · lineup 0.30 | 0.776 [0.70–0.84] · lineup 0.27 |
| 0.95 | 0.265 | 0.940 [0.91–0.97] · lineup 0.63 | 0.951 [0.92–0.97] · lineup 0.65 | 0.933 [0.89–0.97] · lineup 0.66 | 0.912 [0.87–0.95] · lineup 0.50 | 0.889 [0.83–0.94] · lineup 0.58 |

## 3. Checks

- level 0.65 lc_loss: B0 0.729 ≥ B1 0.738 ≥ B2 0.716 → ordering OK
- level 0.65 lc_acc: B0 0.729 ≥ B1 0.706 ≥ B2 0.541 → ordering OK
- level 0.8 lc_loss: B0 0.865 ≥ B1 0.871 ≥ B2 0.808 → ordering OK
- level 0.8 lc_acc: B0 0.865 ≥ B1 0.835 ≥ B2 0.776 → ordering OK
- level 0.95 lc_loss: B0 0.940 ≥ B1 0.951 ≥ B2 0.933 → ordering OK
- level 0.95 lc_acc: B0 0.940 ≥ B1 0.912 ≥ B2 0.889 → ordering OK

Saturation (test AUC at 25/50/100% of training data):
- level 0.65 B0: 0.25→0.599, 0.50→0.650, 1.00→0.729
- level 0.65 B1_lc_loss: 0.25→0.695, 0.50→0.720, 1.00→0.738
- level 0.65 B2_lc_loss: 0.25→0.677, 0.50→0.719, 1.00→0.716
- level 0.65 B1_lc_acc: 0.25→0.675, 0.50→0.706, 1.00→0.706
- level 0.65 B2_lc_acc: 0.25→0.559, 0.50→0.556, 1.00→0.541
- level 0.8 B0: 0.25→0.825, 0.50→0.855, 1.00→0.865
- level 0.8 B1_lc_loss: 0.25→0.834, 0.50→0.875, 1.00→0.871
- level 0.8 B2_lc_loss: 0.25→0.781, 0.50→0.733, 1.00→0.808
- level 0.8 B1_lc_acc: 0.25→0.810, 0.50→0.828, 1.00→0.835
- level 0.8 B2_lc_acc: 0.25→0.789, 0.50→0.822, 1.00→0.776
- level 0.95 B0: 0.25→0.906, 0.50→0.914, 1.00→0.940
- level 0.95 B1_lc_loss: 0.25→0.939, 0.50→0.950, 1.00→0.951
- level 0.95 B2_lc_loss: 0.25→0.927, 0.50→0.908, 1.00→0.933
- level 0.95 B1_lc_acc: 0.25→0.870, 0.50→0.892, 1.00→0.912
- level 0.95 B2_lc_acc: 0.25→0.813, 0.50→0.859, 1.00→0.889

Natural units at each level: target 0.65: n_train≈1858, hidden≈29, alpha≈1.6e-01; target 0.8: n_train≈1774, hidden≈33, alpha≈9.4e-02; target 0.95: n_train≈1566, hidden≈43, alpha≈2.8e-02
