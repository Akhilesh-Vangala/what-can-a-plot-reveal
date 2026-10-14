# Overfitting × learning curve prototype

Sizes: {"cal_healthy": 600, "cal_fail": 300, "main_healthy": 1500, "main_fail": 600, "epochs": 100, "lineups": 8, "null_lineups": 8}

## 1. Severity calibration (B0 detection AUC)

| s | B0 AUC | 95% CI | best |
|---|---|---|---|
| 0.02 | 0.534 | 0.444–0.617 | rf |
| 0.04 | 0.610 | 0.519–0.693 | logreg |
| 0.06 | 0.543 | 0.453–0.632 | logreg |
| 0.08 | 0.718 | 0.636–0.790 | logreg |
| 0.10 | 0.768 | 0.690–0.837 | rf |
| 0.15 | 0.862 | 0.801–0.910 | logreg |
| 0.20 | 0.905 | 0.859–0.945 | logreg |
| 0.30 | 0.969 | 0.940–0.990 | logreg |
| 0.50 | 0.980 | 0.961–0.993 | hgb |
| 0.70 | 1.000 | 0.999–1.000 | logreg |
| 1.00 | 0.999 | 0.998–1.000 | hgb |

Chosen: target 0.65 → s=0.070; target 0.8 → s=0.117; target 0.95 → s=0.270

## 2. Detection by level (held-out AUC [95% CI]; lineup acc = P(real panel ranks first of 20))

| target | s | B0 | B1 loss | B2 loss | B1 acc | B2 acc |
|---|---|---|---|---|---|---|
| 0.65 | 0.070 | 0.691 [0.64–0.75] · lineup 0.18 | 0.690 [0.63–0.75] · lineup 0.18 | – | 0.646 [0.58–0.71] · lineup 0.18 | – |
| 0.8 | 0.117 | 0.810 [0.76–0.85] · lineup 0.30 | 0.820 [0.77–0.86] · lineup 0.30 | – | 0.781 [0.73–0.83] · lineup 0.29 | – |
| 0.95 | 0.270 | 0.956 [0.93–0.98] · lineup 0.74 | 0.952 [0.93–0.97] · lineup 0.68 | – | 0.907 [0.87–0.94] · lineup 0.58 | – |

## 3. Checks

- level 0.65 lc_loss: B0 0.691 ≥ B1 0.690 ≥ B2 – → ordering OK
- level 0.65 lc_acc: B0 0.691 ≥ B1 0.646 ≥ B2 – → ordering OK
- level 0.8 lc_loss: B0 0.810 ≥ B1 0.820 ≥ B2 – → ordering OK
- level 0.8 lc_acc: B0 0.810 ≥ B1 0.781 ≥ B2 – → ordering OK
- level 0.95 lc_loss: B0 0.956 ≥ B1 0.952 ≥ B2 – → ordering OK
- level 0.95 lc_acc: B0 0.956 ≥ B1 0.907 ≥ B2 – → ordering OK

Saturation (test AUC at 25/50/100% of training data):
- level 0.65 B0: 0.25→0.712, 0.50→0.692, 1.00→0.691
- level 0.65 B1_lc_loss: 0.25→0.639, 0.50→0.678, 1.00→0.690
- level 0.65 B1_lc_acc: 0.25→0.626, 0.50→0.638, 1.00→0.646
- level 0.8 B0: 0.25→0.796, 0.50→0.804, 1.00→0.810
- level 0.8 B1_lc_loss: 0.25→0.811, 0.50→0.818, 1.00→0.820
- level 0.8 B1_lc_acc: 0.25→0.782, 0.50→0.780, 1.00→0.781
- level 0.95 B0: 0.25→0.955, 0.50→0.957, 1.00→0.956
- level 0.95 B1_lc_loss: 0.25→0.952, 0.50→0.957, 1.00→0.952
- level 0.95 B1_lc_acc: 0.25→0.916, 0.50→0.909, 1.00→0.907

Natural units at each level: target 0.65: n_train≈1877, hidden≈29, alpha≈1.7e-01; target 0.8: n_train≈1816, hidden≈32, alpha≈1.1e-01; target 0.95: n_train≈1543, hidden≈43, alpha≈2.6e-02
