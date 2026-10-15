# Saturation check (benchmark still gaining > 0.03 AUC from 50% to 100% training data)

## confirm_synthetic: 13 flagged plot cells (all levels)

| failure | level | plot | gain 50%→100% | test AUC | touches a conclusion? |
|---|---|---|---|---|---|
| class_imbalance | 0 | rel_n | +0.036 | 0.587 | no (not strongest level / far from threshold) |
| class_imbalance | 2 | lc_loss | +0.043 | 0.555 | H1 blind-spot cell |
| data_leakage | 1 | pr | +0.037 | 0.735 | no (not strongest level / far from threshold) |
| data_leakage | 1 | roc | +0.088 | 0.730 | no (not strongest level / far from threshold) |
| distribution_shift | 0 | rel | +0.080 | 0.739 | no (not strongest level / far from threshold) |
| distribution_shift | 1 | cm | +0.172 | 0.786 | no (not strongest level / far from threshold) |
| distribution_shift | 1 | cm_norm | +0.031 | 0.758 | no (not strongest level / far from threshold) |
| distribution_shift | 1 | rel_n | +0.086 | 0.858 | no (not strongest level / far from threshold) |
| miscalibration | 0 | rel | +0.038 | 0.598 | no (not strongest level / far from threshold) |
| miscalibration | 0 | rel_n | +0.038 | 0.599 | no (not strongest level / far from threshold) |
| miscalibration | 1 | lc_loss | +0.043 | 0.564 | no (not strongest level / far from threshold) |
| miscalibration | 2 | cm_norm | +0.044 | 0.527 | no (not strongest level / far from threshold) |
| overfitting | 1 | cm_norm | +0.044 | 0.528 | no (not strongest level / far from threshold) |

Cells at the strongest level that touch a conclusion: 1.

## confirm_magic: 13 flagged plot cells (all levels)

| failure | level | plot | gain 50%→100% | test AUC | touches a conclusion? |
|---|---|---|---|---|---|
| class_imbalance | 0 | roc | +0.076 | 0.640 | no (not strongest level / far from threshold) |
| class_imbalance | 1 | roc | +0.062 | 0.748 | no (not strongest level / far from threshold) |
| data_leakage | 0 | rel_n | +0.053 | 0.690 | no (not strongest level / far from threshold) |
| data_leakage | 1 | lc_loss | +0.038 | 0.658 | no (not strongest level / far from threshold) |
| distribution_shift | 0 | rel_n | +0.199 | 0.675 | no (not strongest level / far from threshold) |
| distribution_shift | 1 | dist | +0.217 | 0.861 | no (not strongest level / far from threshold) |
| distribution_shift | 2 | dist | +0.034 | 0.938 | no (not strongest level / far from threshold) |
| label_noise | 2 | lc_acc | +0.055 | 0.698 | near the 0.70 map threshold |
| miscalibration | 0 | cm | +0.030 | 0.513 | no (not strongest level / far from threshold) |
| overfitting | 1 | cm | +0.031 | 0.525 | no (not strongest level / far from threshold) |
| overfitting | 1 | pr | +0.036 | 0.550 | no (not strongest level / far from threshold) |
| overfitting | 1 | roc | +0.034 | 0.549 | no (not strongest level / far from threshold) |
| overfitting | 2 | pr | +0.036 | 0.571 | no (not strongest level / far from threshold) |

Cells at the strongest level that touch a conclusion: 1.

