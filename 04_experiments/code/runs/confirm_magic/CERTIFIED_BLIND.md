# Blind by construction — confirm_magic

30 failing models per failure at the strongest severity, each re-trained next to its healthy twin (same configuration without the planted change; re-trained models match the study exactly).
Entry = largest absolute difference in any plotted value between a failing model and its twin. **0** in every model = the plot is identical, so no reader or benchmark can detect the failure from it.

| failure | lc_loss | lc_acc | rel | rel_n | cm | cm_norm | roc | pr | dist |
|---|---|---|---|---|---|---|---|---|---|
| overfitting | 0.174 | 0.263 | 1.11 | 111 | 45 | 0.0938 | 0.258 | 0.0688 | 2.96 |
| miscalibration | 0.0907 | **0** | 0.119 | 67 | **0** | **0** | **0** | **0** | 1.18 |
| label_noise | 0.0715 | 0.036 | 0.15 | 58 | 17 | 0.0318 | 0.366 | 0.0703 | 1.6 |
| class_imbalance | 0.0729 | 0.109 | 1.09 | 140 | 118 | 0.15 | 0.261 | 0.126 | 2.3 |
| data_leakage | 0.142 | 0.257 | 0.296 | 89 | 44 | 0.0967 | 0.231 | 0.109 | 1.96 |
| distribution_shift | **0** | **0** | 0.462 | 185 | 171 | 0.315 | 0.255 | 0.198 | 2.72 |

Certified blind cells (7): distribution_shift on lc_acc, distribution_shift on lc_loss, miscalibration on cm, miscalibration on cm_norm, miscalibration on lc_acc, miscalibration on pr, miscalibration on roc
