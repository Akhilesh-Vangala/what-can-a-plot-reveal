# Blind by construction — confirm_synthetic

30 failing models per failure at the strongest severity, each re-trained next to its healthy twin (same configuration without the planted change; re-trained models match the study exactly).
Entry = largest absolute difference in any plotted value between a failing model and its twin. **0** in every model = the plot is identical, so no reader or benchmark can detect the failure from it.

| failure | lc_loss | lc_acc | rel | rel_n | cm | cm_norm | roc | pr | dist |
|---|---|---|---|---|---|---|---|---|---|
| overfitting | 0.27 | 0.27 | 1 | 247 | 40 | 0.0823 | 0.24 | 0.133 | 6.36 |
| miscalibration | 0.067 | **0** | 0.271 | 67 | **0** | **0** | 0.0464 | 0.0163 | 2.4 |
| label_noise | 0.162 | 0.062 | 0.6 | 79 | 30 | 0.06 | 0.628 | 0.169 | 2.28 |
| class_imbalance | 0.128 | 0.088 | 0.483 | 115 | 103 | 0.0741 | 0.313 | 0.16 | 2.8 |
| data_leakage | 0.24 | 0.228 | 0.556 | 211 | 47 | 0.089 | 0.507 | 0.203 | 3.94 |
| distribution_shift | **0** | **0** | 0.72 | 303 | 270 | 0.524 | 0.432 | 0.246 | 6.28 |

Certified blind cells (5): distribution_shift on lc_acc, distribution_shift on lc_loss, miscalibration on cm, miscalibration on cm_norm, miscalibration on lc_acc
