# Real-world check: cases and verification

| case | dataset | failure | mechanism | statistic | failure | control | verified |
|---|---|---|---|---|---|---|---|
| elec_shift | electricity | distribution_shift | train on the earliest half of the dates, test on the latest quarter | val_minus_test_acc | 0.085 | 0.007 | yes |
| bank_shift | bank-marketing | distribution_shift | rows are in time order: train on the first half, test on the last quarter | val_minus_test_acc | 0.225 | 0.004 | yes |
| mammo_imb | mammography | class_imbalance | natural 2.3% positive rate | pos_rate | 0.023 | 0.500 | yes |
| ozone_imb | ozone-level-8hr | class_imbalance | natural 6.3% positive rate | pos_rate | 0.063 | 0.516 | yes |
| pc1_imb | pc1 | class_imbalance | natural 6.9% positive rate (software defects) | pos_rate | 0.074 | 0.489 | yes |
| kc1_miscal | kc1 | miscalibration | minority class oversampled to 50% in training only | ece_excess | 0.180 | 0.012 | yes |
| bank_miscal | bank-marketing | miscalibration | minority class oversampled to 50% in training only | ece_excess | 0.114 | 0.005 | yes |
| ozone_leak | ozone-level-8hr | data_leakage | minority oversampled BEFORE the train/val/test split | test_minus_deploy_auc | 0.081 | 0.025 | yes |
| madelon_leak | madelon | data_leakage | top-20 features selected on ALL data (incl. test) before splitting | test_minus_deploy_auc | 0.012 | -0.003 | NO |
| elec_overfit | electricity | overfitting | 256-unit network, no regularization, 300 training rows | val_loss_rise | 0.951 | 0.000 | yes |
| diabetes_overfit | diabetes | overfitting | 256-unit network, no regularization, 400 training rows | val_loss_rise | 1.100 | 0.000 | yes |

10/11 cases verified; 3600 reader items.
