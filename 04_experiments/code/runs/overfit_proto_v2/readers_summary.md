# Reader comparison

| metric | InternVL3-8B | Qwen2.5-VL-32B-Instruct-AWQ | Qwen2.5-VL-7B-Instruct | llava-onevision-qwen2-7b-ov-hf |
|---|---|---|---|---|
| auc_lc_acc_L0 | 0.477 | 0.543 | 0.537 | 0.545 |
| auc_lc_acc_L1 | 0.493 | 0.575 | 0.573 | 0.592 |
| auc_lc_acc_L2 | 0.464 | 0.68 | 0.604 | 0.592 |
| auc_lc_loss_L0 | 0.408 | 0.52 | 0.505 | 0.504 |
| auc_lc_loss_L1 | 0.405 | 0.545 | 0.538 | 0.505 |
| auc_lc_loss_L2 | 0.294 | 0.634 | 0.626 | 0.479 |
| cannot_tell_lc_acc | 0.0 | 1.0 | 0.0 | 0.0 |
| cannot_tell_lc_loss | 0.0 | 0.93 | 0.0 | 0.0 |
| diag_valid | 1320/1320 | 1320/1320 | 1320/1320 | 1319/1320 |
| follows_plot | 9/64 | 13/64 | 10/64 | 6/64 |
| follows_position | 16/64 | 5/64 | 10/64 | 34/64 |
| lineup_hits | 6/48 | 2/48 | 7/48 | 2/48 |
| null_false_alarms | 16/16 | 16/16 | 16/16 | 16/16 |
| p_overfit_healthy_lc_acc | 0.75 | 0.38 | 0.78 | 0.54 |
| p_overfit_healthy_lc_loss | 0.74 | 0.56 | 0.81 | 0.3 |
| probe_rise_corr_lc_acc | 0.72 | 0.5 | 0.49 | 0.22 |
| probe_rise_corr_lc_loss | 0.98 | 0.96 | 0.96 | -0.18 |
| probe_val_err_lc_acc | 0.003 | 0.01 | 0.006 | 0.031 |
| probe_val_err_lc_loss | 0.003 | 0.013 | 0.01 | 0.059 |
| top_position | 18 (30/128) | 20 (25/128) | 17 (45/128) | 16 (73/128) |
