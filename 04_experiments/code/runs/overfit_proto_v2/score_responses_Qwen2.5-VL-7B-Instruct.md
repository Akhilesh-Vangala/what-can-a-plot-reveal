# Reader: Qwen/Qwen2.5-VL-7B-Instruct  (1650 responses)

Unparseable answers: 0

## Lineups (detection)

| set | spec | level | n | hits | null false alarms | choices |
|---|---|---|---|---|---|---|
| lineup | lc_acc | null | 8 | – | 8/8 | [14, 15, 13, 17, 19, 19, 17, 13] |
| lineup | lc_acc | 0 | 8 | 1/8 | – | [12, 17, 15, 17, 14, 17, 12, 17] vs real [3, 2, 10, 17, 8, 9, 5, 12] |
| lineup | lc_acc | 1 | 8 | 0/8 | – | [13, 17, 17, 13, 13, 12, 17, 16] vs real [15, 10, 10, 14, 5, 2, 13, 1] |
| lineup | lc_acc | 2 | 8 | 2/8 | – | [15, 19, 17, 7, 13, 14, 17, 13] vs real [15, 12, 14, 17, 1, 2, 13, 13] |
| lineup | lc_loss | null | 8 | – | 8/8 | [17, 17, 17, 10, 17, 15, 18, 17] |
| lineup | lc_loss | 0 | 8 | 0/8 | – | [17, 18, 17, 17, 18, 17, 15, 7] vs real [11, 1, 12, 9, 8, 13, 19, 10] |
| lineup | lc_loss | 1 | 8 | 2/8 | – | [17, 19, 13, 14, 14, 11, 17, 14] vs real [7, 9, 1, 17, 14, 11, 6, 5] |
| lineup | lc_loss | 2 | 8 | 2/8 | – | [8, 17, 17, 18, 16, 19, 15, 12] vs real [8, 15, 9, 16, 17, 19, 13, 8] |
| lineup_shuffled | lc_acc | null | 8 | – | 8/8 | [17, 7, 14, 17, 12, 13, 17, 17] |
| lineup_shuffled | lc_acc | 0 | 8 | 0/8 | – | [7, 17, 14, 17, 17, 18, 17, 12] vs real [5, 7, 6, 18, 14, 14, 12, 3] |
| lineup_shuffled | lc_acc | 1 | 8 | 0/8 | – | [19, 17, 13, 13, 15, 16, 17, 12] vs real [17, 19, 11, 7, 9, 7, 19, 9] |
| lineup_shuffled | lc_acc | 2 | 8 | 1/8 | – | [7, 17, 10, 17, 17, 12, 13, 17] vs real [2, 10, 10, 16, 18, 1, 11, 5] |
| lineup_shuffled | lc_loss | null | 8 | – | 8/8 | [19, 18, 17, 12, 17, 17, 17, 12] |
| lineup_shuffled | lc_loss | 0 | 8 | 0/8 | – | [14, 14, 15, 14, 14, 17, 14, 18] vs real [19, 18, 17, 1, 17, 14, 20, 6] |
| lineup_shuffled | lc_loss | 1 | 8 | 2/8 | – | [18, 6, 11, 17, 16, 6, 18, 18] vs real [15, 16, 2, 19, 10, 6, 18, 11] |
| lineup_shuffled | lc_loss | 2 | 8 | 3/8 | – | [6, 6, 11, 11, 13, 16, 17, 8] vs real [6, 14, 11, 17, 1, 16, 1, 19] |

Most chosen positions: [(17, 45), (14, 13), (13, 13)] of 128 picks (uniform would be ~6.4 each)
Shuffled control (64 pairs): picked the same PLOT 10/64, the same POSITION 10/64 (chance ≈ 3.2 each)

## Single-plot diagnosis (P(overfitting) as detection score; held-out test items)

| spec | level | reader AUC | B1 AUC | B2 AUC (best) | mean P(overfit) failed / healthy | P(other failures) | cannot_tell | mean top prob |
|---|---|---|---|---|---|---|---|---|
| lc_loss | 0.65 | 0.505 | 0.690 | 0.693 | 0.82 / 0.81 | 0.10 | 0.00 | 0.81 |
| lc_loss | 0.8 | 0.538 | 0.820 | 0.763 | 0.82 / 0.81 | 0.10 | 0.00 | 0.82 |
| lc_loss | 0.95 | 0.626 | 0.952 | 0.956 | 0.85 / 0.81 | 0.09 | 0.00 | 0.82 |
| lc_acc | 0.65 | 0.537 | 0.646 | 0.549 | 0.79 / 0.78 | 0.13 | 0.00 | 0.78 |
| lc_acc | 0.8 | 0.573 | 0.781 | 0.713 | 0.80 / 0.78 | 0.12 | 0.00 | 0.79 |
| lc_acc | 0.95 | 0.604 | 0.907 | 0.919 | 0.81 / 0.78 | 0.12 | 0.00 | 0.79 |

## Prior only (no image)

- lc_loss: {'healthy': 0.1, 'overfitting': 0.2, 'label_noise': 0.1, 'class_imbalance': 0.1, 'data_leakage': 0.1, 'miscalibration': 0.1, 'distribution_shift': 0.1, 'cannot_tell': False, 'reason': 'Insufficient information to determine the exact cause of the issue.'}
- lc_acc: {'healthy': 0.1, 'overfitting': 0.2, 'label_noise': 0.1, 'class_imbalance': 0.1, 'data_leakage': 0.1, 'miscalibration': 0.1, 'distribution_shift': 0.1, 'cannot_tell': False, 'reason': 'Insufficient information to determine the exact cause.'}

## Value probe (can the reader read the plot?)

| spec | n valid | median |error| last train | last val | best val | median epoch error | corr(read rise, true rise) |
|---|---|---|---|---|---|---|
| lc_loss | 100 | 0.014 | 0.010 | 0.015 | 6 | 0.96 |
| lc_acc | 100 | 0.013 | 0.006 | 0.006 | 10 | 0.49 |
