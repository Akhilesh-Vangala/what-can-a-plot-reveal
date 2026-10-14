# Reader: OpenGVLab/InternVL3-8B  (1650 responses)

Unparseable answers: 0

## Lineups (detection)

| set | spec | level | n | hits | null false alarms | choices |
|---|---|---|---|---|---|---|
| lineup | lc_acc | null | 8 | – | 8/8 | [18, 19, 18, 19, 18, 19, 17, 18] |
| lineup | lc_acc | 0 | 8 | 1/8 | – | [19, 13, 10, 13, 13, 17, 10, 17] vs real [3, 2, 10, 17, 8, 9, 5, 12] |
| lineup | lc_acc | 1 | 8 | 0/8 | – | [18, 19, 18, 17, 13, 1, 17, 17] vs real [15, 10, 10, 14, 5, 2, 13, 1] |
| lineup | lc_acc | 2 | 8 | 1/8 | – | [16, 11, 18, 19, 18, 14, 13, 14] vs real [15, 12, 14, 17, 1, 2, 13, 13] |
| lineup | lc_loss | null | 8 | – | 8/8 | [17, 18, 11, 14, 18, 15, 14, 11] |
| lineup | lc_loss | 0 | 8 | 1/8 | – | [13, 19, 13, 17, 18, 13, 15, 15] vs real [11, 1, 12, 9, 8, 13, 19, 10] |
| lineup | lc_loss | 1 | 8 | 2/8 | – | [17, 17, 10, 17, 16, 11, 11, 18] vs real [7, 9, 1, 17, 14, 11, 6, 5] |
| lineup | lc_loss | 2 | 8 | 1/8 | – | [13, 17, 15, 18, 16, 19, 17, 11] vs real [8, 15, 9, 16, 17, 19, 13, 8] |
| lineup_shuffled | lc_acc | null | 8 | – | 8/8 | [18, 12, 18, 3, 1, 18, 17, 13] |
| lineup_shuffled | lc_acc | 0 | 8 | 0/8 | – | [18, 17, 2, 13, 18, 17, 19, 18] vs real [5, 7, 6, 18, 14, 14, 12, 3] |
| lineup_shuffled | lc_acc | 1 | 8 | 0/8 | – | [19, 17, 13, 18, 18, 19, 17, 7] vs real [17, 19, 11, 7, 9, 7, 19, 9] |
| lineup_shuffled | lc_acc | 2 | 8 | 0/8 | – | [7, 3, 18, 17, 1, 5, 17, 17] vs real [2, 10, 10, 16, 18, 1, 11, 5] |
| lineup_shuffled | lc_loss | null | 8 | – | 8/8 | [13, 18, 17, 14, 11, 5, 17, 17] |
| lineup_shuffled | lc_loss | 0 | 8 | 0/8 | – | [14, 14, 11, 18, 18, 17, 15, 18] vs real [19, 18, 17, 1, 17, 14, 20, 6] |
| lineup_shuffled | lc_loss | 1 | 8 | 1/8 | – | [17, 11, 11, 15, 16, 11, 18, 18] vs real [15, 16, 2, 19, 10, 6, 18, 11] |
| lineup_shuffled | lc_loss | 2 | 8 | 1/8 | – | [11, 7, 17, 18, 18, 16, 13, 13] vs real [6, 14, 11, 17, 1, 16, 1, 19] |

Most chosen positions: [(18, 30), (17, 27), (13, 15)] of 128 picks (uniform would be ~6.4 each)
Shuffled control (64 pairs): picked the same PLOT 9/64, the same POSITION 16/64 (chance ≈ 3.2 each)

## Single-plot diagnosis (P(overfitting) as detection score; held-out test items)

| spec | level | reader AUC | B1 AUC | B2 AUC (best) | mean P(overfit) failed / healthy | P(other failures) | cannot_tell | mean top prob |
|---|---|---|---|---|---|---|---|---|
| lc_loss | 0.65 | 0.408 | 0.690 | 0.693 | 0.72 / 0.74 | 0.17 | 0.00 | 0.74 |
| lc_loss | 0.8 | 0.405 | 0.820 | 0.763 | 0.72 / 0.74 | 0.17 | 0.00 | 0.74 |
| lc_loss | 0.95 | 0.294 | 0.952 | 0.956 | 0.68 / 0.74 | 0.18 | 0.00 | 0.73 |
| lc_acc | 0.65 | 0.477 | 0.646 | 0.549 | 0.74 / 0.75 | 0.16 | 0.00 | 0.74 |
| lc_acc | 0.8 | 0.493 | 0.781 | 0.713 | 0.74 / 0.75 | 0.16 | 0.00 | 0.75 |
| lc_acc | 0.95 | 0.464 | 0.907 | 0.919 | 0.74 / 0.75 | 0.16 | 0.00 | 0.74 |

## Prior only (no image)

- lc_loss: {'healthy': 0.1, 'overfitting': 0.3, 'label_noise': 0.2, 'class_imbalance': 0.1, 'data_leakage': 0.1, 'miscalibration': 0.1, 'distribution_shift': 0.1, 'cannot_tell': False, 'reason': 'Insufficient information to determine the exact issue.'}
- lc_acc: {'healthy': 0.1, 'overfitting': 0.3, 'label_noise': 0.2, 'class_imbalance': 0.1, 'data_leakage': 0.1, 'miscalibration': 0.1, 'distribution_shift': 0.1, 'cannot_tell': False, 'reason': 'Insufficient information to determine the exact issue.'}

## Value probe (can the reader read the plot?)

| spec | n valid | median |error| last train | last val | best val | median epoch error | corr(read rise, true rise) |
|---|---|---|---|---|---|---|
| lc_loss | 100 | 0.007 | 0.003 | 0.011 | 21 | 0.98 |
| lc_acc | 100 | 0.004 | 0.003 | 0.004 | 13 | 0.72 |
