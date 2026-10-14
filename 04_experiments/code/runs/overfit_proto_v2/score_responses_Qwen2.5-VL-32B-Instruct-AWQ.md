# Reader: Qwen/Qwen2.5-VL-32B-Instruct-AWQ  (1650 responses)

Unparseable answers: 0

## Lineups (detection)

| set | spec | level | n | hits | null false alarms | choices |
|---|---|---|---|---|---|---|
| lineup | lc_acc | null | 8 | – | 8/8 | [14, 7, 1, 14, 1, 15, 16, 5] |
| lineup | lc_acc | 0 | 8 | 0/8 | – | [10, 4, 20, 11, 13, 5, 13, 13] vs real [3, 2, 10, 17, 8, 9, 5, 12] |
| lineup | lc_acc | 1 | 8 | 1/8 | – | [19, 20, 5, 10, 11, 20, 15, 1] vs real [15, 10, 10, 14, 5, 2, 13, 1] |
| lineup | lc_acc | 2 | 8 | 1/8 | – | [15, 11, 1, 20, 18, 9, 20, 20] vs real [15, 12, 14, 17, 1, 2, 13, 13] |
| lineup | lc_loss | null | 8 | – | 8/8 | [15, 20, 1, 15, 20, 10, 1, 15] |
| lineup | lc_loss | 0 | 8 | 0/8 | – | [2, 19, 5, 19, 20, 10, 10, 20] vs real [11, 1, 12, 9, 8, 13, 19, 10] |
| lineup | lc_loss | 1 | 8 | 0/8 | – | [1, 5, 17, 2, 19, 20, 15, 19] vs real [7, 9, 1, 17, 14, 11, 6, 5] |
| lineup | lc_loss | 2 | 8 | 0/8 | – | [15, 17, 10, 18, 10, 14, 16, 11] vs real [8, 15, 9, 16, 17, 19, 13, 8] |
| lineup_shuffled | lc_acc | null | 8 | – | 8/8 | [10, 6, 17, 3, 4, 20, 15, 1] |
| lineup_shuffled | lc_acc | 0 | 8 | 0/8 | – | [16, 1, 2, 20, 15, 18, 13, 20] vs real [5, 7, 6, 18, 14, 14, 12, 3] |
| lineup_shuffled | lc_acc | 1 | 8 | 0/8 | – | [5, 15, 14, 19, 20, 20, 20, 5] vs real [17, 19, 11, 7, 9, 7, 19, 9] |
| lineup_shuffled | lc_acc | 2 | 8 | 1/8 | – | [10, 3, 15, 1, 20, 5, 10, 5] vs real [2, 10, 10, 16, 18, 1, 11, 5] |
| lineup_shuffled | lc_loss | null | 8 | – | 8/8 | [19, 19, 20, 4, 5, 5, 20, 5] |
| lineup_shuffled | lc_loss | 0 | 8 | 0/8 | – | [5, 14, 20, 19, 1, 15, 19, 19] vs real [19, 18, 17, 1, 17, 14, 20, 6] |
| lineup_shuffled | lc_loss | 1 | 8 | 1/8 | – | [13, 5, 20, 20, 20, 6, 1, 19] vs real [15, 16, 2, 19, 10, 6, 18, 11] |
| lineup_shuffled | lc_loss | 2 | 8 | 0/8 | – | [20, 6, 19, 1, 5, 5, 15, 8] vs real [6, 14, 11, 17, 1, 16, 1, 19] |

Most chosen positions: [(20, 25), (5, 16), (15, 14)] of 128 picks (uniform would be ~6.4 each)
Shuffled control (64 pairs): picked the same PLOT 13/64, the same POSITION 5/64 (chance ≈ 3.2 each)

## Single-plot diagnosis (P(overfitting) as detection score; held-out test items)

| spec | level | reader AUC | B1 AUC | B2 AUC (best) | mean P(overfit) failed / healthy | P(other failures) | cannot_tell | mean top prob |
|---|---|---|---|---|---|---|---|---|
| lc_loss | 0.65 | 0.520 | 0.690 | 0.693 | 0.57 / 0.56 | 0.31 | 0.97 | 0.56 |
| lc_loss | 0.8 | 0.545 | 0.820 | 0.763 | 0.58 / 0.56 | 0.31 | 0.96 | 0.57 |
| lc_loss | 0.95 | 0.634 | 0.952 | 0.956 | 0.62 / 0.56 | 0.31 | 0.93 | 0.58 |
| lc_acc | 0.65 | 0.543 | 0.646 | 0.549 | 0.40 / 0.38 | 0.42 | 1.00 | 0.39 |
| lc_acc | 0.8 | 0.575 | 0.781 | 0.713 | 0.42 / 0.38 | 0.42 | 1.00 | 0.40 |
| lc_acc | 0.95 | 0.680 | 0.907 | 0.919 | 0.45 / 0.38 | 0.41 | 1.00 | 0.41 |

## Prior only (no image)

- lc_loss: {'healthy': 0.1, 'overfitting': 0.2, 'label_noise': 0.1, 'class_imbalance': 0.1, 'data_leakage': 0.1, 'miscalibration': 0.1, 'distribution_shift': 0.1, 'cannot_tell': True, 'reason': 'The plot is hidden, so no specific information about the loss curves is available to make a definitive assessment.'}
- lc_acc: {'healthy': 0.1, 'overfitting': 0.2, 'label_noise': 0.1, 'class_imbalance': 0.1, 'data_leakage': 0.1, 'miscalibration': 0.1, 'distribution_shift': 0.1, 'cannot_tell': True, 'reason': "The plot is hidden, so no specific information about the model's behavior is available."}

## Value probe (can the reader read the plot?)

| spec | n valid | median |error| last train | last val | best val | median epoch error | corr(read rise, true rise) |
|---|---|---|---|---|---|---|
| lc_loss | 100 | 0.017 | 0.013 | 0.016 | 18 | 0.96 |
| lc_acc | 100 | 0.007 | 0.010 | 0.010 | 26 | 0.50 |
