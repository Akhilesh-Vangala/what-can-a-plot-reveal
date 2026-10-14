# Reader: llava-hf/llava-onevision-qwen2-7b-ov-hf  (1650 responses)

Unparseable answers: 0

## Lineups (detection)

| set | spec | level | n | hits | null false alarms | choices |
|---|---|---|---|---|---|---|
| lineup | lc_acc | null | 8 | – | 8/8 | [16, 12, 16, 18, 1, 16, 1, 16] |
| lineup | lc_acc | 0 | 8 | 0/8 | – | [16, 16, 16, 16, 16, 1, 16, 16] vs real [3, 2, 10, 17, 8, 9, 5, 12] |
| lineup | lc_acc | 1 | 8 | 1/8 | – | [18, 16, 1, 16, 16, 1, 16, 1] vs real [15, 10, 10, 14, 5, 2, 13, 1] |
| lineup | lc_acc | 2 | 8 | 0/8 | – | [16, 16, 16, 16, 16, 16, 1, 1] vs real [15, 12, 14, 17, 1, 2, 13, 13] |
| lineup | lc_loss | null | 8 | – | 8/8 | [1, 16, 16, 16, 1, 1, 10, 16] |
| lineup | lc_loss | 0 | 8 | 0/8 | – | [1, 16, 1, 18, 16, 1, 16, 16] vs real [11, 1, 12, 9, 8, 13, 19, 10] |
| lineup | lc_loss | 1 | 8 | 1/8 | – | [1, 1, 1, 16, 15, 16, 16, 16] vs real [7, 9, 1, 17, 14, 11, 6, 5] |
| lineup | lc_loss | 2 | 8 | 0/8 | – | [16, 16, 16, 1, 16, 16, 1, 16] vs real [8, 15, 9, 16, 17, 19, 13, 8] |
| lineup_shuffled | lc_acc | null | 8 | – | 8/8 | [16, 16, 16, 16, 1, 10, 10, 1] |
| lineup_shuffled | lc_acc | 0 | 8 | 0/8 | – | [1, 16, 1, 16, 16, 16, 1, 16] vs real [5, 7, 6, 18, 14, 14, 12, 3] |
| lineup_shuffled | lc_acc | 1 | 8 | 0/8 | – | [16, 1, 16, 16, 10, 1, 1, 1] vs real [17, 19, 11, 7, 9, 7, 19, 9] |
| lineup_shuffled | lc_acc | 2 | 8 | 2/8 | – | [1, 16, 1, 16, 16, 1, 16, 16] vs real [2, 10, 10, 16, 18, 1, 11, 5] |
| lineup_shuffled | lc_loss | null | 8 | – | 8/8 | [10, 15, 16, 1, 1, 1, 1, 16] |
| lineup_shuffled | lc_loss | 0 | 8 | 0/8 | – | [1, 16, 1, 16, 16, 1, 16, 16] vs real [19, 18, 17, 1, 17, 14, 20, 6] |
| lineup_shuffled | lc_loss | 1 | 8 | 0/8 | – | [1, 1, 16, 16, 16, 16, 16, 1] vs real [15, 16, 2, 19, 10, 6, 18, 11] |
| lineup_shuffled | lc_loss | 2 | 8 | 2/8 | – | [16, 16, 16, 16, 1, 1, 1, 16] vs real [6, 14, 11, 17, 1, 16, 1, 19] |

Most chosen positions: [(16, 73), (1, 44), (10, 5)] of 128 picks (uniform would be ~6.4 each)
Shuffled control (64 pairs): picked the same PLOT 6/64, the same POSITION 34/64 (chance ≈ 3.2 each)

## Single-plot diagnosis (P(overfitting) as detection score; held-out test items)

| spec | level | reader AUC | B1 AUC | B2 AUC (best) | mean P(overfit) failed / healthy | P(other failures) | cannot_tell | mean top prob |
|---|---|---|---|---|---|---|---|---|
| lc_loss | 0.65 | 0.504 | 0.690 | 0.693 | 0.30 / 0.30 | 0.60 | 0.00 | 0.30 |
| lc_loss | 0.8 | 0.505 | 0.820 | 0.763 | 0.30 / 0.30 | 0.60 | 0.00 | 0.30 |
| lc_loss | 0.95 | 0.479 | 0.952 | 0.956 | 0.30 / 0.30 | 0.60 | 0.00 | 0.30 |
| lc_acc | 0.65 | 0.545 | 0.646 | 0.549 | 0.58 / 0.54 | 0.35 | 0.00 | 0.55 |
| lc_acc | 0.8 | 0.592 | 0.781 | 0.713 | 0.60 / 0.54 | 0.35 | 0.00 | 0.56 |
| lc_acc | 0.95 | 0.592 | 0.907 | 0.919 | 0.61 / 0.54 | 0.34 | 0.00 | 0.56 |

## Prior only (no image)

- lc_loss: {'healthy': 0.3, 'overfitting': 0.2, 'label_noise': 0.1, 'class_imbalance': 0.1, 'data_leakage': 0.1, 'miscalibration': 0.1, 'distribution_shift': 0.1, 'cannot_tell': False, 'reason': 'Insufficient information to determine the exact state of the model.'}
- lc_acc: {'healthy': 0.3, 'overfitting': 0.2, 'label_noise': 0.1, 'class_imbalance': 0.1, 'data_leakage': 0.1, 'miscalibration': 0.1, 'distribution_shift': 0.1, 'cannot_tell': False, 'reason': 'Insufficient information to determine the exact state of the model.'}

## Value probe (can the reader read the plot?)

| spec | n valid | median |error| last train | last val | best val | median epoch error | corr(read rise, true rise) |
|---|---|---|---|---|---|---|
| lc_loss | 100 | 0.033 | 0.059 | 0.076 | 20 | -0.18 |
| lc_acc | 100 | 0.033 | 0.031 | 0.036 | 20 | 0.22 |
