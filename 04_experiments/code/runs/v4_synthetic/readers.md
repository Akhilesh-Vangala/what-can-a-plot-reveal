# AI readers — synthetic data

| reader | items | balanced acc | per-plot balanced acc | stereotype index (healthy plots) | over-claim index | confident-where-ambiguous | cannot-tell | hedging AUC |
|---|---|---|---|---|---|---|---|---|
| InternVL3-38B-AWQ_core | 4208 | 0.16 | cm 0.14, dist 0.15, lc_loss 0.23, rel 0.14, roc 0.14 | cm 100%→health, dist 99%→miscal, lc_loss 78%→health, rel 100%→miscal, roc 100%→health | 0.43 | 0.40 | 0.22 | 0.60 |
| InternVL3-8B | 12410 | 0.15 | cm 0.14, cm_norm 0.15, dist 0.15, lc_acc 0.14, lc_loss 0.14, pr 0.14, rel 0.14, rel_n 0.14, roc 0.14 | cm 99%→miscal, cm_norm 70%→miscal, dist 64%→miscal, lc_acc 100%→overfi, lc_loss 100%→overfi, pr 100%→distri, rel 100%→miscal, rel_n 100%→miscal, roc 100%→distri | 0.46 | 0.32 | 0.01 | 0.53 |
| InternVL3-8B_iv | 5809 | 0.14 | cm 0.14, dist 0.14, lc_loss 0.14, rel 0.14, roc 0.14 | cm 58%→class_, dist 71%→overfi, lc_loss 100%→overfi, rel 67%→miscal, roc 66%→miscal | 0.32 | 0.18 | 0.31 | 0.57 |
| InternVL3-8B_variants | 2389 | 0.14 | cm 0.14, lc_loss 0.14, rel 0.14 | cm 58%→health, lc_loss 100%→overfi, rel 96%→miscal | 0.57 | 0.42 | 0.00 | 0.49 |
| Qwen2.5-VL-32B-Instruct-AWQ | 13492 | 0.15 | cm 0.16, cm_norm 0.14, dist 0.14, lc_acc 0.15, lc_loss 0.14, pr 0.15, rel 0.14, rel_n 0.14, roc 0.14 | cm 58%→class_, cm_norm 100%→health, dist 100%→miscal, lc_acc 98%→overfi, lc_loss 100%→overfi, pr 75%→class_, rel 99%→miscal, rel_n 97%→miscal, roc 100%→health | 0.43 | 0.32 | 0.59 | 0.65 |
| Qwen2.5-VL-32B-Instruct-AWQ_iv | 5941 | 0.14 | cm 0.13, dist 0.14, lc_loss 0.14, rel 0.14, roc 0.14 | cm 41%→miscal, dist 62%→overfi, lc_loss 34%→overfi, rel 95%→miscal, roc 60%→miscal | 0.40 | 0.22 | 0.50 | 0.42 |
| Qwen2.5-VL-7B-Instruct | 10522 | 0.14 | cm 0.17, cm_norm 0.14, dist 0.14, lc_acc 0.14, lc_loss 0.14, pr 0.14, rel 0.14, rel_n 0.14, roc 0.14 | cm 100%→health, cm_norm 97%→health, dist 97%→overfi, lc_acc 100%→overfi, lc_loss 100%→overfi, pr 100%→health, rel 97%→health, rel_n 98%→health, roc 100%→health | 0.23 | 0.25 | 0.02 | 0.50 |
| Qwen2.5-VL-7B-Instruct_iv | 5612 | 0.15 | cm 0.15, dist 0.15, lc_loss 0.14, rel 0.15, roc 0.14 | cm 97%→health, dist 84%→overfi, lc_loss 100%→overfi, rel 55%→overfi, roc 99%→health | 0.21 | 0.15 | 0.51 | 0.42 |
| Qwen2.5-VL-7B-Instruct_variants | 2389 | 0.15 | cm 0.15, lc_loss 0.16, rel 0.15 | cm 46%→label_, lc_loss 91%→overfi, rel 68%→overfi | 0.42 | 0.28 | 0.01 | 0.60 |
| llava-onevision-qwen2-7b-ov-hf | 11703 | 0.14 | cm 0.13, cm_norm 0.15, dist 0.00, lc_acc 0.14, lc_loss 0.14, pr 0.14, rel 0.14, rel_n 0.14, roc 0.14 | cm 91%→health, cm_norm 87%→health, dist 100%→overfi, lc_acc 100%→overfi, lc_loss 100%→overfi, pr 81%→label_, rel 100%→label_, rel_n 99%→label_, roc 100%→overfi | 0.24 | 0.03 | 0.00 | 0.32 |
| llava-onevision-qwen2-7b-ov-hf_iv | 4278 | 0.14 | cm 0.14, dist 0.15, lc_loss 0.14, rel 0.14, roc 0.14 | cm 94%→health, dist 82%→overfi, lc_loss 100%→health, rel 37%→label_, roc 100%→overfi | 0.32 | 0.15 | 0.17 | 0.50 |
