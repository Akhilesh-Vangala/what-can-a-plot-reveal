# AI readers — magic data

| reader | items | balanced acc | per-plot balanced acc | stereotype index (healthy plots) | over-claim index | confident-where-ambiguous | cannot-tell | hedging AUC |
|---|---|---|---|---|---|---|---|---|
| InternVL3-8B_core | 4185 | 0.14 | cm 0.14, dist 0.14, lc_loss 0.14, rel 0.14, roc 0.14 | cm 98%→miscal, dist 57%→miscal, lc_loss 100%→overfi, rel 100%→miscal, roc 100%→distri | 0.37 | 0.18 | 0.00 | nan |
| Qwen2.5-VL-32B-Instruct-AWQ_core | 4500 | 0.14 | cm 0.13, dist 0.14, lc_loss 0.14, rel 0.15, roc 0.14 | cm 74%→class_, dist 100%→miscal, lc_loss 100%→overfi, rel 92%→miscal, roc 100%→health | 0.46 | 0.34 | 0.56 | 0.58 |
| Qwen2.5-VL-7B-Instruct_core | 3576 | 0.14 | cm 0.20, dist 0.13, lc_loss 0.14, rel 0.15, roc 0.14 | cm 100%→health, dist 98%→overfi, lc_loss 100%→overfi, rel 100%→health, roc 99%→health | 0.17 | 0.14 | 0.02 | 0.46 |
