# Exploratory gap analyses

## Contextual calibration of reader answers

| reader | raw | ÷ no-plot answer | ÷ mean answer per plot |
|---|---|---|---|
| InternVL3-38B-AWQ | 0.165 | nan | 0.150 |
| InternVL3-8B | 0.150 | 0.149 | 0.149 |
| Qwen2.5-VL-32B-Instruct-AWQ | 0.145 | 0.147 | 0.165 |
| Qwen2.5-VL-7B-Instruct | 0.144 | 0.147 | 0.165 |
| llava-onevision-qwen2-7b-ov-hf | 0.142 | 0.147 | 0.149 |

## Gap 4: which cue drives readers' answers (Spearman ρ, held-out core models)

| reader | loss: ρ val-rise | loss: ρ train–val gap | reliability: ρ S-shape | reliability: ρ any deviation | healthy loss reasons citing a gap / plateau | healthy reliability reasons citing a deviation |
|---|---|---|---|---|---|---|
| InternVL3-38B-AWQ | 0.67 | 0.79 | nan | nan | 79% | 92% |
| InternVL3-8B | -0.32 | -0.47 | 0.24 | 0.38 | 12% | 0% |
| Qwen2.5-VL-32B-Instruct-AWQ | 0.48 | 0.43 | 0.03 | 0.09 | 94% | 99% |
| Qwen2.5-VL-7B-Instruct | 0.27 | 0.20 | 0.20 | 0.25 | 97% | 44% |
| llava-onevision-qwen2-7b-ov-hf | 0.06 | 0.03 | nan | nan | 0% | 0% |

Healthy held-out models with a train–val loss gap > 0.05: 84%; healthy reliability diagrams with mean |deviation| > 0.05: 69%.

## Gap 6: real-world cases with EXACT values (no reader), rule trained on synthetic models

| case | textbook plot | detection AUC | failure named | control false alarm | controls mostly called |
|---|---|---|---|---|---|
| elec_shift | dist | 0.03 | 0% | 100% | class_imbalance |
| bank_shift | dist | 0.00 | 0% | 0% | class_imbalance |
| mammo_imb | cm | 1.00 | 100% | 80% | miscalibration |
| ozone_imb | cm | 1.00 | 100% | 75% | distribution_shift |
| pc1_imb | cm | 1.00 | 100% | 75% | distribution_shift |
| kc1_miscal | rel | 0.36 | 0% | 100% | distribution_shift |
| bank_miscal | rel | 0.47 | 0% | 95% | distribution_shift |
| ozone_leak | roc | 0.99 | 90% | 80% | label_noise |
| elec_overfit | lc_loss | 1.00 | 100% | 100% | label_noise |
| diabetes_overfit | lc_loss | 1.00 | 100% | 100% | label_noise |

Overall: detection AUC ≥ 0.70 in 60% of cases, failure named 59%, control false alarm 80%.

## Gap 7: where accuracy curves carry class-imbalance information (fresh models, strongest calibrated level)

- synthetic: final val accuracy AUC 0.52; best single-epoch train accuracy AUC 0.52; healthy accuracy on positives 0.900 vs negatives 0.901
- magic: final val accuracy AUC 0.65; best single-epoch train accuracy AUC 0.65; healthy accuracy on positives 0.792 vs negatives 0.858

## Gap 8: several plots' readings combined (held-out core models)

| reader | loss+CM, exact-value rule | all five, exact-value rule | all five, own-reading rule (5-fold CV) |
|---|---|---|---|
| InternVL3-38B-AWQ | 0.596 | 0.347 | 0.608 |
| InternVL3-8B | 0.569 | 0.252 | 0.594 |
| Qwen2.5-VL-32B-Instruct-AWQ | 0.549 | 0.283 | 0.574 |
| Qwen2.5-VL-7B-Instruct | 0.567 | 0.189 | 0.581 |
| llava-onevision-qwen2-7b-ov-hf | 0.384 | 0.210 | 0.402 |

Exact-value ceilings: lc_loss+cm 0.597, all five 0.760

