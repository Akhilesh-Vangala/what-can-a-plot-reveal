# Read-then-reason: readers read values, a fixed rule diagnoses (exploratory)

## Synthetic core set (strongest level, held-out models)

| reader | lc_loss | rel | cm | roc | dist | all plots | readable answers |
|---|---|---|---|---|---|---|---|
| exact values (ceiling) | 0.349 | 0.456 | 0.416 | 0.339 | 0.406 | 0.393 | — |
| InternVL3-38B-AWQ | 0.349 | 0.261 | 0.409 | 0.207 | 0.157 | 0.277 | 100% |
| InternVL3-8B | 0.341 | 0.163 | 0.409 | 0.200 | 0.121 | 0.247 | 100% |
| Qwen2.5-VL-32B-Instruct-AWQ | 0.327 | 0.299 | 0.409 | 0.216 | 0.211 | 0.292 | 100% |
| Qwen2.5-VL-7B-Instruct | 0.350 | 0.177 | 0.399 | 0.144 | 0.150 | 0.244 | 100% |
| llava-onevision-qwen2-7b-ov-hf | 0.261 | 0.143 | 0.352 | 0.143 | 0.144 | 0.209 | 88% |

Reading fidelity (correlation between read and exact values; selected quantities):

- InternVL3-38B-AWQ: lc_loss:train_last 0.99, lc_loss:val_last 1.00, lc_loss:val_best 0.99, lc_loss:val_best_epoch 0.75, cm:tn 1.00, cm:fp 1.00, cm:fn 1.00, cm:tp 1.00, roc:tpr_at_0.05 0.82, roc:tpr_at_0.1 0.84, roc:tpr_at_0.2 0.82, roc:tpr_at_0.5 0.48
- InternVL3-8B: lc_loss:train_last 0.99, lc_loss:val_last 1.00, lc_loss:val_best 0.95, lc_loss:val_best_epoch 0.35, cm:tn 0.95, cm:fp 0.89, cm:fn 0.85, cm:tp 1.00, roc:tpr_at_0.05 0.82, roc:tpr_at_0.1 0.78, roc:tpr_at_0.2 0.79, roc:tpr_at_0.5 0.36
- Qwen2.5-VL-32B-Instruct-AWQ: lc_loss:train_last 0.95, lc_loss:val_last 0.98, lc_loss:val_best 0.96, lc_loss:val_best_epoch 0.61, cm:tn 1.00, cm:fp 1.00, cm:fn 1.00, cm:tp 1.00, roc:tpr_at_0.05 0.85, roc:tpr_at_0.1 0.80, roc:tpr_at_0.2 0.72, roc:tpr_at_0.5 0.54
- Qwen2.5-VL-7B-Instruct: lc_loss:train_last 0.97, lc_loss:val_last 0.99, lc_loss:val_best 0.96, lc_loss:val_best_epoch 0.70, cm:tn 0.93, cm:fp 0.41, cm:fn 0.50, cm:tp 1.00, roc:tpr_at_0.05 0.80, roc:tpr_at_0.1 0.72, roc:tpr_at_0.2 0.28, roc:tpr_at_0.5 -0.32
- llava-onevision-qwen2-7b-ov-hf: lc_loss:train_last 0.87, lc_loss:val_last 0.68, lc_loss:val_best 0.74, lc_loss:val_best_epoch 0.35, cm:tn 0.13, cm:fp 0.18, cm:fn 0.25, cm:tp 0.55, roc:tpr_at_0.05 -0.05, roc:tpr_at_0.1 -0.04, roc:tpr_at_0.5 0.03, dist:val_below_0_1 0.00

## Positive control (rule trained on synthetic models, applied unchanged)

- InternVL3-38B-AWQ (synthetic): 2/5 primary failures passed (direct reading earlier: ≤ 1/5)
- InternVL3-38B-AWQ (magic): 1/5 primary failures passed (direct reading earlier: ≤ 1/5)
- InternVL3-8B (synthetic): 2/5 primary failures passed (direct reading earlier: ≤ 1/5)
- InternVL3-8B (magic): 1/5 primary failures passed (direct reading earlier: ≤ 1/5)
- Qwen2.5-VL-32B-Instruct-AWQ (synthetic): 2/5 primary failures passed (direct reading earlier: ≤ 1/5)
- Qwen2.5-VL-32B-Instruct-AWQ (magic): 1/5 primary failures passed (direct reading earlier: ≤ 1/5)
- Qwen2.5-VL-7B-Instruct (synthetic): 2/5 primary failures passed (direct reading earlier: ≤ 1/5)
- Qwen2.5-VL-7B-Instruct (magic): 1/5 primary failures passed (direct reading earlier: ≤ 1/5)
- llava-onevision-qwen2-7b-ov-hf (synthetic): 0/5 primary failures passed (direct reading earlier: ≤ 1/5)
- llava-onevision-qwen2-7b-ov-hf (magic): 0/5 primary failures passed (direct reading earlier: ≤ 1/5)

## Real-world cases (rule trained on synthetic models, applied unchanged)

- InternVL3-38B-AWQ: textbook-plot detection AUC ≥ 0.70 in 60% of cases; failure named in 59% of failure models; control false alarm 74%
- InternVL3-8B: textbook-plot detection AUC ≥ 0.70 in 60% of cases; failure named in 60% of failure models; control false alarm 79%
- Qwen2.5-VL-32B-Instruct-AWQ: textbook-plot detection AUC ≥ 0.70 in 60% of cases; failure named in 58% of failure models; control false alarm 80%
- Qwen2.5-VL-7B-Instruct: textbook-plot detection AUC ≥ 0.70 in 60% of cases; failure named in 49% of failure models; control false alarm 80%
- llava-onevision-qwen2-7b-ov-hf: textbook-plot detection AUC ≥ 0.70 in 20% of cases; failure named in 37% of failure models; control false alarm 82%

