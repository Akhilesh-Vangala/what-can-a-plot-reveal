# Robustness checks (exploratory)

## (a) Read-then-reason minus direct diagnosis

Same models and plots, pooled over the 5 core plots; no-answer counted as wrong for the direct reader; 95% bootstrap CI over models (500 resamples).

| reader | gain | 95% CI | items |
|---|---|---|---|
| InternVL3-38B-AWQ | +0.121 | [+0.111, +0.135] | 4500 |
| InternVL3-8B | +0.117 | [+0.100, +0.133] | 3175 |
| Qwen2.5-VL-32B-Instruct-AWQ | +0.146 | [+0.125, +0.164] | 3175 |
| Qwen2.5-VL-7B-Instruct | +0.131 | [+0.117, +0.145] | 3175 |
| llava-onevision-qwen2-7b-ov-hf | +0.123 | [+0.110, +0.137] | 2773 |

## (b) Evidential-map counts under alternative cutoffs (identifiable / detect-only / blind; strongest level)

| run | AUC cutoff | recall cutoff | identifiable | detect-only | blind |
|---|---|---|---|---|---|
| confirm_synthetic | 0.65 | 0.5 | 29 | 10 | 15 |
| confirm_synthetic | 0.70 | 0.5 | 29 | 10 | 15 |
| confirm_synthetic | 0.75 | 0.5 | 27 | 8 | 19 |
| confirm_synthetic | 0.70 | 0.4 | 34 | 5 | 15 |
| confirm_synthetic | 0.70 | 0.6 | 24 | 15 | 15 |
| confirm_magic | 0.65 | 0.5 | 29 | 11 | 14 |
| confirm_magic | 0.70 | 0.5 | 28 | 9 | 17 |
| confirm_magic | 0.75 | 0.5 | 28 | 6 | 20 |
| confirm_magic | 0.70 | 0.4 | 32 | 5 | 17 |
| confirm_magic | 0.70 | 0.6 | 25 | 12 | 17 |
