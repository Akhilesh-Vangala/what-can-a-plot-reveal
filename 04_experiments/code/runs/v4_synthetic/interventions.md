# Design interventions — readers (strongest severity + healthy; 5 core plots; 7 states, chance 0.14)

## InternVL3-8B

| condition | n | balanced acc | Δ vs plain [95% CI] | P(true) | 'overfitting' on healthy loss curves | over-claim | Δ over-claim [95% CI] | confident-where-ambiguous | cannot-tell |
|---|---|---|---|---|---|---|---|---|---|
| plain | 1496 | 0.156 | – | 0.130 | 100% | 0.384 | – | 0.23 | 0.00 |
| badge | 1809 | 0.147 | -0.008 [-0.019, +0.001] | 0.154 | 100% | 0.378 | -0.005 [-0.014, +0.009] | 0.24 | 0.46 |
| companion | 2000 | 0.141 | -0.015 [-0.025, -0.005] | 0.137 | 100% | 0.341 | -0.043 [-0.048, -0.031] | 0.14 | 0.00 |
| badge_companion | 2000 | 0.141 | -0.015 [-0.028, -0.004] | 0.135 | 100% | 0.271 | -0.113 [-0.123, -0.101] | 0.16 | 0.47 |

## Qwen2.5-VL-32B-Instruct-AWQ

| condition | n | balanced acc | Δ vs plain [95% CI] | P(true) | 'overfitting' on healthy loss curves | over-claim | Δ over-claim [95% CI] | confident-where-ambiguous | cannot-tell |
|---|---|---|---|---|---|---|---|---|---|
| plain | 1737 | 0.146 | – | 0.170 | 100% | 0.516 | – | 0.38 | 0.43 |
| badge | 1960 | 0.145 | -0.001 [-0.009, +0.008] | 0.157 | 0% | 0.479 | -0.037 [-0.045, -0.023] | 0.26 | 0.36 |
| companion | 2000 | 0.143 | -0.003 [-0.014, +0.007] | 0.140 | 100% | 0.240 | -0.276 [-0.288, -0.267] | 0.12 | 0.80 |
| badge_companion | 1981 | 0.135 | -0.011 [-0.019, -0.001] | 0.136 | 3% | 0.465 | -0.051 [-0.067, -0.043] | 0.27 | 0.34 |

## Qwen2.5-VL-7B-Instruct

| condition | n | balanced acc | Δ vs plain [95% CI] | P(true) | 'overfitting' on healthy loss curves | over-claim | Δ over-claim [95% CI] | confident-where-ambiguous | cannot-tell |
|---|---|---|---|---|---|---|---|---|---|
| plain | 1361 | 0.144 | – | 0.143 | 100% | 0.231 | – | 0.26 | 0.01 |
| badge | 1612 | 0.154 | +0.010 [+0.004, +0.017] | 0.144 | 100% | 0.237 | +0.006 [-0.002, +0.021] | 0.16 | 0.58 |
| companion | 2000 | 0.135 | -0.009 [-0.014, -0.000] | 0.138 | 100% | 0.217 | -0.014 [-0.022, -0.000] | 0.13 | 0.32 |
| badge_companion | 2000 | 0.148 | +0.003 [-0.003, +0.012] | 0.135 | 100% | 0.185 | -0.046 [-0.055, -0.031] | 0.14 | 0.63 |

## llava-onevision-qwen2-7b-ov-hf

| condition | n | balanced acc | Δ vs plain [95% CI] | P(true) | 'overfitting' on healthy loss curves | over-claim | Δ over-claim [95% CI] | confident-where-ambiguous | cannot-tell |
|---|---|---|---|---|---|---|---|---|---|
| plain | 1328 | 0.139 | – | 0.153 | 100% | 0.203 | – | 0.01 | 0.00 |
| badge | 1034 | 0.136 | -0.004 [-0.014, +0.001] | 0.180 | 0% | 0.368 | +0.165 [+0.150, +0.181] | 0.23 | 0.19 |
| companion | 1981 | 0.142 | +0.002 [-0.001, +0.006] | 0.166 | 1% | 0.251 | +0.048 [+0.036, +0.058] | 0.05 | 0.00 |
| badge_companion | 1263 | 0.127 | -0.013 [-0.022, -0.001] | 0.181 | 0% | 0.391 | +0.188 [+0.175, +0.202] | 0.23 | 0.44 |

