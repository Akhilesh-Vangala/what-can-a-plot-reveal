# Gaps 3 and 5 (exploratory)

## Gap 5: plots redesigned for machine readers (values printed on the plot)

| reader | design | rtr reliability | rtr ROC | rtr histogram | direct reliability | direct ROC | direct histogram | all five plots (rtr) |
|---|---|---|---|---|---|---|---|---|
| InternVL3-38B-AWQ | original | 0.261 | 0.207 | 0.157 | 0.114 | 0.143 | 0.149 | 0.344 |
| InternVL3-38B-AWQ | annotated | 0.287 | 0.297 | 0.394 | 0.129 | 0.143 | 0.141 | 0.319 |
| InternVL3-8B | original | 0.163 | 0.200 | 0.121 | 0.143 | 0.066 | 0.143 | 0.252 |
| InternVL3-8B | annotated | 0.193 | 0.297 | 0.380 | 0.143 | 0.121 | 0.143 | 0.310 |
| Qwen2.5-VL-32B-Instruct-AWQ | original | 0.299 | 0.216 | 0.211 | 0.144 | 0.142 | 0.143 | 0.279 |
| Qwen2.5-VL-32B-Instruct-AWQ | annotated | 0.253 | 0.238 | 0.322 | 0.145 | 0.140 | 0.143 | 0.264 |
| Qwen2.5-VL-7B-Instruct | original | 0.177 | 0.144 | 0.150 | 0.137 | 0.143 | 0.143 | 0.198 |
| Qwen2.5-VL-7B-Instruct | annotated | 0.175 | 0.244 | 0.228 | 0.129 | 0.145 | 0.124 | 0.230 |
| llava-onevision-qwen2-7b-ov-hf | original | 0.143 | 0.143 | 0.144 | 0.143 | 0.124 | 0.000 | 0.148 |
| llava-onevision-qwen2-7b-ov-hf | annotated | 0.168 | 0.143 | 0.301 | 0.143 | 0.143 | 0.009 | 0.191 |
| exact values (ceiling) | — | 0.456 | 0.339 | 0.406 | | | | 0.760 |

## Gap 3: multi-plot diagnosis benchmark (balanced accuracy, strongest severity)

- confirm_synthetic: best single ['rel_n'] 0.627; best pairs lc_acc+rel_n 0.766; lc_loss+rel_n 0.743; lc_loss+dist 0.695; lc_loss+cm+roc 0.703; lc_loss+lc_acc+cm+roc+pr 0.766; all nine 0.819
- confirm_magic: best single ['rel_n'] 0.650; best pairs lc_loss+rel_n 0.781; lc_loss+dist 0.768; lc_acc+rel_n 0.728; lc_loss+cm+roc 0.661; lc_loss+lc_acc+cm+roc+pr 0.693; all nine 0.791
