# Confirmatory results — v4_synthetic, v4_magic

Each hypothesis as preregistered; ✓ = prediction supported. CIs: bootstrap over held-out test runs.

## H1 [v4_synthetic]: theory-predicted blind spots — NOT SUPPORTED

- miscalibration on roc: AUC 0.437 [0.378, 0.493] → NOT blind ✗
- miscalibration on pr: AUC 0.437 [0.377, 0.495] → NOT blind ✗
- miscalibration on cm: AUC 0.500 [0.444, 0.559] → blind ✓
- miscalibration on lc_acc: AUC 0.449 [0.389, 0.509] → blind ✓
- class_imbalance on lc_loss: AUC 0.501 [0.444, 0.557] → blind ✓
- class_imbalance on lc_acc: AUC 0.546 [0.486, 0.601] → blind ✓
- distribution_shift on lc_loss: AUC 0.511 [0.449, 0.573] → blind ✓
- distribution_shift on lc_acc: AUC 0.488 [0.432, 0.545] → blind ✓

## H2 [v4_synthetic]: detection ≠ diagnosis — SUPPORTED

- (a) detect-only cells: 7 (need ≥ 3) → ✓
- detect-only: overfitting on dist (AUC 0.87, lower CI 0.83; recall 0.38)
- detect-only: miscalibration on rel (AUC 0.82, lower CI 0.78; recall 0.25)
- detect-only: miscalibration on rel_n (AUC 0.80, lower CI 0.76; recall 0.19)
- detect-only: miscalibration on dist (AUC 0.80, lower CI 0.76; recall 0.35)
- detect-only: label_noise on lc_acc (AUC 0.76, lower CI 0.71; recall 0.41)
- detect-only: data_leakage on lc_acc (AUC 0.88, lower CI 0.85; recall 0.36)
- detect-only: data_leakage on dist (AUC 0.88, lower CI 0.84; recall 0.32)
- (b) B0* − lc_loss: +0.337 [+0.304, +0.366] ✓
- (b) B0* − lc_acc: +0.455 [+0.421, +0.494] ✓
- (b) B0* − rel: +0.261 [+0.229, +0.294] ✓
- (b) B0* − rel_n: +0.184 [+0.153, +0.215] ✓
- (b) B0* − cm: +0.351 [+0.322, +0.380] ✓
- (b) B0* − cm_norm: +0.442 [+0.407, +0.473] ✓
- (b) B0* − roc: +0.370 [+0.337, +0.400] ✓
- (b) B0* − pr: +0.304 [+0.276, +0.336] ✓
- (b) B0* − dist: +0.307 [+0.272, +0.339] ✓

## H3 [v4_synthetic]: encoding loss in the calibration plot — SUPPORTED

- B0* − reliability (miscalibration detection): +0.159 [+0.125, +0.194]

## H4 [v4_synthetic]: design variants change the evidence — SUPPORTED

- imbalance detection rel_n − rel: +0.364 [+0.298, +0.428] ✓
- diagnosis rel_n − rel: +0.077 [+0.055, +0.100] (expected > 0) ✓
- diagnosis cm_norm − cm: -0.090 [-0.119, -0.065] (expected < 0) ✓

## H5 [v4_synthetic]: readers below the image benchmark — SUPPORTED

- InternVL3-38B-AWQ_core (main/core): reader 0.165 vs B2 0.409, diff [-0.258, -0.227] ✓ below | no-answer 292/4500 (6%); counting no-answer as wrong: 0.155
- InternVL3-8B (main/core): reader 0.150 vs B2 0.410, diff [-0.275, -0.246] ✓ below | no-answer 651/8100 (8%); counting no-answer as wrong: 0.139
- Qwen2.5-VL-32B-Instruct-AWQ (main/core): reader 0.145 vs B2 0.409, diff [-0.280, -0.248] ✓ below | no-answer 7/8100 (0%); counting no-answer as wrong: 0.145
- Qwen2.5-VL-7B-Instruct (main/core): reader 0.146 vs B2 0.420, diff [-0.287, -0.257] ✓ below | no-answer 1798/8100 (22%); counting no-answer as wrong: 0.113
- llava-onevision-qwen2-7b-ov-hf (main/core): reader 0.143 vs B2 0.400, diff [-0.270, -0.240] ✓ below | no-answer 1079/8100 (13%); counting no-answer as wrong: 0.126

## H6 [v4_synthetic]: plot-type stereotype — SUPPORTED

- InternVL3-38B-AWQ_core: χ² p = 0; 'overfitting' on healthy loss curves 22% vs other plots 0% ✓
- InternVL3-8B: χ² p = 0; 'overfitting' on healthy loss curves 100% vs other plots 15% ✓
- Qwen2.5-VL-32B-Instruct-AWQ: χ² p = 0; 'overfitting' on healthy loss curves 100% vs other plots 12% ✓
- Qwen2.5-VL-7B-Instruct: χ² p = 0; 'overfitting' on healthy loss curves 100% vs other plots 32% ✓
- llava-onevision-qwen2-7b-ov-hf: χ² p = 0; 'overfitting' on healthy loss curves 100% vs other plots 31% ✓

## H8 [v4_synthetic]: interventions (benchmark ↑; readers equivalent; badge cue-following) — NOT SUPPORTED

- (a) benchmark: lc_loss 0.44→0.63; rel 0.51→0.59; cm 0.41→0.55; roc 0.40→0.57; dist 0.49→0.66 ✓
- InternVL3-8B: (b) companion − plain 90% CI [-0.029, -0.009] ✓ equivalent | (c) answers naming a badge-listed state: plain 79% → badge 92% ✓
- Qwen2.5-VL-32B-Instruct-AWQ: (b) companion − plain 90% CI [-0.013, +0.005] ✓ equivalent | (c) answers naming a badge-listed state: plain 71% → badge 49% ✗
- Qwen2.5-VL-7B-Instruct: (b) companion − plain 90% CI [-0.013, -0.000] ✓ equivalent | (c) answers naming a badge-listed state: plain 51% → badge 67% ✓
- llava-onevision-qwen2-7b-ov-hf: (b) companion − plain 90% CI [+0.002, +0.008] ✓ equivalent | (c) answers naming a badge-listed state: plain 58% → badge 46% ✗

## H1 [v4_magic]: theory-predicted blind spots — NOT SUPPORTED

- miscalibration on roc: AUC 0.458 [0.403, 0.516] → blind ✓
- miscalibration on pr: AUC 0.460 [0.401, 0.522] → blind ✓
- miscalibration on cm: AUC 0.516 [0.461, 0.574] → blind ✓
- miscalibration on lc_acc: AUC 0.510 [0.454, 0.571] → blind ✓
- class_imbalance on lc_loss: AUC 0.523 [0.463, 0.582] → blind ✓
- class_imbalance on lc_acc: AUC 0.716 [0.665, 0.769] → NOT blind ✗
- distribution_shift on lc_loss: AUC 0.532 [0.474, 0.593] → blind ✓
- distribution_shift on lc_acc: AUC 0.495 [0.435, 0.553] → blind ✓

## H2 [v4_magic]: detection ≠ diagnosis — SUPPORTED

- (a) detect-only cells: 6 (need ≥ 3) → ✓
- detect-only: miscalibration on rel (AUC 0.78, lower CI 0.74; recall 0.32)
- detect-only: miscalibration on rel_n (AUC 0.82, lower CI 0.78; recall 0.35)
- detect-only: miscalibration on dist (AUC 0.80, lower CI 0.76; recall 0.22)
- detect-only: label_noise on lc_acc (AUC 0.82, lower CI 0.77; recall 0.44)
- detect-only: class_imbalance on lc_acc (AUC 0.72, lower CI 0.66; recall 0.23)
- detect-only: data_leakage on rel (AUC 0.80, lower CI 0.76; recall 0.47)
- (b) B0* − lc_loss: +0.391 [+0.358, +0.421] ✓
- (b) B0* − lc_acc: +0.416 [+0.385, +0.451] ✓
- (b) B0* − rel: +0.236 [+0.206, +0.265] ✓
- (b) B0* − rel_n: +0.113 [+0.085, +0.139] ✓
- (b) B0* − cm: +0.323 [+0.291, +0.354] ✓
- (b) B0* − cm_norm: +0.368 [+0.334, +0.400] ✓
- (b) B0* − roc: +0.349 [+0.318, +0.380] ✓
- (b) B0* − pr: +0.310 [+0.282, +0.341] ✓
- (b) B0* − dist: +0.193 [+0.161, +0.222] ✓

## H3 [v4_magic]: encoding loss in the calibration plot — SUPPORTED

- B0* − reliability (miscalibration detection): +0.184 [+0.144, +0.226]

## H4 [v4_magic]: design variants change the evidence — SUPPORTED

- imbalance detection rel_n − rel: +0.266 [+0.218, +0.316] ✓
- diagnosis rel_n − rel: +0.123 [+0.097, +0.153] (expected > 0) ✓
- diagnosis cm_norm − cm: -0.044 [-0.069, -0.016] (expected < 0) ✓

## H5 [v4_magic]: readers below the image benchmark — SUPPORTED

- InternVL3-8B_core (main/core): reader 0.143 vs B2 0.477, diff [-0.352, -0.316] ✓ below | no-answer 315/4500 (7%); counting no-answer as wrong: 0.132
- Qwen2.5-VL-32B-Instruct-AWQ_core (main/core): reader 0.142 vs B2 0.467, diff [-0.342, -0.309] ✓ below | no-answer 0/4500 (0%); counting no-answer as wrong: 0.142
- Qwen2.5-VL-7B-Instruct_core (main/core): reader 0.141 vs B2 0.466, diff [-0.342, -0.309] ✓ below | no-answer 924/4500 (21%); counting no-answer as wrong: 0.113

## H6 [v4_magic]: plot-type stereotype — SUPPORTED

- InternVL3-8B_core: χ² p = 0; 'overfitting' on healthy loss curves 100% vs other plots 0% ✓
- Qwen2.5-VL-32B-Instruct-AWQ_core: χ² p = 0; 'overfitting' on healthy loss curves 100% vs other plots 0% ✓
- Qwen2.5-VL-7B-Instruct_core: χ² p = 9.3e-248; 'overfitting' on healthy loss curves 100% vs other plots 33% ✓

