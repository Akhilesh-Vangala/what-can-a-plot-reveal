# Confirmatory results — confirm_synthetic, confirm_magic

Each hypothesis as preregistered; ✓ = prediction supported. CIs: bootstrap over held-out test runs.

## H1 [confirm_synthetic]: theory-predicted blind spots — NOT SUPPORTED

- miscalibration on roc: AUC 0.523 [0.474, 0.571] → blind ✓
- miscalibration on pr: AUC 0.516 [0.471, 0.561] → blind ✓
- miscalibration on cm: AUC 0.478 [0.428, 0.524] → blind ✓
- miscalibration on lc_acc: AUC 0.491 [0.440, 0.540] → blind ✓
- class_imbalance on lc_loss: AUC 0.500 [0.455, 0.546] → blind ✓
- class_imbalance on lc_acc: AUC 0.590 [0.546, 0.634] → NOT blind ✗ (detectable)
- distribution_shift on lc_loss: AUC 0.505 [0.459, 0.555] → blind ✓
- distribution_shift on lc_acc: AUC 0.510 [0.461, 0.555] → blind ✓

## H2 [confirm_synthetic]: detection ≠ diagnosis — SUPPORTED

- (a) detect-only cells: 10 (need ≥ 3) → ✓
- detect-only: miscalibration on rel (AUC 0.80, lower CI 0.77; recall 0.23)
- detect-only: miscalibration on rel_n (AUC 0.81, lower CI 0.78; recall 0.32)
- detect-only: miscalibration on dist (AUC 0.77, lower CI 0.73; recall 0.12)
- detect-only: label_noise on lc_loss (AUC 0.86, lower CI 0.83; recall 0.42)
- detect-only: label_noise on lc_acc (AUC 0.77, lower CI 0.73; recall 0.48)
- detect-only: label_noise on cm (AUC 0.72, lower CI 0.68; recall 0.40)
- detect-only: label_noise on cm_norm (AUC 0.72, lower CI 0.68; recall 0.46)
- detect-only: data_leakage on dist (AUC 0.86, lower CI 0.83; recall 0.39)
- detect-only: distribution_shift on roc (AUC 0.77, lower CI 0.73; recall 0.42)
- detect-only: distribution_shift on pr (AUC 0.77, lower CI 0.73; recall 0.31)
- (b) B0* − lc_loss: +0.402 [+0.379, +0.426] ✓
- (b) B0* − lc_acc: +0.446 [+0.420, +0.472] ✓
- (b) B0* − rel: +0.283 [+0.259, +0.308] ✓
- (b) B0* − rel_n: +0.182 [+0.159, +0.206] ✓
- (b) B0* − cm: +0.363 [+0.337, +0.386] ✓
- (b) B0* − cm_norm: +0.429 [+0.402, +0.456] ✓
- (b) B0* − roc: +0.314 [+0.288, +0.340] ✓
- (b) B0* − pr: +0.304 [+0.280, +0.327] ✓
- (b) B0* − dist: +0.301 [+0.276, +0.324] ✓

## H3 [confirm_synthetic]: encoding loss in the calibration plot — SUPPORTED

- B0* − reliability (miscalibration detection): +0.168 [+0.138, +0.198]

## H4 [confirm_synthetic]: design variants change the evidence — SUPPORTED

- imbalance detection rel_n − rel: +0.333 [+0.292, +0.381] ✓
- diagnosis rel_n − rel: +0.101 [+0.081, +0.123] (expected > 0) ✓
- diagnosis cm_norm − cm: -0.065 [-0.085, -0.044] (expected < 0) ✓

## H5 [confirm_synthetic]: readers below the image benchmark — SUPPORTED

- InternVL3-38B-AWQ_core (main/core): reader 0.165 vs B2 0.444, diff [-0.294, -0.264] ✓ below | no-answer 251/4500 (6%); counting no-answer as wrong: 0.155
- InternVL3-8B (main/core): reader 0.150 vs B2 0.453, diff [-0.319, -0.288] ✓ below | no-answer 676/8100 (8%); counting no-answer as wrong: 0.139
- Qwen2.5-VL-32B-Instruct-AWQ (main/core): reader 0.145 vs B2 0.448, diff [-0.318, -0.289] ✓ below | no-answer 1/8100 (0%); counting no-answer as wrong: 0.145
- Qwen2.5-VL-7B-Instruct (main/core): reader 0.144 vs B2 0.458, diff [-0.329, -0.300] ✓ below | no-answer 1803/8100 (22%); counting no-answer as wrong: 0.113
- llava-onevision-qwen2-7b-ov-hf (main/core): reader 0.142 vs B2 0.441, diff [-0.314, -0.285] ✓ below | no-answer 1086/8100 (13%); counting no-answer as wrong: 0.124

## H6 [confirm_synthetic]: plot-type stereotype — SUPPORTED

- InternVL3-38B-AWQ_core: χ² p = 0; 'overfitting' on healthy loss curves 22% vs other plots 0% ✓
- InternVL3-8B: χ² p = 0; 'overfitting' on healthy loss curves 100% vs other plots 15% ✓
- Qwen2.5-VL-32B-Instruct-AWQ: χ² p = 0; 'overfitting' on healthy loss curves 100% vs other plots 12% ✓
- Qwen2.5-VL-7B-Instruct: χ² p = 0; 'overfitting' on healthy loss curves 100% vs other plots 32% ✓
- llava-onevision-qwen2-7b-ov-hf: χ² p = 0; 'overfitting' on healthy loss curves 100% vs other plots 31% ✓

## H7a [confirm_synthetic]: fewshot control still below B2 — SUPPORTED

- InternVL3-8B_controls (fewshot): reader 0.135 vs B2 0.449, diff [-0.338, -0.287] ✓ below | no-answer 0/1200 (0%); counting no-answer as wrong: 0.135
- Qwen2.5-VL-32B-Instruct-AWQ_controls (fewshot): reader 0.180 vs B2 0.471, diff [-0.325, -0.254] ✓ below | no-answer 355/1200 (30%); counting no-answer as wrong: 0.130
- Qwen2.5-VL-7B-Instruct_controls (fewshot): reader 0.169 vs B2 0.449, diff [-0.305, -0.253] ✓ below | no-answer 0/1200 (0%); counting no-answer as wrong: 0.169

## H7b [confirm_synthetic]: numbers control still below B2 — SUPPORTED

- InternVL3-8B_controls (numbers): reader 0.143 vs B2 0.495, diff [-0.397, -0.308] ✓ below | no-answer 797/1200 (66%); counting no-answer as wrong: 0.048
- Qwen2.5-VL-32B-Instruct-AWQ_controls (numbers): reader 0.166 vs B2 0.448, diff [-0.308, -0.253] ✓ below | no-answer 50/1200 (4%); counting no-answer as wrong: 0.160
- Qwen2.5-VL-7B-Instruct_controls (numbers): reader 0.140 vs B2 0.444, diff [-0.334, -0.275] ✓ below | no-answer 211/1200 (18%); counting no-answer as wrong: 0.116

## H8 [confirm_synthetic]: interventions (benchmark ↑; readers equivalent; badge cue-following) — NOT SUPPORTED

- (a) benchmark: lc_loss 0.42→0.67; rel 0.53→0.63; cm 0.45→0.65; roc 0.50→0.67; dist 0.54→0.75 ✓
- InternVL3-8B: (b) companion − plain 90% CI [-0.006, +0.017] ✓ equivalent | (c) answers naming a badge-listed state: plain 77% → badge 94% ✓
- Qwen2.5-VL-32B-Instruct-AWQ: (b) companion − plain 90% CI [-0.030, -0.010] ✗ | (c) answers naming a badge-listed state: plain 71% → badge 55% ✗
- Qwen2.5-VL-7B-Instruct: (b) companion − plain 90% CI [-0.004, +0.001] ✓ equivalent | (c) answers naming a badge-listed state: plain 52% → badge 68% ✓
- llava-onevision-qwen2-7b-ov-hf: (b) companion − plain 90% CI [-0.004, +0.008] ✓ equivalent | (c) answers naming a badge-listed state: plain 68% → badge 39% ✗

## H1 [confirm_magic]: theory-predicted blind spots — NOT SUPPORTED

- miscalibration on roc: AUC 0.528 [0.480, 0.576] → blind ✓
- miscalibration on pr: AUC 0.527 [0.479, 0.575] → blind ✓
- miscalibration on cm: AUC 0.540 [0.493, 0.587] → blind ✓
- miscalibration on lc_acc: AUC 0.495 [0.444, 0.547] → blind ✓
- class_imbalance on lc_loss: AUC 0.582 [0.532, 0.632] → NOT blind ✗ (detectable)
- class_imbalance on lc_acc: AUC 0.731 [0.687, 0.773] → NOT blind ✗ (detectable)
- distribution_shift on lc_loss: AUC 0.501 [0.453, 0.550] → blind ✓
- distribution_shift on lc_acc: AUC 0.493 [0.444, 0.542] → blind ✓

## H2 [confirm_magic]: detection ≠ diagnosis — SUPPORTED

- (a) detect-only cells: 9 (need ≥ 3) → ✓
- detect-only: overfitting on dist (AUC 0.90, lower CI 0.88; recall 0.48)
- detect-only: miscalibration on rel (AUC 0.81, lower CI 0.77; recall 0.41)
- detect-only: miscalibration on rel_n (AUC 0.83, lower CI 0.80; recall 0.39)
- detect-only: miscalibration on dist (AUC 0.81, lower CI 0.77; recall 0.20)
- detect-only: label_noise on lc_loss (AUC 0.80, lower CI 0.77; recall 0.49)
- detect-only: class_imbalance on lc_acc (AUC 0.73, lower CI 0.69; recall 0.28)
- detect-only: class_imbalance on rel (AUC 0.72, lower CI 0.68; recall 0.24)
- detect-only: data_leakage on rel (AUC 0.83, lower CI 0.79; recall 0.46)
- detect-only: distribution_shift on roc (AUC 0.72, lower CI 0.68; recall 0.30)
- (b) B0* − lc_loss: +0.414 [+0.387, +0.441] ✓
- (b) B0* − lc_acc: +0.414 [+0.388, +0.441] ✓
- (b) B0* − rel: +0.273 [+0.250, +0.298] ✓
- (b) B0* − rel_n: +0.135 [+0.112, +0.158] ✓
- (b) B0* − cm: +0.349 [+0.325, +0.373] ✓
- (b) B0* − cm_norm: +0.398 [+0.371, +0.426] ✓
- (b) B0* − roc: +0.356 [+0.329, +0.381] ✓
- (b) B0* − pr: +0.325 [+0.302, +0.350] ✓
- (b) B0* − dist: +0.212 [+0.189, +0.235] ✓

## H3 [confirm_magic]: encoding loss in the calibration plot — SUPPORTED

- B0* − reliability (miscalibration detection): +0.157 [+0.122, +0.191]

## H4 [confirm_magic]: design variants change the evidence — SUPPORTED

- imbalance detection rel_n − rel: +0.251 [+0.210, +0.292] ✓
- diagnosis rel_n − rel: +0.138 [+0.114, +0.162] (expected > 0) ✓
- diagnosis cm_norm − cm: -0.049 [-0.069, -0.026] (expected < 0) ✓

## H5 [confirm_magic]: readers below the image benchmark — SUPPORTED

- InternVL3-8B_core (main/core): reader 0.144 vs B2 0.448, diff [-0.322, -0.284] ✓ below | no-answer 286/4500 (6%); counting no-answer as wrong: 0.136
- Qwen2.5-VL-32B-Instruct-AWQ_core (main/core): reader 0.141 vs B2 0.440, diff [-0.317, -0.282] ✓ below | no-answer 0/4500 (0%); counting no-answer as wrong: 0.141
- Qwen2.5-VL-7B-Instruct_core (main/core): reader 0.143 vs B2 0.442, diff [-0.317, -0.280] ✓ below | no-answer 935/4500 (21%); counting no-answer as wrong: 0.114

## H6 [confirm_magic]: plot-type stereotype — SUPPORTED

- InternVL3-8B_core: χ² p = 0; 'overfitting' on healthy loss curves 100% vs other plots 0% ✓
- Qwen2.5-VL-32B-Instruct-AWQ_core: χ² p = 0; 'overfitting' on healthy loss curves 100% vs other plots 0% ✓
- Qwen2.5-VL-7B-Instruct_core: χ² p = 2.2e-239; 'overfitting' on healthy loss curves 100% vs other plots 33% ✓

