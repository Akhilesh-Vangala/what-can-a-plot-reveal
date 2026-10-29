# Project status and context

*What Can a Plot Reveal? Detection vs. Diagnosis in ML Diagnostic Plots for Human and AI Readers*
DS-GA 3001 Visualization for Machine Learning (Prof. Claudio Silva), Fall 2026, Team 10:
Akhilesh Vangala (sv3129@nyu.edu), Sanvi Jagtap (sj5129@nyu.edu), Lucas Yao (ly2808@nyu.edu).
Goal: the course project, then a workshop paper (possibly a full paper).

Last updated: 2026-10-29.

---

## 1. Course deliverables

| Deliverable | Due | Weight | Status |
|---|---|---|---|
| Proposal (4 pages, `Team10.pdf`) | Oct 20 | 10% | Submitted. **Score 9/10 (A−)**, feedback below |
| Mid-term update (1 page, preliminary results) | **Nov 3** | 10% | To write; must answer the feedback |
| Presentation | Dec 1 / Dec 8 | — | Not started |
| Final report (8 pages) + repo + demo video | Dec 14 | 25% | Not started; the Explorer is the demo |

---

## 2. Instructor feedback on the proposal (verbatim)

> **Feedback for DS-GA 3001: Project proposal**
>
> **Overall Feedback**
>
> 1. **Stating the visualization insight** - A systematic study of what diagnostic plots can reveal is itself a
>    visualization contribution. The paper will be stronger if it states that insight directly (which plots hide which
>    failures, and what a redesign changes), and if the Explorer and the redesigns, which are still to be built, turn
>    it into something practitioners can use.
> 2. **What is new this term** - The proposal says the pipeline and the confirmatory run are complete. A line in the
>    update on which parts predate the course and which are the course deliverables would help me see the plan, and it
>    clarifies the contribution of this team's work.
> 3. **The human reader study** - Twelve to twenty participants with a mixed-effects model is a controlled user study,
>    and those require IRB approval, which is not feasible in this course. An expert or formative study at a smaller
>    scale, reported qualitatively, would fit the rules, and the same applies to the Explorer task study with
>    practitioners. The AI-reader results can carry the quantitative evidence.
> 4. **What "blind" means** - A cell is called blind when a learned benchmark (including a ResNet-18 on the rendered
>    image) fails to detect the failure. That is evidence about the estimator as much as about the plot. A sensitivity
>    analysis on the AUC and recall thresholds, and perhaps a stronger or different benchmark model, would make the
>    "blind" claims easier to defend.
> 5. **Reader results near chance** - A seven-way forced choice may be hard enough that near-chance performance is
>    partly a task-design effect. A simpler check (healthy vs. failing) could show whether the readers are blind or the
>    task is too hard, which would affect how you present the result.
> 6. **Scope** - Nine plots, six failures, two data sources, five VLMs plus frontier models, human readers, redesigns
>    and the Explorer is a lot. Marking what is core for the paper and what is stretch would help.
>
> **Score:** 9 / 10 - A-

---

## 3. Response to each point

| # | Point | Response | Status |
|---|---|---|---|
| 1 | State the visualization insight | Lead the update and the paper with the insight: which plots hide which failures (including failures that are provably invisible in a plot) and what each redesign changes. Show that the Explorer and the redesigns are already built (screenshot + headline redesign result). | To write (update) |
| 2 | What is new this term | All work was done this term: idea and manual pilot early October, first cluster job Oct 7, confirmatory run Oct 12–13, repository Oct 14. "Complete" in the proposal meant the core pipeline built in the first weeks of term. **Team to confirm** before the update states it. | To write; team confirms |
| 3 | No controlled human study | Controlled study dropped (archived). Replaced by 4–6 qualitative expert sessions: think-aloud on 12 chosen plots, an Explorer task, a short interview; reported as themes and quotes, no statistics. Protocol: `04_experiments/human_study/PROTOCOL.md`. | Materials ready; recruit 4–6 people |
| 4 | What "blind" means | (a) Cells **blind by construction**: failing model re-trained next to its healthy twin; plotted values identical → no reader or estimator can ever detect the failure. (b) Full **cutoff sensitivity** with a grade per blind cell. (c) **Classifier-free two-sample test** per cell and a **stronger, different image benchmark** (B3, ConvNeXt-Tiny at native resolution). "Blind" is split into *invisible* (plot unchanged) and *weak* (a difference exists, but no estimator reaches AUC 0.70). | **Done** (results below) |
| 5 | Near chance may be a task-design effect | Two simpler tasks for all five readers on the same held-out items: **binary** (healthy vs. problem; AUC per failure × plot, compared with the benchmark on the same items) and **two-alternative forced choice** (which of two plots is the failing model; chance 0.5). | **Done** (results below) |
| 6 | Core vs. stretch | Proposed split below. | To confirm with the team, then state in the update |

**Proposed scope split (point 6)**

- **Core (paper and course):** evidential map (9 plots × 6 failures × 2 data sources) with the invisible/weak
  distinction; benchmarks B0/B1/B2(+B3) with the robustness checks; five open AI readers (7-way, binary, 2AFC);
  redesigns (bin sizes drawn, values printed) evaluated by benchmarks and readers; the Plot Evidence Explorer;
  4–6 formative expert sessions (qualitative) covering the plots and the Explorer task — these replace the controlled
  human study and the Explorer user test, as the instructor asked.
- **Supporting (already done, appendix or short sections):** real-world check, read-then-reason, multi-plot benchmark,
  positive control, value probe, badges and companion views.
- **Stretch:** frontier models (paid APIs, only if the instructor approves).

**First results for point 4 (done locally)**

- Blind by construction (identical plots in all 30 twin pairs per failure): miscalibration on accuracy curves,
  confusion matrix (counts and row-normalized), ROC and PR; distribution shift on loss and accuracy curves.
  7 cells on real data, 5 on synthetic (ROC/PR there differ only through float32 rounding ties; benchmark AUC 0.52).
  Reports: `04_experiments/code/runs/confirm_*/CERTIFIED_BLIND.md`.
- Cutoff sensitivity (AUC 0.60–0.80 × recall 0.30–0.70): synthetic blind cells 15 = 5 certified + 10 robust +
  0 borderline; real 17 = 7 certified + 7 robust + 3 borderline (labels that depend on the cutoff: label noise on
  accuracy curves and on the row-normalized confusion matrix, shift on PR). Report: `04_experiments/code/runs/BLIND_SENSITIVITY.md`.


**Final results for points 4 and 5 (2026-10-29; `04_experiments/code/runs/REVIEW_RESULTS.md`)**

- Point 4: the stronger image benchmark B3 never beats the earlier benchmarks by more than 0.05. Blind cells,
  synthetic: 5 invisible (proven) + 4 invisible (no difference found) + 6 weak; real: 7 + 5 + 4, and one cell revised
  to detectable (distribution shift on PR, B3 AUC 0.72).
- Point 5: readers stay near chance on the simpler tasks — binary detection AUC 0.50–0.54 (benchmark 0.80 on the
  same items), 2AFC 0.51–0.58 (benchmark 0.79 on the same pairs). The largest reader detects only textbook pairs
  (overfitting on loss curves 0.86). The near-chance result is not a task-design effect.

---

## 4. What has been done (timeline)

| Date | Work |
|---|---|
| Early Oct | Idea selection; manual pilot (`04_experiments/manual_pilot_oct2/`); literature check |
| Oct 7–9 | Cluster set-up (NYU HPC Cloud Bursting, A100); prototype; multi-failure pilot (v3); full pilot (v4, synthetic + real) |
| Oct 9–10 | Design interventions (evidence badges, companion views); job system rebuilt after cluster failures (`04_experiments/hpc/README.md`) |
| Oct 11 | Analysis plan and hypotheses H1–H8 fixed before the confirmatory run |
| Oct 12–13 | **Confirmatory study**: 38,400 models (fresh seeds), synthetic + MagicTelescope, 5 readers, controls, interventions |
| Oct 13–14 | Scoring with Holm correction; positive control and value probe; real-world check (10 verified cases on 8 datasets) |
| Oct 14–15 | Read-then-reason; annotated machine-reader designs; multi-plot benchmark; gap analyses; consistency pass (all reports reproduce exactly) |
| Oct 15–16 | Plot Evidence Explorer (map, cell, my-plots and reader views); human-study materials |
| Oct 20 | Proposal submitted |
| Oct 28 | Instructor feedback received (9/10); response started: blind-by-construction proof, cutoff sensitivity, new cluster stage `review`, formative expert-session protocol |
| Oct 29 | Review checks complete: two-sample tests, B3, binary and 2AFC tasks for all five readers (points 4 and 5 answered) |

Main results: see `README.md` (study design and results) and `04_experiments/FINDINGS_LOG.md` (dated log).
Deviations from the plan and their reasons: `04_experiments/DEVIATIONS.md`.

**Compute used:** about 45–50 of the 300 A100 GPU-hours on the Cloud Bursting allocation. Cost: $0 (open models only).

---

## 5. Where things are

| What | Where |
|---|---|
| Study code (models, failures, plots, benchmarks) | `04_experiments/code/plotreveal/`, `run_v4.py` |
| Scoring and reports | `04_experiments/code/score_*.py`, `04_experiments/code/runs/*/` |
| Review checks (this round) | `certify_blind.py`, `blind_sensitivity.py`, `review_prepare.py`, `score_review.py`, `hpc/b3_strong.py` |
| Cluster jobs | `04_experiments/hpc/` (`run_stage.sh <stage>`; bundle built by `04_experiments/make_bundle.sh`) |
| Explorer | `04_experiments/explorer/index.html` |
| Expert sessions | `04_experiments/human_study/` (`PROTOCOL.md`, `formative/plots/`) |
| Figures | `04_experiments/figures/` |

---

## 6. Decisions and rules

- **Open models first.** Paid APIs (frontier models) only after the instructor approves. No Gemini key is used.
- **No controlled human study** (instructor: needs IRB). Expert sessions are qualitative only.
- **Analysis discipline:** hypotheses fixed before the confirmatory run; everything added later is labelled
  exploratory; all results reported, including null results; every deviation logged in `DEVIATIONS.md`.
- **Reproducibility:** every model and plot is regenerated exactly from seeds; re-trained models are checked against
  the stored run (max |Δ deployment accuracy| = 0).
- **Paper text:** written by the team (venue policies on AI-written text apply).

---

## 7. Next steps

| # | Step | Who | When |
|---|---|---|---|
| 1 | ~~Run cluster stage `review`~~ — done Oct 29 | — | — |
| 2 | Update the evidential-map figure and the Explorer with the invisible/weak labels | — | Before the update |
| 3 | Confirm what predates the course (point 2) and the core/stretch split (point 6) | Team | Before Nov 1 |
| 4 | Write the 1-page mid-term update answering all six points | Team | **Nov 3** |
| 5 | Recruit 4–6 experts; run the sessions (40 min each); code notes into themes | Team | Nov 4–20 |
| 6 | Revise the Explorer from session feedback | — | After sessions |
| 7 | Ask the instructor about frontier models (stretch) and the paper venue | Team | With the update |
| 8 | Presentation | Team | Dec 1 / 8 |
| 9 | Final report (8 pages), repo clean-up, demo video | Team | Dec 14 |
