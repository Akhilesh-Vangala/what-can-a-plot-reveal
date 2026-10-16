# Human reader study — protocol

Status: materials ready; **data collection waits for the IRB route** (exempt / course-project determination).
The design and analysis below are fixed before any data are collected.

## Question

Do people diagnose ML failures from standard diagnostic plots better than chance and better than AI readers, and do
redesigned plots (bin sizes drawn; key values printed) help them? People, five open AI readers and the plot-content
benchmark are compared on **identical items**.

## Participants

- Target **N = 20** people with ML training (graduate students or practitioners who have trained classifiers).
- Recruitment: course and lab mailing lists; voluntary; no compensation planned (to be confirmed with the IRB route).
- Power (two proportions, before clustering): 20 participants give ≈ 0.82 power to detect a +0.13 accuracy effect of a
  redesign (baseline 0.40); 12 participants give ≈ 0.61 for the same effect. Clustering by participant lowers this.

## Materials (`make_human_study.py`)

- 140 held-out models from the confirmatory study (strongest severity; 20 per state: healthy + six failures), all
  read by the AI readers. Re-trained from their seeds; identical to the study models (checked: max |Δ deploy accuracy| = 0).
- Plot families and versions: loss curves; confusion matrix; reliability diagram (standard / bin sizes drawn / values
  printed); ROC (standard / values printed); validation-vs-test histogram (standard / values printed).
- Four counterbalanced forms (A–D), 40 items each (8 per family, states balanced), each model at most once per form;
  the version of a redesign-eligible item rotates across forms. Two attention checks per form (blatant class imbalance
  on a confusion matrix; blatant leakage on a ROC curve). Item order randomised per form.
- Task wording and failure definitions are identical to the AI readers' prompt.

## Procedure

1. Participant receives the study folder (or link), a participant code and a form letter (rotating A–D).
2. `survey.html` (works offline in any browser): consent → instructions with definitions → 42 plots, one at a time:
   seven-way answer, optional "the plot is insufficient to tell", confidence 1–5; time per plot recorded; no going back.
3. Participant downloads `responses_<code>_<form>.json` and sends it to the team. Expected duration 20–30 minutes.

## Analysis (`score_humans.py`, fixed in advance)

- Exclusion: missing either attention check, or answering fewer than 80% of items.
- Primary: balanced accuracy (7 states; chance 1/7) overall and per family.
- Redesign effect: logistic GEE clustered by participant, correct ~ redesigned + family + state, on the three families
  with redesigns; reported as log-odds with 95% CI.
- Comparison on the same items: people (majority answer per item) vs each AI reader vs the plot-content benchmark (B1*).
- Confidence: accuracy by confidence level; abstention rate.
- All results reported, including null effects; anything beyond this list is labelled exploratory.

## Data handling

Answers are stored under participant codes only; no names or contact details in the response files. The key linking
codes to people (if any) is kept separately by one team member and deleted after the study.
