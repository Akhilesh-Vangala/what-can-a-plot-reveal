# Human reader study — how to run

**Do not collect data until the IRB route is confirmed** and the consent text in `survey.html` is replaced by the
approved version (see `PROTOCOL.md`).

1. Give each participant ONLY the `survey/` folder (`survey.html`, `forms.js`, `images/`) — zipped, or hosted as a static page —
   plus a participant code (e.g. P01) and a form letter (rotate A, B, C, D).
   Direct link form: `survey/survey.html?form=B&pid=P01`.
2. They complete the survey (20–30 min) and send back `responses_<code>_<form>.json`.
3. Put returned files in `responses/` and run, from `04_experiments/code/`:
   `python score_humans.py ../human_study/responses` → `HUMAN_RESULTS.md`, `human_answers.csv`.

Do **not** share `answer_key.json` with participants. Regenerate everything with `python make_human_study.py`.
