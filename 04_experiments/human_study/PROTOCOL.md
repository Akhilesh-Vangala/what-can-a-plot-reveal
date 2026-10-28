# Formative expert sessions — protocol

Replaces the controlled human-reader study (archived in `archive/`), following the instructor's review: a controlled
user study needs IRB approval, which is not feasible in the course; a small expert/formative study reported
qualitatively fits the rules. The quantitative evidence comes from the benchmarks and the five AI readers.
**No statistics are computed on these sessions and no accuracy is reported as a measurement.**

## Purpose

1. **Plots:** how experienced practitioners read standard diagnostic plots — which cues they use, whether they notice
   when a plot cannot show a failure, and whether the redesigns (bin sizes drawn; values printed) change what they say.
2. **Explorer:** whether the Plot Evidence Explorer helps with a realistic task ("given the plots I have, which
   failures can I rule out?") and what is confusing or missing. Findings feed the Explorer's final design.

## Participants

4–6 people who have trained and evaluated classifiers (graduate students, TAs or practitioners), invited personally.
Voluntary, no compensation, can stop at any time. No names, contact details or recordings are kept: one team member
takes written notes under a code (E1, E2, ...).

## Session (about 40 minutes, in person or video call, one participant and one or two team members)

1. **Introduction (3 min).** Purpose, voluntary participation, notes only, no recording. Verbal agreement.
2. **Think-aloud on 12 plots (15 min).** `formative/plots/P01.png`–`P12.png`, in order, one at a time. For each plot the
   participant says aloud what they see and what, if anything, is wrong with the model; "the plot cannot tell" is a
   valid answer. Prompts only: "What are you looking at?", "What would you check next?" No feedback on answers.
   The set (`formative/key.json`, team only) covers every plot family, standard and redesigned versions, a healthy
   model, failures the plot identifies, and one failure the plot cannot show at all (identical to a healthy plot).
3. **Explorer task (15 min).** Open `explorer/index.html`. Scenario: "Your team only looked at loss curves and a
   confusion matrix before shipping." Tasks: (a) which failures could those plots have missed? (b) which single plot
   would you add, and why? (c) look at one cell of your choice in the cell view and explain what it shows. Think aloud.
4. **Interview (7 min).** What was surprising? Would you trust an AI assistant's critique of these plots? What would
   make the Explorer useful in your own work? What is missing or confusing?

## Notes template (one per participant)

Per plot: answer given (state or "cannot tell"), cue mentioned, confidence in words, anything surprising.
Explorer: task answers, where they hesitated, misunderstandings, suggestions. Interview: short quotes.

## Analysis (qualitative)

Notes are coded into four themes fixed in advance — (1) cues used and textbook heuristics (e.g. "validation loss above
training loss means overfitting"); (2) awareness of what a plot cannot show; (3) effect of the redesigns; (4) Explorer
usability and trust in AI critique — plus any theme that emerges. Reported as themes with representative quotes and
counts of participants ("4 of 5 experts..."), never as accuracy or significance. Explorer changes made in response
are listed in the final report.

## Materials

- `formative/plots/` — the 12 plots (regenerate with `code/make_formative.py`).
- `formative/key.json` — true states and the reason each plot is in the set (team only; not shown to participants).
- `../explorer/index.html` — the Plot Evidence Explorer.
- `survey/` — the earlier self-paced survey; not used for data collection.
