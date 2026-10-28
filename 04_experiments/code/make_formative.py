"""Plots for the formative expert sessions (qualitative; replaces the controlled human study, see
human_study/PROTOCOL.md).

  python make_formative.py   -> ../human_study/formative/plots/P01.png ... P12.png, key.json (team only)

Twelve plots from the held-out pool the AI readers also answered: every plot family, standard and redesigned versions,
and cells chosen to provoke the questions the sessions are about (a failure the plot shows clearly, a failure the plot
cannot show at all, a healthy model).
"""
import json
import pathlib
import shutil

ROOT = pathlib.Path(__file__).parent.parent / "human_study"
# (plot version, true state, why it is in the set)
WANT = [
    ("lc_loss", "healthy", "healthy loss curves: AI readers call every one 'overfitting'"),
    ("lc_loss", "overfitting", "a failure the plot identifies"),
    ("cm", "class_imbalance", "a failure the plot identifies"),
    ("cm", "miscalibration", "invisible by construction: identical to the healthy twin's plot"),
    ("rel", "miscalibration", "the textbook plot for this failure, yet it only detects (AUC 0.80, recall 0.23)"),
    ("rel_n", "class_imbalance", "redesign: drawing bin sizes lifts imbalance detection from 0.64 to 0.97"),
    ("rel_ann", "miscalibration", "redesign: values printed"),
    ("roc", "data_leakage", "a failure the plot identifies"),
    ("roc_ann", "label_noise", "redesign: values printed"),
    ("dist", "distribution_shift", "a failure the plot identifies"),
    ("dist_ann", "distribution_shift", "redesign: values printed"),
    ("rel", "healthy", "healthy reliability diagram"),
]


def main():
    key = json.load(open(ROOT / "answer_key.json"))
    out = ROOT / "formative" / "plots"
    out.mkdir(parents=True, exist_ok=True)
    chosen, used = [], set()
    for n, (spec, truth, why) in enumerate(WANT, 1):
        qid = sorted(q for q, v in key.items() if v["spec"] == spec and v["truth"] == truth and v["run_id"] not in used)[0]
        used.add(key[qid]["run_id"])
        shutil.copy(ROOT / "survey" / "images" / f"{qid}.png", out / f"P{n:02d}.png")
        chosen.append(dict(plot=f"P{n:02d}", spec=spec, truth=truth, why=why, run_id=key[qid]["run_id"]))
    json.dump(chosen, open(ROOT / "formative" / "key.json", "w"), indent=1)
    print("\n".join(f"{c['plot']}  {c['spec']:8s} {c['truth']:18s} {c['why']}" for c in chosen))


if __name__ == "__main__":
    main()
