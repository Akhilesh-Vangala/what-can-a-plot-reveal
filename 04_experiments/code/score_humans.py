"""Score the human reader study (analysis fixed before data collection; see human_study/PROTOCOL.md).

  python score_humans.py ../human_study/responses   -> ../human_study/HUMAN_RESULTS.md (+ .json)

Exclusions: a participant who misses either attention check, or answers fewer than 80% of items, is excluded.
Measures (7-way, chance 1/7): balanced accuracy overall and per plot family; standard vs redesigned version per family
(logistic GEE clustered by participant: correct ~ redesigned + C(family) + C(state)); people vs AI readers vs the
benchmark on the same items; accuracy by confidence level; abstention ("insufficient to tell").
"""
import collections
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from score_confirm import CLASSES, Run, reader_vec

ROOT = pathlib.Path(__file__).parent
RUN = ROOT / "runs" / "confirm_synthetic"
STANDARD = {"lc_loss", "cm", "rel", "roc", "dist"}
BASE = {"rel_n": "rel", "rel_ann": "rel", "roc_ann": "roc", "dist_ann": "dist"}
AI = {"InternVL3-38B-AWQ": ["InternVL3-38B-AWQ_core", "InternVL3-38B-AWQ_ann"], "InternVL3-8B": ["InternVL3-8B", "InternVL3-8B_ann"],
      "Qwen2.5-VL-32B-Instruct-AWQ": ["Qwen2.5-VL-32B-Instruct-AWQ", "Qwen2.5-VL-32B-Instruct-AWQ_ann"],
      "Qwen2.5-VL-7B-Instruct": ["Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-7B-Instruct_ann"],
      "llava-onevision-qwen2-7b-ov-hf": ["llava-onevision-qwen2-7b-ov-hf", "llava-onevision-qwen2-7b-ov-hf_ann"]}


def bal(df, col="correct"):
    return float(df.groupby("truth")[col].mean().mean()) if len(df) else float("nan")


def main(resp_dir):
    resp_dir = pathlib.Path(resp_dir)
    study = resp_dir.parent
    key = json.load(open(study / "answer_key.json"))
    recs, excluded = [], []
    for fp in sorted(resp_dir.glob("*.json")):
        d = json.load(open(fp))
        pid, form = d["meta"]["participant"], d["meta"]["form"]
        att = [a for a in d["answers"] if key[a["item_id"]]["family"] == "attention"]
        att_ok = len(att) == 2 and all(a["answer"] == key[a["item_id"]]["truth"] for a in att)
        n_items = len([a for a in d["answers"] if key[a["item_id"]]["family"] != "attention"])
        if not att_ok or n_items < 0.8 * 40:
            excluded.append((pid, "attention check" if not att_ok else "incomplete"))
            continue
        for a in d["answers"]:
            k = key[a["item_id"]]
            if k["family"] == "attention":
                continue
            recs.append(dict(participant=pid, form=form, item_id=a["item_id"], run_id=k["run_id"], spec=k["spec"], family=k["family"],
                             truth=k["truth"], answer=a["answer"], correct=int(a["answer"] == k["truth"]),
                             redesigned=int(k["spec"] not in STANDARD), confidence=a["confidence"], cannot=int(a["cannot_tell"]),
                             seconds=a["seconds"]))
    df = pd.DataFrame(recs)
    L = ["# Human reader study — results", "", f"Participants analysed: {df['participant'].nunique() if len(df) else 0}; "
         f"excluded: {len(excluded)} {excluded if excluded else ''}; answers: {len(df)}.", ""]
    if not len(df):
        (study / "HUMAN_RESULTS.md").write_text("\n".join(L) + "\nNo usable responses.\n")
        print("\n".join(L))
        return
    L += [f"Overall balanced accuracy: **{bal(df):.3f}** (chance {1 / 7:.3f}); abstention {df.cannot.mean():.0%}; "
          f"median time per plot {df.seconds.median():.0f} s.", "", "## By plot family and version", "",
          "| family | version | answers | balanced accuracy |", "|---|---|---|---|"]
    for (fam, spec), g in df.groupby(["family", "spec"]):
        L.append(f"| {fam} | {spec} | {len(g)} | {bal(g):.3f} |")
    red = df[df.family.isin(["reliability", "roc", "histogram"])]
    if red.redesigned.nunique() == 2 and red.participant.nunique() > 1:
        m = smf.gee("correct ~ redesigned + C(family) + C(truth)", groups="participant", data=red, family=sm.families.Binomial(),
                    cov_struct=sm.cov_struct.Exchangeable()).fit()
        b, se = m.params["redesigned"], m.bse["redesigned"]
        L += ["", f"Redesign effect (log-odds, GEE clustered by participant): {b:+.2f} [95% CI {b - 1.96 * se:+.2f}, {b + 1.96 * se:+.2f}]"]
    # AI readers and benchmark on the same items
    R = Run(RUN)
    post = {sp: R.dia_post(R.top, sp) for sp in STANDARD}
    ai = collections.defaultdict(dict)
    for model, files in AI.items():
        for f in files:
            p = RUN / f"responses_{f}.jsonl"
            if not p.exists():
                continue
            for line in open(p):
                r = json.loads(line)
                if r.get("task") == "diagnosis":
                    v = reader_vec(r)
                    if v is not None:
                        ai[model][(r["run_id"], r["spec"])] = CLASSES[int(np.argmax(v))]
    L += ["", "## People vs AI readers vs benchmark on the same items", "", "| reader | items covered | balanced accuracy |", "|---|---|---|"]
    items = df.drop_duplicates("item_id")
    L.append(f"| people (majority answer per item) | {len(items)} | "
             f"{bal(df.groupby('item_id').agg(truth=('truth', 'first'), correct=('correct', 'mean')).assign(correct=lambda x: (x.correct > .5).astype(int))):.3f} |")
    for model, ans in ai.items():
        hit = items[[(r, s) in ans for r, s in zip(items.run_id, items.spec)]].copy()
        hit["correct"] = [int(ans[(r, s)] == t) for r, s, t in zip(hit.run_id, hit.spec, hit.truth)]
        L.append(f"| {model} | {len(hit)} | {bal(hit):.3f} |")
    bench = items.copy()
    bench["correct"] = [int(R.classes[int(np.argmax(post[BASE.get(s, s)][r]))] == t) if r in post[BASE.get(s, s)] else np.nan
                        for r, s, t in zip(bench.run_id, bench.spec, bench.truth)]
    L.append(f"| benchmark B1* (plot content) | {bench.correct.notna().sum()} | {bal(bench.dropna(subset=['correct'])):.3f} |")
    L += ["", "## Confidence vs accuracy (people)", "", "| confidence | answers | accuracy |", "|---|---|---|"]
    for c, g in df.groupby("confidence"):
        L.append(f"| {c} | {len(g)} | {g.correct.mean():.2f} |")
    (study / "HUMAN_RESULTS.md").write_text("\n".join(L) + "\n")
    df.to_csv(study / "human_answers.csv", index=False)
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT.parent / "human_study" / "responses")
