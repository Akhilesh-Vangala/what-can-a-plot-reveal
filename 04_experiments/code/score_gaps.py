"""Score gaps 3 and 5 (exploratory).

  python score_gaps.py runs/confirm_synthetic [runs/confirm_magic]   -> runs/confirm_synthetic/GAPS_RESULTS.md

Gap 5 (machine-reader designs), same held-out core models:
  reading fidelity      read vs exact values, original vs annotated design
  read-then-reason      rule on exact values (trained on synthetic training models) applied to readings
  direct diagnosis      the reader's own 7-way answer on the original vs the annotated plot
  all five plots        loss + confusion matrix readings combined with the three annotated designs vs the originals
Gap 3 (multi-plot benchmark): best single plot, best pair, dashboards, all nine (plotsets.json).
"""
import json
import pathlib
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import rtr
from annotate import BASE
from score_confirm import CLASSES, reader_vec

READERS = {"InternVL3-38B-AWQ": "InternVL3-38B-AWQ_core", "InternVL3-8B": "InternVL3-8B", "Qwen2.5-VL-32B-Instruct-AWQ":
           "Qwen2.5-VL-32B-Instruct-AWQ", "Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B-Instruct",
           "llava-onevision-qwen2-7b-ov-hf": "llava-onevision-qwen2-7b-ov-hf"}


def bal(y, p):
    return float(np.mean([np.mean(p[y == c] == c) for c in np.unique(y)])) if len(y) else float("nan")


def feats(vals, S):
    out = []
    for sp in S:
        f = rtr.features(vals.get(sp), sp) if vals.get(sp) is not None else None
        if f is None or any(np.isnan(f)):
            return None
        out += f
    return out


def main(syn, mag=None):
    syn = pathlib.Path(syn)
    V = json.load(open(syn / "rtr_values.json"))
    rules = {}

    def rule(S):
        if S not in rules:
            tr = [(feats(v["values"], S), CLASSES.index(v["failure"])) for v in V.values() if v["split"] == "train"]
            rules[S] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced")).fit(
                np.array([a for a, _ in tr]), [b for _, b in tr])
        return rules[S]
    L = ["# Gaps 3 and 5 (exploratory)", "", "## Gap 5: plots redesigned for machine readers (values printed on the plot)", ""]
    if (syn / "reader_items_ann.json").exists():
        L += ["| reader | design | rtr reliability | rtr ROC | rtr histogram | direct reliability | direct ROC | direct histogram | all five plots (rtr) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for m, f in READERS.items():
            orig, ann, dorig, dann = {}, {}, {}, {}
            p_ext, p_ann = syn / f"responses_{m}_extract.jsonl", syn / f"responses_{m}_ann.jsonl"
            if not p_ann.exists():
                continue
            for line in open(p_ext):
                r = json.loads(line)
                d = rtr.from_reader(r.get("parsed"), r["spec"])
                if d:
                    orig.setdefault(r["run_id"], {})[r["spec"]] = d
            for line in open(p_ann):
                r = json.loads(line)
                base = BASE[r["spec"]]
                if r["task"] == "extract":
                    d = rtr.from_reader(r.get("parsed"), base)
                    if d:
                        ann.setdefault(r["run_id"], {})[base] = d
                else:
                    v = reader_vec(r)
                    dann[(r["run_id"], base)] = int(np.argmax(v)) if v is not None else -1
            for line in open(syn / f"responses_{f}.jsonl"):
                r = json.loads(line)
                if r.get("task") == "diagnosis" and r["spec"] in BASE.values() and r["run_id"] in V:
                    v = reader_vec(r)
                    dorig[(r["run_id"], r["spec"])] = int(np.argmax(v)) if v is not None else -1
            row = {}
            for label, src, dsrc in (("original", orig, dorig), ("annotated", ann, dann)):
                cells = []
                for sp in ("rel", "roc", "dist"):
                    ks = [k for k in src if feats(src[k], (sp,)) is not None]
                    y = np.array([CLASSES.index(V[k]["failure"]) for k in ks])
                    cells.append(bal(y, rule((sp,)).predict(np.array([feats(src[k], (sp,)) for k in ks]))) if ks else float("nan"))
                for sp in ("rel", "roc", "dist"):
                    ks = [k for k in dsrc if k[1] == sp]
                    y = np.array([CLASSES.index(V[k[0]]["failure"]) for k in ks])
                    cells.append(bal(y, np.array([dsrc[k] for k in ks])) if ks else float("nan"))
                S5 = ("lc_loss", "rel", "cm", "roc", "dist")
                comb = {k: dict(orig.get(k, {}), **(src.get(k, {}) if label == "annotated" else {})) for k in orig}
                ks = [k for k in comb if feats(comb[k], S5) is not None]
                y = np.array([CLASSES.index(V[k]["failure"]) for k in ks])
                cells.append(bal(y, rule(S5).predict(np.array([feats(comb[k], S5) for k in ks]))) if ks else float("nan"))
                row[label] = cells
                L.append(f"| {m} | {label} | " + " | ".join(f"{c:.3f}" for c in cells) + " |")
        ids = [k for k, v in V.items() if v["split"] == "test"]
        y = np.array([CLASSES.index(V[k]["failure"]) for k in ids])
        ceil = [bal(y, rule((sp,)).predict(np.array([feats(V[k]["values"], (sp,)) for k in ids]))) for sp in ("rel", "roc", "dist")]
        c5 = bal(y, rule(("lc_loss", "rel", "cm", "roc", "dist")).predict(np.array([feats(V[k]["values"], ("lc_loss", "rel", "cm", "roc", "dist")) for k in ids])))
        L.append("| exact values (ceiling) | — | " + " | ".join(f"{c:.3f}" for c in ceil) + " | | | | " + f"{c5:.3f} |")
    else:
        L.append("(annotated reader results not present)")
    L += ["", "## Gap 3: multi-plot diagnosis benchmark (balanced accuracy, strongest severity)", ""]
    for run in [syn] + ([pathlib.Path(mag)] if mag else []):
        f = run / "plotsets.json"
        if not f.exists():
            L.append(f"- {run.name}: plotsets.json not present")
            continue
        S = json.load(open(f))
        single = max((s for s in S if len(s["plots"]) == 1), key=lambda s: s["balanced_acc"])
        pairs = sorted((s for s in S if len(s["plots"]) == 2), key=lambda s: -s["balanced_acc"])
        dash = [s for s in S if len(s["plots"]) >= 3]
        L.append(f"- {run.name}: best single {single['plots']} {single['balanced_acc']:.3f}; best pairs " +
                 "; ".join(f"{'+'.join(p['plots'])} {p['balanced_acc']:.3f}" for p in pairs[:3]) + "; " +
                 "; ".join(f"{'+'.join(d['plots']) if len(d['plots']) < 9 else 'all nine'} {d['balanced_acc']:.3f}" for d in dash))
    (syn / "GAPS_RESULTS.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(*sys.argv[1:3])
