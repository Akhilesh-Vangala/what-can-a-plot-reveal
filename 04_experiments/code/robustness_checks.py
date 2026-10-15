"""Robustness checks (exploratory): (a) read-then-reason gain over direct diagnosis with bootstrap CIs; (b) evidential-map
counts under alternative cutoffs.

  python robustness_checks.py   -> runs/confirm_synthetic/ROBUSTNESS.md
"""
import json
import pathlib

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import rtr
from score_confirm import CLASSES, Run, auc_of, reader_vec

RUNS = pathlib.Path(__file__).parent / "runs"
SPECS = ("lc_loss", "rel", "cm", "roc", "dist")
FILES = {"InternVL3-38B-AWQ": "InternVL3-38B-AWQ_core", "InternVL3-8B": "InternVL3-8B", "Qwen2.5-VL-32B-Instruct-AWQ": "Qwen2.5-VL-32B-Instruct-AWQ",
         "Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B-Instruct", "llava-onevision-qwen2-7b-ov-hf": "llava-onevision-qwen2-7b-ov-hf"}


def bal(y, p):
    return float(np.mean([np.mean(p[y == c] == c) for c in np.unique(y)]))


def main():
    V = json.load(open(RUNS / "confirm_synthetic/rtr_values.json"))
    rules = {}
    for sp in SPECS:
        tr = [(rtr.features(v["values"][sp], sp), CLASSES.index(v["failure"])) for v in V.values() if v["split"] == "train"]
        rules[sp] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced")).fit(
            np.nan_to_num(np.array([a for a, _ in tr], float)), [b for _, b in tr])
    rng = np.random.default_rng(0)
    L = ["# Robustness checks (exploratory)", "", "## (a) Read-then-reason minus direct diagnosis",
         "", "Same models and plots, pooled over the 5 core plots; no-answer counted as wrong for the direct reader; "
         "95% bootstrap CI over models (500 resamples).", "", "| reader | gain | 95% CI | items |", "|---|---|---|---|"]
    for m, f in FILES.items():
        ext = {}
        for line in open(RUNS / f"confirm_synthetic/responses_{m}_extract.jsonl"):
            r = json.loads(line)
            d = rtr.from_reader(r.get("parsed"), r["spec"])
            ft = rtr.features(d, r["spec"]) if d else None
            if ft is not None and not any(np.isnan(ft)):
                q = rules[r["spec"]].predict_proba(np.array([ft]))[0]
                full = np.zeros(len(CLASSES))
                full[rules[r["spec"]].classes_] = q
                ext[(r["run_id"], r["spec"])] = int(np.argmax(full))
        dire = {}
        for line in open(RUNS / f"confirm_synthetic/responses_{f}.jsonl"):
            r = json.loads(line)
            if r.get("task") == "diagnosis" and r["spec"] in SPECS and r["run_id"] in V:
                v = reader_vec(r)
                dire[(r["run_id"], r["spec"])] = int(np.argmax(v)) if v is not None else -1
        keys = [k for k in ext if k in dire]
        by = {}
        for k in keys:
            by.setdefault(k[0], []).append(k)
        runs = sorted(by)

        def diff(rs):
            ks = [k for r in rs for k in by[r]]
            y = np.array([CLASSES.index(V[k[0]]["failure"]) for k in ks])
            return bal(y, np.array([ext[k] for k in ks])) - bal(y, np.array([dire[k] for k in ks]))
        bs = [diff(list(rng.choice(runs, len(runs)))) for _ in range(500)]
        L.append(f"| {m} | {diff(runs):+.3f} | [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}] | {len(keys)} |")
    L += ["", "## (b) Evidential-map counts under alternative cutoffs (identifiable / detect-only / blind; strongest level)", "",
          "| run | AUC cutoff | recall cutoff | identifiable | detect-only | blind |", "|---|---|---|---|---|---|"]
    for name in ("confirm_synthetic", "confirm_magic"):
        R = Run(RUNS / name)
        A, Rc = [], []
        for fl in R.meta["failures"]:
            for s in R.meta["specs"]:
                sc, _, ids = R.det_scores(fl, R.top, s)
                A.append(auc_of(sc, R.truth, ids))
                post = R.dia_post(R.top, s)
                pred = {k: R.classes[int(np.argmax(v))] for k, v in post.items()}
                Rc.append(np.mean([pred[k] == fl for k in post if R.truth[k] == fl]))
        A, Rc = np.array(A), np.array(Rc)
        for ta, trc in ((.65, .5), (.70, .5), (.75, .5), (.70, .4), (.70, .6)):
            i, d = int(((A >= ta) & (Rc >= trc)).sum()), int(((A >= ta) & (Rc < trc)).sum())
            L.append(f"| {name} | {ta:.2f} | {trc:.1f} | {i} | {d} | {len(A) - i - d} |")
    (RUNS / "confirm_synthetic/ROBUSTNESS.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
