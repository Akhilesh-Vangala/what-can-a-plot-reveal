"""Exploratory gap analyses (regenerates the numbers logged in FINDINGS_LOG.md on 2026-10-14/15).

  python gap_analyses.py   -> runs/confirm_synthetic/GAP_ANALYSES.md

  calibration   contextual calibration of reader answers (divide by no-plot answer / mean answer per plot type)
  gap 4         which plot cue drives readers' answers (reason texts + exact plot values)
  gap 6         real-world false alarms with EXACT values (domain shift vs misreading)
  gap 7         where accuracy curves carry class-imbalance information (fresh models; per-class difficulty)
  gap 8         multi-plot read-then-reason; rule trained on each reader's own readings (5-fold CV)
"""
import collections
import json
import pathlib
import re

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import rtr
from make_positive import CANONICAL
from plotreveal.world import apply_failure, healthy_config, train_run
from score_confirm import CLASSES, Run, bal_acc, reader_vec

RUNS = pathlib.Path(__file__).parent / "runs"
S = RUNS / "confirm_synthetic"
DIRECT = {"InternVL3-38B-AWQ": "InternVL3-38B-AWQ_core", "InternVL3-8B": "InternVL3-8B", "Qwen2.5-VL-32B-Instruct-AWQ":
          "Qwen2.5-VL-32B-Instruct-AWQ", "Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B-Instruct",
          "llava-onevision-qwen2-7b-ov-hf": "llava-onevision-qwen2-7b-ov-hf"}
CORE = ("lc_loss", "rel", "cm", "roc", "dist")


def bal(y, p):
    return float(np.mean([np.mean(p[y == c] == c) for c in np.unique(y)]))


def rows(name):
    return [json.loads(x) for x in open(S / f"responses_{name}.jsonl")]


def calibration(L):
    R = Run(S)
    L += ["## Contextual calibration of reader answers", "", "| reader | raw | ÷ no-plot answer | ÷ mean answer per plot |", "|---|---|---|---|"]
    for m, f in DIRECT.items():
        rr = rows(f)
        prior = {r["spec"]: reader_vec(r) for r in rr if r.get("task") == "prior_only"}
        V = {(r["run_id"], r["spec"]): reader_vec(r) for r in rr if r.get("task") == "diagnosis" and R.level.get(r["run_id"]) in (None, R.top)}
        V = {k: v for k, v in V.items() if v is not None}
        mean = {sp: np.mean([v for k, v in V.items() if k[1] == sp], 0) for sp in {k[1] for k in V}}
        tr = {k: R.truth[k[0]] for k in V}

        def score(fn):
            pred = {k: CLASSES[int(np.argmax(w))] for k, v in V.items() if (w := fn(k, v)) is not None}
            return bal_acc(pred, tr, list(pred), R.classes)
        raw = score(lambda k, v: v)
        pc = score(lambda k, v: (v + 1e-3) / (prior[k[1]] + 1e-3) if prior.get(k[1]) is not None else None) if prior else float("nan")
        mc = score(lambda k, v: (v + 1e-3) / (mean[k[1]] + 1e-3))
        L.append(f"| {m} | {raw:.3f} | {pc:.3f} | {mc:.3f} |")
    L.append("")


def gap4(L, V):
    def s_shape(pts):
        P = np.array(pts)
        x, d = P[:, 0], P[:, 1] - P[:, 0]
        return (d[x < .5].mean() if (x < .5).any() else 0) - (d[x >= .5].mean() if (x >= .5).any() else 0), np.abs(d).mean()
    L += ["## Gap 4: which cue drives readers' answers (Spearman ρ, held-out core models)", "",
          "| reader | loss: ρ val-rise | loss: ρ train–val gap | reliability: ρ S-shape | reliability: ρ any deviation | healthy loss reasons citing a gap / plateau | healthy reliability reasons citing a deviation |",
          "|---|---|---|---|---|---|---|"]
    for m, f in DIRECT.items():
        rise, gap, po, S_, A_, pm = [], [], [], [], [], []
        hr_loss, hr_rel = [], []
        for r in rows(f):
            if r.get("task") != "diagnosis" or r["run_id"] not in V:
                continue
            v = reader_vec(r)
            if v is None:
                continue
            reason = str((r.get("parsed") or {}).get("reason", "")).lower()
            healthy = V[r["run_id"]]["failure"] == "healthy"
            if r["spec"] == "lc_loss":
                val = V[r["run_id"]]["values"]["lc_loss"]
                rise.append(val["val_last"] - val["val_best"])
                gap.append(val["val_last"] - val["train_last"])
                po.append(v[CLASSES.index("overfitting")])
                if healthy:
                    hr_loss.append(bool(re.search(r"gap|higher than|diverg|separat|plateau|converg|stabil|flatten", reason)))
            elif r["spec"] == "rel":
                s, a = s_shape(V[r["run_id"]]["values"]["rel"]["points"])
                S_.append(s)
                A_.append(a)
                pm.append(v[CLASSES.index("miscalibration")])
                if healthy:
                    hr_rel.append(bool(re.search(r"deviat|diagonal|not align|off the", reason)))
        rho = lambda a, b: spearmanr(a, b).correlation if len(set(b)) > 1 else float("nan")  # noqa: E731
        L.append(f"| {m} | {rho(rise, po):.2f} | {rho(gap, po):.2f} | {rho(S_, pm):.2f} | {rho(A_, pm):.2f} | "
                 f"{np.mean(hr_loss):.0%} | {np.mean(hr_rel):.0%} |")
    H = [v for v in V.values() if v["split"] == "test" and v["failure"] == "healthy"]
    gaps = [h["values"]["lc_loss"]["val_last"] - h["values"]["lc_loss"]["train_last"] for h in H]
    devs = [s_shape(h["values"]["rel"]["points"])[1] for h in H]
    L += ["", f"Healthy held-out models with a train–val loss gap > 0.05: {np.mean(np.array(gaps) > .05):.0%}; healthy reliability "
          f"diagrams with mean |deviation| > 0.05: {np.mean(np.array(devs) > .05):.0%}.", ""]


def gap6(L, V):
    G = np.load(RUNS / "realworld/features.npz")
    T = json.load(open(RUNS / "realworld/truth.json"))
    C = json.load(open(RUNS / "realworld/cases.json"))
    idx = {r: i for i, r in enumerate(G["ids"])}
    L += ["## Gap 6: real-world cases with EXACT values (no reader), rule trained on synthetic models", "",
          "| case | textbook plot | detection AUC | failure named | control false alarm | controls mostly called |", "|---|---|---|---|---|---|"]
    out = []
    for cid, cv in C.items():
        if not cv["verified"]:
            continue
        sp = CANONICAL[cv["failure"]]
        tr = [(rtr.features(v["values"][sp], sp), CLASSES.index(v["failure"])) for v in V.values() if v["split"] == "train"]
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced")).fit(
            np.nan_to_num(np.array([a for a, _ in tr], float)), [b for _, b in tr])
        rids = [r for r, t in T.items() if t["case"] == cid]
        X = np.nan_to_num(np.array([rtr.features(rtr.truth_from_vector(G[sp][idx[r]], sp), sp) for r in rids], float))
        P = m.predict_proba(X)
        pred = [CLASSES[m.classes_[int(np.argmax(p))]] for p in P]
        isf = np.array([T[r]["condition"] == "failure" for r in rids])
        auc = roc_auc_score(isf, P[:, list(m.classes_).index(CLASSES.index(cv["failure"]))])
        named = np.mean([pred[i] in T[rids[i]]["present"] for i in np.where(isf)[0]])
        fa = np.mean([pred[i] not in T[rids[i]]["present"] for i in np.where(~isf)[0]])
        top = collections.Counter(pred[i] for i in np.where(~isf)[0]).most_common(1)[0][0]
        out.append((auc, named, fa))
        L.append(f"| {cid} | {sp} | {auc:.2f} | {named:.0%} | {fa:.0%} | {top} |")
    o = np.array(out)
    L += ["", f"Overall: detection AUC ≥ 0.70 in {np.mean(o[:, 0] >= .7):.0%} of cases, failure named {o[:, 1].mean():.0%}, "
          f"control false alarm {o[:, 2].mean():.0%}.", ""]


def gap7(L):
    L += ["## Gap 7: where accuracy curves carry class-imbalance information (fresh models, strongest calibrated level)", ""]
    for src, run in (("synthetic", "confirm_synthetic"), ("magic", "confirm_magic")):
        sev = json.load(open(RUNS / run / "meta.json"))["severities"]["class_imbalance"][2]
        n = 150
        H = Parallel(n_jobs=8)(delayed(train_run)(healthy_config("gap7", "healthy", i, epochs=100, source=src)) for i in range(n))
        I = Parallel(n_jobs=8)(delayed(train_run)(apply_failure(healthy_config("gap7", "imb", i, epochs=100, source=src),
                                                                "class_imbalance", sev)) for i in range(n))
        y = np.r_[np.zeros(n), np.ones(n)]
        tr = np.array([s["train_acc"] for s in H + I])
        va = np.array([s["val_acc"] for s in H + I])
        pos = np.median([np.mean(s["test_p"][s["test_y"] == 1] >= .5) for s in H])
        neg = np.median([np.mean(s["test_p"][s["test_y"] == 0] < .5) for s in H])
        L.append(f"- {src}: final val accuracy AUC {roc_auc_score(y, va[:, -1]):.2f}; best single-epoch train accuracy AUC "
                 f"{max(abs(roc_auc_score(y, tr[:, e]) - .5) for e in range(100)) + .5:.2f}; healthy accuracy on positives "
                 f"{pos:.3f} vs negatives {neg:.3f}")
    L.append("")


def gap8(L, V):
    dims = {sp: len(rtr.features(next(iter(V.values()))["values"][sp], sp)) for sp in CORE}

    def feats(vals, St):
        out = []
        for sp in St:
            f = rtr.features(vals.get(sp), sp) if vals.get(sp) is not None else None
            out += f if f is not None and not any(np.isnan(f)) else [np.nan] * dims[sp]
        return out
    mk = lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced", C=.5))  # noqa: E731
    sets = [("lc_loss", "cm"), CORE]
    exact = {St: mk().fit(np.array([feats(v["values"], St) for v in V.values() if v["split"] == "train"]),
                          [CLASSES.index(v["failure"]) for v in V.values() if v["split"] == "train"]) for St in sets}
    L += ["## Gap 8: several plots' readings combined (held-out core models)", "",
          "| reader | loss+CM, exact-value rule | all five, exact-value rule | all five, own-reading rule (5-fold CV) |", "|---|---|---|---|"]
    for m in DIRECT:
        O = {}
        for r in rows(f"{m}_extract"):
            d = rtr.from_reader(r.get("parsed"), r["spec"])
            if d:
                O.setdefault(r["run_id"], {})[r["spec"]] = d
        ks = sorted(O)
        y = np.array([CLASSES.index(V[k]["failure"]) for k in ks])
        cells = [bal(y, exact[St].predict(np.array([feats(O[k], St) for k in ks], float))) for St in sets]
        X = np.array([feats(O[k], CORE) for k in ks], float)
        p = np.zeros(len(y), int)
        for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
            p[b] = mk().fit(X[a], y[a]).predict(X[b])
        L.append(f"| {m} | {cells[0]:.3f} | {cells[1]:.3f} | {bal(y, p):.3f} |")
    ids = [k for k, v in V.items() if v["split"] == "test"]
    yt = np.array([CLASSES.index(V[k]["failure"]) for k in ids])
    L += ["", "Exact-value ceilings: " + ", ".join(f"{'+'.join(St) if len(St) < 5 else 'all five'} "
                                                 f"{bal(yt, exact[St].predict(np.array([feats(V[k]['values'], St) for k in ids], float))):.3f}" for St in sets), ""]


def main():
    V = json.load(open(S / "rtr_values.json"))
    L = ["# Exploratory gap analyses", ""]
    calibration(L)
    gap4(L, V)
    gap6(L, V)
    gap7(L)
    gap8(L, V)
    (S / "GAP_ANALYSES.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
