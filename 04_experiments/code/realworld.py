"""Step 3 — real-world check: real datasets + documented real-world bad practices (no severity knobs).

  python realworld.py [--seeds 20] [--jobs 8]   -> runs/realworld/ (img/, truth.json, cases.json, features.npz,
                                                   reader_items.json, report.md)

Each case pairs a FAILURE condition (a real mechanism documented in the literature) with a HEALTHY control on the same
data. A case counts only if its failure is verified objectively (the signature statistic differs from the control in the
expected direction by a pre-set margin); unverified cases are reported and excluded. Models, training loop (two-layer MLP,
100 epochs of partial_fit), plots and mark vectors are the same as in the main study, so the synthetic-trained benchmarks
can be applied to these plots unchanged. Label noise is excluded: no clean real tabular source with documented label
errors was available, and simulating it would not be a real-world case.
"""
import argparse
import json
import pathlib
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.datasets import fetch_openml
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import f_classif
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier

from plotreveal import plots
from plotreveal.seeds import seed_for

ROOT = pathlib.Path(__file__).parent
SPECS = ("lc_loss", "lc_acc", "rel", "rel_n", "cm", "cm_norm", "roc", "pr", "dist")
EPOCHS = 100
T0 = time.time()

# (case id, dataset, failure, mechanism, reference for the mechanism)
CASES = [
    ("elec_shift", "electricity", "distribution_shift", "train on the earliest half of the dates, test on the latest quarter",
     "temporal concept drift (Gama et al. 2014)"),
    ("bank_shift", "bank-marketing", "distribution_shift", "rows are in time order: train on the first half, test on the last quarter",
     "temporal shift in the bank-marketing campaign data (Moro et al. 2014)"),
    ("mammo_imb", "mammography", "class_imbalance", "natural 2.3% positive rate", "natural class imbalance"),
    ("ozone_imb", "ozone-level-8hr", "class_imbalance", "natural 6.3% positive rate", "natural class imbalance"),
    ("pc1_imb", "pc1", "class_imbalance", "natural 6.9% positive rate (software defects)", "natural class imbalance"),
    ("kc1_miscal", "kc1", "miscalibration", "minority class oversampled to 50% in training only",
     "resampling miscalibrates probabilities (Xenopoulos et al. 2022, Fig. 8)"),
    ("bank_miscal", "bank-marketing", "miscalibration", "minority class oversampled to 50% in training only",
     "resampling miscalibrates probabilities (Xenopoulos et al. 2022, Fig. 8)"),
    ("ozone_leak", "ozone-level-8hr", "data_leakage", "minority oversampled BEFORE the train/val/test split",
     "overlap leakage (Yang et al. ASE 2022)"),
    ("madelon_leak", "madelon", "data_leakage", "top-20 features selected on ALL data (incl. test) before splitting",
     "feature-selection leakage (Hastie et al., ESL 7.10.2; Yang et al. ASE 2022)"),
    ("elec_overfit", "electricity", "overfitting", "256-unit network, no regularization, 300 training rows",
     "overparameterized model on small data"),
    ("diabetes_overfit", "diabetes", "overfitting", "256-unit network, no regularization, 400 training rows",
     "overparameterized model on small data"),
]
POSITIVE = {"electricity": "UP", "bank-marketing": "2", "mammography": "1", "ozone-level-8hr": "2", "pc1": "true",
            "kc1": "true", "madelon": "1", "diabetes": "tested_positive"}
# verification: (statistic, direction, minimum failure-minus-control difference in medians)
VERIFY = dict(distribution_shift=("val_minus_test_acc", +1, 0.03), class_imbalance=("pos_rate", -1, 0.20),
              miscalibration=("ece_excess", +1, 0.03), data_leakage=("test_minus_deploy_auc", +1, 0.03),
              overfitting=("val_loss_rise", +1, 0.05))
# conditions present in a model, measured with fixed thresholds (real data can carry several at once)
PRESENT = dict(class_imbalance=lambda t: t["pos_rate"] < 0.20,
               miscalibration=lambda t: t["ece_excess"] >= 0.05,
               distribution_shift=lambda t: t["val_minus_test_acc"] >= 0.05,
               data_leakage=lambda t: t["test_minus_deploy_auc"] >= 0.05,
               overfitting=lambda t: t["val_loss_rise"] >= 0.10)
_DATA = {}


def log(m):
    print(f"[{time.time() - T0:5.0f}s] {m}", flush=True)


def load(name):
    if name not in _DATA:
        d = fetch_openml(name=name, version=1, as_frame=True, parser="auto")
        X = d.data.copy()
        for c in X.columns:  # categorical -> integer codes (bank-marketing)
            if not pd.api.types.is_numeric_dtype(X[c]):
                X[c] = X[c].astype("category").cat.codes
        y = (d.target.astype(str) == POSITIVE[name]).astype(int).to_numpy()
        order = np.argsort(X["date"].to_numpy(), kind="stable") if name == "electricity" else np.arange(len(y))
        _DATA[name] = (X.to_numpy(np.float64)[order], y[order])
    return _DATA[name]


def ece(p, y, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def ece_excess(p, y, seed, reps=200):
    """ECE minus its expected value for a perfectly calibrated model with the same predictions and sample size
    (small test sets inflate ECE through noise alone)."""
    rng = np.random.default_rng(seed)
    null = np.mean([ece(p, (rng.random(len(p)) < p).astype(int)) for _ in range(reps)])
    return ece(p, y) - float(null)


def train(Xtr, ytr, Xva, yva, Xte, yte, Xdep, ydep, hidden, alpha, lr, seed):
    """Same training loop and state as plotreveal.world.train_run."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr, Xva, Xte, Xdep = [(a - mu) / sd for a in (Xtr, Xva, Xte, Xdep)]
    clf = MLPClassifier(hidden_layer_sizes=(hidden, hidden), alpha=alpha, learning_rate_init=lr,
                        batch_size=min(128, len(ytr)), random_state=seed)
    pr = lambda X: np.clip(clf.predict_proba(X)[:, 1], 1e-7, 1 - 1e-7)  # noqa: E731
    logs = {k: np.empty(EPOCHS, np.float32) for k in ("train_loss", "val_loss", "train_acc", "val_acc")}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for e in range(EPOCHS):
            clf.partial_fit(Xtr, ytr, classes=[0, 1])
            ptr, pva = pr(Xtr), pr(Xva)
            logs["train_loss"][e], logs["val_loss"][e] = log_loss(ytr, ptr, labels=[0, 1]), log_loss(yva, pva, labels=[0, 1])
            logs["train_acc"][e], logs["val_acc"][e] = ((ptr >= .5) == ytr).mean(), ((pva >= .5) == yva).mean()
    pte, pdep = pr(Xte), pr(Xdep)
    return dict(**logs, test_p=pte.astype(np.float32), test_y=yte.astype(np.int8), val_p=pva.astype(np.float32),
                deploy_acc=float(((pdep >= .5) == ydep).mean()), test_acc=float(((pte >= .5) == yte).mean()),
                test_auc=float(roc_auc_score(yte, pte)), deploy_auc=float(roc_auc_score(ydep, pdep)),
                test_loss=float(log_loss(yte, pte, labels=[0, 1])), test_pos_rate=float(yte.mean()),
                test_ece=ece(pte, yte), val_conf=float(np.maximum(pva, 1 - pva).mean()))


def oversample(idx, y, rng):
    pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
    return np.r_[neg, rng.choice(pos, len(neg), replace=True)]


def split(pool, n_tr, n_va, n_te, rng):
    p = rng.permutation(pool)
    return p[:n_tr], p[n_tr:n_tr + n_va], p[n_tr + n_va:n_tr + n_va + n_te]


def build(case, failure_on, s):
    """Indices / features for one model. failure_on=False gives the healthy control on the same data."""
    cid, ds, fail = case[0], case[1], case[2]
    X, y = load(ds)
    rng = np.random.default_rng(seed_for("realworld", cid, failure_on, s))
    n = len(y)
    hidden, alpha, lr = int(rng.choice([16, 24, 32])), float(10 ** rng.uniform(-1, 0)), float(10 ** rng.uniform(-3.3, -2.7))
    if fail == "distribution_shift":
        early, late = np.arange(n // 2), np.arange(3 * n // 4, n)
        if failure_on:
            tr, va, _ = split(early, 2000, 500, 0, rng)
            te = rng.choice(late, 1000, replace=False)
            dep = rng.choice(np.setdiff1d(late, te), 1000, replace=False)
        else:
            pool = np.r_[early, late]
            tr, va, te = split(pool, 2000, 500, 1000, rng)
            dep = rng.choice(np.setdiff1d(pool, np.r_[tr, va, te]), 1000, replace=False)
    elif fail == "class_imbalance":
        if failure_on:
            pool = np.arange(n)
        else:  # balanced subsample: all positives + as many negatives
            pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
            pool = np.r_[pos, rng.choice(neg, len(pos), replace=False)]
        m = len(pool)
        tr, va, te = split(pool, int(.55 * m), int(.15 * m), int(.3 * m), rng)
        dep = te
    elif fail == "miscalibration":
        tr, va, te = split(np.arange(n), int(.55 * min(n, 6000)), int(.15 * min(n, 6000)), int(.3 * min(n, 6000)), rng)
        dep = te
        if failure_on:
            tr = oversample(tr, y, rng)
    elif fail == "data_leakage" and ds == "ozone-level-8hr":
        p = rng.permutation(n)
        dep, rest = p[:500], p[500:]
        if failure_on:  # oversample, then split: copies of the same minority rows land in train and test
            rest = rng.permutation(oversample(rest, y, rng))
            m = len(rest)
            tr, va, te = rest[:int(.55 * m)], rest[int(.55 * m):int(.7 * m)], rest[int(.7 * m):]
        else:
            m = len(rest)
            tr, va, te = split(rest, int(.55 * m), int(.15 * m), int(.3 * m), rng)
    elif fail == "data_leakage":  # madelon: feature selection before the split
        p = rng.permutation(n)
        dep, rest = p[:500], p[500:]
        tr, va, te = split(rest, 1300, 300, 500, rng)
        sel_rows = rest if failure_on else tr
        F, _ = f_classif(X[sel_rows], y[sel_rows])
        cols = np.argsort(-np.nan_to_num(F))[:20]
        return X[:, cols], y, tr, va, te, dep, hidden, alpha, lr
    elif fail == "overfitting":
        p = rng.permutation(n)
        if ds == "electricity":  # small labelled subset of a large dataset
            tr, va, te, dep = p[:300], p[300:800], p[800:1800], p[1800:2800]
        else:
            tr, va, te = p[:400], p[400:550], p[550:]
            dep = te
        if failure_on:
            hidden, alpha, lr = 256, 1e-6, 3e-3
        else:  # control must be genuinely healthy: small, regularized network
            hidden, alpha, lr = 16, 1.0, 1e-3
    return X, y, tr, va, te, dep, hidden, alpha, lr


def one(case, failure_on, s, img):
    X, y, tr, va, te, dep, h, a, lr = build(case, failure_on, s)
    st = train(X[tr], y[tr], X[va], y[va], X[te], y[te], X[dep], y[dep], h, a, lr, seed_for("rw-model", case[0], s))
    rid = f"{case[0]}_{'fail' if failure_on else 'ctrl'}_{s}"
    for sp in SPECS:
        plots.render(plots.mark_table(st, sp), sp, path=img / f"{rid}_{sp}.png")
    vl = np.asarray(st["val_loss"])
    stats = dict(val_minus_test_acc=float(st["val_acc"][-1] - st["test_acc"]), pos_rate=st["test_pos_rate"], ece=st["test_ece"],
                 ece_excess=ece_excess(st["test_p"].astype(float), st["test_y"].astype(int), seed_for("rw-ece", rid)),
                 test_minus_deploy_auc=st["test_auc"] - st["deploy_auc"], val_loss_rise=float(vl[-1] - vl.min()),
                 test_acc=st["test_acc"], deploy_acc=st["deploy_acc"], n_test=len(te), n_train=len(tr))
    feats = {sp: plots.marks_vector(plots.mark_table(st, sp), sp).astype(np.float32) for sp in SPECS}
    return rid, stats, feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", default="runs/realworld")
    a = ap.parse_args()
    out = ROOT / a.out
    img = out / "img"
    img.mkdir(parents=True, exist_ok=True)
    for ds in sorted({c[1] for c in CASES}):
        load(ds)
    log(f"{len(CASES)} cases × 2 conditions × {a.seeds} seeds")
    tasks = [(c, f, s) for c in CASES for f in (True, False) for s in range(a.seeds)]
    res = Parallel(n_jobs=a.jobs, batch_size=2)(delayed(one)(c, f, s, img) for c, f, s in tasks)
    truth, feats = {}, {sp: [] for sp in SPECS}
    ids = []
    for (c, f, s), (rid, stats, fv) in zip(tasks, res):
        truth[rid] = dict(case=c[0], dataset=c[1], failure=c[2] if f else "healthy", condition="failure" if f else "control", **stats)
        truth[rid]["present"] = [k for k, fn in PRESENT.items() if fn(stats)] or ["healthy"]
        ids.append(rid)
        for sp in SPECS:
            feats[sp].append(fv[sp])
    cases, L = {}, ["# Real-world check: cases and verification", "",
                    "| case | dataset | failure | mechanism | statistic | failure | control | verified |", "|---|---|---|---|---|---|---|---|"]
    for c in CASES:
        stat, sign, margin = VERIFY[c[2]]
        fm = float(np.median([t[stat] for t in truth.values() if t["case"] == c[0] and t["condition"] == "failure"]))
        cm = float(np.median([t[stat] for t in truth.values() if t["case"] == c[0] and t["condition"] == "control"]))
        ok = sign * (fm - cm) >= margin
        cases[c[0]] = dict(dataset=c[1], failure=c[2], mechanism=c[3], reference=c[4], statistic=stat, failure_median=fm,
                           control_median=cm, verified=bool(ok))
        L.append(f"| {c[0]} | {c[1]} | {c[2]} | {c[3]} | {stat} | {fm:.3f} | {cm:.3f} | {'yes' if ok else 'NO'} |")
        log(f"{c[0]:17s} {stat:22s} failure {fm:.3f} control {cm:.3f} → {'verified' if ok else 'NOT verified'}")
    keep = {k for k, v in cases.items() if v["verified"]}
    items = [dict(item_id=f"{rid}_{sp}", run_id=rid, task="diagnosis", spec=sp, image=f"img/{rid}_{sp}.png")
             for rid in ids if truth[rid]["case"] in keep for sp in SPECS]
    json.dump(truth, open(out / "truth.json", "w"), indent=0)
    json.dump(cases, open(out / "cases.json", "w"), indent=1)
    json.dump(items, open(out / "reader_items.json", "w"), indent=0)
    np.savez_compressed(out / "features.npz", ids=np.array(ids), **{sp: np.stack(v) for sp, v in feats.items()})
    (out / "report.md").write_text("\n".join(L) + f"\n\n{len(keep)}/{len(CASES)} cases verified; {len(items)} reader items.\n")
    log(f"done: {len(keep)}/{len(CASES)} cases verified, {len(items)} reader items → {out}")


if __name__ == "__main__":
    main()
