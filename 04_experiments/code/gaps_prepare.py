"""Cluster CPU job for gaps 3 and 5 (exploratory).

  python gaps_prepare.py ann  runs/confirm_synthetic          -> img_ann/, reader_items_ann.json (gap 5)
  python gaps_prepare.py sets runs/confirm_synthetic [--jobs 24] -> plotsets.json                  (gap 3)

ann:  for the held-out core models, render the three annotated designs (annotate.py) from the stored mark vectors and
      write reader items: extraction (read-then-reason) and direct diagnosis, both on the annotated plots.
sets: the diagnosis benchmark (same families, validation selection and prior correction as the study) trained on the
      concatenated mark vectors of a SET of plots, strongest severity: every single plot, every pair, and three
      common dashboards. Per set: balanced accuracy, per-failure recall, and per-failure detection AUC (healthy vs
      that failure, from the class probabilities).
"""
import itertools
import json
import pathlib
import sys

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

import annotate
from plotreveal import benchmarks as bm
from plotreveal.seeds import seed_for
from score_confirm import Run

SPECS = ["lc_loss", "lc_acc", "rel", "rel_n", "cm", "cm_norm", "roc", "pr", "dist"]
DASHBOARDS = [("lc_loss", "cm", "roc"), ("lc_loss", "lc_acc", "cm", "roc", "pr"), tuple(SPECS)]


def ann(run):
    run = pathlib.Path(run)
    F = np.load(run / "features.npz")
    idx = {r: i for i, r in enumerate(F["ids"])}
    out = run / "img_ann"
    out.mkdir(exist_ok=True)
    core = list(dict.fromkeys(it["run_id"] for it in json.load(open(run / "reader_items_core.json"))))
    items = []
    for rid in core:
        for sa, sp in annotate.BASE.items():
            annotate.render(F[sp][idx[rid]], sa, out / f"{rid}_{sa}.png")
            for task in ("extract", "diagnosis"):
                items.append(dict(item_id=f"{rid}_{sa}_{task}", run_id=rid, task=task, spec=sa, image=f"img_ann/{rid}_{sa}.png"))
    json.dump(items, open(run / "reader_items_ann.json", "w"), indent=0)
    print(f"{len(core)} models × 3 annotated designs → {len(items)} reader items", flush=True)


def one_set(S, X, y, split, classes, ids_te, truth):
    r = bm.best_of_k_multi(X, y, split, seed=seed_for("plotsets", *S), classes=classes)
    P, cm = r["test_proba"], np.array(r["confusion"])
    rec = {c: float(cm[i, i] / max(cm[i].sum(), 1)) for i, c in enumerate(classes)}
    det = {}
    yt = y[split["test"]]
    for i, c in enumerate(classes[1:], 1):
        m = (yt == 0) | (yt == i)
        det[c] = float(roc_auc_score(yt[m] == i, P[m, i]))
    return dict(plots=list(S), balanced_acc=r["balanced_acc"], recall=rec, detect_auc=det, best=r["best"])


def sets(run, jobs=24):
    run = pathlib.Path(run)
    R = Run(run)
    j = R.top
    F = np.load(run / "features.npz")
    pos = {r: i for i, r in enumerate(F["ids"])}
    keys = [("healthy", "None")] + [(f, str(j)) for f in R.meta["failures"]]
    rows, y, split_lab = [], [], []
    for c, (f, lv) in enumerate(keys):
        for rid, s in R.meta["groups"][f"{f}|{lv}"].items():
            rows.append(pos[rid])
            y.append(c)
            split_lab.append(s)
    rows, y, split_lab = np.array(rows), np.array(y), np.array(split_lab)
    split = {s: np.where(split_lab == s)[0] for s in ("train", "val", "test")}
    classes = ["healthy"] + list(R.meta["failures"])
    todo = [(s,) for s in SPECS] + list(itertools.combinations(SPECS, 2)) + DASHBOARDS
    Xs = {S: np.hstack([F[sp][rows] for sp in S]) for S in todo}
    res = Parallel(n_jobs=jobs)(delayed(one_set)(S, Xs[S], y, split, classes, None, None) for S in todo)
    json.dump(res, open(run / "plotsets.json", "w"), indent=1)
    best_pair = max((r for r in res if len(r["plots"]) == 2), key=lambda r: r["balanced_acc"])
    print(f"{run.name}: {len(res)} plot sets; best single {max(r['balanced_acc'] for r in res if len(r['plots']) == 1):.3f}, "
          f"best pair {best_pair['plots']} {best_pair['balanced_acc']:.3f}, all nine "
          f"{[r for r in res if len(r['plots']) == 9][0]['balanced_acc']:.3f}", flush=True)


if __name__ == "__main__":
    mode, run = sys.argv[1], sys.argv[2]
    jobs = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--jobs=")), 24))
    ann(run) if mode == "ann" else sets(run, jobs)
