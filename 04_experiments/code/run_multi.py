"""Multi-failure study: 6 failures × 7 diagnostic plots, detection AND diagnosis benchmarks.
(The Oct 7 cluster run "multi_v3" used 4 failures × 6 plots — the code zipped for that run is the reference for it.)

  python run_multi.py --smoke                 # minutes, checks the pipeline
  python run_multi.py --name multi_v3         # full run (designed for a 24–48 core cluster node)

Steps
  1. Severity calibration per failure: B0 detection AUC on a grid -> severities at AUC 0.65 / 0.80 / 0.95
  2. Main set: healthy pool + each failure at each calibrated level (fresh seeds); 60/20/20 splits
  3. Render every plot type for every run (mark tables -> PNG)
  4. Detection: B0 / B1(plot) healthy-vs-failure AUC per failure × level × plot
     Diagnosis: B0 / B1(plot) 5-way "which state?" per level (healthy + 4 failures) -> accuracy, AUC, confusion
  5. results.json, meta.json (splits), manifest.jsonl, reader_items.json, report.md
B0 = every plot's mark vector + held-out scalars, so B0 is a superset of every B1 by construction (gap 2.5).
"""
import argparse
import json
import pathlib
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.isotonic import IsotonicRegression

from plotreveal import benchmarks as bm
from plotreveal import plots
from plotreveal.seeds import rng_for, seed_for
from plotreveal.world import FAILURES, apply_failure, healthy_config, train_run

ROOT = pathlib.Path(__file__).parent
SPECS = ("lc_loss", "lc_acc", "rel", "cm", "roc", "pr", "dist")
TARGETS = (0.65, 0.80, 0.95)
GRID = [0.01, 0.02, 0.04, 0.07, 0.1, 0.15, 0.25, 0.4, 0.6, 1.0]
FULL = dict(cal_healthy=600, cal_fail=250, main_healthy=1500, main_fail=500, epochs=100, reader_levels=(1, 2),
            reader_healthy=300, reader_fail=100)
SMOKE = dict(cal_healthy=60, cal_fail=30, main_healthy=150, main_fail=50, epochs=30, reader_levels=(1, 2),
             reader_healthy=10, reader_fail=4)
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def features(states, spec):
    return np.array([plots.marks_vector(plots.mark_table(s, spec), spec) for s in states])


def b0(states):
    sc = np.array([[s["test_loss"], s["test_acc"], s["test_pos_rate"], s["test_ece"], s["val_conf"]] for s in states])
    return np.hstack([features(states, sp) for sp in SPECS] + [sc])


def split_of(n, *seed):
    perm = rng_for("split", *seed).permutation(n)
    a, b = int(0.6 * n), int(0.8 * n)
    return dict(train=perm[:a], val=perm[a:b], test=perm[b:])


def combine(splits, sizes):
    off = np.cumsum([0] + list(sizes[:-1]))
    return {k: np.concatenate([s[k] + o for s, o in zip(splits, off)]) for k in ("train", "val", "test")}


def calibrate(P, jobs, manifest):
    log("calibration: healthy pool")
    hc = [healthy_config("cal", "healthy", i, epochs=P["epochs"]) for i in range(P["cal_healthy"])]
    Xh = b0(Parallel(n_jobs=jobs)(delayed(train_run)(c) for c in hc))
    manifest += [dict(role="cal", run_id=f"cal_healthy_{i}", **c) for i, c in enumerate(hc)]
    out = {}
    for f in FAILURES:
        rows = []
        for s in GRID:
            fc = [apply_failure(healthy_config("cal", f, s, i, epochs=P["epochs"]), f, s) for i in range(P["cal_fail"])]
            X = np.vstack([Xh, b0(Parallel(n_jobs=jobs)(delayed(train_run)(c) for c in fc))])
            y = np.r_[np.zeros(len(Xh)), np.ones(len(fc))].astype(int)
            r = bm.best_of_k(X, y, bm.make_split(y, seed_for("cal-split", f, s)), seed=seed_for("cal", f, s),
                             saturation=False)
            rows.append(dict(severity=s, b0_auc=r["test_auc"], ci=r["ci"]))
            log(f"  {f:16s} s={s:<5} B0 AUC {r['test_auc']:.3f}")
        sev = np.array(GRID)
        fit = IsotonicRegression(increasing=True).fit_transform(sev, [r["b0_auc"] for r in rows])
        chosen = []
        for t in TARGETS:
            if fit.max() < t:
                chosen.append(dict(target=t, severity=float(sev[-1]), reachable=False))
            elif fit[0] >= t:
                chosen.append(dict(target=t, severity=float(sev[0]), reachable=True, at_floor=True))
            else:
                k = int(np.argmax(fit >= t))
                fr = (t - fit[k - 1]) / (fit[k] - fit[k - 1])
                chosen.append(dict(target=t, severity=float(sev[k - 1] + fr * (sev[k] - sev[k - 1])), reachable=True))
        out[f] = dict(grid=rows, fitted=fit.tolist(), chosen=chosen)
        log(f"  {f}: chosen " + ", ".join(f"{c['target']}→{c['severity']:.3f}" for c in chosen))
    return out


def b0_best(row):
    """Anyone holding the full state can also draw any plot, so the B0 estimate is the best (on validation) of the
    state-feature model and every plot's model; the test score of the chosen candidate is reported."""
    cand = max(["B0"] + list(SPECS), key=lambda k: row[k]["val_auc"])
    return dict(row[cand], via=cand)


def _render(state, rid, img_dir):
    for sp in SPECS:
        plots.render(plots.mark_table(state, sp), sp, path=img_dir / f"{rid}_{sp}.png")
    return rid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args()
    P = SMOKE if a.smoke else FULL
    out = ROOT / "runs" / (a.name or ("multi_smoke" if a.smoke else "multi_v3"))
    img = out / "img"
    img.mkdir(parents=True, exist_ok=True)
    manifest = []

    cal = calibrate(P, a.jobs, manifest)
    json.dump(cal, open(out / "calibration.json", "w"), indent=1)

    # ---- main set ----
    log("main set: training")
    groups = {("healthy", None): [healthy_config("main", "healthy", i, epochs=P["epochs"]) for i in range(P["main_healthy"])]}
    for f in FAILURES:
        for j, c in enumerate(cal[f]["chosen"]):
            groups[(f, j)] = [apply_failure(healthy_config("main", f, j, i, epochs=P["epochs"]), f, c["severity"])
                              for i in range(P["main_fail"])]
    ids, cfgs, gkey = [], [], []
    for (f, j), cs in groups.items():
        for i, c in enumerate(cs):
            ids.append(f"m_{f}_L{j}_{i}" if j is not None else f"m_healthy_{i}")
            cfgs.append(c)
            gkey.append((f, j))
    states = Parallel(n_jobs=a.jobs, batch_size=8)(delayed(train_run)(c) for c in cfgs)
    manifest += [dict(role="main", run_id=r, level=k[1], **c) for r, c, k in zip(ids, cfgs, gkey)]
    by = {k: [i for i, g in enumerate(gkey) if g == k] for k in groups}
    splits = {k: split_of(len(v), "main", *map(str, k)) for k, v in by.items()}
    meta = dict(specs=list(SPECS), failures=list(FAILURES), targets=list(TARGETS),
                groups={f"{k[0]}|{k[1]}": {ids[i]: name for name, idx in splits[k].items() for i in np.array(v)[idx]}
                        for k, v in by.items()},
                severities={f: [c["severity"] for c in cal[f]["chosen"]] for f in FAILURES})
    json.dump(meta, open(out / "meta.json", "w"))

    log("rendering plots")
    Parallel(n_jobs=a.jobs, batch_size=16)(delayed(_render)(st, rid, img) for st, rid in zip(states, ids))

    log("features")
    F = {sp: features(states, sp) for sp in SPECS}
    F["B0"] = b0(states)

    def stack(keys):
        rows = [np.array(by[k]) for k in keys]
        idx = np.concatenate(rows)
        sp = combine([splits[k] for k in keys], [len(r) for r in rows])
        return idx, sp

    # ---- detection ----
    res = dict(detection=[], diagnosis=[])
    for f in FAILURES:
        for j in range(len(TARGETS)):
            idx, sp = stack([("healthy", None), (f, j)])
            y = np.r_[np.zeros(len(by[("healthy", None)])), np.ones(len(by[(f, j)]))].astype(int)
            row = dict(failure=f, level=j, target=TARGETS[j], severity=cal[f]["chosen"][j]["severity"])
            for lv in ["B0"] + list(SPECS):
                r = bm.best_of_k(F[lv][idx], y, sp, seed=seed_for("det", f, j, lv), saturation=(lv == "B0"))
                r.pop("test_scores")
                row[lv] = r
            row["B0_state_only"] = row["B0"]
            row["B0"] = b0_best(row)
            res["detection"].append(row)
            log(f"detect {f:16s} L{j}: B0 {row['B0']['test_auc']:.3f} | " +
                " ".join(f"{sp} {row[sp]['test_auc']:.2f}" for sp in SPECS))

    # ---- diagnosis (5-way) ----
    classes = ["healthy"] + list(FAILURES)
    for j in range(len(TARGETS)):
        keys = [("healthy", None)] + [(f, j) for f in FAILURES]
        idx, sp = stack(keys)
        y = np.concatenate([np.full(len(by[k]), c) for c, k in enumerate(keys)])
        row = dict(level=j, target=TARGETS[j])
        for lv in ["B0"] + list(SPECS):
            row[lv] = bm.best_of_k_multi(F[lv][idx], y, sp, seed=seed_for("diag", j, lv), classes=classes)
        row["B0_state_only"] = row["B0"]
        row["B0"] = b0_best(row)
        res["diagnosis"].append(row)
        log(f"diagnose L{j}: acc B0 {row['B0']['test_acc']:.3f} | " +
            " ".join(f"{s} {row[s]['test_acc']:.2f}" for s in SPECS))

    json.dump(res, open(out / "results.json", "w"), indent=1, default=float)
    with open(out / "manifest.jsonl", "w") as fh:
        for m in manifest:
            fh.write(json.dumps(m) + "\n")

    # ---- reader items: held-out test plots (healthy + each failure at the reader levels) ----
    r = rng_for("reader-items")
    pick = list(r.choice(splits[("healthy", None)]["test"], min(P["reader_healthy"], len(splits[("healthy", None)]["test"])),
                         replace=False))
    picked = [ids[by[("healthy", None)][i]] for i in pick]
    for f in FAILURES:
        for j in P["reader_levels"]:
            te = splits[(f, j)]["test"]
            picked += [ids[by[(f, j)][i]] for i in r.choice(te, min(P["reader_fail"], len(te)), replace=False)]
    items = [dict(item_id=f"{rid}_{sp}", run_id=rid, task="diagnosis", spec=sp, image=f"img/{rid}_{sp}.png")
             for sp in SPECS for rid in picked]
    items += [dict(item_id=f"prior_{sp}", task="prior_only", spec=sp, image=None) for sp in SPECS]
    json.dump(items, open(out / "reader_items.json", "w"), indent=0)
    report(out, cal, res)
    log(f"done → {out}  ({len(items)} reader items)")


def report(out, cal, res):
    L = ["# Multi-failure study (v3)", "", "## Calibrated severities (B0 detection AUC targets 0.65 / 0.80 / 0.95)", ""]
    for f, c in cal.items():
        L.append(f"- {f}: " + ", ".join(f"{x['target']}→s={x['severity']:.3f}{'' if x['reachable'] else ' (NOT reachable)'}"
                                        for x in c["chosen"]))
    L += ["", "## Detection (held-out AUC, healthy vs failure)", "",
          "| failure | level | B0 | " + " | ".join(SPECS) + " |", "|---|---|---|" + "---|" * len(SPECS)]
    for r in res["detection"]:
        L.append(f"| {r['failure']} | {r['target']} | {r['B0']['test_auc']:.3f} | " +
                 " | ".join(f"{r[s]['test_auc']:.3f}" for s in SPECS) + " |")
    L += ["", "## Diagnosis (5-way: healthy + 4 failures; held-out accuracy, chance = 0.20)", "",
          "| level | B0 | " + " | ".join(SPECS) + " |", "|---|---|" + "---|" * len(SPECS)]
    for r in res["diagnosis"]:
        L.append(f"| {r['target']} | {r['B0']['test_acc']:.3f} | " + " | ".join(f"{r[s]['test_acc']:.3f}" for s in SPECS) + " |")
    L += ["", "Confusion matrices (rows = truth, cols = predicted; order: " + ", ".join(res["diagnosis"][0]["B0"]["classes"]) + ")"]
    for r in res["diagnosis"]:
        for s in ["B0"] + list(SPECS):
            L.append(f"- level {r['target']} {s}: {r[s]['confusion']}")
    (out / "report.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
