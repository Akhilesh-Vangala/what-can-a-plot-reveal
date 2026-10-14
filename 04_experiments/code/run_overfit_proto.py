"""Overfitting × learning-curve prototype (first decision point).

  python run_overfit_proto.py --smoke          # minutes: checks the pipeline end to end
  python run_overfit_proto.py                  # full prototype (~30–60 min on a laptop)

Steps
  1. Calibrate severity: B0 detection AUC on a severity grid -> severities hitting AUC targets (gap 1.3)
  2. Main item set at the calibrated severities with fresh seeds
  3. B0 (full state) / B1 (mark table) / B2 (image) detection benchmarks per severity and plot variant
  4. Lineups for readers (1 failed + 19 healthy, and null lineups) with a hidden key
  5. results.json, manifest.jsonl, report.md, figures

Everything is seeded (seeds.seed_for) and recorded in the manifest, so every item can be regenerated.
"""
import argparse
import json
import pathlib
import time

import matplotlib
import numpy as np
from joblib import Parallel, delayed
from sklearn.isotonic import IsotonicRegression

from plotreveal import benchmarks as bm
from plotreveal import plots
from plotreveal.generator import apply_failure, b0_features, healthy_config, train_run
from plotreveal.seeds import rng_for, seed_for

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).parent
FAILURE = "overfitting"
SPEC_NAMES = ("lc_loss", "lc_acc")

FULL = dict(cal_healthy=600, cal_fail=300, grid=[0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0],
            main_healthy=1500, main_fail=600, epochs=100, lineups=8, null_lineups=8)
SMOKE = dict(cal_healthy=40, cal_fail=20, grid=[0.3, 0.7, 1.0],
             main_healthy=120, main_fail=40, epochs=30, lineups=1, null_lineups=1)
TARGETS = (0.65, 0.80, 0.95)


def log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:6.0f}s] {msg}", flush=True)


def train_all(cfgs, jobs):
    return Parallel(n_jobs=jobs)(delayed(train_run)(c) for c in cfgs)


def combine(split_h, split_f, n_h):
    """Healthy items 0..n_h-1 and failed items n_h.. keep their own stratified splits."""
    return {k: np.r_[split_h[k], split_f[k] + n_h] for k in split_h}


def calibrate(P, jobs, out, manifest):
    log("calibration: training healthy pool")
    h_cfgs = [healthy_config("cal", "healthy", i, epochs=P["epochs"]) for i in range(P["cal_healthy"])]
    h_states = train_all(h_cfgs, jobs)
    manifest += [dict(role="cal", idx=i, **c) for i, c in enumerate(h_cfgs)]
    Xh = np.array([b0_features(s) for s in h_states])
    rows = []
    for s in P["grid"]:
        f_cfgs = [apply_failure(healthy_config("cal", FAILURE, s, i, epochs=P["epochs"]), FAILURE, s)
                  for i in range(P["cal_fail"])]
        f_states = train_all(f_cfgs, jobs)
        manifest += [dict(role="cal", idx=i, **c) for i, c in enumerate(f_cfgs)]
        X = np.vstack([Xh, [b0_features(st) for st in f_states]])
        y = np.r_[np.zeros(len(Xh)), np.ones(len(f_states))].astype(int)
        r = bm.best_of_k(X, y, bm.make_split(y, seed_for("cal-split", s)), seed=seed_for("cal-bench", s),
                         saturation=False)
        rows.append(dict(severity=float(s), b0_auc=r["test_auc"], ci=r["ci"], best=r["best"]))
        log(f"  s={s:.2f}  B0 AUC={r['test_auc']:.3f} {np.round(r['ci'], 3).tolist()}  ({r['best']})")

    sev = np.array([r["severity"] for r in rows])
    fitted = IsotonicRegression(increasing=True).fit_transform(sev, [r["b0_auc"] for r in rows])
    chosen = []
    for t in TARGETS:
        if fitted.max() < t:
            chosen.append(dict(target=t, severity=float(sev[-1]), reachable=False))
        elif fitted.min() >= t:
            chosen.append(dict(target=t, severity=float(sev[0]), reachable=True, at_grid_floor=True))
        else:
            k = int(np.argmax(fitted >= t))  # first grid point reaching the target; interpolate from the previous one
            frac = (t - fitted[k - 1]) / (fitted[k] - fitted[k - 1])
            chosen.append(dict(target=t, severity=float(sev[k - 1] + frac * (sev[k] - sev[k - 1])), reachable=True))
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.errorbar(sev, [r["b0_auc"] for r in rows], yerr=np.abs(np.array([r["ci"] for r in rows]).T
                                                              - [r["b0_auc"] for r in rows]), fmt="o", label="B0 AUC")
    ax.plot(sev, fitted, "-", label="isotonic fit")
    for c in chosen:
        ax.axhline(c["target"], ls=":", c="gray")
        ax.axvline(c["severity"], ls=":", c="gray")
    ax.set_xlabel("overfitting severity s")
    ax.set_ylabel("B0 detection AUC")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "calibration.png", dpi=130)
    plt.close(fig)
    return dict(grid=rows, fitted=fitted.tolist(), chosen=chosen)


def _render_item(state, spec, path):
    marks = plots.mark_table(state, spec)
    return plots.downsample(plots.render(marks, spec, path=path))


def main_study(P, levels, jobs, out, manifest, skip_b2=False):
    img_dir = out / "img"
    img_dir.mkdir(exist_ok=True)
    log("main set: training healthy models")
    h_cfgs = [healthy_config("main", "healthy", i, epochs=P["epochs"]) for i in range(P["main_healthy"])]
    h_states = train_all(h_cfgs, jobs)
    h_ids = [f"main_healthy_{i}" for i in range(len(h_cfgs))]
    manifest += [dict(role="main", run_id=r, **c) for r, c in zip(h_ids, h_cfgs)]
    perm = rng_for("main-split", "healthy").permutation(len(h_states))
    n_tr, n_va = int(0.6 * len(perm)), int(0.2 * len(perm))
    split_h = dict(train=perm[:n_tr], val=perm[n_tr:n_tr + n_va], test=perm[n_tr + n_va:])
    split_of = lambda sp_dict, ids: {ids[i]: name for name, idx in sp_dict.items() for i in idx}  # noqa: E731
    meta = dict(specs=list(SPEC_NAMES), healthy=split_of(split_h, h_ids), levels=[])

    log("main set: rendering healthy plots")
    h_imgs = {sp: np.array(Parallel(n_jobs=jobs)(delayed(_render_item)(st, sp, img_dir / f"{rid}_{sp}.png")
                                                 for st, rid in zip(h_states, h_ids))) for sp in SPEC_NAMES}
    h_marks = {sp: [plots.mark_table(st, sp) for st in h_states] for sp in SPEC_NAMES}
    for sp in SPEC_NAMES:
        h_marks[sp][0].to_csv(out / f"example_marks_{sp}.csv", index=False)

    results, key = [], []
    for j, lv in enumerate(levels):
        s = lv["severity"]
        log(f"level {j + 1}/{len(levels)} (target AUC {lv['target']}, s={s:.3f}): training failed models")
        f_cfgs = [apply_failure(healthy_config("main", FAILURE, j, i, epochs=P["epochs"]), FAILURE, s)
                  for i in range(P["main_fail"])]
        f_states = train_all(f_cfgs, jobs)
        f_ids = [f"main_{FAILURE}_L{j}_{i}" for i in range(len(f_cfgs))]
        manifest += [dict(role="main", run_id=r, level=j, **c) for r, c in zip(f_ids, f_cfgs)]
        perm = rng_for("main-split", FAILURE, j).permutation(len(f_states))
        n_tr, n_va = int(0.6 * len(perm)), int(0.2 * len(perm))
        split_f = dict(train=perm[:n_tr], val=perm[n_tr:n_tr + n_va], test=perm[n_tr + n_va:])
        split = combine(split_h, split_f, len(h_states))
        meta["levels"].append(dict(level=j, target=lv["target"], severity=s, failed=split_of(split_f, f_ids)))
        y = np.r_[np.zeros(len(h_states)), np.ones(len(f_states))].astype(int)
        te_fail = y[split["test"]] == 1

        def summarise(r):
            sc = r.pop("test_scores")
            r["lineup_acc"] = bm.lineup_accuracy(sc[te_fail], sc[~te_fail], seed=seed_for("lineup-acc", j))
            return r

        res = dict(level=j, target=lv["target"], severity=s,
                   natural_units=dict(n_train_mean=float(np.mean([c["n_train"] for c in f_cfgs])),
                                      hidden_mean=float(np.mean([c["hidden"] for c in f_cfgs])),
                                      alpha_median=float(np.median([c["alpha"] for c in f_cfgs]))))
        X0 = np.array([b0_features(st) for st in h_states + f_states])
        res["B0"] = summarise(bm.best_of_k(X0, y, split, seed=seed_for("bench", "B0", j)))
        log(f"  B0 AUC {res['B0']['test_auc']:.3f}")
        for sp in SPEC_NAMES:
            f_marks = [plots.mark_table(st, sp) for st in f_states]
            X1 = np.array([plots.marks_vector(m) for m in h_marks[sp] + f_marks])
            res[f"B1_{sp}"] = summarise(bm.best_of_k(X1, y, split, seed=seed_for("bench", "B1", sp, j)))
            f_imgs = np.array(Parallel(n_jobs=jobs)(delayed(_render_item)(st, sp, img_dir / f"{rid}_{sp}.png")
                                                    for st, rid in zip(f_states, f_ids)))
            if skip_b2:
                log(f"  {sp}: B1 AUC {res[f'B1_{sp}']['test_auc']:.3f} | B2 skipped (runs on GPU)")
            else:
                imgs = np.concatenate([h_imgs[sp], f_imgs])
                res[f"B2_{sp}"] = summarise(bm.image_benchmark(imgs, y, split, seed=seed_for("bench", "B2", sp, j)))
                log(f"  {sp}: B1 AUC {res[f'B1_{sp}']['test_auc']:.3f} | B2 AUC {res[f'B2_{sp}']['test_auc']:.3f} "
                    f"({res[f'B2_{sp}']['best']})")

            # lineups for readers: real panel from the failed test items, nulls from healthy test items
            lr = rng_for("lineups", sp, j)
            f_test = split["test"][te_fail] - len(h_states)
            for k in range(P["lineups"]):
                real = int(lr.choice(f_test))
                nulls = lr.choice(split_h["test"], 19, replace=False)
                pos = int(lr.integers(20))
                panel_ids = [h_ids[n] for n in nulls]
                panel_ids.insert(pos, f_ids[real])
                panels = [h_marks[sp][n] for n in nulls]
                panels.insert(pos, f_marks[real])
                item = f"LU_{sp}_L{j}_{k}"
                plots.render_lineup(panels, sp, out / "lineups" / f"{item}.png")
                key.append(dict(item_id=item, spec=sp, level=j, severity=s, target_auc=lv["target"],
                                null_lineup=False, real_panel=pos + 1, panel_runs=panel_ids))
        res["ordering"] = {}
        for sp in SPEC_NAMES:
            b0, b1 = res["B0"]["test_auc"], res[f"B1_{sp}"]["test_auc"]
            b2 = res[f"B2_{sp}"]["test_auc"] if f"B2_{sp}" in res else None
            res["ordering"][sp] = dict(B0=b0, B1=b1, B2=b2,
                                       violation=bool(b1 > b0 + 0.02 or (b2 is not None and b2 > b1 + 0.02)))
        results.append(res)

    for sp in SPEC_NAMES:  # null lineups: 20 healthy test plots
        lr = rng_for("null-lineups", sp)
        for k in range(P["null_lineups"]):
            nulls = lr.choice(split_h["test"], 20, replace=False)
            item = f"LU_{sp}_null_{k}"
            plots.render_lineup([h_marks[sp][n] for n in nulls], sp, out / "lineups" / f"{item}.png")
            key.append(dict(item_id=item, spec=sp, level=None, severity=0.0, null_lineup=True, real_panel=None,
                            panel_runs=[h_ids[n] for n in nulls]))
    json.dump(meta, open(out / "meta.json", "w"), indent=0)
    return results, key


def report(out, P, cal, results, smoke):
    fig, axs = plt.subplots(1, len(SPEC_NAMES), figsize=(9, 3.4), sharey=True)
    for ax, sp in zip(axs, SPEC_NAMES):
        x = [r["target"] for r in results]
        for lvl, mk in (("B0", "o"), (f"B1_{sp}", "s"), (f"B2_{sp}", "^")):
            if lvl not in results[0]:
                continue
            v = np.array([r[lvl]["test_auc"] for r in results])
            ci = np.array([r[lvl]["ci"] for r in results]).T
            ax.errorbar(x, v, yerr=np.abs(ci - v), marker=mk, capsize=3, label=lvl.split("_")[0])
        ax.plot([0.6, 1], [0.6, 1], ":", c="gray")
        ax.set_title(sp)
        ax.set_xlabel("calibration target (B0 AUC)")
    axs[0].set_ylabel("held-out detection AUC")
    axs[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "levels.png", dpi=130)
    plt.close(fig)

    L = [f"# Overfitting × learning curve prototype{' (SMOKE — not results)' if smoke else ''}", "",
         f"Sizes: {json.dumps({k: v for k, v in P.items() if k != 'grid'})}", "",
         "## 1. Severity calibration (B0 detection AUC)", "", "| s | B0 AUC | 95% CI | best |", "|---|---|---|---|"]
    L += [f"| {r['severity']:.2f} | {r['b0_auc']:.3f} | {r['ci'][0]:.3f}–{r['ci'][1]:.3f} | {r['best']} |"
          for r in cal["grid"]]
    L += ["", "Chosen: " + "; ".join(f"target {c['target']} → s={c['severity']:.3f}"
                                     + ("" if c["reachable"] else " (NOT reachable)") for c in cal["chosen"]), "",
          "## 2. Detection by level (held-out AUC [95% CI]; lineup acc = P(real panel ranks first of 20))", "",
          "| target | s | B0 | B1 loss | B2 loss | B1 acc | B2 acc |", "|---|---|---|---|---|---|---|"]
    fmt = lambda r: "–" if r is None else f"{r['test_auc']:.3f} [{r['ci'][0]:.2f}–{r['ci'][1]:.2f}] · lineup {r['lineup_acc']:.2f}"
    for r in results:
        L.append(f"| {r['target']} | {r['severity']:.3f} | {fmt(r['B0'])} | {fmt(r['B1_lc_loss'])} | "
                 f"{fmt(r.get('B2_lc_loss'))} | {fmt(r['B1_lc_acc'])} | {fmt(r.get('B2_lc_acc'))} |")
    L += ["", "## 3. Checks", ""]
    for r in results:
        for sp, o in r["ordering"].items():
            b2 = "–" if o["B2"] is None else f"{o['B2']:.3f}"
            L.append(f"- level {r['target']} {sp}: B0 {o['B0']:.3f} ≥ B1 {o['B1']:.3f} ≥ B2 {b2} → "
                     + ("**ordering violated (benchmark under-trained?)**" if o["violation"] else "ordering OK"))
    L += ["", "Saturation (test AUC at 25/50/100% of training data):"]
    for r in results:
        for lvl in ["B0"] + [f"B{b}_{sp}" for sp in SPEC_NAMES for b in (1, 2)]:
            if lvl in r and "saturation" in r[lvl]:
                L.append(f"- level {r['target']} {lvl}: " + ", ".join(f"{f:.2f}→{a:.3f}" for f, a in r[lvl]["saturation"]))
    L += ["", "Natural units at each level: " + "; ".join(
        f"target {r['target']}: n_train≈{r['natural_units']['n_train_mean']:.0f}, hidden≈{r['natural_units']['hidden_mean']:.0f}, "
        f"alpha≈{r['natural_units']['alpha_median']:.1e}" for r in results)]
    (out / "report.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--skip-b2", action="store_true", help="skip the image benchmark (run it on the GPU instead)")
    a = ap.parse_args()
    P = SMOKE if a.smoke else FULL
    out = ROOT / "runs" / (a.name or ("overfit_proto_smoke" if a.smoke else "overfit_proto"))
    (out / "lineups").mkdir(parents=True, exist_ok=True)
    manifest = []
    cal = calibrate(P, a.jobs, out, manifest)
    levels = [c for c in cal["chosen"]]
    results, key = main_study(P, levels, a.jobs, out, manifest, skip_b2=a.skip_b2)
    json.dump(dict(params=P, calibration=cal, results=results), open(out / "results.json", "w"), indent=1, default=float)
    json.dump(key, open(out / "lineups" / "key.json", "w"), indent=1)
    with open(out / "manifest.jsonl", "w") as fh:
        for m in manifest:
            fh.write(json.dumps(m) + "\n")
    report(out, P, cal, results, a.smoke)
    log(f"done → {out}/report.md")


if __name__ == "__main__":
    main()
