"""v4 study: 6 failures × 9 diagnostic plots (incl. 2 design variants), synthetic or real data, with the audit fixes.

  python run_v4.py --source synthetic --name v4_synthetic [--smoke]
  python run_v4.py --source magic     --name v4_magic     [--smoke]

Changes vs multi_v3 (audit 2026-10-08, see FINDINGS_LOG.md):
  - balanced accuracy for diagnosis (classes are imbalanced by design), prior-corrected benchmark decisions
  - reliability-diagram B1 no longer contains undrawn bin counts; the "rel_n" design draws them (encoding variant)
  - calibration uses 5-fold cross-validated B0 AUC (less noise); saturation reported for every benchmark
  - B0 = validation-best of {state model, every plot model}; B1* = best of {mark model, image model} at scoring time
  - per-item test scores/probabilities saved (item_scores.json) for the formal over-claiming measure
  - larger samples; 6 failures (adds data leakage, distribution shift) and a validation-vs-test score plot
  - reader item sets: main (levels 1–2), core (strongest level, 5 plots, for the slow 38B reader), prompt variants
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
SPECS = ("lc_loss", "lc_acc", "rel", "rel_n", "cm", "cm_norm", "roc", "pr", "dist")
CORE = ("lc_loss", "rel", "cm", "roc", "dist")
VARIANT_SPECS = ("lc_loss", "rel", "cm")
TARGETS = (0.65, 0.80, 0.95)
GRID = [0.005, 0.01, 0.02, 0.04, 0.07, 0.1, 0.15, 0.25, 0.4, 0.6, 1.0]
FULL = dict(cal_healthy=600, cal_fail=300, main_healthy=2000, main_fail=600, epochs=100,
            reader_levels=(1, 2), reader_healthy=300, reader_fail=100, variant_healthy=100, variant_fail=50)
CONFIRM = dict(cal_healthy=600, cal_fail=300, main_healthy=3000, main_fail=900, epochs=100,
               reader_levels=(1, 2), reader_healthy=300, reader_fail=100, variant_healthy=100, variant_fail=50)
SMOKE = dict(cal_healthy=60, cal_fail=30, main_healthy=200, main_fail=60, epochs=30,
             reader_levels=(1, 2), reader_healthy=10, reader_fail=4, variant_healthy=4, variant_fail=2)
T0 = time.time()
TAG = ""  # seed namespace: "" = development runs; "confirm" = preregistered confirmatory run (all-new models)


def T(name):
    return f"{TAG}:{name}" if TAG else name


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def features(states, spec):
    return np.array([plots.marks_vector(plots.mark_table(s, spec), spec) for s in states])


def b0(states):
    """Full pipeline state: every plot's marks + held-out scalars (NOT the clean deployment set, which is ground truth)."""
    sc = np.array([[s["test_loss"], s["test_acc"], s["test_pos_rate"], s["test_ece"], s["val_conf"]] for s in states])
    return np.hstack([features(states, sp) for sp in SPECS] + [sc])


def split_of(n, *seed):
    perm = rng_for("split", *seed).permutation(n)
    a, b = int(0.6 * n), int(0.8 * n)
    return dict(train=perm[:a], val=perm[a:b], test=perm[b:])


def combine(splits, sizes):
    off = np.cumsum([0] + list(sizes[:-1]))
    return {k: np.concatenate([s[k] + o for s, o in zip(splits, off)]) for k in ("train", "val", "test")}


def best_on_val(row, keys):
    cand = max(keys, key=lambda k: row[k]["val_auc"])
    return dict({k: v for k, v in row[cand].items() if k not in ("test_scores", "test_proba")}, via=cand)


def marks_text(m, sp):
    """The drawn values of a plot as compact text (numbers-only control)."""
    kind = plots.SPECS[sp]["kind"]
    if kind == "line":
        piv = m.pivot(index="x", columns="series", values="y")
        return "epoch," + ",".join(piv.columns) + "\n" + "\n".join(
            f"{int(e)}," + ",".join(f"{v:.4f}" for v in row) for e, row in piv.iterrows())
    if kind == "reliability":
        cols = ["x", "y", "n"] if plots.SPECS[sp].get("show_counts") else ["x", "y"]
        return "mean_predicted,fraction_positive" + (",count" if "n" in cols else "") + "\n" + "\n".join(
            ",".join(f"{getattr(r, c):.4f}" if c != "n" else str(int(r.n)) for c in cols) for r in m.itertuples())
    if kind == "confusion":
        return "true_label,predicted_label,value\n" + "\n".join(f"{int(r.row)},{int(r.col)},{r.value:.4g}" for r in m.itertuples())
    if kind in ("roc", "pr"):
        ln = m[m.mark == "line_vertex"]
        txt = ("false_positive_rate,true_positive_rate" if kind == "roc" else "recall,precision") + "\n" + "\n".join(
            f"{x:.2f},{y:.4f}" for x, y in zip(ln.x, ln.y))
        b = m[m.mark == "baseline"]
        return txt + (f"\npositive_rate,{b.y.iloc[0]:.4f}" if len(b) else "")
    if kind == "dist":
        piv = m.pivot(index="x", columns="series", values="y")
        return "bin_center," + ",".join(f"{c}_density" for c in piv.columns) + "\n" + "\n".join(
            f"{x:.3f}," + ",".join(f"{v:.3f}" for v in row) for x, row in piv.iterrows())
    raise ValueError(kind)


def calibrate(P, src, jobs, manifest):
    log("calibration: healthy pool")
    hc = [healthy_config(T("cal"), "healthy", i, epochs=P["epochs"], source=src) for i in range(P["cal_healthy"])]
    Xh = b0(Parallel(n_jobs=jobs)(delayed(train_run)(c) for c in hc))
    manifest += [dict(role="cal", run_id=f"cal_healthy_{i}", **c) for i, c in enumerate(hc)]
    out = {}
    for f in FAILURES:
        rows = []
        for s in GRID:
            fc = [apply_failure(healthy_config(T("cal"), f, s, i, epochs=P["epochs"], source=src), f, s)
                  for i in range(P["cal_fail"])]
            X = np.vstack([Xh, b0(Parallel(n_jobs=jobs)(delayed(train_run)(c) for c in fc))])
            y = np.r_[np.zeros(len(Xh)), np.ones(len(fc))].astype(int)
            auc, se, fam = bm.cv_auc(X, y, seed=seed_for(T("cal"), src, f, s))
            rows.append(dict(severity=s, b0_auc=auc, se=se, family=fam))
            log(f"  {f:18s} s={s:<6} B0 CV-AUC {auc:.3f} ± {se:.3f}")
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
        log(f"  {f}: chosen " + ", ".join(f"{c['target']}→{c['severity']:.4f}" for c in chosen))
    return out


def _render(state, rid, img_dir):
    for sp in SPECS:
        plots.render(plots.mark_table(state, sp), sp, path=img_dir / f"{rid}_{sp}.png")
    return rid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic")
    ap.add_argument("--name", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--size", choices=["full", "confirm", "smoke"], default="full")
    ap.add_argument("--seed-tag", default="", help='"confirm" for the preregistered confirmatory run')
    ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args()
    global TAG
    TAG = a.seed_tag
    P = SMOKE if (a.smoke or a.size == "smoke") else (CONFIRM if a.size == "confirm" else FULL)
    src = a.source
    out = ROOT / "runs" / (a.name or f"v4_{src}{'_smoke' if a.smoke else ''}")
    img = out / "img"
    img.mkdir(parents=True, exist_ok=True)
    manifest = []

    cal = calibrate(P, src, a.jobs, manifest)
    json.dump(cal, open(out / "calibration.json", "w"), indent=1)

    log("main set: training")
    groups = {("healthy", None): [healthy_config(T("main"), "healthy", i, epochs=P["epochs"], source=src)
                                  for i in range(P["main_healthy"])]}
    for f in FAILURES:
        for j, c in enumerate(cal[f]["chosen"]):
            groups[(f, j)] = [apply_failure(healthy_config(T("main"), f, j, i, epochs=P["epochs"], source=src), f,
                                            c["severity"]) for i in range(P["main_fail"])]
    ids, cfgs, gkey = [], [], []
    for (f, j), cs in groups.items():
        for i, c in enumerate(cs):
            ids.append(f"m_{f}_L{j}_{i}" if j is not None else f"m_healthy_{i}")
            cfgs.append(c)
            gkey.append((f, j))
    states = Parallel(n_jobs=a.jobs, batch_size=8)(delayed(train_run)(c) for c in cfgs)
    manifest += [dict(role="main", run_id=r, level=k[1], **c) for r, c, k in zip(ids, cfgs, gkey)]
    by = {k: [i for i, g in enumerate(gkey) if g == k] for k in groups}
    splits = {k: split_of(len(v), T("main"), src, *map(str, k)) for k, v in by.items()}
    meta = dict(source=src, specs=list(SPECS), core_specs=list(CORE), failures=list(FAILURES), targets=list(TARGETS),
                groups={f"{k[0]}|{k[1]}": {ids[i]: name for name, idx in splits[k].items() for i in np.array(v)[idx]}
                        for k, v in by.items()},
                severities={f: [c["severity"] for c in cal[f]["chosen"]] for f in FAILURES},
                deploy_acc={ids[i]: states[i]["deploy_acc"] for i in range(len(ids))})
    json.dump(meta, open(out / "meta.json", "w"))

    log("rendering plots")
    Parallel(n_jobs=a.jobs, batch_size=16)(delayed(_render)(st, rid, img) for st, rid in zip(states, ids))
    log("features")
    F = {sp: features(states, sp) for sp in SPECS}
    F["B0"] = b0(states)
    np.savez_compressed(out / "features.npz", ids=np.array(ids), **{k: v.astype(np.float32) for k, v in F.items()})

    def stack(keys):
        rows = [np.array(by[k]) for k in keys]
        return np.concatenate(rows), combine([splits[k] for k in keys], [len(r) for r in rows])

    res, scores = dict(detection=[], diagnosis=[]), {}
    for f in FAILURES:
        for j in range(len(TARGETS)):
            idx, sp = stack([("healthy", None), (f, j)])
            y = np.r_[np.zeros(len(by[("healthy", None)])), np.ones(len(by[(f, j)]))].astype(int)
            row = dict(failure=f, level=j, target=TARGETS[j], severity=cal[f]["chosen"][j]["severity"])
            test_ids = [ids[i] for i in idx[sp["test"]]]
            for lv in ["B0"] + list(SPECS):
                r = bm.best_of_k(F[lv][idx], y, sp, seed=seed_for(T("det"), src, f, j, lv), saturation=True)
                scores[f"det|{f}|{j}|{lv}"] = dict(ids=test_ids, score=r["test_scores"].tolist())
                row[lv] = r
            row["B0_state_only"] = {k: v for k, v in row["B0"].items() if k != "test_scores"}
            row["B0"] = best_on_val(row, ["B0"] + list(SPECS))
            for lv in SPECS:
                row[lv] = {k: v for k, v in row[lv].items() if k != "test_scores"}
            res["detection"].append(row)
            log(f"detect {f:18s} L{j}: B0 {row['B0']['test_auc']:.3f} | " +
                " ".join(f"{s} {row[s]['test_auc']:.2f}" for s in SPECS))

    classes = ["healthy"] + list(FAILURES)
    for j in range(len(TARGETS)):
        keys = [("healthy", None)] + [(f, j) for f in FAILURES]
        idx, sp = stack(keys)
        y = np.concatenate([np.full(len(by[k]), c) for c, k in enumerate(keys)])
        test_ids = [ids[i] for i in idx[sp["test"]]]
        row = dict(level=j, target=TARGETS[j])
        for lv in ["B0"] + list(SPECS):
            r = bm.best_of_k_multi(F[lv][idx], y, sp, seed=seed_for(T("diag"), src, j, lv), classes=classes)
            scores[f"diag|{j}|{lv}"] = dict(ids=test_ids, proba=np.round(r["test_proba"], 5).tolist(), prior=r["prior"])
            row[lv] = r
        row["B0_state_only"] = {k: v for k, v in row["B0"].items() if k != "test_proba"}
        row["B0"] = best_on_val(row, ["B0"] + list(SPECS))
        for lv in SPECS:
            row[lv] = {k: v for k, v in row[lv].items() if k != "test_proba"}
        res["diagnosis"].append(row)
        log(f"diagnose L{j}: balanced acc B0 {row['B0']['balanced_acc']:.3f} | " +
            " ".join(f"{s} {row[s]['balanced_acc']:.2f}" for s in SPECS))

    json.dump(res, open(out / "results.json", "w"), indent=1, default=float)
    json.dump(scores, open(out / "item_scores.json", "w"))
    with open(out / "manifest.jsonl", "w") as fh:
        for m in manifest:
            fh.write(json.dumps(m) + "\n")

    # ---- reader item sets (held-out test plots only) ----
    r = rng_for(T("reader-items"), src)

    def pick(key, n):
        te = splits[key]["test"]
        return [ids[by[key][i]] for i in r.choice(te, min(n, len(te)), replace=False)]

    h = pick(("healthy", None), P["reader_healthy"])
    main_runs = h + [x for f in FAILURES for j in P["reader_levels"] for x in pick((f, j), P["reader_fail"])]
    top = len(TARGETS) - 1
    core_runs = h + [x for f in FAILURES for x in pick((f, top), P["reader_fail"])]
    var_runs = h[:P["variant_healthy"]] + [x for f in FAILURES for x in pick((f, top), P["variant_fail"])]
    item = lambda rid, sp, **kw: dict(item_id=f"{rid}_{sp}" + (f"_{kw['variant']}" if "variant" in kw else ""),  # noqa
                                      run_id=rid, task="diagnosis", spec=sp, image=f"img/{rid}_{sp}.png", **kw)
    main_items = [item(x, sp) for sp in SPECS for x in main_runs]
    main_items += [dict(item_id=f"prior_{sp}", task="prior_only", spec=sp, image=None) for sp in SPECS]
    json.dump(main_items, open(out / "reader_items.json", "w"), indent=0)
    json.dump([item(x, sp) for sp in CORE for x in core_runs], open(out / "reader_items_core.json", "w"), indent=0)
    json.dump([item(x, sp, variant=v) for v in ("paraphrase", "describe_first") for sp in VARIANT_SPECS for x in var_runs],
              open(out / "reader_items_variants.json", "w"), indent=0)
    # controls: few-shot (one labelled TRAIN example per state, same plot type) and numbers-only (the drawn values as
    # text, no image) — the fairest chance we can give a reader, and a perception-vs-reasoning test (gap 4.1)
    ex_keys = [("healthy", None)] + [(f, top) for f in FAILURES]
    examples = {sp: [dict(label=k[0], image=f"img/{ids[by[k][int(r.choice(splits[k]['train']))]]}_{sp}.png")
                     for k in ex_keys] for sp in VARIANT_SPECS}
    pos = {rid: i for i, rid in enumerate(ids)}
    ctl = [dict(item(x, sp, variant="fewshot"), examples=examples[sp]) for sp in VARIANT_SPECS for x in var_runs]
    ctl += [dict(item_id=f"{x}_{sp}_numbers", run_id=x, task="diagnosis", spec=sp, variant="numbers", image=None,
                 marks_text=marks_text(plots.mark_table(states[pos[x]], sp), sp)) for sp in VARIANT_SPECS for x in var_runs]
    json.dump(ctl, open(out / "reader_items_controls.json", "w"), indent=0)
    report(out, cal, res)
    log(f"done → {out}  (reader items: main {len(main_items)}, core {len(CORE) * len(core_runs)}, "
        f"variants {2 * len(VARIANT_SPECS) * len(var_runs)})")


def report(out, cal, res):
    L = [f"# v4 study", "", "## Calibrated severities (B0 5-fold CV detection AUC targets 0.65 / 0.80 / 0.95)", ""]
    for f, c in cal.items():
        L.append(f"- {f}: " + ", ".join(f"{x['target']}→s={x['severity']:.4f}{'' if x['reachable'] else ' (NOT reachable)'}"
                                        for x in c["chosen"]))
    L += ["", "## Detection (held-out AUC, healthy vs failure)", "",
          "| failure | level | B0 | " + " | ".join(SPECS) + " |", "|---|---|---|" + "---|" * len(SPECS)]
    for r in res["detection"]:
        L.append(f"| {r['failure']} | {r['target']} | {r['B0']['test_auc']:.3f} ({r['B0']['via']}) | " +
                 " | ".join(f"{r[s]['test_auc']:.3f}" for s in SPECS) + " |")
    L += ["", "## Diagnosis (7-way; balanced accuracy, chance = 1/7 = 0.143)", "",
          "| level | B0 | " + " | ".join(SPECS) + " |", "|---|---|" + "---|" * len(SPECS)]
    for r in res["diagnosis"]:
        L.append(f"| {r['target']} | {r['B0']['balanced_acc']:.3f} | " +
                 " | ".join(f"{r[s]['balanced_acc']:.3f}" for s in SPECS) + " |")
    (out / "report.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
