"""Score the instructor-review checks (points 4 and 5).

  python score_review.py runs/confirm_synthetic runs/confirm_magic   -> runs/REVIEW_RESULTS.md (+ .json)

Point 4, "blind" cells. For every cell blind at the default cutoffs (AUC < 0.70): the certified flag (certify_blind.py),
the kernel two-sample test (two_sample.json; Holm-adjusted over all 54 cells of a run) and the stronger image
benchmark B3 (b3_results.json). Final grade:
  invisible  plot identical to the healthy twin (certified), or no estimator above 0.70 and the two-sample test
             finds no difference (Holm p >= 0.05);
  weak       the plot's distribution does differ, but no estimator (B1 families, B2, B3) reaches AUC 0.70;
  revised    B3 reaches AUC >= 0.70 (95% CI above 0.5): the cell was blind only for the weaker estimators.
Point 5, task difficulty. Binary: AUC of the reader's p_problem, healthy vs each failure, on the same held-out items
as the benchmark (B1*), whose AUC on those items is the ceiling. 2AFC: share of pairs where the reader picks the
failing model (chance 0.5); the benchmark's 2AFC accuracy on the same pairs is P(score failing > score healthy).
"""
import collections
import json
import pathlib
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from score_confirm import Run

READERS = ["Qwen2.5-VL-7B-Instruct", "InternVL3-8B", "llava-onevision-qwen2-7b-ov-hf", "Qwen2.5-VL-32B-Instruct-AWQ",
           "InternVL3-38B-AWQ"]
RNG = np.random.default_rng(0)


def holm(ps):
    order = np.argsort(ps)
    adj, m, run = np.empty(len(ps)), len(ps), 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def boot_ci(fn, units, n=1000):
    units = np.array(units, dtype=object)
    v = [fn(units[RNG.integers(0, len(units), len(units))]) for _ in range(n)]
    v = [x for x in v if x is not None and not np.isnan(x)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def blind_part(R, L):
    sens = json.load(open(R.p.parent / "blind_sensitivity.json"))[R.p.name]["blind"]
    ts_f, b3_f = R.p / "two_sample.json", R.p / "b3_results.json"
    ts = json.load(open(ts_f)) if ts_f.exists() else None
    b3 = json.load(open(b3_f)) if b3_f.exists() else None
    if ts:
        adj = holm(np.array([t["p"] for t in ts]))
        ts = {(t["failure"], t["plot"]): dict(t, p_holm=float(a)) for t, a in zip(ts, adj)}
        L += [f"Two-sample test detects a difference (Holm p < 0.05) in {sum(t['p_holm'] < 0.05 for t in ts.values())}/{len(ts)} cells.", ""]
    b3i = {}
    if b3:
        b3d = {(d["failure"], d["spec"]): d for d in b3["detection"]}
        items = json.load(open(R.p / "b3_item_scores.json"))
        top = R.top
        for (f, sp), d in b3d.items():
            src = items[f"det|{f}|{top}|{sp}"]
            sc, ids = dict(zip(src["ids"], src["score"])), src["ids"]
            y = lambda u: np.array([0 if R.truth[i] == "healthy" else 1 for i in u])  # noqa: E731
            ci = boot_ci(lambda u: roc_auc_score(y(u), [sc[i] for i in u]) if len(set(y(u))) == 2 else None, ids, n=500)
            b3i[(f, sp)] = dict(auc=d["test_auc"], ci=ci)
        star = {}
        for f, sp in b3i:
            sc, y, ids = R.det_scores(f, R.top, sp)
            star[(f, sp)] = roc_auc_score(y, [sc[i] for i in ids])
        b2 = [R.b2det[(f, R.top, sp)]["test_auc"] for f, sp in b3i if (f, R.top, sp) in R.b2det]
        L += [f"B3 (ConvNeXt-Tiny, native resolution) over all {len(b3i)} cells: mean AUC {np.mean([v['auc'] for v in b3i.values()]):.3f} "
              f"(B2 ResNet-18 {np.mean(b2):.3f}; B1* {np.mean(list(star.values())):.3f}). "
              f"B3 exceeds B1* by more than 0.05 in {sum(b3i[k]['auc'] > star[k] + 0.05 for k in b3i)} cells.", ""]
    L += ["| failure | plot | B1* AUC | B3 AUC [95% CI] | two-sample p (Holm) | earlier grade | final |", "|---|---|---|---|---|---|---|"]
    rows = []
    for c in sens:
        k = (c["failure"], c["plot"])
        b = b3i.get(k)
        t = ts.get(k) if ts else None
        if c["grade"] == "certified":
            g = "invisible"
        elif b and b["auc"] >= 0.70 and b["ci"][0] > 0.5:
            g = "revised (B3 detects)"
        elif t is not None and t["p_holm"] >= 0.05 and max(c["best_val"], b["auc"] if b else 0) < 0.70:
            g = "invisible"
        elif b is None or t is None:
            g = "pending"
        else:
            g = "weak"
        rows.append(dict(failure=k[0], plot=k[1], b1=c["auc"], b3=b, two_sample=t, earlier=c["grade"], final=g))
        L.append(f"| {k[0]} | {k[1]} | {c['auc']:.2f} | " + (f"{b['auc']:.2f} [{b['ci'][0]:.2f}, {b['ci'][1]:.2f}]" if b else "–") +
                 " | " + (f"{t['p_holm']:.3g}" if t else "–") + f" | {c['grade']} | **{g}** |")
    n = collections.Counter(r["final"] for r in rows)
    L += ["", "Final: " + ", ".join(f"{v} {k}" for k, v in sorted(n.items())), ""]
    return rows


def task_part(R, L):
    out = {}
    det = {}
    for f in R.meta["failures"]:
        for sp in R.meta["core_specs"]:
            sc, _, ids = R.det_scores(f, R.top, sp)
            det[(f, sp)] = sc
    for task in ("binary", "2afc"):
        res = {}
        for m in READERS:
            fp = R.p / f"responses_{m}_{task}.jsonl"
            if fp.exists():
                res[m] = [json.loads(x) for x in open(fp)]
        if not res:
            L += [f"({task}: no answers yet)", ""]
            continue
        if task == "binary":
            L += ["### Binary task: healthy or problem? (detection AUC per failure x plot cell, 30 cells; chance 0.5)", "",
                  "Reader and benchmark (B1*) are scored on the same held-out items of each cell. "
                  "Cells above chance = 95% bootstrap CI above 0.5.", "",
                  "| reader | answered | mean AUC | cells above chance | best cell | benchmark mean AUC | "
                  + " | ".join(R.meta["failures"]) + " |", "|---|---|---|---|---|---|" + "---|" * len(R.meta["failures"])]
            for m, rows in res.items():
                p = {(r["run_id"], r["spec"]): (r.get("parsed") or {}).get("p_problem") for r in rows}
                p = {k: float(v) for k, v in p.items() if isinstance(v, (int, float)) and 0 <= v <= 1}
                cell = {}
                for f in R.meta["failures"]:
                    for sp in R.meta["core_specs"]:
                        ks = [k for k in p if k[1] == sp and R.truth[k[0]] in ("healthy", f) and k[0] in det[(f, sp)]]
                        y = np.array([R.truth[k[0]] == f for k in ks], int)
                        if len(set(y)) < 2:
                            continue
                        a = roc_auc_score(y, [p[k] for k in ks])
                        bench = roc_auc_score(y, [det[(f, sp)][k[0]] for k in ks])
                        ci = boot_ci(lambda u: roc_auc_score([R.truth[k[0]] == f for k in u], [p[tuple(k)] for k in u])
                                     if len({R.truth[k[0]] == f for k in u}) == 2 else None, [tuple(k) for k in ks], n=300)
                        cell[(f, sp)] = dict(auc=float(a), ci=ci, bench=float(bench))
                best = max(cell, key=lambda k: cell[k]["auc"])
                above = sum(v["ci"][0] > 0.5 for v in cell.values())
                byf = {f: np.mean([v["auc"] for k, v in cell.items() if k[0] == f]) for f in R.meta["failures"]}
                out[f"binary|{m}"] = dict(cells={f"{k[0]}|{k[1]}": v for k, v in cell.items()}, answered=len(p), items=len(rows),
                                          share_problem=float(np.mean([v >= 0.5 for v in p.values()])))
                L.append(f"| {m} | {len(p)}/{len(rows)} | {np.mean([v['auc'] for v in cell.values()]):.2f} | {above}/{len(cell)} | "
                         f"{best[0]} on {best[1]} {cell[best]['auc']:.2f} | {np.mean([v['bench'] for v in cell.values()]):.2f} | "
                         + " | ".join(f"{byf[f]:.2f}" for f in R.meta["failures"]) + " |")
            L.append("")
        else:
            items = {it["item_id"]: it for it in json.load(open(R.p / "reader_items_2afc.json"))}
            bench_hit = [det[(it["failure"], it["spec"])][it["failing_run"]] > det[(it["failure"], it["spec"])][it["healthy_run"]]
                         for it in items.values() if it["failing_run"] in det[(it["failure"], it["spec"])] and it["healthy_run"] in det[(it["failure"], it["spec"])]]
            L += ["### Two-alternative forced choice: which of two plots is the failing model? (chance 0.50)", "",
                  f"Benchmark (B1*) on the same pairs: **{np.mean(bench_hit):.2f}** ({len(bench_hit)} pairs).", "",
                  "| reader | answered | accuracy [95% CI] | picks A | best plot | per failure (accuracy) |", "|---|---|---|---|---|---|"]
            for m, rows in res.items():
                ans = {r["item_id"]: str((r.get("parsed") or {}).get("answer", "")).strip().upper()[:1] for r in rows}
                ans = {k: v for k, v in ans.items() if v in ("A", "B")}
                hit = {k: v == items[k]["failing_side"] for k, v in ans.items()}
                ks = list(hit)
                acc = float(np.mean(list(hit.values()))) if hit else np.nan
                ci = boot_ci(lambda u: np.mean([hit[k] for k in u]), ks) if ks else (np.nan, np.nan)
                bysp = {sp: np.mean([hit[k] for k in ks if items[k]["spec"] == sp]) for sp in R.meta["core_specs"]}
                byf = {f: np.mean([hit[k] for k in ks if items[k]["failure"] == f]) for f in R.meta["failures"]}
                bsp = max(bysp, key=bysp.get)
                out[f"2afc|{m}"] = dict(acc=acc, ci=ci, picks_A=float(np.mean([v == "A" for v in ans.values()])), by_plot=bysp, by_failure=byf, answered=len(ans))
                L.append(f"| {m} | {len(ans)}/{len(rows)} | {acc:.2f} [{ci[0]:.2f}, {ci[1]:.2f}] | {np.mean([v == 'A' for v in ans.values()]):.0%} | "
                         f"{bsp} {bysp[bsp]:.2f} | " + ", ".join(f"{f} {v:.2f}" for f, v in byf.items()) + " |")
            out["2afc|benchmark"] = float(np.mean(bench_hit))
            L.append("")
    return out


def main(paths):
    L = ["# Instructor-review checks (points 4 and 5)", ""]
    js = {}
    for p in paths:
        R = Run(p)
        L += [f"## {R.p.name}", "", "### Point 4: what the blind cells are", ""]
        js[R.p.name] = dict(blind=blind_part(R, L))
        if (R.p / "reader_items_2afc.json").exists():
            L += ["### Point 5: is the 7-way task too hard?", ""]
            js[R.p.name]["tasks"] = task_part(R, L)
    out = pathlib.Path(paths[0]).parent
    (out / "REVIEW_RESULTS.md").write_text("\n".join(L) + "\n")
    json.dump(js, open(out / "review_results.json", "w"), indent=1, default=str)
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1:] or ["runs/confirm_synthetic", "runs/confirm_magic"])
