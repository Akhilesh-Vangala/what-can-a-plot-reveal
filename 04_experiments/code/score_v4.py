"""Score a v4 run: evidential map, benchmarks, and AI readers (incl. formal over-claiming vs the benchmark posterior).

  python score_v4.py runs/v4_synthetic      -> evidential_map.md/.png, readers.md, readers_summary.json

Benchmarks: B1* = validation-best of {mark-table model (B1), image model (B2)} per cell; B0* = validation-best of
{state model, every B1*}. Diagnosis metric: balanced accuracy (mean per-class recall; chance = 1/#classes).
Evidential map labels (provisional thresholds, to be preregistered): detected if AUC >= 0.70; diagnosed if the
failure's recall in the multi-way benchmark >= 0.50.
Readers (diagnosis items, argmax over healthy + 6 failures):
  balanced accuracy; stereotype index = share of the single most common answer on HEALTHY plots of each type;
  over-claiming index = reader prob. on its top cause − benchmark prob. on that cause (benchmark posterior from the
  validation-best B1* model of that plot, prior-corrected to equal class priors, same held-out item);
  confident-where-ambiguous = share of items with reader top prob ≥ 0.7 while benchmark top prob < 0.5;
  hedging AUC = AUC of the reader's "cannot tell" against benchmark entropy (0.5 = hedging unrelated to ambiguity).
"""
import collections
import json
import pathlib
import sys

import matplotlib
import numpy as np
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DETECT_T, DIAG_T = 0.70, 0.50


def bal(conf):
    c = np.array(conf, dtype=float)
    return float(np.mean(c.diagonal() / np.maximum(c.sum(1), 1)))


def main(run):
    run = pathlib.Path(run)
    meta = json.load(open(run / "meta.json"))
    res = json.load(open(run / "results.json"))
    b2 = json.load(open(run / "b2_results.json")) if (run / "b2_results.json").exists() else None
    item1 = json.load(open(run / "item_scores.json"))
    item2 = json.load(open(run / "b2_item_scores.json")) if (run / "b2_item_scores.json").exists() else {}
    specs, fails = meta["specs"], meta["failures"]
    classes = ["healthy"] + fails
    top = len(meta["targets"]) - 1
    b2det = {(r["failure"], r["level"], r["spec"]): r for r in b2["detection"]} if b2 else {}
    b2dia = {(r["level"], r["spec"]): r for r in b2["diagnosis"]} if b2 else {}

    def star_det(row, f, j, sp):
        a, b = row[sp], b2det.get((f, j, sp))
        return (a, "B1") if b is None or a["val_auc"] >= b["val_auc"] else (b, "B2")

    def star_dia(row, j, sp):
        a, b = row[sp], b2dia.get((j, sp))
        return (a, "B1") if b is None or a["val_auc"] >= b["val_auc"] else (b, "B2")

    # ---------- evidential map ----------
    L = [f"# Evidential map — {meta['source']} data", "",
         f"B1* = best on validation of mark-table / image model. Detected: AUC ≥ {DETECT_T}; diagnosed: recall ≥ {DIAG_T} "
         f"(chance {1 / len(classes):.2f}).", ""]
    det_rows = {(r["failure"], r["level"]): r for r in res["detection"]}
    dia_rows = {r["level"]: r for r in res["diagnosis"]}
    grid = {}
    for j in range(len(meta["targets"])):
        L += [f"## Level {j} (B0 target {meta['targets'][j]})", "", "| failure | B0* | " + " | ".join(specs) + " |",
              "|---|---|" + "---|" * len(specs)]
        drow = dia_rows[j]
        for i, f in enumerate(fails):
            row = det_rows[(f, j)]
            cells = []
            for sp in specs:
                d, _ = star_det(row, f, j, sp)
                dg, _ = star_dia(drow, j, sp)
                rec = np.array(dg["confusion"], dtype=float)
                recall = rec[i + 1, i + 1] / max(rec[i + 1].sum(), 1)
                lab = "ID" if d["test_auc"] >= DETECT_T and recall >= DIAG_T else (
                    "detect-only" if d["test_auc"] >= DETECT_T else "–")
                grid[(j, f, sp)] = (d["test_auc"], recall, lab)
                cells.append(f"{d['test_auc']:.2f} / {recall:.2f} {lab}")
            L.append(f"| {f} | {row['B0']['test_auc']:.2f} | " + " | ".join(cells) + " |")
        L += ["", "Balanced diagnosis accuracy (B1*): " +
              ", ".join(f"{sp} {star_dia(drow, j, sp)[0].get('balanced_acc', bal(star_dia(drow, j, sp)[0]['confusion'])):.2f}"
                        for sp in specs) + f"; B0 {drow['B0']['balanced_acc']:.2f}", ""]
    L += ["## Rendering loss at the strongest level (B1 mark-table AUC → B2 image AUC; only differences > 0.05)", ""]
    for f in fails:
        row = det_rows[(f, top)]
        for sp in specs:
            b = b2det.get((f, top, sp))
            if b and row[sp]["test_auc"] - b["test_auc"] > 0.05:
                L.append(f"- {f} on {sp}: {row[sp]['test_auc']:.2f} → {b['test_auc']:.2f}")
    (run / "evidential_map.md").write_text("\n".join(L) + "\n")

    fig, axs = plt.subplots(1, 2, figsize=(1.2 * len(specs) + 4, 0.55 * len(fails) + 1.8))
    for ax, k, title, lo in ((axs[0], 0, "detection AUC", 0.5), (axs[1], 1, "diagnosis recall", 0)):
        m = np.array([[grid[(top, f, sp)][k] for sp in specs] for f in fails])
        im = ax.imshow(m, vmin=lo, vmax=1, cmap="viridis")
        for a in range(len(fails)):
            for b in range(len(specs)):
                ax.text(b, a, f"{m[a, b]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if m[a, b] < (lo + 1) / 2 else "black")
        ax.set_xticks(range(len(specs)), specs, rotation=35, fontsize=8)
        ax.set_yticks(range(len(fails)), fails, fontsize=8)
        ax.set_title(f"{title} (strongest level, B1*)", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(run / "evidential_map.png", dpi=150)
    plt.close(fig)

    # ---------- benchmark posteriors per item (for over-claiming) ----------
    level_of, truth = {}, {}
    for key, members in meta["groups"].items():
        f, lv = key.split("|")
        for rid in members:
            truth[rid] = f
            level_of[rid] = None if lv == "None" else int(lv)
    post = {}  # (run_id, spec, level) -> prior-corrected benchmark posterior over classes

    for j in range(len(meta["targets"])):
        for sp in specs:
            _, which = star_dia(dia_rows[j], j, sp)
            src = item1.get(f"diag|{j}|{sp}") if which == "B1" else item2.get(f"diag|{j}|{sp}")
            if not src:
                continue
            P = np.array(src["proba"]) / np.maximum(np.array(src["prior"]), 1e-9)
            P = P / P.sum(1, keepdims=True)
            for rid, p in zip(src["ids"], P):
                post[(rid, sp, j)] = p

    # ---------- readers ----------
    R = [f"# AI readers — {meta['source']} data", "",
         "| reader | items | balanced acc | per-plot balanced acc | stereotype index (healthy plots) | over-claim index | "
         "confident-where-ambiguous | cannot-tell | hedging AUC |", "|---|---|---|---|---|---|---|---|---|"]
    summary = {}
    for fp in sorted(run.glob("responses_*.jsonl")):
        rows = [json.loads(x) for x in open(fp)]
        model = rows[0]["model"].split("/")[-1] + fp.stem.replace("responses_" + rows[0]["model"].split("/")[-1], "")
        recs = collections.defaultdict(list)
        per_plot = collections.defaultdict(lambda: collections.defaultdict(list))
        healthy_ans = collections.defaultdict(collections.Counter)
        over, conf_amb, ct, ent = [], [], [], []
        for r in rows:
            if r["task"] != "diagnosis" or not isinstance(r.get("parsed"), dict):
                continue
            try:
                v = np.array([float(r["parsed"].get(c, 0) or 0) for c in classes])
            except (TypeError, ValueError):
                continue
            if v.sum() <= 0:
                continue
            v = v / v.sum()
            t, sp = truth[r["run_id"]], r["spec"]
            ans = classes[int(v.argmax())]
            recs[t].append(ans == t)
            per_plot[sp][t].append(ans == t)
            if t == "healthy":
                healthy_ans[sp][ans] += 1
            lv = level_of[r["run_id"]]
            lvs = [lv] if lv is not None else list(range(len(meta["targets"])))
            for j in lvs:
                p = post.get((r["run_id"], sp, j))
                if p is not None:
                    over.append(v.max() - p[v.argmax()])
                    conf_amb.append(v.max() >= 0.7 and p.max() < 0.5)
                    ct.append(bool(r["parsed"].get("cannot_tell")))
                    ent.append(float(-(p * np.log(p + 1e-12)).sum()))
                    break
        ba = float(np.mean([np.mean(recs[c]) for c in classes if recs[c]]))
        pp = {sp: float(np.mean([np.mean(d[c]) for c in classes if d[c]])) for sp, d in per_plot.items()}
        st = {sp: max(c.values()) / sum(c.values()) for sp, c in healthy_ans.items()}
        hed = roc_auc_score(ct, ent) if len(set(ct)) == 2 else float("nan")
        summary[model] = dict(balanced_acc=ba, per_plot=pp, stereotype=st, over_claim=float(np.mean(over)) if over else None,
                              confident_where_ambiguous=float(np.mean(conf_amb)) if conf_amb else None,
                              cannot_tell=float(np.mean(ct)) if ct else None, hedging_auc=hed,
                              healthy_top_answer={sp: c.most_common(1)[0][0] for sp, c in healthy_ans.items()})
        R.append(f"| {model} | {sum(len(x) for x in recs.values())} | {ba:.2f} | " +
                 ", ".join(f"{sp} {a:.2f}" for sp, a in sorted(pp.items())) + " | " +
                 ", ".join(f"{sp} {s:.0%}→{summary[model]['healthy_top_answer'][sp][:6]}" for sp, s in sorted(st.items())) +
                 f" | {np.mean(over) if over else float('nan'):.2f} | {np.mean(conf_amb) if conf_amb else float('nan'):.2f} | "
                 f"{np.mean(ct) if ct else float('nan'):.2f} | {hed:.2f} |")
    (run / "readers.md").write_text("\n".join(R) + "\n")
    json.dump(summary, open(run / "readers_summary.json", "w"), indent=1)
    print("\n".join(L[:3]))
    print("\n".join(R))


if __name__ == "__main__":
    main(sys.argv[1])
