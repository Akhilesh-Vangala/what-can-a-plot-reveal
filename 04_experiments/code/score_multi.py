"""Score the multi-failure run: evidential map (detection vs diagnosis per failure × plot) and AI readers vs B2.

  python score_multi.py runs/multi_v3        -> evidential_map.md, readers_multi.md, figures

Evidential map, per failure × plot (benchmark = best of B1 mark table / B2 image available):
  detection   = held-out AUC healthy vs failure (strongest level)
  diagnosis   = recall of that failure in the 5-way benchmark confusion matrix (chance 0.20)
  label       = "identifiable" (detected AND diagnosed), "detect-only" (detected, not diagnosed), "neither"
Thresholds are PROVISIONAL (to be fixed in the preregistration before the full study): detect AUC >= 0.70,
diagnosis recall >= 0.50.
Readers: 5-way diagnosis from their probabilities (argmax over healthy + 4 planted failures; mass on the 2 unplanted
causes reported separately), accuracy vs B2 accuracy on the same plot/level, confidence, Brier, cannot-tell use.
"""
import collections
import json
import pathlib
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DETECT_T, DIAG_T = 0.70, 0.50
ALL = ["healthy", "overfitting", "label_noise", "class_imbalance", "data_leakage", "miscalibration", "distribution_shift"]


def recalls(conf):
    c = np.array(conf, dtype=float)
    return c.diagonal() / np.maximum(c.sum(1), 1)


def main(run):
    run = pathlib.Path(run)
    meta = json.load(open(run / "meta.json"))
    res = json.load(open(run / "results.json"))
    b2 = json.load(open(run / "b2_multi_results.json")) if (run / "b2_multi_results.json").exists() else None
    specs, fails, classes = meta["specs"], meta["failures"], ["healthy"] + meta["failures"]
    top = len(meta["targets"]) - 1

    det = {(r["failure"], r["level"], sp): r[sp]["test_auc"] for r in res["detection"] for sp in specs}
    det.update({(r["failure"], r["level"], "B0"): r["B0"]["test_auc"] for r in res["detection"]})
    diag = {(r["level"], sp): r[sp] for r in res["diagnosis"] for sp in specs + ["B0"]}
    if b2:
        b2det = {(r["failure"], r["level"], r["spec"]): r["test_auc"] for r in b2["detection"]}
        b2diag = {(r["level"], r["spec"]): r for r in b2["diagnosis"]}

    L = ["# Evidential map (multi-failure run)", "",
         f"Provisional thresholds: detected if AUC ≥ {DETECT_T}; diagnosed if 5-way recall ≥ {DIAG_T} (chance 0.20).", ""]
    grid = np.zeros((len(fails), len(specs)), dtype=object)
    for lv in range(len(meta["targets"])):
        L += [f"## Level {lv} (B0 detection target {meta['targets'][lv]})", "",
              "| failure | " + " | ".join(specs) + " |", "|---|" + "---|" * len(specs)]
        for i, f in enumerate(fails):
            cells = []
            for k, sp in enumerate(specs):
                d1 = det[(f, lv, sp)]
                d2 = b2det.get((f, lv, sp)) if b2 else None
                dd = max(d1, d2) if d2 is not None else d1
                r1 = recalls(diag[(lv, sp)]["confusion"])[i + 1]
                r2 = recalls(b2diag[(lv, sp)]["confusion"])[i + 1] if b2 else None
                rr = max(r1, r2) if r2 is not None else r1
                lab = "identifiable" if dd >= DETECT_T and rr >= DIAG_T else ("detect-only" if dd >= DETECT_T else "neither")
                if lv == top:
                    grid[i, k] = (dd, rr, lab)
                cells.append(f"det {dd:.2f} · diag {rr:.2f} · **{lab}**")
            L.append(f"| {f} | " + " | ".join(cells) + " |")
        L.append("")
        L.append("5-way diagnosis accuracy (B1 / B2): " + ", ".join(
            f"{sp} {diag[(lv, sp)]['test_acc']:.2f}" + (f"/{b2diag[(lv, sp)]['test_acc']:.2f}" if b2 else "") for sp in specs)
            + f"; B0 {diag[(lv, 'B0')]['test_acc']:.2f}")
        L.append("")
    L += ["## Most-confused pairs at the strongest level (benchmark B1 confusion; share of the row)", ""]
    for sp in specs:
        c = np.array(diag[(top, sp)]["confusion"], dtype=float)
        c = c / np.maximum(c.sum(1, keepdims=True), 1)
        np.fill_diagonal(c, 0)
        a, b = np.unravel_index(c.argmax(), c.shape)
        L.append(f"- {sp}: {classes[a]} → predicted {classes[b]} ({c[a, b]:.0%})")
    (run / "evidential_map.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))

    # heatmap figure (strongest level)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, j, title in ((axs[0], 0, "detection AUC"), (axs[1], 1, "diagnosis recall (5-way)")):
        m = np.array([[grid[i, k][j] for k in range(len(specs))] for i in range(len(fails))], dtype=float)
        im = ax.imshow(m, vmin=0 if j else 0.5, vmax=1, cmap="viridis")
        for i in range(len(fails)):
            for k in range(len(specs)):
                ax.text(k, i, f"{m[i, k]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if m[i, k] < (0.6 if j else 0.75) else "black")
        ax.set_xticks(range(len(specs)), specs, rotation=30)
        ax.set_yticks(range(len(fails)), fails)
        ax.set_title(f"{title}, strongest level")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(run / "evidential_map.png", dpi=140)
    plt.close(fig)

    # ---- readers ----
    groups = meta["groups"]
    truth = {}
    for key, members in groups.items():
        f, lv = key.split("|")
        for rid in members:
            truth[rid] = (f, None if lv == "None" else int(lv))
    R = ["# AI readers on the multi-failure items", "",
         "| reader | spec | level | n | acc (5-way) | B2 acc | B1 acc | P(true) | mean top prob | P(unplanted causes) | cannot_tell | Brier |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    summary = collections.defaultdict(dict)
    for fpath in sorted(run.glob("responses_*.jsonl")):
        rows = [json.loads(x) for x in open(fpath)]
        model = rows[0]["model"].split("/")[-1]
        for sp in specs:
            for lv in sorted({t[1] for t in truth.values() if t[1] is not None}):
                items = [r for r in rows if r["task"] == "diagnosis" and r["spec"] == sp and isinstance(r["parsed"], dict)
                         and (truth[r["run_id"]][1] in (None, lv))]
                if not items:
                    continue
                P, y, ct, other = [], [], [], []
                for r in items:
                    try:
                        v = np.array([float(r["parsed"].get(c, 0) or 0) for c in ALL])
                    except (TypeError, ValueError):
                        continue
                    if v.sum() <= 0:
                        continue
                    v = v / v.sum()
                    P.append(v[[ALL.index(c) for c in classes]])
                    other.append(v[[ALL.index(c) for c in ALL if c not in classes]].sum())
                    y.append(classes.index(truth[r["run_id"]][0]))
                    ct.append(bool(r["parsed"].get("cannot_tell")))
                P, y = np.array(P), np.array(y)
                acc = float((P.argmax(1) == y).mean())
                onehot = np.eye(len(classes))[y]
                brier = float(((P - onehot) ** 2).sum(1).mean())
                b2a = b2diag[(lv, sp)]["test_acc"] if b2 else float("nan")
                R.append(f"| {model} | {sp} | {lv} | {len(y)} | {acc:.2f} | {b2a:.2f} | {diag[(lv, sp)]['test_acc']:.2f} | "
                         f"{P[np.arange(len(y)), y].mean():.2f} | {P.max(1).mean():.2f} | {np.mean(other):.2f} | "
                         f"{np.mean(ct):.2f} | {brier:.2f} |")
                summary[model][f"{sp}_L{lv}"] = acc
    (run / "readers_multi.md").write_text("\n".join(R) + "\n")
    print("\n".join(R))


if __name__ == "__main__":
    main(sys.argv[1])
