"""Plot specs -> mark tables (B1 input) -> rendered images (B2 / reader input).

Every image is drawn *from its mark table*, so B1 sees exactly what is drawn (gap 2.4).
Each spec is tagged with the level its design choice acts on (gap 2.6):
  encoding  = changes which numbers are drawn (e.g. loss vs accuracy curves)
  rendering = changes only how the same numbers are drawn (colour, size, resolution)
Titles are neutral and never name a failure (gap 4.3).
"""
import io

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

COLORS = {"train": "#1f77b4", "validation": "#ff7f0e"}

SPECS = {
    "lc_loss": dict(kind="line", series={"train": "train_loss", "validation": "val_loss"},
                    xlabel="epoch", ylabel="loss", level="encoding"),
    "lc_acc": dict(kind="line", series={"train": "train_acc", "validation": "val_acc"},
                   xlabel="epoch", ylabel="accuracy", level="encoding"),
    "rel": dict(kind="reliability", xlabel="predicted probability", ylabel="observed fraction positive",
                level="encoding", bins=10),
    "rel_n": dict(kind="reliability", xlabel="predicted probability", ylabel="observed fraction positive",
                  level="encoding", bins=10, show_counts=True),  # design variant: point area = bin size
    "cm": dict(kind="confusion", xlabel="predicted label", ylabel="true label", level="encoding"),
    "cm_norm": dict(kind="confusion", xlabel="predicted label", ylabel="true label", level="encoding",
                    normalize=True),  # design variant: row-normalised (rates instead of counts)
    "roc": dict(kind="roc", xlabel="false positive rate", ylabel="true positive rate", level="encoding"),
    "pr": dict(kind="pr", xlabel="recall", ylabel="precision", level="encoding"),
    "dist": dict(kind="dist", xlabel="predicted probability", ylabel="density", level="encoding", bins=20),
}
GRID = np.linspace(0, 1, 51)


def _roc_pr(p, y):
    order = np.argsort(-p, kind="stable")
    ys = y[order].astype(float)
    tp, fp = np.r_[0, np.cumsum(ys)], np.r_[0, np.cumsum(1 - ys)]
    P, N = max(ys.sum(), 1), max(len(ys) - ys.sum(), 1)
    tpr, fpr = tp / P, fp / N
    roc = np.array([tpr[fpr <= f].max() for f in GRID])                      # ROC on a fixed FPR grid
    prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 1.0)
    pr = np.array([prec[tpr >= r].max() if (tpr >= r).any() else 0.0 for r in GRID])  # interpolated precision
    return roc, pr


def mark_table(state, spec_name):
    """One row per drawn mark with its channel values (B1 sees exactly what is drawn)."""
    spec = SPECS[spec_name]
    kind = spec["kind"]
    if kind == "reliability":
        p, y, b = state["test_p"], state["test_y"], spec["bins"]
        idx = np.minimum((p * b).astype(int), b - 1)
        rows = [dict(mark="point", series="model", bin=k, x=float(p[idx == k].mean()), y=float(y[idx == k].mean()),
                     n=int((idx == k).sum()), color="#1f77b4", size=4.0) for k in range(b) if (idx == k).any()]
        return pd.DataFrame(rows)
    if kind == "confusion":
        p, y = state["test_p"], state["test_y"]
        pred = (p >= 0.5).astype(int)
        cnt = {(t, q): int(((y == t) & (pred == q)).sum()) for t in (0, 1) for q in (0, 1)}
        if spec.get("normalize"):  # only the row rates are drawn
            val = {k: v / max(cnt[(k[0], 0)] + cnt[(k[0], 1)], 1) for k, v in cnt.items()}
        else:
            val = cnt
        return pd.DataFrame([dict(mark="cell", series="rate" if spec.get("normalize") else "count", row=t, col=q,
                                  x=q, y=t, value=val[(t, q)], color="value", size=1.0)
                             for t in (0, 1) for q in (0, 1)])
    if kind == "dist":  # score distributions, validation vs test (the plot made for shift)
        edges = np.linspace(0, 1, spec["bins"] + 1)
        rows = []
        for label, key, color in (("validation", "val_p", "#ff7f0e"), ("test", "test_p", "#2ca02c")):
            h, _ = np.histogram(state[key], bins=edges, density=True)
            rows += [dict(mark="bar", series=label, bin=k, x=float((edges[k] + edges[k + 1]) / 2), y=float(h[k]),
                          color=color, size=1.0) for k in range(spec["bins"])]
        return pd.DataFrame(rows)
    if kind in ("roc", "pr"):
        roc, pr = _roc_pr(state["test_p"], state["test_y"])
        df = pd.DataFrame({"mark": "line_vertex", "series": "model", "x": GRID, "y": roc if kind == "roc" else pr,
                           "color": "#1f77b4", "size": 1.5})
        if kind == "pr":  # positive-rate baseline is data-dependent, so it is a mark
            base = float(state["test_y"].mean())
            df = pd.concat([df, pd.DataFrame({"mark": "baseline", "series": "positive rate", "x": [0.0, 1.0],
                                              "y": [base, base], "color": "gray", "size": 1.0})], ignore_index=True)
        return df
    rows = []
    for label, key in spec["series"].items():
        ys = np.asarray(state[key])
        rows.append(pd.DataFrame({"mark": "line_vertex", "series": label, "x": np.arange(1, len(ys) + 1),
                                  "y": ys, "color": COLORS[label], "size": 1.5}))
    return pd.concat(rows, ignore_index=True)


def marks_vector(marks, spec_name=None):
    """Fixed-length vector of a mark table (same layout for every plot of a spec)."""
    kind = SPECS[spec_name]["kind"] if spec_name else "line"
    if kind == "reliability":  # bin counts are NOT drawn in the default design, so they are not in B1
        show_n = SPECS[spec_name].get("show_counts", False)
        v = np.full((SPECS[spec_name]["bins"], 3 if show_n else 2), -1.0)
        for r in marks.itertuples():
            v[r.bin] = (r.x, r.y, r.n) if show_n else (r.x, r.y)
        return v.ravel()
    if kind == "confusion":
        return marks.sort_values(["row", "col"]).value.to_numpy(dtype=float)
    if kind == "dist":
        return marks.sort_values(["series", "bin"]).y.to_numpy()
    if kind in ("roc", "pr"):
        return np.r_[marks[marks.mark == "line_vertex"].y.to_numpy(), marks[marks.mark == "baseline"].y.to_numpy()[:1]]
    return marks.sort_values(["series", "x"]).y.to_numpy()


def _draw(ax, marks, spec, ticks=True, ylim=None):
    kind = spec["kind"]
    if kind == "reliability":
        ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1)
        if spec.get("show_counts"):
            ax.plot(marks.x, marks.y, color="#1f77b4", lw=1.0)
            ax.scatter(marks.x, marks.y, s=4 + 300 * marks.n / max(marks.n.sum(), 1), color="#1f77b4", zorder=3)
        else:
            ax.plot(marks.x, marks.y, marker="o", ms=4, color="#1f77b4", lw=1.2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    elif kind == "confusion":
        m = np.zeros((2, 2))
        for r in marks.itertuples():
            m[r.row, r.col] = r.value
        ax.imshow(m, cmap="Blues", vmin=0, vmax=1 if spec.get("normalize") else None)
        for r in marks.itertuples():
            txt = f"{r.value:.0%}" if spec.get("normalize") else str(int(r.value))
            ax.text(r.col, r.row, txt, ha="center", va="center", fontsize=9,
                    color="white" if r.value > (0.5 if spec.get("normalize") else m.max() / 2) else "black")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
    elif kind == "dist":
        edges = np.linspace(0, 1, SPECS["dist"]["bins"] + 1)
        for label, g in marks.groupby("series", sort=False):
            ax.stairs(g.sort_values("bin").y.to_numpy(), edges, color=g.color.iloc[0], lw=1.4, label=label)
        ax.set_xlim(0, 1)
        if ylim is not None:
            ax.set_ylim(*ylim)
    elif kind in ("roc", "pr"):
        line = marks[marks.mark == "line_vertex"]
        ax.plot(line.x, line.y, color="#1f77b4", lw=1.5)
        if kind == "roc":
            ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1)
        else:
            b = marks[marks.mark == "baseline"]
            ax.plot(b.x, b.y, ls="--", color="gray", lw=1, label="positive rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
    if kind != "line":
        if ticks:
            ax.set_xlabel(spec["xlabel"], fontsize=8)
            ax.set_ylabel(spec["ylabel"], fontsize=8)
            ax.tick_params(labelsize=7)
            if kind in ("pr", "dist"):
                ax.legend(fontsize=7, frameon=False)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        return
    for label, g in marks.groupby("series", sort=False):
        ax.plot(g.x, g.y, color=g.color.iloc[0], lw=g["size"].iloc[0], label=label)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if ticks:
        ax.set_xlabel(spec["xlabel"], fontsize=8)
        ax.set_ylabel(spec["ylabel"], fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, frameon=False)
    else:
        ax.set_xticks([])
        ax.set_yticks([])


def _to_array(fig, dpi):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


def render(marks, spec_name, path=None, dpi=100, size=(3.2, 2.4)):
    """Single diagnostic plot as a reader would see it. Returns an RGB array; saves a PNG if `path`."""
    fig, ax = plt.subplots(figsize=size)
    ax.set_title("Plot", fontsize=9)
    _draw(ax, marks, SPECS[spec_name])
    fig.tight_layout()
    img = _to_array(fig, dpi)
    if path is not None:
        Image.fromarray(img).save(path)
    return img


def render_lineup(panel_marks, spec_name, path, dpi=100):
    """20-panel lineup on shared axes, no tick labels (lineup convention)."""
    ys = np.concatenate([m.y.to_numpy() for m in panel_marks])
    pad = 0.04 * (ys.max() - ys.min())
    ylim = (ys.min() - pad, ys.max() + pad)
    fig, axs = plt.subplots(4, 5, figsize=(12, 9.6))
    for i, (ax, m) in enumerate(zip(axs.flat, panel_marks)):
        _draw(ax, m, SPECS[spec_name], ticks=False, ylim=ylim)
        ax.set_title(str(i + 1), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def downsample(img, width=128):
    """Reader-resolution image -> small benchmark input (B2 sees a downsampled copy of the reader's image)."""
    h, w = img.shape[:2]
    return np.asarray(Image.fromarray(img).resize((width, int(round(h * width / w))), Image.BILINEAR))
