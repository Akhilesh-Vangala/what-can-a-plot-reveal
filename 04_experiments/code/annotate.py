"""Gap 5 — plot designs for machine readers: the same data, with the values readers misjudge printed on the plot.

Readers read printed numbers almost perfectly (confusion-matrix cells, r = 1.00) but misjudge positions and areas
(reliability points, ROC at given FPRs, histogram mass). The annotated variants print exactly those values; nothing
else changes. Mark tables are rebuilt exactly from the stored mark vectors (plots.marks_vector).

  rel_ann   reliability diagram, each point labelled with its observed fraction, x positions given as text
  roc_ann   ROC curve with markers and printed TPR at FPR 0.05, 0.1, 0.2, 0.5
  dist_ann  validation/test score histograms with a printed table of mass below 0.1, above 0.5, above 0.9
"""
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import rtr  # noqa: E402
from plotreveal import plots  # noqa: E402

BASE = {"rel_ann": "rel", "roc_ann": "roc", "dist_ann": "dist"}


def marks_from_vector(v, spec):
    v = np.asarray(v, float)
    if spec == "rel":
        pts = v.reshape(-1, 2)
        return pd.DataFrame([dict(mark="point", series="model", bin=i, x=x, y=y, n=1, color="#1f77b4", size=4.0)
                             for i, (x, y) in enumerate(pts) if x >= 0])
    if spec == "roc":
        return pd.DataFrame({"mark": "line_vertex", "series": "model", "x": plots.GRID, "y": v[:51], "color": "#1f77b4", "size": 1.5})
    if spec == "dist":
        edges = np.linspace(0, 1, 21)
        rows = []
        for k, (label, color) in enumerate((("test", "#2ca02c"), ("validation", "#ff7f0e"))):
            rows += [dict(mark="bar", series=label, bin=b, x=float((edges[b] + edges[b + 1]) / 2), y=float(v[k * 20 + b]),
                          color=color, size=1.0) for b in range(20)]
        return pd.DataFrame(rows)
    raise ValueError(spec)


def render(v, spec_ann, path, dpi=100, size=(3.2, 2.4)):
    spec = BASE[spec_ann]
    m = marks_from_vector(v, spec)
    truth = rtr.truth_from_vector(v, spec)
    fig, ax = plt.subplots(figsize=size if spec != "dist" else (3.6, 2.6))
    ax.set_title("Plot", fontsize=9)
    plots._draw(ax, m, plots.SPECS[spec])
    if spec == "rel":
        for r in m.itertuples():
            ax.annotate(f"{r.y:.2f}", (r.x, r.y), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=5.5)
        ax.set_xticks(np.round(m.x.to_numpy(), 2))
        ax.tick_params(axis="x", labelsize=4.5, rotation=90)
    elif spec == "roc":
        for f in rtr.FPR_AT:
            ax.plot([f], [truth[f"tpr_at_{f}"]], "o", ms=3, color="#d62728")
        txt = "FPR   TPR\n" + "\n".join(f"{f:<5} {truth[f'tpr_at_{f}']:.2f}" for f in rtr.FPR_AT)
        ax.text(0.97, 0.05, txt, transform=ax.transAxes, ha="right", va="bottom", fontsize=5.5, family="monospace",
                bbox=dict(boxstyle="round", fc="white", ec="#bbbbbb", lw=.5))
        ax.grid(alpha=.3, lw=.5)
    elif spec == "dist":
        rows = [("validation", truth["val_below_0_1"], truth["val_above_0_5"], truth["val_above_0_9"]),
                ("test", truth["test_below_0_1"], truth["test_above_0_5"], truth["test_above_0_9"])]
        txt = "share   <0.1   >0.5   >0.9\n" + "\n".join(f"{n:<10s}{a:5.2f}  {b:5.2f}  {c:5.2f}" for n, a, b, c in rows)
        ax.text(0.5, 0.97, txt, transform=ax.transAxes, ha="center", va="top", fontsize=5, family="monospace",
                bbox=dict(boxstyle="round", fc="white", ec="#bbbbbb", lw=.5))
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
