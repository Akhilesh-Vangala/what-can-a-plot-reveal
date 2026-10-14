"""Build lineup items: published human residual-plot lineups + planted-flaw ML diagnostic lineups.

Every item is a 20-panel PNG plus a key entry. Keys never go into prompts.
"""
import json
import pathlib
import pickle

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from plotreveal.seeds import rng_for  # noqa: E402

ROOT = pathlib.Path(__file__).parent
DATA = ROOT / "data"


def _rng(*parts):
    """Seeded from the item family, so every item can be regenerated (gap 2.7)."""
    return rng_for("legacy-lineups", *parts)


def _grid(panels, path, kind):
    fig, axs = plt.subplots(4, 5, figsize=(12, 9.6))
    for i, ax in enumerate(axs.flat):
        x, y = panels[i]
        if kind == "resid":
            ax.scatter(x, y, s=3, alpha=0.6, color="black")
            ax.axhline(0, color="red", lw=1)
        elif kind == "reliability":
            ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1)
            ax.plot(x, y, marker="o", ms=3, color="black", lw=1)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        elif kind == "learning_curve":
            train, val = y
            ax.plot(x, train, color="tab:blue", lw=1.2)
            ax.plot(x, val, color="tab:orange", lw=1.2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(str(i + 1), fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


# ---------- published human lineups ----------

def load_human():
    import pyreadr
    import rdata

    survey = pyreadr.read_r(str(DATA / "vi_survey.rda"))["vi_survey"]
    cache = DATA / "vi_lineup.pkl"
    if cache.exists():
        lineups = pickle.load(open(cache, "rb"))
    else:
        parsed = rdata.parser.parse_file(str(DATA / "vi_lineup.rds"))
        lineups = rdata.conversion.convert(parsed, default_encoding="utf8")
        pickle.dump(lineups, open(cache, "wb"))
    return survey, lineups


def human_items(out_dir, n_per_band=8, n_null=8):
    """Stratified sample of published lineups by human detection rate, plus null lineups."""
    rng = _rng("human")
    survey, lineups = load_human()
    sv = survey[survey.attention_check == False]  # noqa: E712
    per = sv.groupby("unique_lineup_id").agg(
        human_rate=("detect", "mean"), null=("null_lineup", "first"),
        type=("type", "first"), n_humans=("detect", "size")).reset_index()
    real = per[per.null == False].copy()  # noqa: E712
    real["band"] = np.digitize(real.human_rate, [0.1, 0.35, 0.6, 0.85])
    picks = []
    for b in range(5):
        pool = real[real.band == b].unique_lineup_id.values
        picks += list(rng.choice(pool, min(n_per_band, len(pool)), replace=False))
    nulls = per[per.null == True].unique_lineup_id.values  # noqa: E712
    picks += list(rng.choice(nulls, min(n_null, len(nulls)), replace=False))

    items = []
    for lid in picks:
        d = lineups[lid]["data"]
        ks = sorted(d["k"].unique())
        perm = rng.permutation(len(ks))
        panels, real_pos = [None] * 20, None
        for pos, k in zip(perm, ks):
            dd = d[d.k == k]
            panels[pos] = (dd[".fitted"].values, dd[".resid"].values)
            if not bool(dd["null"].iloc[0]):
                real_pos = int(pos) + 1
        row = per[per.unique_lineup_id == lid].iloc[0]
        item_id = f"H_{lid}"
        _grid(panels, out_dir / f"{item_id}.png", "resid")
        items.append({"item_id": item_id, "family": "residual", "source": "visage",
                      "lineup": str(lid), "real_panel": None if row.null else real_pos,
                      "null_lineup": bool(row.null), "human_rate": float(row.human_rate),
                      "n_humans": int(row.n_humans), "departure": str(row.type)})
    return items


# ---------- planted-flaw ML diagnostic lineups ----------

def _reliability(s, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(s, edges) - 1, 0, bins - 1)
    xs, ys = [], []
    for b in range(bins):
        m = idx == b
        if m.sum():
            xs.append(s[m].mean())
            ys.append(y[m].mean())
    return np.array(xs), np.array(ys)


def calibration_items(out_dir, temps=(1.0, 1.15, 1.3, 1.6, 2.5), reps=4, n=600, bins=10):
    """Overconfidence via temperature T on one panel; T=1.0 is a null lineup."""
    rng = _rng("calibration")
    items = []
    for T in temps:
        for r in range(reps):
            real_pos = int(rng.integers(20))
            panels = []
            for i in range(20):
                s = rng.beta(2, 2, size=n)
                p = s
                if i == real_pos and T != 1.0:
                    p = 1 / (1 + np.exp(-np.log(s / (1 - s)) / T))
                panels.append(_reliability(s, rng.random(n) < p, bins))
            item_id = f"C_T{T}_r{r}"
            _grid(panels, out_dir / f"{item_id}.png", "reliability")
            items.append({"item_id": item_id, "family": "reliability", "source": "planted",
                          "flaw": "miscalibration", "severity": T, "null_lineup": T == 1.0,
                          "real_panel": None if T == 1.0 else real_pos + 1, "bins": bins, "n": n})
    return items


def learning_curve_items(out_dir, gaps=(0.0, 0.03, 0.06, 0.12, 0.25), reps=4, epochs=30):
    """Overfitting: validation loss turns upward by `gap` on one panel; gap=0 is a null lineup."""
    rng = _rng("learning_curve")
    items = []
    x = np.arange(epochs)
    for g in gaps:
        for r in range(reps):
            real_pos = int(rng.integers(20))
            panels = []
            for i in range(20):
                rate = rng.uniform(0.08, 0.15)
                floor = rng.uniform(0.15, 0.3)
                train = floor * 0.6 + np.exp(-rate * x) + rng.normal(0, 0.01, epochs)
                val = floor + np.exp(-rate * x) + rng.normal(0, 0.02, epochs)
                if i == real_pos and g > 0:
                    turn = int(epochs * rng.uniform(0.35, 0.55))
                    val = val + g * np.clip((x - turn) / (epochs - turn), 0, None)
                panels.append((x, (train, val)))
            item_id = f"L_g{g}_r{r}"
            _grid(panels, out_dir / f"{item_id}.png", "learning_curve")
            items.append({"item_id": item_id, "family": "learning_curve", "source": "planted",
                          "flaw": "overfitting", "severity": g, "null_lineup": g == 0.0,
                          "real_panel": None if g == 0.0 else real_pos + 1})
    return items


def build(out_name="pilot", human=True, calibration=True, learning=True):
    out_dir = ROOT / "runs" / out_name / "img"
    out_dir.mkdir(parents=True, exist_ok=True)
    items = []
    if human:
        items += human_items(out_dir)
    if calibration:
        items += calibration_items(out_dir)
    if learning:
        items += learning_curve_items(out_dir)
    key_path = ROOT / "runs" / out_name / "key.json"
    json.dump(items, open(key_path, "w"), indent=1)
    print(f"{len(items)} items -> {out_dir}  (key: {key_path})")
    print(pd.DataFrame(items).groupby(["family", "null_lineup"]).size())
    return items


if __name__ == "__main__":
    build()
