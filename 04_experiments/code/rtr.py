"""Read-then-reason: the quantities a reader is asked to read off each plot, computed (a) exactly from a plot's stored
mark vector and (b) from a reader's extraction answer, plus the fixed features the diagnosis rule uses.

The reader never diagnoses here: it only reports drawn values (prompts in hpc/vlm_read.py, EXTRACT). The rule is
trained on exact values from synthetic training models and applied unchanged to what readers read.
"""
import numpy as np

FPR_AT = (0.05, 0.1, 0.2, 0.5)
REC_AT = (0.2, 0.5, 0.8)
GRID = np.linspace(0, 1, 51)


def _f(x, default=np.nan):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------- exact values from mark vectors (plots.marks_vector)
def truth_from_vector(v, spec):
    v = np.asarray(v, float)
    if spec in ("lc_loss", "lc_acc"):
        tr, va = v[:len(v) // 2], v[len(v) // 2:]
        best = np.argmin(va) if spec == "lc_loss" else np.argmax(va)
        return dict(train_last=tr[-1], val_last=va[-1], val_best=va[best], val_best_epoch=best + 1)
    if spec in ("rel", "rel_n"):
        k = 3 if spec == "rel_n" else 2
        pts = [p for p in v.reshape(-1, k) if p[0] >= 0]
        tot = sum(p[2] for p in pts) if spec == "rel_n" else 1
        return dict(points=[[p[0], p[1]] + ([p[2] / max(tot, 1)] if spec == "rel_n" else []) for p in pts])
    if spec in ("cm", "cm_norm"):
        tn, fp, fn, tp = v[:4]
        return dict(tn=tn, fp=fp, fn=fn, tp=tp)
    if spec == "roc":
        y = v[:51]
        return {f"tpr_at_{f}": float(np.interp(f, GRID, y)) for f in FPR_AT}
    if spec == "pr":
        y = v[:51]
        out = {f"precision_at_{r}": float(np.interp(r, GRID, y)) for r in REC_AT}
        out["positive_rate_line"] = v[51] if len(v) > 51 else np.nan
        return out
    if spec == "dist":  # marks sorted by series: "test" then "validation", 20 density bars each (width 0.05)
        test, val = v[:20] * 0.05, v[20:40] * 0.05
        q = lambda h, a, b: float(h[int(a * 20):int(b * 20)].sum() / max(h.sum(), 1e-9))  # noqa: E731
        return dict(val_below_0_1=q(val, 0, .1), val_above_0_9=q(val, .9, 1), val_above_0_5=q(val, .5, 1),
                    test_below_0_1=q(test, 0, .1), test_above_0_9=q(test, .9, 1), test_above_0_5=q(test, .5, 1))
    raise ValueError(spec)


# ---------------------------------------------------------------- reader answers -> same dict
def from_reader(parsed, spec):
    if not isinstance(parsed, dict):
        return None
    if spec in ("rel", "rel_n"):
        pts = parsed.get("points")
        if not isinstance(pts, list):
            return None
        good = []
        for p in pts:
            if isinstance(p, dict):
                p = [p.get("x"), p.get("y")] + ([p.get("share")] if spec == "rel_n" else [])
            if isinstance(p, (list, tuple)) and len(p) >= (3 if spec == "rel_n" else 2):
                q = [_f(a) for a in p[:3 if spec == "rel_n" else 2]]
                if not any(np.isnan(q)):
                    good.append(q)
        return dict(points=good) if good else None
    keys = dict(lc_loss=("train_last", "val_last", "val_best", "val_best_epoch"),
                lc_acc=("train_last", "val_last", "val_best", "val_best_epoch"),
                cm=("tn", "fp", "fn", "tp"), cm_norm=("tn", "fp", "fn", "tp"),
                roc=tuple(f"tpr_at_{f}" for f in FPR_AT),
                pr=tuple(f"precision_at_{r}" for r in REC_AT) + ("positive_rate_line",),
                dist=("val_below_0_1", "val_above_0_9", "val_above_0_5", "test_below_0_1", "test_above_0_9", "test_above_0_5"))[spec]
    out = {k: _f(parsed.get(k)) for k in keys}
    if spec == "cm_norm":  # readers may answer percentages
        if any(out[k] > 1.5 for k in keys if not np.isnan(out[k])):
            out = {k: out[k] / 100 for k in keys}
    return None if all(np.isnan(x) for x in out.values()) else out


# ---------------------------------------------------------------- fixed rule features
def features(d, spec):
    if d is None:
        return None
    if spec in ("lc_loss", "lc_acc"):
        a = [d["train_last"], d["val_last"], d["val_best"], d["val_best_epoch"] / 100]
        return a + [d["val_last"] - d["val_best"], d["val_last"] - d["train_last"]]
    if spec in ("rel", "rel_n"):
        P = np.array(d["points"], float)
        x, y = P[:, 0], P[:, 1]
        dev = y - x
        lo, hi = dev[x < .5], dev[x >= .5]
        f = [dev.mean(), np.abs(dev).mean(), lo.mean() if len(lo) else 0.0, hi.mean() if len(hi) else 0.0,
             len(P) / 10, x.min(), x.max()]
        if spec == "rel_n":
            s = P[:, 2] / max(P[:, 2].sum(), 1e-9)
            f += [s[(x < .1) | (x > .9)].sum(), s.max(), (s * x).sum()]
        return f
    if spec in ("cm", "cm_norm"):
        tn, fp, fn, tp = (d[k] for k in ("tn", "fp", "fn", "tp"))
        if spec == "cm":
            n = max(tn + fp + fn + tp, 1e-9)
            return [(fn + tp) / n, (fp + tp) / n, (tn + tp) / n, tp / max(tp + fn, 1e-9), tn / max(tn + fp, 1e-9)]
        return [tn, fp, fn, tp]
    if spec == "roc":
        t = [d[f"tpr_at_{f}"] for f in FPR_AT]
        return t + [float(np.mean(t))]
    if spec == "pr":
        return [d[f"precision_at_{r}"] for r in REC_AT] + [d["positive_rate_line"]]
    if spec == "dist":
        v = [d[k] for k in ("val_below_0_1", "val_above_0_9", "val_above_0_5", "test_below_0_1", "test_above_0_9", "test_above_0_5")]
        return v + [v[3] - v[0], v[4] - v[1], v[5] - v[2]]
    raise ValueError(spec)
