"""Instructor-review checks that need the cluster copy of a run (features.npz, img/).

  python review_prepare.py mmd   runs/confirm_synthetic runs/confirm_magic  -> <run>/two_sample.json
  python review_prepare.py items runs/confirm_synthetic                     -> <run>/reader_items_binary.json,
                                                                               reader_items_2afc.json, img_2afc/

mmd (point 4, "blind" without a learned estimator): for every failure x plot at the strongest severity, a kernel
  two-sample test of "the plotted values have the same distribution for healthy and failing models". Statistic: the
  largest (biased) MMD^2 over three RBF bandwidths (0.25x, 1x, 4x the median distance) on standardized plotted
  values, 500 healthy vs 500 failing models; p-value from 2,000 label permutations. No model is trained, so a
  non-significant result cannot be blamed on a weak estimator; a significant one with a low benchmark AUC means the
  plot carries a weak signal, not none.
items (point 5, is the 7-way task too hard?): on the same held-out core items the readers already diagnosed,
  binary: one plot -> probability that the model has any problem (healthy vs failing; scored as AUC per failure);
  2afc:   two plots side by side, one healthy and one failing model -> which one has the problem (chance = 0.5).
  For an ideal observer, 2AFC accuracy equals the detection AUC, so both compare directly with the benchmark.
"""
import json
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PER_CELL = 30  # 2AFC pairs per failure x plot


def mmd_test(X, Y, rng, n_perm=2000):
    Z = np.r_[X, Y]
    sd = Z.std(0)
    keep = sd > 1e-9
    if not keep.any():
        return dict(stat=0.0, p=1.0, constant=True)
    Z = (Z[:, keep] - Z[:, keep].mean(0)) / sd[keep]
    sq = (Z ** 2).sum(1)
    D = np.maximum(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0)
    med = np.median(D[np.triu_indices(len(Z), 1)])
    med = med if med > 0 else 1.0
    n, m = len(X), len(Y)
    W = np.empty((len(Z), n_perm + 1))
    base = np.r_[np.full(n, 1 / n), np.full(m, -1 / m)]
    W[:, 0] = base
    for k in range(n_perm):
        W[:, k + 1] = base[rng.permutation(len(Z))]
    stats = []
    for s in (0.25, 1.0, 4.0):
        K = np.exp(-D / (s * med))
        stats.append(np.einsum("ik,ik->k", W, K @ W))
    T = np.max(np.array(stats), 0)  # max over bandwidths, permutation-calibrated
    return dict(stat=float(T[0]), p=float((1 + (T[1:] >= T[0]).sum()) / (n_perm + 1)), constant=False)


def mmd(paths):
    for p in paths:
        run = pathlib.Path(p)
        meta = json.load(open(run / "meta.json"))
        F = np.load(run / "features.npz")
        pos = {r: i for i, r in enumerate(F["ids"])}
        top = len(meta["targets"]) - 1
        rng = np.random.default_rng(0)
        healthy = list(meta["groups"]["healthy|None"])
        out = []
        for f in meta["failures"]:
            sick = list(meta["groups"][f"{f}|{top}"])
            h = [pos[r] for r in rng.choice(healthy, 500, replace=False)]
            s = [pos[r] for r in rng.choice(sick, min(500, len(sick)), replace=False)]
            for sp in meta["specs"]:
                r = mmd_test(F[sp][h].astype(float), F[sp][s].astype(float), np.random.default_rng(1))
                out.append(dict(failure=f, plot=sp, n_healthy=len(h), n_failing=len(s), **r))
                print(f"{run.name} {f:18s} {sp:8s} MMD {r['stat']:.4f} p {r['p']:.4f}", flush=True)
        json.dump(out, open(run / "two_sample.json", "w"), indent=1)
        print("saved", run / "two_sample.json")


def items(path):
    run = pathlib.Path(path)
    meta = json.load(open(run / "meta.json"))
    core = json.load(open(run / "reader_items_core.json"))
    truth = {r: k.split("|")[0] for k, m in meta["groups"].items() for r in m}
    json.dump([dict(it, item_id=it["item_id"], task="binary") for it in core], open(run / "reader_items_binary.json", "w"))
    out_dir = run / "img_2afc"
    out_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(2026)
    try:  # large labels so every reader can see which panel is A and which is B
        font = ImageFont.load_default(size=22)
    except TypeError:  # Pillow < 10.1: no sized default font -> draw small and upscale (see label())
        font = None
    pairs = []
    for sp in sorted({it["spec"] for it in core}):
        ids = sorted({it["run_id"] for it in core if it["spec"] == sp})
        healthy = [r for r in ids if truth[r] == "healthy"]
        for f in meta["failures"]:
            sick = list(rng.choice(sorted(r for r in ids if truth[r] == f), PER_CELL, replace=False))
            hl = list(rng.choice(healthy, PER_CELL, replace=False))
            sides = rng.permutation(np.r_[np.zeros(PER_CELL // 2, int), np.ones(PER_CELL - PER_CELL // 2, int)])
            for k, (s, h, side) in enumerate(zip(sick, hl, sides)):
                left, right = (s, h) if side == 0 else (h, s)
                a, b = (Image.open(run / "img" / f"{r}_{sp}.png").convert("RGB") for r in (left, right))
                W, H = a.width + b.width + 30, max(a.height, b.height) + 34
                c = Image.new("RGB", (W, H), "white")
                c.paste(a, (0, 34))
                c.paste(b, (a.width + 30, 34))
                d = ImageDraw.Draw(c)
                d.line([(a.width + 15, 34), (a.width + 15, H)], fill=(180, 180, 180), width=2)
                for x0, w, lab in ((0, a.width, "A"), (a.width + 30, b.width, "B")):
                    if font is not None:
                        d.text((x0 + w // 2, 17), lab, fill="black", font=font, anchor="mm", stroke_width=1, stroke_fill="black")
                    else:
                        t = Image.new("RGB", (8, 11), "white")
                        ImageDraw.Draw(t).text((1, 0), lab, fill="black")
                        t = t.resize((24, 33), Image.NEAREST)
                        c.paste(t, (x0 + w // 2 - 12, 0))
                name = f"pair_{sp}_{f}_{k}"
                c.save(out_dir / f"{name}.png")
                pairs.append(dict(item_id=name, task="2afc", spec=sp, failure=f, failing_side="AB"[side],
                                  failing_run=s, healthy_run=h, image=f"img_2afc/{name}.png"))
    json.dump(pairs, open(run / "reader_items_2afc.json", "w"))
    print(f"binary items: {len(core)}; 2AFC pairs: {len(pairs)} ({PER_CELL} per failure x plot, failing side balanced)")


if __name__ == "__main__":
    {"mmd": lambda: mmd(sys.argv[2:]), "items": lambda: items(sys.argv[2])}[sys.argv[1]]()
