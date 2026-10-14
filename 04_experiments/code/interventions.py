"""Design interventions built FROM the measured evidential map of a v4 run (H6).

  python interventions.py runs/v4_synthetic      (after run_v4.py and b2_v4.py)

1. Evidence badge (epistemic, rendering-level): a note under the plot stating what this plot type can identify, can only
   signal, and cannot show — derived from the strongest-level evidential map (B1* = val-best of mark/image model).
2. Companion view (informational, encoding-level): the plot paired with the plot that best resolves its most-confused
   pair of states; benchmark for the pair = multi-way diagnosis on the concatenated mark features (features.npz).
Outputs: img_iv/ (badge, companion, badge+companion images), badges.json, companions.json,
         companion_benchmarks.json, reader_items_interventions.json (same held-out runs as the prompt-variant set).
"""
import json
import pathlib
import sys
import textwrap

import matplotlib
import numpy as np
from PIL import Image

from plotreveal import benchmarks as bm
from plotreveal.seeds import seed_for

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUN = pathlib.Path(sys.argv[1])
CORE = ("lc_loss", "rel", "cm", "roc", "dist")
STANDARD = ("lc_loss", "lc_acc", "rel", "cm", "roc", "pr", "dist")  # companions come from standard designs
NAMES = {"overfitting": "overfitting", "miscalibration": "miscalibration", "label_noise": "label noise",
         "class_imbalance": "class imbalance", "data_leakage": "data leakage", "distribution_shift": "distribution shift"}
DETECT_T, DIAG_T = 0.70, 0.50


def star(a, b):
    return a if b is None or a["val_auc"] >= b["val_auc"] else b


def load_map():
    meta = json.load(open(RUN / "meta.json"))
    res = json.load(open(RUN / "results.json"))
    b2 = json.load(open(RUN / "b2_results.json"))
    top = len(meta["targets"]) - 1
    b2det = {(r["failure"], r["spec"]): r for r in b2["detection"] if r["level"] == top}
    b2dia = {r["spec"]: r for r in b2["diagnosis"] if r["level"] == top}
    det = {r["failure"]: r for r in res["detection"] if r["level"] == top}
    dia = [r for r in res["diagnosis"] if r["level"] == top][0]
    lab, conf = {}, {}
    for sp in meta["specs"]:
        c = np.array(star(dia[sp], b2dia.get(sp))["confusion"], dtype=float)
        conf[sp] = c / np.maximum(c.sum(1, keepdims=True), 1)
        for i, f in enumerate(meta["failures"]):
            auc = star(det[f][sp], b2det.get((f, sp)))["test_auc"]
            rec = conf[sp][i + 1, i + 1]
            lab[(f, sp)] = "id" if auc >= DETECT_T and rec >= DIAG_T else ("signal" if auc >= DETECT_T else "none")
    return meta, lab, conf, top


def badge_text(meta, lab, sp):
    groups = {k: [NAMES[f] for f in meta["failures"] if lab[(f, sp)] == k] for k in ("id", "signal", "none")}
    parts = []
    if groups["id"]:
        parts.append("Can identify: " + ", ".join(groups["id"]) + ".")
    if groups["signal"]:
        parts.append("Can signal a problem but cannot tell which: " + ", ".join(groups["signal"]) + ".")
    if groups["none"]:
        parts.append("Cannot show: " + ", ".join(groups["none"]) + ".")
    return "Evidence note for this plot type — " + " ".join(parts)


def companion_for(meta, conf, sp):
    classes = ["healthy"] + meta["failures"]
    c = conf[sp].copy()
    np.fill_diagonal(c, 0)
    a, b = np.unravel_index(c.argmax(), c.shape)  # most-confused pair (true a read as b)
    best = max((s for s in STANDARD if s != sp and not s.startswith(sp[:2])),
               key=lambda s: min(conf[s][a, a], conf[s][b, b]))
    return best, (classes[a], classes[b])


def add_badge(img, text):
    w, h = img.size
    lines = textwrap.wrap(text, width=int(w / 6.2))
    fig = plt.figure(figsize=(w / 100, (h + 14 * len(lines) + 10) / 100), dpi=100)
    ax = fig.add_axes([0, 1 - h / (h + 14 * len(lines) + 10), 1, h / (h + 14 * len(lines) + 10)])
    ax.imshow(img)
    ax.axis("off")
    fig.text(0.02, (14 * len(lines) + 4) / (h + 14 * len(lines) + 10), "\n".join(lines), fontsize=7.2, va="top",
             family="sans-serif", bbox=dict(facecolor="#fff7e0", edgecolor="#d9a400", boxstyle="round,pad=0.3"))
    fig.canvas.draw()
    out = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    plt.close(fig)
    return out


def side_by_side(a, b):
    c = Image.new("RGB", (a.width + b.width, max(a.height, b.height)), "white")
    c.paste(a, (0, 0))
    c.paste(b, (a.width, 0))
    return c


def load_features(meta, top):
    """features.npz from run_v4 if present; otherwise rebuild the needed runs (healthy + strongest level) from the
    manifest — every run is seeded, so on the same software stack the models and plots are reproduced exactly."""
    if (RUN / "features.npz").exists():
        F = np.load(RUN / "features.npz")
        return {k: F[k] for k in F.files}
    from joblib import Parallel, delayed

    from plotreveal import plots
    from plotreveal.world import train_run
    keys = ["healthy|None"] + [f"{f}|{top}" for f in meta["failures"]]
    need = [r for k in keys for r in meta["groups"][k]]
    man = {m["run_id"]: m for m in map(json.loads, open(RUN / "manifest.jsonl")) if m.get("role") == "main"}
    cfgs = [{k: v for k, v in man[r].items() if k not in ("role", "run_id", "level")} for r in need]
    print(f"features.npz missing — rebuilding {len(need)} runs from the manifest", flush=True)
    states = Parallel(n_jobs=-1, batch_size=8)(delayed(train_run)(c) for c in cfgs)
    F = {"ids": np.array(need)}
    for sp in STANDARD + ("rel_n", "cm_norm"):
        if sp in meta["specs"]:
            F[sp] = np.array([plots.marks_vector(plots.mark_table(st, sp), sp) for st in states], dtype=np.float32)
    np.savez_compressed(RUN / "features_rebuilt.npz", **F)
    return F


def main():
    meta, lab, conf, top = load_map()
    badges = {sp: badge_text(meta, lab, sp) for sp in meta["specs"]}
    comps = {sp: dict(zip(("companion", "pair"), companion_for(meta, conf, sp))) for sp in CORE}
    json.dump(badges, open(RUN / "badges.json", "w"), indent=1)
    json.dump(comps, open(RUN / "companions.json", "w"), indent=1)
    for sp in CORE:
        print(f"{sp}: companion {comps[sp]['companion']} (resolves {comps[sp]['pair']})\n   badge: {badges[sp]}")

    # companion benchmark: multi-way diagnosis on concatenated mark features vs the single plot
    F = load_features(meta, top)
    ids = list(F["ids"])
    pos = {r: i for i, r in enumerate(ids)}
    keys = ["healthy|None"] + [f"{f}|{top}" for f in meta["failures"]]
    members = [(k, r, s) for k in keys for r, s in meta["groups"][k].items()]
    rows = np.array([pos[r] for _, r, _ in members])
    y = np.array([keys.index(k) for k, _, _ in members])
    split = {s: np.array([i for i, (_, _, t) in enumerate(members) if t == s]) for s in ("train", "val", "test")}
    bench = {}
    for sp in CORE:
        c = comps[sp]["companion"]
        single = bm.best_of_k_multi(F[sp][rows], y, split, seed=seed_for("iv", sp))
        pair = bm.best_of_k_multi(np.hstack([F[sp][rows], F[c][rows]]), y, split, seed=seed_for("iv", sp, c))
        bench[sp] = dict(companion=c, single_balanced_acc=single["balanced_acc"], pair_balanced_acc=pair["balanced_acc"])
        print(f"benchmark {sp}: single {single['balanced_acc']:.3f} → with {c} {pair['balanced_acc']:.3f}")
    json.dump(bench, open(RUN / "companion_benchmarks.json", "w"), indent=1)

    # images + reader items on the same held-out runs as the prompt-variant set
    out = RUN / "img_iv"
    out.mkdir(exist_ok=True)
    runs = sorted({it["run_id"] for it in json.load(open(RUN / "reader_items_variants.json"))})
    items = []
    for rid in runs:
        for sp in CORE:
            base = Image.open(RUN / "img" / f"{rid}_{sp}.png").convert("RGB")
            comp = Image.open(RUN / "img" / f"{rid}_{comps[sp]['companion']}.png").convert("RGB")
            imgs = {"badge": add_badge(base, badges[sp]), "companion": side_by_side(base, comp)}
            imgs["badge_companion"] = add_badge(imgs["companion"],
                                                badges[sp].replace("for this plot type", "for the left plot"))
            for cond, im in imgs.items():
                path = f"img_iv/{rid}_{sp}_{cond}.png"
                im.save(RUN / path)
                items.append(dict(item_id=f"{rid}_{sp}_{cond}", run_id=rid, task="diagnosis", spec=sp, condition=cond,
                                  companion=comps[sp]["companion"] if "companion" in cond else None,
                                  badge="badge" in cond, image=path))
    json.dump(items, open(RUN / "reader_items_interventions.json", "w"), indent=0)
    print(f"{len(items)} intervention items")


if __name__ == "__main__":
    main()
