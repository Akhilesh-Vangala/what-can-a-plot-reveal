"""Human reader study materials (ready to run once the IRB route is confirmed).

  python make_human_study.py   -> ../human_study/survey/ (participant files: survey.html, forms.js, images/)
                                  + ../human_study/answer_key.json, items.json (team only)

Items are held-out confirmatory models (strongest severity) that the AI readers also answered, so people, AI readers and
benchmarks are compared on identical items. Models are re-trained from their seeds (identical to the study run; checked
against the stored deployment accuracy) and rendered in standard and redesigned versions:

  family     standard   redesigns
  loss       lc_loss    —
  confusion  cm         —
  reliability rel       rel_n (bin sizes drawn), rel_ann (values printed)
  roc        roc        roc_ann (values printed)
  histogram  dist       dist_ann (values printed)

Four counterbalanced forms (A–D): 40 items each (8 per family, states balanced), every model at most once per form, and
for families with a redesign the version rotates across forms so each model is seen in each version by someone. Two
attention checks per form use blatant failures from the positive control design (failure on its textbook plot).
"""
import json
import pathlib
import re

import numpy as np
from joblib import Parallel, delayed

import annotate
from plotreveal import plots
from plotreveal.seeds import rng_for
from plotreveal.world import apply_failure, healthy_config, train_run

ROOT = pathlib.Path(__file__).parent
OUT = ROOT.parent / "human_study"
RUN = ROOT / "runs" / "confirm_synthetic"
STATES = ["healthy", "overfitting", "miscalibration", "label_noise", "class_imbalance", "data_leakage", "distribution_shift"]
FAMILIES = {"loss": ["lc_loss"], "confusion": ["cm"], "reliability": ["rel", "rel_n", "rel_ann"], "roc": ["roc", "roc_ann"],
            "histogram": ["dist", "dist_ann"]}
PER_STATE = 20          # models per state in the pool (140 models)
FORMS = "ABCD"
PER_FAMILY = 8          # items per family per form (40 items)


def config(rid, meta):
    m = re.match(r"m_(.+)_L(\d)_(\d+)$", rid)
    if m:
        f, j, i = m.group(1), int(m.group(2)), int(m.group(3))
        return apply_failure(healthy_config("confirm:main", f, j, i, epochs=100, source="synthetic"), f, meta["severities"][f][j])
    return healthy_config("confirm:main", "healthy", int(rid.split("_")[-1]), epochs=100, source="synthetic")


def render_model(rid, meta, img):
    st = train_run(config(rid, meta))
    for sp in ("lc_loss", "cm", "rel", "rel_n", "roc", "dist"):
        plots.render(plots.mark_table(st, sp), sp, path=img / f"{rid}_{sp}.png", dpi=110)
    for sa, sp in annotate.BASE.items():
        annotate.render(plots.marks_vector(plots.mark_table(st, sp), sp), sa, img / f"{rid}_{sa}.png", dpi=110)
    return rid, abs(st["deploy_acc"] - meta["deploy_acc"][rid])


def attention_items(img):
    """Two blatant cases (positive-control strength) per form, re-used across forms."""
    from make_positive import SEVERITY
    out = []
    for k, (f, sp) in enumerate((("class_imbalance", "cm"), ("data_leakage", "roc"))):
        st = train_run(apply_failure(healthy_config("human-attention", f, k, epochs=100, source="synthetic"), f, SEVERITY[f]))
        name = f"attention_{f}"
        plots.render(plots.mark_table(st, sp), sp, path=img / f"{name}_{sp}.png", dpi=110)
        out.append(dict(item_id=f"{name}_{sp}", run_id=name, spec=sp, family="attention", truth=f, image=f"images/{name}_{sp}.png"))
    return out


def main():
    meta = json.load(open(RUN / "meta.json"))
    V = json.load(open(RUN / "rtr_values.json"))
    core = list(dict.fromkeys(it["run_id"] for it in json.load(open(RUN / "reader_items_core.json"))))
    rng = rng_for("human-study")
    pool = {s: list(rng.choice(sorted(r for r in core if V[r]["failure"] == s), PER_STATE, replace=False)) for s in STATES}
    models = [r for s in STATES for r in pool[s]]
    img = OUT / "survey" / "images"
    img.mkdir(parents=True, exist_ok=True)
    res = Parallel(n_jobs=8)(delayed(render_model)(r, meta, img) for r in models)
    worst = max(d for _, d in res)
    if worst > 0.01:
        raise SystemExit(f"re-trained models do not match the study (max |Δ deploy_acc| = {worst:.4f})")
    attention = attention_items(img)

    forms = {}
    for fi, form in enumerate(FORMS):
        r2 = rng_for("human-form", form)
        used, items = set(), []
        for fam, versions in FAMILIES.items():
            # 8 items: one per state + one extra (rotating state), models not yet used in this form
            states = STATES + [STATES[(fi + list(FAMILIES).index(fam)) % len(STATES)]]
            for k, s in enumerate(states[:PER_FAMILY]):
                cand = [r for r in pool[s] if r not in used]
                rid = cand[(fi * 3 + k + list(FAMILIES).index(fam) * 5) % len(cand)]
                used.add(rid)
                ver = versions[(fi + k) % len(versions)]  # rotate the version across forms
                items.append(dict(item_id=f"{rid}_{ver}", run_id=rid, spec=ver, family=fam, truth=s, image=f"images/{rid}_{ver}.png"))
        items += attention
        order = r2.permutation(len(items))
        forms[form] = [items[i] for i in order]
    # participant-facing ids and file names are opaque: nothing in them reveals the model's state
    import hashlib
    opaque = {}
    for f in forms.values():
        for it in f:
            if it["item_id"] not in opaque:
                opaque[it["item_id"]] = "q" + hashlib.sha256(("plotreveal-human|" + it["item_id"]).encode()).hexdigest()[:10]
    for old, new in opaque.items():
        src = img / (old + ".png")
        if src.exists():
            src.rename(img / (new + ".png"))
    key = {opaque[it["item_id"]]: dict(truth=it["truth"], family=it["family"], spec=it["spec"], run_id=it["run_id"], item=it["item_id"])
           for f in forms.values() for it in f}
    public = {f: [dict(item_id=opaque[it["item_id"]], spec=it["spec"], image=f"images/{opaque[it['item_id']]}.png")
                  for it in items] for f, items in forms.items()}
    for f in forms.values():
        for it in f:
            it["image"] = f"images/{opaque[it['item_id']]}.png"
    (OUT / "survey" / "forms.js").write_text("window.FORMS = " + json.dumps(public) + ";\n")
    json.dump(key, open(OUT / "answer_key.json", "w"), indent=1)
    json.dump(dict(pool=pool, per_form=len(forms["A"])), open(OUT / "items.json", "w"), indent=1)
    used = {it["image"].split("/")[-1] for f in forms.values() for it in f}
    for p in img.iterdir():  # ship only the images the forms use
        if p.name not in used:
            p.unlink()
    n_ver = {}
    for f in forms.values():
        for it in f:
            n_ver[it["spec"]] = n_ver.get(it["spec"], 0) + 1
    print(f"{len(models)} models re-trained exactly (max |Δ deploy_acc| = {worst:.5f}); {len(forms)} forms × {len(forms['A'])} items; "
          f"versions shown across forms: {n_ver}")


if __name__ == "__main__":
    main()
