"""Example plots for the explorer's cell view: a few held-out confirmatory models per state and severity, re-trained from
their seeds (identical to the cluster run; verified against the stored deployment accuracy) and rendered.

  python make_examples.py runs/confirm_synthetic synthetic [--per 2 --healthy 3]
      -> ../explorer/examples/<run>/<run_id>_<plot>.png  +  ../explorer/examples_<run>.js (index of examples)
"""
import json
import pathlib
import re
import sys

import numpy as np
from joblib import Parallel, delayed

from plotreveal import plots
from plotreveal.world import apply_failure, healthy_config, train_run

ROOT = pathlib.Path(__file__).parent
SPECS = ("lc_loss", "lc_acc", "rel", "rel_n", "cm", "cm_norm", "roc", "pr", "dist")


def config(rid, src, meta):
    m = re.match(r"m_(.+)_L(\d)_(\d+)$", rid)
    if m:
        f, j, i = m.group(1), int(m.group(2)), int(m.group(3))
        return apply_failure(healthy_config("confirm:main", f, j, i, epochs=100, source=src), f, meta["severities"][f][j])
    return healthy_config("confirm:main", "healthy", int(rid.split("_")[-1]), epochs=100, source=src)


def one(rid, src, meta, out):
    st = train_run(config(rid, src, meta))
    for sp in SPECS:
        plots.render(plots.mark_table(st, sp), sp, path=out / f"{rid}_{sp}.png", dpi=80)
    return rid, abs(st["deploy_acc"] - meta["deploy_acc"][rid])


def main(run, src, per=2, n_healthy=3):
    run = pathlib.Path(run)
    meta = json.load(open(run / "meta.json"))
    out = ROOT.parent / "explorer" / "examples" / run.name
    out.mkdir(parents=True, exist_ok=True)
    pick, index = [], {}
    for key, members in meta["groups"].items():
        f, lv = key.split("|")
        test = sorted([r for r, s in members.items() if s == "test"], key=lambda r: int(r.split("_")[-1]))
        chosen = test[:n_healthy] if f == "healthy" else test[:per]
        pick += chosen
        index.setdefault(f, {})["None" if lv == "None" else lv] = chosen
    res = Parallel(n_jobs=8)(delayed(one)(r, src, meta, out) for r in pick)
    worst = max(d for _, d in res)
    if worst > 0.01:
        raise SystemExit(f"re-trained examples do not match the confirmatory models (max |Δ deploy_acc| = {worst:.4f})")
    js = ROOT.parent / "explorer" / f"examples_{run.name}.js"
    js.write_text(f"window.EXAMPLES = window.EXAMPLES || {{}};\nwindow.EXAMPLES[{json.dumps(run.name)}] = {json.dumps(index)};\n")
    print(f"{len(pick)} example models × {len(SPECS)} plots → {out} (reproduced exactly: max |Δ deploy_acc| = {worst:.5f})")


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    o = dict(x.lstrip("-").split("=") for x in sys.argv[1:] if x.startswith("--"))
    main(a[0], a[1], int(o.get("per", 2)), int(o.get("healthy", 3)))
