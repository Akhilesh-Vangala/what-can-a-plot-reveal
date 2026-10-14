"""Read-then-reason, step 1 (cluster CPU): exact values for the rule + extraction item sets for the readers.

  python rtr_prepare.py runs/confirm_synthetic runs/positive runs/realworld
    -> runs/confirm_synthetic/rtr_values.json     exact values for training runs (healthy + strongest level, train split)
                                                   and core test runs, all 9 plot types
    -> <run>/reader_items_extract.json            extraction items (same images as the diagnosis items)
"""
import json
import pathlib
import sys

import numpy as np

import rtr
from score_confirm import Run


def main(syn, pos, rw):
    syn, pos, rw = map(pathlib.Path, (syn, pos, rw))
    R = Run(syn)
    j = R.top
    F = np.load(syn / "features.npz")
    idx = {r: i for i, r in enumerate(F["ids"])}
    keep = {}
    for key, members in R.meta["groups"].items():
        f, lv = key.split("|")
        if lv not in ("None", str(j)):
            continue
        for rid, split in members.items():
            keep[rid] = dict(failure=f, split=split)
    out = {}
    for rid, info in keep.items():
        out[rid] = dict(info, values={sp: {k: (np.asarray(v).tolist() if isinstance(v, list) else float(v))
                                           for k, v in rtr.truth_from_vector(F[sp][idx[rid]], sp).items()}
                                      for sp in R.meta["specs"]})
    json.dump(out, open(syn / "rtr_values.json", "w"))
    print(f"exact values: {len(out)} runs ({sum(v['split'] == 'train' for v in out.values())} train)", flush=True)
    for run, src in ((syn, "reader_items_core.json"), (pos, "reader_items.json"), (rw, "reader_items.json")):
        items = [dict(it, item_id=it["item_id"] + "_extract", task="extract") for it in json.load(open(run / src))
                 if it.get("task") == "diagnosis"]
        json.dump(items, open(run / "reader_items_extract.json", "w"), indent=0)
        print(f"{run.name}: {len(items)} extraction items", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
