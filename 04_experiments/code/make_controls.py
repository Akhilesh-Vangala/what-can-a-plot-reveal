"""Reader controls for a prototype run (gaps 3.3, 4.1).

  python make_controls.py runs/overfit_proto_v2

1. Shuffled lineups: every lineup re-drawn with its panels in a new (seeded) order -> lineups_shuffled/ + key.
   Same plots, different positions: separates "sees the odd panel" from "prefers a position".
2. Value-extraction probe: single plots where the reader must read numbers off the plot; true values saved
   locally (probe_truth.json) and only the item list goes to the cluster (probe_items.json).
Panel models are re-trained from the manifest configs (all runs are seeded, so they are reproduced exactly).
"""
import json
import pathlib
import sys

import numpy as np
from joblib import Parallel, delayed

from plotreveal import plots
from plotreveal.generator import train_run
from plotreveal.seeds import rng_for

RUN = pathlib.Path(sys.argv[1])
N_PROBE = 50  # per class per spec


def main():
    manifest = {m["run_id"]: m for m in map(json.loads, open(RUN / "manifest.jsonl")) if "run_id" in m}
    key = json.load(open(RUN / "lineups" / "key.json"))
    meta = json.load(open(RUN / "meta.json"))

    # probe items: healthy test plots + strongest-level failed test plots
    r = rng_for("probe-items")
    h_test = sorted(k for k, s in meta["healthy"].items() if s == "test")
    f_test = sorted(k for k, s in meta["levels"][-1]["failed"].items() if s == "test")
    probe_runs = list(r.choice(h_test, N_PROBE, replace=False)) + list(r.choice(f_test, N_PROBE, replace=False))

    need = sorted({p for k in key for p in k["panel_runs"]} | set(probe_runs))
    cfgs = {rid: {k: v for k, v in manifest[rid].items() if k not in ("role", "run_id", "level", "idx")} for rid in need}
    print(f"re-training {len(need)} models from the manifest", flush=True)
    states = dict(zip(need, Parallel(n_jobs=-1)(delayed(train_run)(cfgs[rid]) for rid in need)))

    out = RUN / "lineups_shuffled"
    out.mkdir(exist_ok=True)
    skey = []
    for k in key:
        rr = rng_for("shuffle", k["item_id"])
        order = rr.permutation(20)
        runs = [k["panel_runs"][i] for i in order]
        real = None if k["null_lineup"] else int(np.where(order == k["real_panel"] - 1)[0][0]) + 1
        if real is not None and real == k["real_panel"]:  # force a different position
            order = np.roll(order, 1)
            runs = [k["panel_runs"][i] for i in order]
            real = int(np.where(order == k["real_panel"] - 1)[0][0]) + 1
        item = k["item_id"].replace("LU_", "LS_")
        plots.render_lineup([plots.mark_table(states[x], k["spec"]) for x in runs], k["spec"], out / f"{item}.png")
        skey.append(dict(k, item_id=item, original=k["item_id"], real_panel=real, panel_runs=runs))
    json.dump(skey, open(out / "key.json", "w"), indent=1)

    truth, items = {}, []
    for sp in meta["specs"]:
        for rid in probe_runs:
            st = states[rid]
            if sp == "lc_loss":
                t = dict(train_last=st["train_loss"][-1], val_last=st["val_loss"][-1],
                         val_best=st["val_loss"].min(), val_best_epoch=int(st["val_loss"].argmin()) + 1)
            else:
                t = dict(train_last=st["train_acc"][-1], val_last=st["val_acc"][-1],
                         val_best=st["val_acc"].max(), val_best_epoch=int(st["val_acc"].argmax()) + 1)
            iid = f"probe_{rid}_{sp}"
            truth[iid] = dict(run_id=rid, spec=sp, failed=rid in f_test, **{a: float(b) for a, b in t.items()})
            items.append(dict(item_id=iid, run_id=rid, spec=sp))
    json.dump(items, open(RUN / "probe_items.json", "w"), indent=1)
    json.dump(truth, open(RUN / "probe_truth.json", "w"), indent=1)

    # check reproduction against the original single-plot image for one probe item per spec
    from PIL import Image
    for sp in meta["specs"]:
        rid = probe_runs[0]
        a = np.asarray(Image.open(RUN / "img" / f"{rid}_{sp}.png").convert("RGB")).astype(int)
        b = plots.render(plots.mark_table(states[rid], sp), sp).astype(int)
        print(f"reproduction check {sp}: max pixel diff {np.abs(a - b).max()}", flush=True)
    print(f"{len(skey)} shuffled lineups, {len(items)} probe items")


if __name__ == "__main__":
    main()
