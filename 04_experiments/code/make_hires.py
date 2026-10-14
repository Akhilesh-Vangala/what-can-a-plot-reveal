"""Resolution experiment (rendering-level, gap 2.6): re-render the probe plots of a v2 run at 2× resolution.

  python make_hires.py runs/overfit_proto_v2   -> img_hires/ + reader_items_hires.json (probe + diagnosis tasks)
Same mark tables, same figure size, dpi 200 instead of 100 (every mark and label gets 4× the pixels).
Answers on the standard images already exist, so the comparison is item-by-item.
"""
import json
import pathlib
import sys

from joblib import Parallel, delayed

from plotreveal import plots
from plotreveal.generator import train_run

RUN = pathlib.Path(sys.argv[1])
manifest = {m["run_id"]: m for m in map(json.loads, open(RUN / "manifest.jsonl")) if "run_id" in m}
probe = json.load(open(RUN / "probe_items.json"))
runs = sorted({p["run_id"] for p in probe})
cfgs = [{k: v for k, v in manifest[r].items() if k not in ("role", "run_id", "level", "idx")} for r in runs]
states = dict(zip(runs, Parallel(n_jobs=-1)(delayed(train_run)(c) for c in cfgs)))
out = RUN / "img_hires"
out.mkdir(exist_ok=True)
items = []
for p in probe:
    rid, sp = p["run_id"], p["spec"]
    img = f"img_hires/{rid}_{sp}.png"
    plots.render(plots.mark_table(states[rid], sp), sp, path=RUN / img, dpi=200)
    items.append(dict(item_id=f"hires_probe_{rid}_{sp}", run_id=rid, task="probe", spec=sp, image=img))
    items.append(dict(item_id=f"hires_diag_{rid}_{sp}", run_id=rid, task="diagnosis", spec=sp, image=img))
json.dump(items, open(RUN / "reader_items_hires.json", "w"), indent=0)
print(len(items), "items;", len(list(out.glob("*.png"))), "hi-res images")
