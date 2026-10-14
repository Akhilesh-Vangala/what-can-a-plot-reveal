"""Export a run's benchmark + reader results for the D3 explorer (04_experiments/explorer).

  python export_explorer.py runs/multi_v3 v3          -> 04_experiments/explorer/data_v3.js
  python export_explorer.py runs/v4_synthetic v4syn

Benchmark per cell = B1* (validation-best of mark-table / image model); diagnosis = balanced, prior-corrected where
available. Works for the v3 file layout (b2_multi_results.json) and v4 (b2_results.json).
"""
import collections
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
ALL = ["healthy", "overfitting", "miscalibration", "label_noise", "class_imbalance", "data_leakage", "distribution_shift"]


def star(a, b):
    return a if b is None or a["val_auc"] >= b["val_auc"] else b


def main(run, tag):
    run = pathlib.Path(run)
    meta = json.load(open(run / "meta.json"))
    res = json.load(open(run / "results.json"))
    b2f = run / ("b2_results.json" if (run / "b2_results.json").exists() else "b2_multi_results.json")
    b2 = json.load(open(b2f)) if b2f.exists() else {"detection": [], "diagnosis": []}
    specs, fails = meta["specs"], meta["failures"]
    classes = ["healthy"] + fails
    b2det = {(r["failure"], r["level"], r["spec"]): r for r in b2["detection"]}
    b2dia = {(r["level"], r["spec"]): r for r in b2["diagnosis"]}
    cells, confusions, bench_bal = [], {}, {}
    for r in res["detection"]:
        f, j = r["failure"], r["level"]
        drow = [d for d in res["diagnosis"] if d["level"] == j][0]
        for sp in specs:
            d = star(r[sp], b2det.get((f, j, sp)))
            g = star(drow[sp], b2dia.get((j, sp)))
            c = np.array(g["confusion"], dtype=float)
            rec = c[classes.index(f), classes.index(f)] / max(c[classes.index(f)].sum(), 1)
            cells.append(dict(failure=f, plot=sp, level=j, detect=round(d["test_auc"], 3), diagnose=round(float(rec), 3),
                              label="identifiable" if d["test_auc"] >= 0.7 and rec >= 0.5 else
                              ("detect-only" if d["test_auc"] >= 0.7 else "neither")))
    for d in res["diagnosis"]:
        for sp in specs:
            g = star(d[sp], b2dia.get((d["level"], sp)))
            c = np.array(g["confusion"], dtype=float)
            confusions[f"{d['level']}|{sp}"] = (c / np.maximum(c.sum(1, keepdims=True), 1)).round(3).tolist()
            bench_bal[f"{d['level']}|{sp}"] = round(float(np.mean(c.diagonal() / np.maximum(c.sum(1), 1))), 3)

    truth = {rid: k.split("|")[0] for k, m in meta["groups"].items() for rid in m}
    readers = {}
    for fp in sorted(run.glob("responses_*.jsonl")):
        rows = [json.loads(x) for x in open(fp)]
        if not rows:
            continue
        name = rows[0]["model"].split("/")[-1]
        suffix = fp.stem.replace("responses_", "").replace(name, "").strip("_")
        if suffix in ("iv", "hires", "controls", "probe", "extract", "variants"):
            continue  # interventions / resolution files are shown elsewhere
        name = name + (f" ({suffix})" if suffix else "")
        conf = collections.defaultdict(lambda: np.zeros((len(classes), len(classes) + 1)))
        for r in rows:
            if r.get("task") != "diagnosis" or not isinstance(r.get("parsed"), dict) or r["spec"] not in specs:
                continue
            try:
                v = np.array([float(r["parsed"].get(c, 0) or 0) for c in ALL])
            except (TypeError, ValueError):
                continue
            if v.sum() <= 0:
                continue
            ans = ALL[int(v.argmax())]
            col = classes.index(ans) if ans in classes else len(classes)  # last column = a state not planted here
            conf[r["spec"]][classes.index(truth[r["run_id"]]), col] += 1
        readers[name] = {sp: dict(confusion=(c / np.maximum(c.sum(1, keepdims=True), 1)).round(3).tolist(),
                                  balanced_acc=round(float(np.mean(c.diagonal()[:len(classes)] /
                                                                   np.maximum(c.sum(1), 1))), 3))
                         for sp, c in conf.items()}
    data = dict(tag=tag, source=meta.get("source", "synthetic"), failures=fails, classes=classes, plots=specs,
                levels=meta["targets"], cells=cells, confusions=confusions, bench_balanced=bench_bal, readers=readers)
    out = ROOT / "04_experiments" / "explorer" / f"data_{tag}.js"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"window.EXPLORER_DATA = window.EXPLORER_DATA || {{}};\nwindow.EXPLORER_DATA[{json.dumps(tag)}] = "
                   + json.dumps(data) + ";\n")
    print(f"wrote {out}: {len(cells)} cells, {len(readers)} readers")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
