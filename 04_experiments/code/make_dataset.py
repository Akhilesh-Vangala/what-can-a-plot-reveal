"""Package the rendered plots that the AI readers diagnosed as a public dataset.

  python make_dataset.py runs/confirm_synthetic   -> runs/confirm_synthetic/plotreveal_confirm_synthetic.zip

One zip per run: every image of the main and core reader item sets (reader_items.json, reader_items_core.json),
labels.csv and a README. Labels come from meta.json (true state and severity level of each model). in_core = the
image is in the held-out core set on which the 7-way, binary and 2AFC reader results are reported. Plots are
regenerated exactly from seeds by run_v4.py, so the zip is a convenience copy, not the only source.
"""
import csv
import io
import json
import pathlib
import sys
import zipfile

PLOTS = dict(lc_loss="loss curves (train/validation)", lc_acc="accuracy curves (train/validation)",
             rel="reliability diagram", rel_n="reliability diagram with bin sizes", cm="confusion matrix (counts)",
             cm_norm="confusion matrix (row-normalized)", roc="ROC curve", pr="precision-recall curve",
             dist="validation vs test score histogram")
SOURCE = dict(confirm_synthetic="synthetic tabular task (sklearn make_classification)",
              confirm_magic="MagicTelescope (OpenML, 19,020 events)")

README = """# What Can a Plot Reveal? Rendered diagnostic plots ({run})

Data source: {source}. {n} PNG images of standard diagnostic plots of small binary classifiers (two-layer MLPs), each
model either healthy or with exactly one planted failure: overfitting, miscalibration, label noise, class imbalance,
data leakage, distribution shift. These are the plots diagnosed by the AI readers in the study.

labels.csv columns:
  image           path inside this zip
  model_id        trained model (several plots per model)
  plot            plot type: {plots}
  state           healthy or the planted failure
  level           severity level 0-2 of the failure (empty for healthy)
  in_core         1 if the image is in the held-out core set used for the 7-way, binary and 2AFC reader results

Every plot is regenerated exactly from seeds by the code in the repository (04_experiments/code/run_v4.py).
Repository: https://github.com/Akhilesh-Vangala/what-can-a-plot-reveal
"""


def main(path):
    R = pathlib.Path(path)
    meta = json.load(open(R / "meta.json"))
    truth = {r: k.split("|") for k, m in meta["groups"].items() for r in m}
    core_items = json.load(open(R / "reader_items_core.json"))
    core = {it["image"] for it in core_items}
    items = [it for it in json.load(open(R / "reader_items.json")) + core_items if it.get("image")]  # no-image prior items dropped
    images = sorted({it["image"]: it for it in items}.values(), key=lambda it: it["image"])
    missing = [it["image"] for it in images if not (R / it["image"]).exists()]
    if missing:
        sys.exit(f"{len(missing)} images missing, e.g. {missing[0]}")
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["image", "model_id", "plot", "state", "level", "in_core"])
    for it in images:
        state, level = truth[it["run_id"]]
        w.writerow([it["image"], it["run_id"], it["spec"], state, "" if level == "None" else level, int(it["image"] in core)])
    out = R / f"plotreveal_{R.name}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        z.writestr("README.md", README.format(run=R.name, source=SOURCE.get(R.name, R.name), n=len(images),
                                              plots="; ".join(f"{k} = {v}" for k, v in PLOTS.items())))
        z.writestr("labels.csv", buf.getvalue())
        for it in images:
            z.write(R / it["image"], it["image"])
    print(f"wrote {out}: {len(images)} images, {sum(it['image'] in core for it in images)} in the core set, "
          f"{out.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main(sys.argv[1])
