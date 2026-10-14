"""Positive control + value probe (credibility checks for the confirmatory reader results; exploratory, not preregistered).

  python make_positive.py positive --out runs/positive [--n 50] [--jobs 24]
      Blatant cases: each failure at textbook strength (far beyond the calibrated levels), shown on its textbook plot,
      against healthy models on the same plot types. Both data sources. Same diagnosis prompt as the main study.
      If readers cannot name THESE, the setup (not the subtlety of the cases) is the problem.
  python make_positive.py probe --run runs/confirm_synthetic [--per-failure 25 --healthy 100] [--jobs 24]
      Value probe on confirmatory models: read numbers off loss / accuracy curves (perception check). The runs are
      re-trained from manifest.jsonl (deterministic; verified against meta.json deploy_acc) to get the true values.
"""
import argparse
import json
import pathlib
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

from plotreveal import plots
from plotreveal.seeds import rng_for
from plotreveal.world import FAILURES, apply_failure, healthy_config, train_run

T0 = time.time()
EPOCHS = 100
# Textbook strength (calibrated confirmatory levels, strongest: overfitting 0.25-0.34, miscalibration 0.04-0.07,
# label noise 0.04-0.06, imbalance 0.28, leakage 0.14-0.30, shift 0.25-0.82):
SEVERITY = dict(overfitting=1.0,         # 15% of the data, 4x width, no regularisation
                miscalibration=1.0,      # logits x5 (x3 looked healthy-like on accurate synthetic models)
                label_noise=0.5,         # 20% of labels flipped
                class_imbalance=0.9,     # positive rate ~0.5 -> ~0.07
                data_leakage=1.0,        # label-encoding feature, strength 3
                distribution_shift=1.0)  # test features shifted by 3 SD
# Severities fixed 2026-10-13 after visual inspection of rendered examples, BEFORE any reader saw these items.
# Pass rule (fixed now): readers "can do the task" if, on >= 4 of the 5 failures with a textbook single-plot signature
# (all but label noise, which shows only as a lower accuracy plateau), detection AUC >= 0.80 AND the failure is named
# (argmax) in >= 50% of its items. Label noise is reported separately.
PRIMARY = ("overfitting", "miscalibration", "class_imbalance", "data_leakage", "distribution_shift")
# the plot on which each failure is textbook-visible
CANONICAL = dict(overfitting="lc_loss", miscalibration="rel", label_noise="lc_acc", class_imbalance="cm",
                 data_leakage="roc", distribution_shift="dist")


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def blatancy(st):
    """Objective size of each failure's signature in the state (reported next to reader accuracy)."""
    vl = np.asarray(st["val_loss"])
    return dict(val_loss_rise=float(vl[-1] - vl.min()), train_val_gap=float(st["val_loss"][-1] - st["train_loss"][-1]),
                ece=st["test_ece"], pos_rate=st["test_pos_rate"], test_acc=st["test_acc"],
                test_auc=float(roc_auc_score(st["test_y"], st["test_p"])), deploy_acc=st["deploy_acc"],
                train_acc_last=float(st["train_acc"][-1]),
                score_shift=float(abs(np.mean(st["test_p"]) - np.mean(st["val_p"]))))


def _one(rid, cfg, specs, img):
    st = train_run(cfg)
    for sp in specs:
        plots.render(plots.mark_table(st, sp), sp, path=img / f"{rid}_{sp}.png")
    return rid, blatancy(st)


def positive(out, n, jobs):
    img = out / "img"
    img.mkdir(parents=True, exist_ok=True)
    tasks = []
    for src in ("synthetic", "magic"):
        for i in range(n):
            tasks.append((f"{src}_healthy_{i}", healthy_config("positive", src, "healthy", i, epochs=EPOCHS, source=src),
                          list(CANONICAL.values()), src, "healthy"))
        for f in FAILURES:
            for i in range(n):
                cfg = apply_failure(healthy_config("positive", src, f, i, epochs=EPOCHS, source=src), f, SEVERITY[f])
                tasks.append((f"{src}_{f}_{i}", cfg, [CANONICAL[f]], src, f))
    log(f"training + rendering {len(tasks)} runs")
    res = dict(Parallel(n_jobs=jobs, batch_size=4)(delayed(_one)(rid, c, sp, img) for rid, c, sp, _, _ in tasks))
    truth = {rid: dict(source=src, failure=f, severity=SEVERITY.get(f, 0.0), specs=sp, **res[rid])
             for rid, _, sp, src, f in tasks}
    items = [dict(item_id=f"{rid}_{sp}", run_id=rid, task="diagnosis", spec=sp, image=f"img/{rid}_{sp}.png")
             for rid, _, specs, _, _ in tasks for sp in specs]
    json.dump(truth, open(out / "truth.json", "w"), indent=0)
    json.dump(items, open(out / "reader_items.json", "w"), indent=0)
    json.dump(dict(severity=SEVERITY, canonical=CANONICAL, primary=PRIMARY, n=n), open(out / "design.json", "w"), indent=1)
    for src in ("synthetic", "magic"):
        for f in ("healthy",) + FAILURES:
            v = [t for t in truth.values() if t["source"] == src and t["failure"] == f]
            log(f"{src:9s} {f:18s} n={len(v)} " + " ".join(
                f"{k}={np.median([t[k] for t in v]):.3f}" for k in ("val_loss_rise", "train_val_gap", "ece", "pos_rate",
                                                                   "test_acc", "test_auc", "score_shift")))
    log(f"done: {len(items)} reader items → {out}")


def probe(run, per_failure, n_healthy, jobs):
    meta = json.load(open(run / "meta.json"))
    core = json.load(open(run / "reader_items_core.json"))
    runs = list(dict.fromkeys(it["run_id"] for it in core))  # 300 healthy + 100 per failure (strongest level)
    truth_of = {r: k.split("|")[0] for k, m in meta["groups"].items() for r in m}
    r = rng_for("probe", run.name)
    pick = list(r.choice([x for x in runs if truth_of[x] == "healthy"], n_healthy, replace=False))
    for f in FAILURES:
        pick += list(r.choice([x for x in runs if truth_of[x] == f], per_failure, replace=False))
    want = set(pick)
    cfg = {}
    with open(run / "manifest.jsonl") as fh:
        for line in fh:
            m = json.loads(line)
            if m.get("role") == "main" and m["run_id"] in want:
                cfg[m["run_id"]] = {k: v for k, v in m.items() if k not in ("role", "run_id", "level")}
    log(f"re-training {len(pick)} confirmatory runs from the manifest")
    states = Parallel(n_jobs=jobs, batch_size=4)(delayed(train_run)(cfg[x]) for x in pick)
    dev = [abs(s["deploy_acc"] - meta["deploy_acc"][x]) for s, x in zip(states, pick)]
    log(f"reproducibility: max |deploy_acc - stored| = {max(dev):.5f} (must be ~0)")
    if max(dev) > 0.01:
        raise SystemExit("re-trained runs do not match the confirmatory runs — not writing probe items")
    truth, items = {}, []
    for x, s in zip(pick, states):
        for sp, (tr, va, best) in (("lc_loss", ("train_loss", "val_loss", np.argmin)),
                                   ("lc_acc", ("train_acc", "val_acc", np.argmax))):
            v = np.asarray(s[va])
            e = int(best(v))
            truth[f"{x}_{sp}_probe"] = dict(run_id=x, spec=sp, failure=truth_of[x], train_last=float(s[tr][-1]),
                                            val_last=float(v[-1]), val_best=float(v[e]), val_best_epoch=e + 1)
            items.append(dict(item_id=f"{x}_{sp}_probe", run_id=x, task="probe", spec=sp, image=f"img/{x}_{sp}.png"))
    json.dump(truth, open(run / "probe_truth.json", "w"), indent=0)
    json.dump(items, open(run / "reader_items_probe.json", "w"), indent=0)
    log(f"done: {len(items)} probe items → {run}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["positive", "probe"])
    ap.add_argument("--out", default="runs/positive")
    ap.add_argument("--run", default="runs/confirm_synthetic")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--per-failure", type=int, default=25)
    ap.add_argument("--healthy", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args()
    if a.mode == "positive":
        positive(pathlib.Path(a.out), a.n, a.jobs)
    else:
        probe(pathlib.Path(a.run), a.per_failure, a.healthy, a.jobs)
