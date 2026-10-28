"""Which blind cells are blind by construction (no estimator and no reader can ever detect them)?

  python certify_blind.py [N_PER_FAILURE]   -> runs/<run>/CERTIFIED_BLIND.md (+ certified_blind.json)

Each failing model is generated from a healthy configuration plus one planted change. Re-training the same
configuration without the change gives its healthy twin, which is an ordinary draw from the healthy population. If the
plotted values (mark table) of the failing model equal its twin's exactly, for every model, then the plot has the
same distribution under the failure as under health: detection AUC is 0.5 for any reader or benchmark. This turns
an empirical "blind" label into a proof for those cells; the remaining blind cells stay empirical (see
blind_sensitivity.py).
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
RUNS = {"confirm_synthetic": "synthetic", "confirm_magic": "magic"}


def twin_diff(rid, meta, source):
    f, j, i = re.match(r"m_(.+)_L(\d)_(\d+)$", rid).groups()
    j, i = int(j), int(i)
    base = healthy_config("confirm:main", f, j, i, epochs=100, source=source)
    sick, twin = train_run(apply_failure(base, f, meta["severities"][f][j])), train_run(base)
    if abs(sick["deploy_acc"] - meta["deploy_acc"][rid]) > 1e-9:
        raise SystemExit(f"{rid}: re-trained model differs from the study run")
    out = {}
    for sp in meta["specs"]:
        a = plots.marks_vector(plots.mark_table(sick, sp), sp)
        b = plots.marks_vector(plots.mark_table(twin, sp), sp)
        out[sp] = float(np.max(np.abs(a - b))) if a.shape == b.shape else float("inf")
    return f, out


def main(n):
    for run, source in RUNS.items():
        meta = json.load(open(ROOT / "runs" / run / "meta.json"))
        top = len(meta["severities"][meta["failures"][0]]) - 1
        rng = np.random.default_rng(0)
        ids = []
        for f in meta["failures"]:
            pool = sorted(r for r in meta["deploy_acc"] if r.startswith(f"m_{f}_L{top}_"))
            ids += list(rng.choice(pool, min(n, len(pool)), replace=False))
        res = Parallel(n_jobs=8)(delayed(twin_diff)(r, meta, source) for r in ids)
        worst = {f: {sp: max(d[sp] for g, d in res if g == f) for sp in meta["specs"]} for f in meta["failures"]}
        cert = sorted([f, sp] for f in worst for sp in worst[f] if worst[f][sp] == 0.0)
        L = [f"# Blind by construction — {run}", "",
             f"{n} failing models per failure at the strongest severity, each re-trained next to its healthy twin "
             "(same configuration without the planted change; re-trained models match the study exactly).",
             "Entry = largest absolute difference in any plotted value between a failing model and its twin. "
             "**0** in every model = the plot is identical, so no reader or benchmark can detect the failure from it.", "",
             "| failure | " + " | ".join(meta["specs"]) + " |", "|---|" + "---|" * len(meta["specs"])]
        for f in meta["failures"]:
            L.append(f"| {f} | " + " | ".join("**0**" if worst[f][sp] == 0 else f"{worst[f][sp]:.3g}" for sp in meta["specs"]) + " |")
        L += ["", f"Certified blind cells ({len(cert)}): " + ", ".join(f"{f} on {sp}" for f, sp in cert)]
        (ROOT / "runs" / run / "CERTIFIED_BLIND.md").write_text("\n".join(L) + "\n")
        json.dump(dict(n_per_failure=n, max_diff=worst, certified=cert), open(ROOT / "runs" / run / "certified_blind.json", "w"), indent=1)
        print("\n".join(L), "\n")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
