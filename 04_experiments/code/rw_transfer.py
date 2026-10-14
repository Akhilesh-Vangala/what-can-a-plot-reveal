"""Step 3b — does the evidential map transfer to real models?

  python rw_transfer.py runs/confirm_synthetic runs/realworld   -> runs/realworld/transfer.json + TRANSFER.md

For every verified real-world case and every plot, the mark-table benchmark (B1) is rebuilt as in the
confirmatory run (same family, parameters and seed, trained on the same synthetic training models at the strongest
severity) and applied, unchanged, to the real models' plots:
  detection  AUC of the case's failure detector, real failure models vs. their healthy controls (same data);
  diagnosis  7-way prediction (prior-corrected argmax) counted correct if it names a condition measured as present.
The map's prediction for each (failure, plot) cell comes from the synthetic run: identifiable / detect-only / blind.
Caveat: count-based marks (confusion-matrix counts, reliability bin sizes) scale with test-set size, which differs
between real datasets and the synthetic world; row-normalized and rate-based plots are not affected.
"""
import json
import pathlib
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from plotreveal.benchmarks import _make
from plotreveal.seeds import seed_for
from score_confirm import Run, auc_of

TAG, SRC = "confirm", "synthetic"


def as_proportions(Z):
    """Count-based marks -> proportions, so test-set size cannot drive detection (confusion-matrix counts -> shares of
    the total; reliability bin sizes -> shares of all examples). Other plots are unchanged."""
    Z = {k: np.array(Z[k]) for k in Z.files}
    Z["cm"] = Z["cm"] / np.maximum(Z["cm"].sum(1, keepdims=True), 1e-9)
    n = Z["rel_n"].reshape(len(Z["rel_n"]), -1, 3).copy()
    cnt = np.where(n[:, :, 2] < 0, 0, n[:, :, 2])
    n[:, :, 2] = np.where(n[:, :, 2] < 0, -1, cnt / np.maximum(cnt.sum(1, keepdims=True), 1e-9))
    Z["rel_n"] = n.reshape(len(n), -1)
    return Z


def main(syn, rw, proportions=False):
    syn, rw = pathlib.Path(syn), pathlib.Path(rw)
    R = Run(syn)
    j = R.top
    F = np.load(syn / "features.npz")
    pos = {r: i for i, r in enumerate(F["ids"])}
    G = np.load(rw / "features.npz")
    if proportions:
        F, G = as_proportions(F), as_proportions(G)
    rpos = {r: i for i, r in enumerate(G["ids"])}
    bad = [sp for sp in R.meta["specs"] if F[sp].shape[1] != G[sp].shape[1]]
    if bad:  # e.g. different number of training epochs -> learning-curve vectors of different length
        raise SystemExit(f"feature length mismatch for {bad}: synthetic {[F[s].shape[1] for s in bad]} vs real "
                         f"{[G[s].shape[1] for s in bad]} — both must use the same plot settings (epochs)")
    truth = json.load(open(rw / "truth.json"))
    cases = {k: v for k, v in json.load(open(rw / "cases.json")).items() if v["verified"]}
    res, dia_res = json.load(open(syn / "results.json")), {}
    det = {(r["failure"], r["level"]): r for r in res["detection"]}
    dia = {r["level"]: r for r in res["diagnosis"]}
    split_of = {rid: s for m in R.meta["groups"].values() for rid, s in m.items()}
    healthy = [r for r, s in R.meta["groups"]["healthy|None"].items() if s == "train"]
    out, L = [], ["# Real-world transfer of the evidential map (B1, strongest severity)", "",
                  "| case | failure | plot | map label (synthetic) | real detection AUC | real diagnosis (names a present condition) |",
                  "|---|---|---|---|---|---|"]
    for sp in R.meta["specs"]:
        # 7-way diagnosis model, rebuilt as in the confirmatory run
        keys = [("healthy", None)] + [(f, j) for f in R.meta["failures"]]
        ids, y = [], []
        for c, k in enumerate(keys):
            g = R.meta["groups"][f"{k[0]}|{k[1]}"]
            ids += [r for r, s in g.items() if s == "train"]
            y += [c] * sum(1 for s in g.values() if s == "train")
        d = dia[j][sp]
        mdl = _make(d["best"], d["params"], seed_for(f"{TAG}:diag", SRC, j, sp)).fit(F[sp][[pos[r] for r in ids]], np.array(y))
        prior = np.array(d["prior"])
        classes = ["healthy"] + list(R.meta["failures"])
        for cid, cv in cases.items():
            f = cv["failure"]
            rids = [r for r, t in truth.items() if t["case"] == cid]
            Xr = G[sp][[rpos[r] for r in rids]]
            # detection: rebuild the (healthy vs f) detector
            dr = det[(f, j)][sp]
            fids = [r for r, s in R.meta["groups"][f"{f}|{j}"].items() if s == "train"]
            Xs = F[sp][[pos[r] for r in healthy + fids]]
            ys = np.r_[np.zeros(len(healthy)), np.ones(len(fids))]
            dm = _make(dr["best"], dr["params"], seed_for(f"{TAG}:det", SRC, f, j, sp)).fit(Xs, ys)
            yr = np.array([1 if truth[r]["condition"] == "failure" else 0 for r in rids])
            auc = float(roc_auc_score(yr, dm.predict_proba(Xr)[:, 1]))
            P = mdl.predict_proba(Xr) / np.maximum(prior, 1e-9)
            pred = [classes[int(np.argmax(p))] for p in P]
            fail_rows = [i for i, r in enumerate(rids) if truth[r]["condition"] == "failure"]
            named = float(np.mean([pred[i] in truth[rids[i]]["present"] for i in fail_rows]))
            names_planted = float(np.mean([pred[i] == f for i in fail_rows]))
            # synthetic map label for this cell
            sc, _, sids = R.det_scores(f, j, sp)
            a_syn = auc_of(sc, R.truth, sids)
            post = R.dia_post(j, sp)
            pr = {k: classes[int(np.argmax(v))] for k, v in post.items()}
            rec = np.mean([pr[k] == f for k in post if R.truth[k] == f])
            label = "identifiable" if a_syn >= .7 and rec >= .5 else ("detect-only" if a_syn >= .7 else "blind")
            out.append(dict(case=cid, failure=f, spec=sp, map_label=label, syn_auc=a_syn, syn_recall=float(rec),
                            real_auc=auc, real_names_present=named, real_names_planted=names_planted))
            L.append(f"| {cid} | {f} | {sp} | {label} ({a_syn:.2f}) | {auc:.2f} | {named:.0%} (planted {names_planted:.0%}) |")
    # agreement summary: does the map's detectability predict real detectability?
    lab = np.array([o["map_label"] != "blind" for o in out])
    real = np.array([o["real_auc"] >= .7 for o in out])
    agree = float(np.mean(lab == real))
    L += ["", f"Map says detectable vs. real detection AUC ≥ 0.70: agreement {agree:.0%} over {len(out)} case-plot pairs "
          f"(map detectable & real detectable {int((lab & real).sum())}, map blind & real blind {int((~lab & ~real).sum())}, "
          f"map detectable but real not {int((lab & ~real).sum())}, map blind but real detectable {int((~lab & real).sum())})."]
    suffix = "_prop" if proportions else ""
    json.dump(out, open(rw / f"transfer{suffix}.json", "w"), indent=1)
    (rw / f"TRANSFER{suffix.upper()}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[-1:]))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], proportions="--proportions" in sys.argv)
