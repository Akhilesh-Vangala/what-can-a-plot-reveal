"""Secondary analyses named in the analysis plan, on the confirmatory runs.

  python score_secondary.py runs/confirm_synthetic runs/confirm_magic   -> <first run>/SECONDARY_RESULTS.md (+ .json)

1. No-plot (prior-only) baseline: each reader answered the same question with the plot hidden. We report how often its
   answer WITH the plot equals its answer WITHOUT it (same plot type), and the balanced accuracy of the no-plot answer.
2. Over-claiming: reader probability on its top answer minus the benchmark (B1*, prior-corrected) posterior on that
   same answer for the same item (as in score_v4.py); share of confident answers (≥ 0.7) where the benchmark is
   uncertain (top posterior < 0.5); hedging AUC = AUC of "cannot tell" against benchmark entropy (0.5 = unrelated).
3. Regression of reader correctness ~ plot + failure + reader. The plan named a mixed-effects logistic model with
   random intercepts for run and reader; with ~1,500 run levels the variational fit did not finish on a laptop, so we
   fit the standard alternative: a logistic GEE clustered by model run (exchangeable correlation), readers as fixed
   effects. Coefficients are population-averaged log-odds with cluster-robust standard errors.
"""
import collections
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
import statsmodels.formula.api as smf

from score_confirm import CLASSES, Run, bal_acc, reader_vec


def readers_of(R):
    return {k: v for k, v in R.responses("responses_*.jsonl").items()
            if not any(k.endswith(s) for s in ("_iv", "_variants", "_controls", "_hires", "_probe"))}


def no_plot(R, L, out):
    L += [f"### No-plot baseline — {R.p.name}", "",
          "| reader | answer unchanged by seeing the plot | no-plot balanced acc | with-plot balanced acc |", "|---|---|---|---|"]
    for name, rows in readers_of(R).items():
        prior = {r["spec"]: reader_vec(r) for r in rows if r.get("task") == "prior_only"}
        prior = {s: CLASSES[int(np.argmax(v))] for s, v in prior.items() if v is not None}
        if not prior:
            L.append(f"| {name} | (no no-plot items in this set) | | |")
            continue
        same, pred_p, pred_r, tr = [], {}, {}, {}
        for r in rows:
            if r.get("task") != "diagnosis" or r["spec"] not in prior:
                continue
            v = reader_vec(r)
            if v is None:
                continue
            k = (r["run_id"], r["spec"])
            pred_r[k], pred_p[k], tr[k] = CLASSES[int(np.argmax(v))], prior[r["spec"]], R.truth[r["run_id"]]
            same.append(pred_r[k] == pred_p[k])
        ks = list(tr)
        a_p, a_r = bal_acc(pred_p, tr, ks, R.classes), bal_acc(pred_r, tr, ks, R.classes)
        out[f"{R.p.name}|{name}|no_plot"] = dict(unchanged=float(np.mean(same)), no_plot_acc=a_p, with_plot_acc=a_r)
        L.append(f"| {name} | {np.mean(same):.0%} | {a_p:.3f} | {a_r:.3f} |")
    L.append("")


def over_claim(R, L, out):
    L += [f"### Over-claiming and hedging — {R.p.name}", "",
          "| reader | items | over-claim index | confident where benchmark uncertain | cannot-tell rate | hedging AUC |",
          "|---|---|---|---|---|---|"]
    posts = {j: {sp: R.dia_post(j, sp) for sp in R.meta["specs"]} for j in range(R.top + 1)}
    for name, rows in readers_of(R).items():
        over, amb, ct, ent = [], [], [], []
        for r in rows:
            if r.get("task") != "diagnosis":
                continue
            p_raw = r.get("parsed")
            v = reader_vec(r)
            j = R.level.get(r["run_id"])
            j = R.top if j is None else j
            p = posts[j][r["spec"]].get(r["run_id"])
            if p is None or not isinstance(p_raw, dict):
                continue
            ct.append(bool(p_raw.get("cannot_tell")))
            ent.append(float(-(p * np.log(p + 1e-12)).sum()))
            if v is not None:
                over.append(v.max() - p[int(np.argmax(v))])
                amb.append(v.max() >= 0.7 and p.max() < 0.5)
        hed = roc_auc_score(ct, ent) if len(set(ct)) == 2 else float("nan")
        out[f"{R.p.name}|{name}|over_claim"] = dict(n=len(over), over_claim=float(np.mean(over)), confident_ambiguous=float(np.mean(amb)),
                                                     cannot_tell=float(np.mean(ct)), hedging_auc=hed)
        L.append(f"| {name} | {len(over)} | {np.mean(over):+.2f} | {np.mean(amb):.0%} | {np.mean(ct):.0%} | {hed:.2f} |")
    L.append("")


def mixed(R, L, out):
    recs = []
    for name, rows in readers_of(R).items():
        for r in rows:
            if r.get("task") != "diagnosis":
                continue
            v = reader_vec(r)
            if v is None:
                continue
            recs.append(dict(correct=int(CLASSES[int(np.argmax(v))] == R.truth[r["run_id"]]), plot=r["spec"],
                             failure=R.truth[r["run_id"]], run=r["run_id"], reader=name.replace("_core", "")))
    d = pd.DataFrame(recs)
    d["plot"] = pd.Categorical(d["plot"], categories=["lc_loss"] + sorted(set(d["plot"]) - {"lc_loss"}))
    d["failure"] = pd.Categorical(d["failure"], categories=["healthy"] + sorted(set(d["failure"]) - {"healthy"}))
    m = smf.gee("correct ~ C(plot) + C(failure) + C(reader)", groups="run", data=d,
                family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
    fe = pd.DataFrame({"coef": m.params, "sd": m.bse})
    rho = float(m.model.cov_struct.dep_params)
    L += [f"### Regression of reader correctness (logistic GEE, clustered by model run) — {R.p.name}", "",
          f"{len(d):,} reader answers, {d['run'].nunique():,} model runs, {d['reader'].nunique()} readers; within-run "
          f"correlation {rho:.2f}. Log-odds of a correct answer; reference = loss curves, healthy model, first reader.",
          "", "| term | log-odds | SE | 95% CI excludes 0 |", "|---|---|---|---|"]
    for t, row in fe.iterrows():
        lo, hi = row.coef - 1.96 * row.sd, row.coef + 1.96 * row.sd
        if abs(row.coef) > 10:  # separation: no correct answers in this category, the estimate diverges
            L.append(f"| {t} | never correct (no estimate) | — | — |")
        else:
            L.append(f"| {t} | {row.coef:+.2f} | {row.sd:.2f} | {'yes' if lo > 0 or hi < 0 else 'no'} |")
    L.append("")
    vc = {"within_run_correlation": rho}
    out[f"{R.p.name}|mixed"] = dict(n=len(d), fixed={t: [float(a), float(b)] for t, (a, b) in fe.iterrows()},
                                    random_sd=vc)


def main(paths):
    runs = [Run(p) for p in paths]
    L, out = ["# Secondary analyses (analysis plan §4–5)", ""], {}
    for R in runs:
        no_plot(R, L, out)
        over_claim(R, L, out)
        mixed(R, L, out)
    (runs[0].p / "SECONDARY_RESULTS.md").write_text("\n".join(L) + "\n")
    json.dump(out, open(runs[0].p / "secondary_results.json", "w"), indent=1, default=float)
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1:])
