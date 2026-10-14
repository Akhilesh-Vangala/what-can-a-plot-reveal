"""Score the design interventions (H7): plain vs evidence badge vs companion view vs both, same runs × plots × readers.

  python score_interventions.py runs/v4_synthetic      -> interventions.md, interventions_summary.json

"plain" answers come from the main v4 reader files (same held-out runs and plots); intervention answers from
responses_<model>_iv.jsonl. Metrics per reader × condition (strongest severity + healthy, 5 core plots, 7 states):
balanced accuracy; P(true state); stereotype = share of "overfitting" on HEALTHY loss curves; over-claiming index vs the
benchmark posterior of the plain plot (prior-corrected B1*, same item); confident-where-ambiguous; cannot-tell rate.
Paired bootstrap (over runs) CI for the change vs plain.
"""
import collections
import json
import pathlib
import sys

import numpy as np

CLASSES = ["healthy", "overfitting", "miscalibration", "label_noise", "class_imbalance", "data_leakage",
           "distribution_shift"]
CONDS = ["plain", "badge", "companion", "badge_companion"]


def probs(p):
    try:
        v = np.array([float(p.get(c, 0) or 0) for c in CLASSES])
    except (TypeError, ValueError, AttributeError):
        return None
    return v / v.sum() if v.sum() > 0 else None


def main(run):
    run = pathlib.Path(run)
    meta = json.load(open(run / "meta.json"))
    res = json.load(open(run / "results.json"))
    b2 = json.load(open(run / "b2_results.json"))
    top = len(meta["targets"]) - 1
    truth = {rid: k.split("|")[0] for k, m in meta["groups"].items() for rid in m}
    item1, item2 = json.load(open(run / "item_scores.json")), json.load(open(run / "b2_item_scores.json"))
    dia = [r for r in res["diagnosis"] if r["level"] == top][0]
    b2d = {r["spec"]: r for r in b2["diagnosis"] if r["level"] == top}
    post = {}
    for sp in meta["specs"]:
        use_b1 = dia[sp]["val_auc"] >= b2d[sp]["val_auc"]
        src = (item1 if use_b1 else item2).get(f"diag|{top}|{sp}")
        P = np.array(src["proba"]) / np.maximum(np.array(src["prior"]), 1e-9)
        P = P / P.sum(1, keepdims=True)
        for rid, p in zip(src["ids"], P):
            post[(rid, sp)] = p

    out, L = {}, ["# Design interventions — readers (strongest severity + healthy; 5 core plots; 7 states, chance 0.14)", ""]
    for fp in sorted(run.glob("responses_*_iv.jsonl")):
        model = fp.stem.replace("responses_", "").replace("_iv", "")
        iv = [json.loads(x) for x in open(fp)]
        keys = {(r["run_id"], r["spec"]) for r in iv}
        plain = [dict(json.loads(x), condition="plain") for x in open(run / f"responses_{model}.jsonl")]
        rows = [r for r in plain if r.get("task") == "diagnosis" and (r["run_id"], r["spec"]) in keys] + iv
        per = collections.defaultdict(list)
        for r in rows:
            v = probs(r.get("parsed"))
            if v is None:
                continue
            t = truth[r["run_id"]]
            p = post.get((r["run_id"], r["spec"]))
            per[r["condition"]].append(dict(run=r["run_id"], spec=r["spec"], truth=t, ans=CLASSES[int(v.argmax())],
                                            ptrue=v[CLASSES.index(t)], top=v.max(),
                                            over=(v.max() - p[v.argmax()]) if p is not None else np.nan,
                                            amb=(v.max() >= 0.7 and p.max() < 0.5) if p is not None else np.nan,
                                            ct=bool(r["parsed"].get("cannot_tell"))))

        def metrics(items):
            by = collections.defaultdict(list)
            for it in items:
                by[it["truth"]].append(it["ans"] == it["truth"])
            hl = [it["ans"] == "overfitting" for it in items if it["truth"] == "healthy" and it["spec"] == "lc_loss"]
            return dict(balanced_acc=float(np.mean([np.mean(by[c]) for c in CLASSES if by[c]])),
                        p_true=float(np.mean([it["ptrue"] for it in items])),
                        stereotype_lc_loss=float(np.mean(hl)) if hl else float("nan"),
                        over_claim=float(np.nanmean([it["over"] for it in items])),
                        confident_where_ambiguous=float(np.nanmean([it["amb"] for it in items])),
                        cannot_tell=float(np.mean([it["ct"] for it in items])), n=len(items))

        rng = np.random.default_rng(0)
        runs = sorted({it["run"] for it in per["plain"]})
        res_m = {}
        for c in CONDS:
            if not per[c]:
                continue
            m = metrics(per[c])
            if c != "plain":  # paired bootstrap over runs: change in balanced accuracy and over-claiming vs plain
                byrun = {k: collections.defaultdict(list) for k in ("plain", c)}
                for k in ("plain", c):
                    for it in per[k]:
                        byrun[k][it["run"]].append(it)
                d_acc, d_over = [], []
                for _ in range(500):
                    smp = rng.choice(runs, len(runs))
                    a = [it for r in smp for it in byrun["plain"][r]]
                    b = [it for r in smp for it in byrun[c][r]]
                    ma, mb = metrics(a), metrics(b)
                    d_acc.append(mb["balanced_acc"] - ma["balanced_acc"])
                    d_over.append(mb["over_claim"] - ma["over_claim"])
                m["delta_acc_ci"] = [float(np.percentile(d_acc, 2.5)), float(np.percentile(d_acc, 97.5))]
                m["delta_over_ci"] = [float(np.percentile(d_over, 2.5)), float(np.percentile(d_over, 97.5))]
            res_m[c] = m
        out[model] = res_m
        L += [f"## {model}", "", "| condition | n | balanced acc | Δ vs plain [95% CI] | P(true) | 'overfitting' on healthy loss curves | "
              "over-claim | Δ over-claim [95% CI] | confident-where-ambiguous | cannot-tell |", "|---|---|---|---|---|---|---|---|---|---|"]
        for c, m in res_m.items():
            da = (f"{m['balanced_acc'] - res_m['plain']['balanced_acc']:+.3f} "
                  f"[{m['delta_acc_ci'][0]:+.3f}, {m['delta_acc_ci'][1]:+.3f}]") if c != "plain" else "–"
            do = (f"{m['over_claim'] - res_m['plain']['over_claim']:+.3f} "
                  f"[{m['delta_over_ci'][0]:+.3f}, {m['delta_over_ci'][1]:+.3f}]") if c != "plain" else "–"
            L.append(f"| {c} | {m['n']} | {m['balanced_acc']:.3f} | {da} | {m['p_true']:.3f} | {m['stereotype_lc_loss']:.0%} | "
                     f"{m['over_claim']:.3f} | {do} | {m['confident_where_ambiguous']:.2f} | {m['cannot_tell']:.2f} |")
        L.append("")
    (run / "interventions.md").write_text("\n".join(L) + "\n")
    json.dump(out, open(run / "interventions_summary.json", "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1])
