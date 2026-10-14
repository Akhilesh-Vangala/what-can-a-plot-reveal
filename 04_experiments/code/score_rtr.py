"""Read-then-reason, step 3 (local): score extraction fidelity and rule diagnoses from readers' readings.

  python score_rtr.py runs/confirm_synthetic runs/positive runs/realworld   -> runs/confirm_synthetic/RTR_RESULTS.md (+ .json)

Rule: per plot type, logistic regression (standardized, balanced classes) on rtr.features of EXACT values from the
synthetic training models (healthy + strongest level, train split). It never sees reader data or test models.
Synthetic core set: rule on readers' readings vs. rule on exact values of the same models (ceiling) vs. the readers'
direct diagnosis (H5). Positive control and real world: same metrics and pass rule as score_positive / score_realworld,
with the rule's probabilities in place of the reader's.
"""
import collections
import json
import pathlib
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import rtr
from make_positive import CANONICAL, PRIMARY
from score_confirm import CLASSES


def bal(y, p):
    return float(np.mean([np.mean(p[y == c] == c) for c in np.unique(y)]))


def responses(run, suffix="_extract"):
    out = {}
    for fp in sorted(run.glob(f"responses_*{suffix}.jsonl")):
        out[fp.stem.replace("responses_", "").replace(suffix, "")] = {json.loads(x)["item_id"]: json.loads(x) for x in open(fp)}
    return out


def main(syn, pos, rw):
    syn, pos, rw = map(pathlib.Path, (syn, pos, rw))
    V = json.load(open(syn / "rtr_values.json"))
    specs = ["lc_loss", "lc_acc", "rel", "rel_n", "cm", "cm_norm", "roc", "pr", "dist"]
    rules = {}
    for sp in specs:
        tr = [(rtr.features(v["values"][sp], sp), CLASSES.index(v["failure"])) for v in V.values() if v["split"] == "train"]
        X = np.nan_to_num(np.array([a for a, _ in tr], float))
        rules[sp] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced")).fit(X, [b for _, b in tr])
    L, out = ["# Read-then-reason: readers read values, a fixed rule diagnoses (exploratory)", ""], {}

    def rule_proba(sp, d):
        f = rtr.features(d, sp)
        if f is None or any(np.isnan(f)):
            return None
        q = rules[sp].predict_proba(np.array([f]))[0]
        full = np.zeros(len(CLASSES))
        full[rules[sp].classes_] = q  # map the rule's class columns onto the fixed 7-state order
        return full

    # ---------------- synthetic core
    ext = responses(syn)
    core_specs = ("lc_loss", "rel", "cm", "roc", "dist")
    L += ["## Synthetic core set (strongest level, held-out models)", "",
          "| reader | " + " | ".join(core_specs) + " | all plots | readable answers |", "|---|" + "---|" * (len(core_specs) + 2)]
    ceil = {}
    for sp in core_specs:
        ids = [k for k, v in V.items() if v["split"] == "test"]
        y = np.array([CLASSES.index(V[k]["failure"]) for k in ids])
        p = rules[sp].predict(np.nan_to_num(np.array([rtr.features(V[k]["values"][sp], sp) for k in ids], float)))
        ceil[sp] = bal(y, p)
    L.append("| exact values (ceiling) | " + " | ".join(f"{ceil[s]:.3f}" for s in core_specs) + f" | {np.mean(list(ceil.values())):.3f} | — |")
    fidelity = collections.defaultdict(dict)
    for model, rows in ext.items():
        accs, n_ok, n_all, ys, ps = {}, 0, 0, [], []
        for sp in core_specs:
            y, p = [], []
            for r in rows.values():
                if r["spec"] != sp or r["run_id"] not in V:
                    continue
                n_all += 1
                d = rtr.from_reader(r.get("parsed"), sp)
                pr = rule_proba(sp, d) if d else None
                if pr is None:
                    continue
                n_ok += 1
                y.append(CLASSES.index(V[r["run_id"]]["failure"]))
                p.append(int(np.argmax(pr)))
                if sp in ("lc_loss", "cm", "roc", "dist"):
                    for k, tv in V[r["run_id"]]["values"][sp].items():
                        fidelity[model].setdefault(f"{sp}:{k}", []).append((tv, d.get(k, np.nan)))
            accs[sp] = bal(np.array(y), np.array(p)) if y else float("nan")
            ys += y
            ps += p
        out[f"syn|{model}"] = dict(per_plot=accs, readable=n_ok / max(n_all, 1))
        L.append(f"| {model} | " + " | ".join(f"{accs[s]:.3f}" for s in core_specs) + f" | {np.nanmean(list(accs.values())):.3f} | {n_ok / max(n_all, 1):.0%} |")
    L += ["", "Reading fidelity (correlation between read and exact values; selected quantities):", ""]
    for model, fd in fidelity.items():
        cs = []
        for q, pairs in fd.items():
            a = np.array([p for p in pairs if not np.isnan(p[1])], float)
            if len(a) > 10 and a[:, 0].std() > 0 and a[:, 1].std() > 0:
                cs.append((q, np.corrcoef(a[:, 0], a[:, 1])[0, 1]))
        L.append(f"- {model}: " + ", ".join(f"{q} {c:.2f}" for q, c in cs[:12]))
    L.append("")

    # ---------------- positive control and real world
    for name, run, truthfile in (("Positive control", pos, "truth.json"), ("Real-world cases", rw, "truth.json")):
        truth = json.load(open(run / truthfile))
        cases = json.load(open(run / "cases.json")) if (run / "cases.json").exists() else None
        L += [f"## {name} (rule trained on synthetic models, applied unchanged)", ""]
        for model, rows in responses(run).items():
            cells, passed = [], 0
            groups = collections.defaultdict(list)
            for r in rows.values():
                t = truth[r["run_id"]]
                key = (t.get("case") or t["source"], r["spec"])
                groups[key].append(r)
            for (grp, sp), R in sorted(groups.items()):
                if cases is not None:
                    if not cases.get(grp, {}).get("verified") or sp != CANONICAL[cases[grp]["failure"]]:
                        continue
                    f = cases[grp]["failure"]
                    is_f = lambda t: t["condition"] == "failure"  # noqa: E731
                else:
                    continue
                y, s, named, fa = [], [], [], []
                for r in R:
                    t = truth[r["run_id"]]
                    pr = rule_proba(sp, rtr.from_reader(r.get("parsed"), sp))
                    if pr is None:
                        continue
                    top = CLASSES[int(np.argmax(pr))]
                    y.append(int(is_f(t)))
                    s.append(pr[CLASSES.index(f)])
                    (named if is_f(t) else fa).append(top in t["present"] if is_f(t) else top not in t["present"])
                auc = roc_auc_score(y, s) if len(set(y)) == 2 else float("nan")
                cells.append(dict(case=grp, failure=f, spec=sp, auc=auc, named=np.mean(named) if named else np.nan,
                                  false_alarm=np.mean(fa) if fa else np.nan))
            if cases is None:
                # positive control: failure-specific canonical plots, pass rule as in score_positive
                for src in ("synthetic", "magic"):
                    n_pass = 0
                    for f, sp in CANONICAL.items():
                        R = [r for r in rows.values() if r["spec"] == sp and truth[r["run_id"]]["source"] == src
                             and truth[r["run_id"]]["failure"] in ("healthy", f)]
                        y, s, nm = [], [], []
                        for r in R:
                            pr = rule_proba(sp, rtr.from_reader(r.get("parsed"), sp))
                            if pr is None:
                                continue
                            isf = truth[r["run_id"]]["failure"] == f
                            y.append(int(isf))
                            s.append(1 - pr[0])
                            if isf:
                                nm.append(CLASSES[int(np.argmax(pr))] == f)
                        auc = roc_auc_score(y, s) if len(set(y)) == 2 else float("nan")
                        ok = auc >= .8 and (np.mean(nm) if nm else 0) >= .5
                        n_pass += ok and f in PRIMARY
                        cells.append(dict(source=src, failure=f, spec=sp, auc=auc, named=float(np.mean(nm)) if nm else np.nan, passed=bool(ok)))
                    L.append(f"- {model} ({src}): {n_pass}/5 primary failures passed (direct reading earlier: ≤ 1/5)")
            else:
                L.append(f"- {model}: textbook-plot detection AUC ≥ 0.70 in {np.mean([c['auc'] >= .7 for c in cells]):.0%} of cases; "
                         f"failure named in {np.nanmean([c['named'] for c in cells]):.0%} of failure models; "
                         f"control false alarm {np.nanmean([c['false_alarm'] for c in cells]):.0%}")
            out[f"{name}|{model}"] = cells
        L.append("")
    (syn / "RTR_RESULTS.md").write_text("\n".join(L) + "\n")
    json.dump(out, open(syn / "rtr_results.json", "w"), indent=1, default=float)
    print("\n".join(L))


if __name__ == "__main__":
    main(*sys.argv[1:4])
