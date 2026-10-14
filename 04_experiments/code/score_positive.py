"""Score the positive control and the value probe (see make_positive.py).

  python score_positive.py runs/positive runs/confirm_synthetic   -> runs/positive/POSITIVE_RESULTS.md (+ .json)
"""
import collections
import json
import pathlib
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from make_positive import CANONICAL, PRIMARY
from score_confirm import CLASSES, reader_vec

RNG = np.random.default_rng(0)


def boot_auc(y, s, n=1000):
    y, s = np.asarray(y), np.asarray(s)
    v = []
    for _ in range(n):
        i = RNG.integers(0, len(y), len(y))
        if len(set(y[i])) == 2:
            v.append(roc_auc_score(y[i], s[i]))
    return np.percentile(v, [2.5, 97.5])


def positive(P):
    truth = json.load(open(P / "truth.json"))
    L, out = ["## Positive control: blatant failures on their textbook plot", "",
              "Detection AUC = healthy vs the failure on the same plot type (score 1 − P(healthy)). Named = share of "
              "failure items whose top answer is the failure. Healthy false alarm = share of healthy items on that plot "
              "whose top answer is not healthy. Pass rule (fixed before reading): AUC ≥ 0.80 and named ≥ 50% on ≥ 4 of "
              f"the 5 primary failures ({', '.join(PRIMARY)}).", ""], {}
    for fp in sorted(P.glob("responses_*.jsonl")):
        model = fp.stem.replace("responses_", "")
        rows = [json.loads(x) for x in open(fp)]
        for src in ("synthetic", "magic"):
            R = [r for r in rows if truth[r["run_id"]]["source"] == src]
            if not R:
                continue
            L += [f"### {model} — {src}", "", "| failure (plot) | detection AUC [95% CI] | named | healthy false alarm "
                  "| top answer on failure items | no answer |", "|---|---|---|---|---|---|"]
            passes, res = 0, {}
            for f, sp in CANONICAL.items():
                items = [r for r in R if r["spec"] == sp and truth[r["run_id"]]["failure"] in ("healthy", f)]
                y, s, top_f, top_h, noans = [], [], collections.Counter(), [], 0
                for r in items:
                    v = reader_vec(r)
                    is_f = truth[r["run_id"]]["failure"] == f
                    if v is None:
                        noans += 1
                        continue
                    y.append(int(is_f))
                    s.append(1 - v[0])
                    if is_f:
                        top_f[CLASSES[int(np.argmax(v))]] += 1
                    else:
                        top_h.append(CLASSES[int(np.argmax(v))] != "healthy")
                auc = roc_auc_score(y, s) if len(set(y)) == 2 else float("nan")
                lo, hi = boot_auc(y, s) if len(set(y)) == 2 else (float("nan"), float("nan"))
                named = top_f[f] / max(sum(top_f.values()), 1)
                fa = float(np.mean(top_h)) if top_h else float("nan")
                ok = auc >= 0.80 and named >= 0.5
                passes += ok and f in PRIMARY
                res[f] = dict(spec=sp, auc=auc, ci=[lo, hi], named=named, false_alarm=fa, no_answer=noans,
                              top=dict(top_f.most_common(3)), passed=bool(ok))
                L.append(f"| {f} ({sp}){'' if f in PRIMARY else ' †'} | {auc:.2f} [{lo:.2f}, {hi:.2f}] | {named:.0%} "
                         f"| {fa:.0%} | " + ", ".join(f"{k} {v}" for k, v in top_f.most_common(3)) + f" | {noans} |")
            verdict = passes >= 4
            out[f"{model}|{src}"] = dict(cells=res, primary_passed=int(passes), can_do_task=bool(verdict))
            L += ["", f"**{passes}/5 primary failures passed → {'CAN do the task when the signal is blatant' if verdict else 'CANNOT do the task even when the signal is blatant'}**"
                  " († label noise: no unique single-plot signature, reported only)", ""]
    return L, out


def probe(C):
    tf = C / "probe_truth.json"
    if not tf.exists():
        return ["## Value probe: not run"], {}
    truth = json.load(open(tf))
    L, out = ["## Value probe (confirmatory models): read numbers off the curves", "",
              "Correlation and median absolute error between read and true values; 'rise' = val_last − val_best "
              "(the overfitting signature). AUC = overfitting vs healthy from the reader's OWN read numbers vs from the "
              "true numbers.", ""], {}
    for fp in sorted(C.glob("responses_*_probe.jsonl")):
        model = fp.stem.replace("responses_", "").replace("_probe", "")
        rows = [json.loads(x) for x in open(fp)]
        L += [f"### {model}", "", "| plot | quantity | corr | median abs error | parsed |", "|---|---|---|---|---|"]
        res = {}
        for sp in ("lc_loss", "lc_acc"):
            R = [r for r in rows if r["spec"] == sp]
            got = []
            for r in R:
                p, t = r.get("parsed"), truth[r["item_id"]]
                try:
                    got.append((t, {k: float(p[k]) for k in ("train_last", "val_last", "val_best", "val_best_epoch")}))
                except (TypeError, KeyError, ValueError):
                    pass
            for q in ("train_last", "val_last", "val_best", "val_best_epoch"):
                a = np.array([t[q] for t, _ in got])
                b = np.array([g[q] for _, g in got])
                c = float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 0 and b.std() > 0 else float("nan")
                med = float(np.median(np.abs(a - b))) if len(a) else float("nan")
                res[f"{sp}|{q}"] = dict(corr=c, mae=med, n=len(a))
                L.append(f"| {sp} | {q} | {c:.2f} | {med:.3g} | {len(got)}/{len(R)} |")
            sign = 1 if sp == "lc_loss" else -1
            sub = [(t, g) for t, g in got if t["failure"] in ("healthy", "overfitting")]
            if len({t["failure"] for t, _ in sub}) == 2:
                y = [int(t["failure"] == "overfitting") for t, _ in sub]
                own = roc_auc_score(y, [sign * (g["val_last"] - g["val_best"]) for _, g in sub])
                true = roc_auc_score(y, [sign * (t["val_last"] - t["val_best"]) for t, _ in sub])
                res[f"{sp}|overfit_auc"] = dict(own=own, true=true)
                L.append(f"| {sp} | overfitting AUC from rise: own reading {own:.2f} vs true values {true:.2f} | | | |")
        out[model] = res
        L.append("")
    return L, out


if __name__ == "__main__":
    P, C = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    a, ra = positive(P)
    b, rb = probe(C)
    L = ["# Positive control and value probe (exploratory credibility checks)", ""] + a + b
    (P / "POSITIVE_RESULTS.md").write_text("\n".join(L) + "\n")
    json.dump(dict(positive=ra, probe=rb), open(P / "positive_results.json", "w"), indent=1, default=float)
    print("\n".join(L))
