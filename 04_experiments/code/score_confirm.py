"""Confirmatory analysis: every preregistered hypothesis (05_paper/PREREGISTRATION_DRAFT.md) with its test and verdict.

  python score_confirm.py runs/confirm_synthetic [runs/confirm_magic]   -> <first run>/CONFIRM_RESULTS.md (+ .json)

Benchmarks: B1* = validation-best of mark-table (B1) / image (B2) model per cell; B0* = validation-best of state model and
all plot models (as stored in results.json "via"). CIs: bootstrap over held-out test runs (1,000 resamples, seed 0).
Readers: argmax over 7 states; unparseable / all-zero answers = no answer (excluded; counts reported).
"""
import collections
import json
import pathlib
import sys

import numpy as np
from scipy.stats import chi2_contingency
from sklearn.metrics import roc_auc_score

CLASSES = ["healthy", "overfitting", "miscalibration", "label_noise", "class_imbalance", "data_leakage",
           "distribution_shift"]
NB = 1000
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- loading
class Run:
    def __init__(self, path):
        self.p = pathlib.Path(path)
        self.meta = json.load(open(self.p / "meta.json"))
        self.res = json.load(open(self.p / "results.json"))
        self.b2 = json.load(open(self.p / "b2_results.json"))
        self.i1 = json.load(open(self.p / "item_scores.json"))
        self.i2 = json.load(open(self.p / "b2_item_scores.json"))
        self.top = len(self.meta["targets"]) - 1
        self.truth = {r: k.split("|")[0] for k, m in self.meta["groups"].items() for r in m}
        self.level = {r: (None if k.split("|")[1] == "None" else int(k.split("|")[1]))
                      for k, m in self.meta["groups"].items() for r in m}
        self.classes = ["healthy"] + self.meta["failures"]
        self.det = {(r["failure"], r["level"]): r for r in self.res["detection"]}
        self.dia = {r["level"]: r for r in self.res["diagnosis"]}
        self.b2det = {(r["failure"], r["level"], r["spec"]): r for r in self.b2["detection"]}
        self.b2dia = {(r["level"], r["spec"]): r for r in self.b2["diagnosis"]}

    # detection scores of a cell for B1* (val-best of mark/image) or B0*
    def det_scores(self, f, j, sp):
        if sp == "B0":
            via = self.det[(f, j)]["B0"].get("via", "B0")
            src = self.i1[f"det|{f}|{j}|{via}"]
        else:
            a, b = self.det[(f, j)][sp], self.b2det.get((f, j, sp))
            src = self.i1[f"det|{f}|{j}|{sp}"] if b is None or a["val_auc"] >= b["val_auc"] else self.i2[f"det|{f}|{j}|{sp}"]
        y = np.array([0 if self.truth[r] == "healthy" else 1 for r in src["ids"]])
        return dict(zip(src["ids"], src["score"])), y, src["ids"]

    # prior-corrected diagnosis posterior per test run for B1* / B0* / B2
    def dia_post(self, j, sp, which="star"):
        if sp == "B0":
            via = self.dia[j]["B0"].get("via", "B0")
            src = self.i1[f"diag|{j}|{via}"]
        elif which == "B2":
            src = self.i2[f"diag|{j}|{sp}"]
        else:
            a, b = self.dia[j][sp], self.b2dia.get((j, sp))
            src = self.i1[f"diag|{j}|{sp}"] if b is None or a["val_auc"] >= b["val_auc"] else self.i2[f"diag|{j}|{sp}"]
        P = np.array(src["proba"]) / np.maximum(np.array(src["prior"]), 1e-9)
        P = P / P.sum(1, keepdims=True)
        return dict(zip(src["ids"], P))

    def responses(self, pattern):
        out = {}
        for fp in sorted(self.p.glob(pattern)):
            rows = [json.loads(x) for x in open(fp)]
            if rows:
                out[fp.stem.replace("responses_", "")] = rows
        return out


def reader_vec(r):
    p = r.get("parsed")
    if not isinstance(p, dict):
        return None
    try:
        v = np.array([float(p.get(c, 0) or 0) for c in CLASSES])
    except (TypeError, ValueError):
        return None
    return v / v.sum() if v.sum() > 0 else None


# ---------------------------------------------------------------- statistics
def boot(fn, units, n=NB):
    units = np.array(units)
    vals = [fn(units[RNG.integers(0, len(units), len(units))]) for _ in range(n)]
    vals = [v for v in vals if v is not None and not np.isnan(v)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def auc_of(scores, truth, ids):
    y = np.array([0 if truth[i] == "healthy" else 1 for i in ids])
    if len(set(y)) < 2:
        return None
    return roc_auc_score(y, [scores[i] for i in ids])


def bal_acc(pred, truth_of, ids, classes):
    rec = []
    for c in classes:
        m = [pred[i] == c for i in ids if truth_of[i] == c]
        if m:
            rec.append(np.mean(m))
    return float(np.mean(rec)) if rec else None


# ---------------------------------------------------------------- hypotheses
def h1(R):
    cells = [("miscalibration", s) for s in ("roc", "pr", "cm", "lc_acc")] + \
            [(f, s) for f in ("class_imbalance", "distribution_shift") for s in ("lc_loss", "lc_acc")]
    rows, ok = [], True
    for f, s in cells:
        sc, _, ids = R.det_scores(f, R.top, s)
        a = auc_of(sc, R.truth, ids)
        lo, hi = boot(lambda u: auc_of(sc, R.truth, u), ids)
        inc = lo <= 0.5 <= hi
        ok &= inc
        rows.append(f"{f} on {s}: AUC {a:.3f} [{lo:.3f}, {hi:.3f}] → {'blind ✓' if inc else ('NOT blind ✗ (below chance: selection artefact, no usable signal)' if hi < 0.5 else 'NOT blind ✗ (detectable)')}")
    return ok, rows


def h2(R):
    j = R.top
    rows, n_detect_only = [], 0
    for i, f in enumerate(R.meta["failures"]):
        for s in R.meta["specs"]:
            sc, _, ids = R.det_scores(f, j, s)
            a = auc_of(sc, R.truth, ids)
            lo, _ = boot(lambda u: auc_of(sc, R.truth, u), ids, n=300)
            post = R.dia_post(j, s)
            pred = {k: R.classes[int(np.argmax(v))] for k, v in post.items()}
            fid = [k for k in post if R.truth[k] == f]
            rec = np.mean([pred[k] == f for k in fid]) if fid else 0
            if a >= 0.70 and lo > 0.5 and rec < 0.5:
                n_detect_only += 1
                rows.append(f"detect-only: {f} on {s} (AUC {a:.2f}, lower CI {lo:.2f}; recall {rec:.2f})")
    a_ok = n_detect_only >= 3
    rows.insert(0, f"(a) detect-only cells: {n_detect_only} (need ≥ 3) → {'✓' if a_ok else '✗'}")
    p0 = R.dia_post(j, "B0")
    pred0 = {k: R.classes[int(np.argmax(v))] for k, v in p0.items()}
    b_ok = True
    for s in R.meta["specs"]:
        ps = R.dia_post(j, s)
        preds = {k: R.classes[int(np.argmax(v))] for k, v in ps.items()}
        ids = sorted(set(p0) & set(ps))
        d = (bal_acc(pred0, R.truth, ids, R.classes) or 0) - (bal_acc(preds, R.truth, ids, R.classes) or 0)
        lo, hi = boot(lambda u: (bal_acc(pred0, R.truth, u, R.classes) or 0) - (bal_acc(preds, R.truth, u, R.classes) or 0), ids, n=500)
        b_ok &= lo > 0
        rows.append(f"(b) B0* − {s}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}] {'✓' if lo > 0 else '✗'}")
    return a_ok and b_ok, rows


def h3(R):
    s0, _, ids = R.det_scores("miscalibration", R.top, "B0")
    s1, _, ids1 = R.det_scores("miscalibration", R.top, "rel")
    ids = sorted(set(ids) & set(ids1))
    f = lambda u: (auc_of(s0, R.truth, u) or 0) - (auc_of(s1, R.truth, u) or 0)  # noqa: E731
    d, (lo, hi) = f(ids), boot(f, ids)
    return lo > 0, [f"B0* − reliability (miscalibration detection): {d:+.3f} [{lo:+.3f}, {hi:+.3f}]"]


def h4(R):
    j, rows, ok = R.top, [], True
    a, _, ia = R.det_scores("class_imbalance", j, "rel_n")
    b, _, ib = R.det_scores("class_imbalance", j, "rel")
    ids = sorted(set(ia) & set(ib))
    f = lambda u: (auc_of(a, R.truth, u) or 0) - (auc_of(b, R.truth, u) or 0)  # noqa: E731
    lo, hi = boot(f, ids)
    ok &= lo > 0
    rows.append(f"imbalance detection rel_n − rel: {f(ids):+.3f} [{lo:+.3f}, {hi:+.3f}] {'✓' if lo > 0 else '✗'}")
    for x, y, sign in (("rel_n", "rel", 1), ("cm_norm", "cm", -1)):
        px, py = R.dia_post(j, x), R.dia_post(j, y)
        cx = {k: R.classes[int(np.argmax(v))] for k, v in px.items()}
        cy = {k: R.classes[int(np.argmax(v))] for k, v in py.items()}
        ids = sorted(set(px) & set(py))
        g = lambda u: (bal_acc(cx, R.truth, u, R.classes) or 0) - (bal_acc(cy, R.truth, u, R.classes) or 0)  # noqa: E731
        lo, hi = boot(g, ids, n=500)
        good = lo > 0 if sign > 0 else hi < 0
        ok &= good
        rows.append(f"diagnosis {x} − {y}: {g(ids):+.3f} [{lo:+.3f}, {hi:+.3f}] (expected {'>' if sign > 0 else '<'} 0) "
                    f"{'✓' if good else '✗'}")
    return ok, rows


def reader_vs_b2(R, rows_by_reader, label):
    """Balanced accuracy of each reader vs B2 (image benchmark) on the same strongest-level items."""
    j, out, ok = R.top, [], True
    for name, rows in rows_by_reader.items():
        pred, abst, ids_by_spec, noans = {}, {}, collections.defaultdict(list), 0
        for r in rows:
            if r.get("task") != "diagnosis" or R.level.get(r["run_id"]) not in (None, j):
                continue
            v = reader_vec(r)
            if v is None:
                noans += 1
                abst[(r["run_id"], r["spec"])] = "no_answer"
            else:
                pred[(r["run_id"], r["spec"])] = CLASSES[int(np.argmax(v))]
            ids_by_spec[r["spec"]].append(r["run_id"])
        if not pred:
            continue
        b2p = {}
        for s in ids_by_spec:
            post = R.dia_post(j, s, which="B2")
            for rid in ids_by_spec[s]:
                if rid in post:
                    b2p[(rid, s)] = R.classes[int(np.argmax(post[rid]))]
        keys = [k for k in pred if k in b2p]
        tr = {k: R.truth[k[0]] for k in keys}
        runs = sorted({k[0] for k in keys})
        byrun = collections.defaultdict(list)
        for k in keys:
            byrun[k[0]].append(k)

        def diff(u):
            ks = [k for r in u for k in byrun[r]]
            a = bal_acc(pred, tr, ks, R.classes)
            b = bal_acc(b2p, tr, ks, R.classes)
            return None if a is None or b is None else a - b
        ra, rb = bal_acc(pred, tr, keys, R.classes), bal_acc(b2p, tr, keys, R.classes)
        allk = [k for k in {**pred, **abst} if k in b2p]
        tra = {k: R.truth[k[0]] for k in allk}
        sens = bal_acc({**pred, **abst}, tra, allk, R.classes)  # sensitivity: no answer counted as wrong
        lo, hi = boot(diff, runs, n=500)
        ok &= hi < 0
        out.append(f"{name} ({label}): reader {ra:.3f} vs B2 {rb:.3f}, diff [{lo:+.3f}, {hi:+.3f}] "
                   f"{'✓ below' if hi < 0 else '✗'} | no-answer {noans}/{len(allk)} ({noans / max(len(allk), 1):.0%}); "
                   f"counting no-answer as wrong: {sens:.3f}")
    return ok, out


def h6(R, readers):
    rows, ok = [], True
    for name, rr in readers.items():
        tab = collections.defaultdict(collections.Counter)
        for r in rr:
            if r.get("task") != "diagnosis" or R.truth.get(r["run_id"]) != "healthy":
                continue
            v = reader_vec(r)
            if v is not None:
                tab[r["spec"]][CLASSES[int(np.argmax(v))]] += 1
        specs = sorted(tab)
        M = np.array([[tab[s][c] for c in CLASSES] for s in specs])
        M = M[:, M.sum(0) > 0]
        if M.shape[0] < 2 or M.shape[1] < 2:
            rows.append(f"{name}: degenerate table (single answer everywhere) — stereotype maximal")
            continue
        chi2, p, _, _ = chi2_contingency(M)
        lc = tab.get("lc_loss", collections.Counter())
        sh_lc = lc["overfitting"] / max(sum(lc.values()), 1)
        other = sum(tab[s]["overfitting"] for s in specs if s != "lc_loss") / max(sum(sum(tab[s].values()) for s in specs if s != "lc_loss"), 1)
        good = p < 0.05 and sh_lc > other
        ok &= good
        rows.append(f"{name}: χ² p = {p:.2g}; 'overfitting' on healthy loss curves {sh_lc:.0%} vs other plots {other:.0%} "
                    f"{'✓' if good else '✗'}")
    return ok, rows


def h8(R):
    rows, ok = [], True
    cb = json.load(open(R.p / "companion_benchmarks.json"))
    a_ok = all(v["pair_balanced_acc"] > v["single_balanced_acc"] for v in cb.values())
    rows.append("(a) benchmark: " + "; ".join(f"{k} {v['single_balanced_acc']:.2f}→{v['pair_balanced_acc']:.2f}"
                                               for k, v in cb.items()) + f" {'✓' if a_ok else '✗'}")
    badges = json.load(open(R.p / "badges.json"))
    listed = {sp: [c for c in CLASSES if c.replace("_", " ") in txt.split("Cannot show")[0]] for sp, txt in badges.items()}
    b_ok, c_ok = True, True
    for fp in sorted(R.p.glob("responses_*_iv.jsonl")):
        model = fp.stem.replace("responses_", "").replace("_iv", "")
        iv = [json.loads(x) for x in open(fp)]
        keys = {(r["run_id"], r["spec"]) for r in iv}
        plain = {(r["run_id"], r["spec"]): r for r in map(json.loads, open(R.p / f"responses_{model}.jsonl"))
                 if r.get("task") == "diagnosis" and (r["run_id"], r["spec"]) in keys}
        cond = collections.defaultdict(dict)
        for r in iv:
            cond[r["condition"]][(r["run_id"], r["spec"])] = r
        cond["plain"] = plain

        def answers(c):
            out = {}
            for k, r in cond[c].items():
                v = reader_vec(r)
                if v is not None:
                    out[k] = CLASSES[int(np.argmax(v))]
            return out
        ap, ac, ab = answers("plain"), answers("companion"), answers("badge")
        common = [k for k in ap if k in ac]
        runs = sorted({k[0] for k in common})
        byrun = collections.defaultdict(list)
        for k in common:
            byrun[k[0]].append(k)
        tr = {k: R.truth[k[0]] for k in common}

        def d_acc(u):
            ks = [k for r in u for k in byrun[r]]
            a, b = bal_acc(ac, tr, ks, R.classes), bal_acc(ap, tr, ks, R.classes)
            return None if a is None or b is None else a - b
        lo90 = np.percentile([d_acc(np.array(runs)[RNG.integers(0, len(runs), len(runs))]) for _ in range(500)], [5, 95])
        equiv = -0.03 < lo90[0] and lo90[1] < 0.03
        b_ok &= equiv
        cb_keys = [k for k in ap if k in ab]
        sh = lambda A: np.mean([A[k] in listed[k[1]] for k in cb_keys])  # noqa: E731
        moved = sh(ab) - sh(ap)
        c_ok &= moved > 0
        rows.append(f"{model}: (b) companion − plain 90% CI [{lo90[0]:+.3f}, {lo90[1]:+.3f}] {'✓ equivalent' if equiv else '✗'}"
                    f" | (c) answers naming a badge-listed state: plain {sh(ap):.0%} → badge {sh(ab):.0%} {'✓' if moved > 0 else '✗'}")
    return a_ok and b_ok and c_ok, rows


def main(paths):
    runs = [Run(p) for p in paths]
    R = runs[0]
    L = [f"# Confirmatory results — {', '.join(r.p.name for r in runs)}", "",
         "Each hypothesis as preregistered; ✓ = prediction supported. CIs: bootstrap over held-out test runs.", ""]
    verdicts = {}

    def section(name, title, fn, *args):
        try:
            ok, rows = fn(*args)
            ok = bool(ok)
        except (KeyError, FileNotFoundError) as e:
            ok, rows = None, [f"not available: {e}"]
        verdicts[name] = None if ok is None else bool(ok)
        L.extend([f"## {name}: {title} — {'SUPPORTED' if ok else ('NOT SUPPORTED' if ok is False else 'n/a')}", ""]
                 + [f"- {x}" for x in rows] + [""])

    for Rx in runs:
        tag = Rx.p.name
        section(f"H1 [{tag}]", "theory-predicted blind spots", h1, Rx)
        section(f"H2 [{tag}]", "detection ≠ diagnosis", h2, Rx)
        section(f"H3 [{tag}]", "encoding loss in the calibration plot", h3, Rx)
        section(f"H4 [{tag}]", "design variants change the evidence", h4, Rx)
        main_readers = {k: v for k, v in Rx.responses("responses_*.jsonl").items()
                        if not any(k.endswith(s) for s in ("_iv", "_variants", "_controls", "_hires"))}
        section(f"H5 [{tag}]", "readers below the image benchmark", reader_vs_b2, Rx, main_readers, "main/core")
        section(f"H6 [{tag}]", "plot-type stereotype", h6, Rx, main_readers)
        ctl = Rx.responses("responses_*_controls.jsonl")
        if ctl:
            for variant in ("fewshot", "numbers"):
                sub = {k: [r for r in v if r.get("variant") == variant] for k, v in ctl.items()}
                section(f"H7{'a' if variant == 'fewshot' else 'b'} [{tag}]", f"{variant} control still below B2",
                        reader_vs_b2, Rx, sub, variant)
        if (Rx.p / "companion_benchmarks.json").exists():
            section(f"H8 [{tag}]", "interventions (benchmark ↑; readers equivalent; badge cue-following)", h8, Rx)
    (R.p / "CONFIRM_RESULTS.md").write_text("\n".join(L) + "\n")
    json.dump(verdicts, open(R.p / "confirm_verdicts.json", "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1:])
