"""Score AI readers on the prototype items against the key and the B0/B1/B2 benchmarks.

  python score_reader.py runs/overfit_proto_v2                      # every responses_*.jsonl + summary.md
  python score_reader.py runs/overfit_proto_v2 responses_X.jsonl    # one reader

Per reader: lineup hits / null false alarms / position concentration, shuffled-lineup control (does the reader follow
the plot or the position?), single-plot diagnosis vs B1/B2, over-claiming signs, prior-only answer, value probe.
"""
import collections
import json
import pathlib
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

CAUSES = ["healthy", "overfitting", "label_noise", "class_imbalance", "data_leakage", "miscalibration",
          "distribution_shift"]


def probs(p):
    try:
        v = np.array([float(p.get(c, 0) or 0) for c in CAUSES])
    except (TypeError, ValueError):
        return None
    return v / v.sum() if v.sum() > 0 else None


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def score_one(run, resp_name):
    resp = [json.loads(line) for line in open(run / resp_name)]
    meta = json.load(open(run / "meta.json"))
    keys = {k["item_id"]: k for k in json.load(open(run / "lineups" / "key.json"))}
    if (run / "lineups_shuffled" / "key.json").exists():
        keys.update({k["item_id"]: k for k in json.load(open(run / "lineups_shuffled" / "key.json"))})
    res = json.load(open(run / "results.json"))["results"]
    b2 = {(r["spec"], r["level"]): r for r in json.load(open(run / "b2_gpu_results.json"))} \
        if (run / "b2_gpu_results.json").exists() else {}
    level_of = {r: lv["level"] for lv in meta["levels"] for r in lv["failed"]}
    model = resp[0]["model"]
    S = dict(model=model)
    L = [f"# Reader: {model}  ({len(resp)} responses)", "",
         f"Unparseable answers: {sum(r['parsed'] is None for r in resp)}"]

    # --- lineups (original + shuffled) ---
    L += ["", "## Lineups (detection)", "", "| set | spec | level | n | hits | null false alarms | choices |",
          "|---|---|---|---|---|---|---|"]
    groups = collections.defaultdict(list)
    choice = {}
    for r in resp:
        if r["task"] in ("lineup", "lineup_shuffled"):
            k = keys[r["item_id"]]
            c = (r["parsed"] or {}).get("choice") if isinstance(r["parsed"], dict) else None
            c = int(c) if _num(c) is not None else None
            choice[r["item_id"]] = c
            groups[(r["task"], r["spec"], "null" if k["null_lineup"] else k["level"])].append((c, k))
    tot = collections.Counter()
    for (task, spec, lv), rows in sorted(groups.items(), key=lambda x: str(x[0])):
        ch = [c for c, _ in rows]
        if lv == "null":
            fa = sum(c not in (0, None) for c in ch)
            tot[f"{task}_fa"] += fa
            tot[f"{task}_null"] += len(rows)
            L.append(f"| {task} | {spec} | null | {len(rows)} | – | {fa}/{len(rows)} | {ch} |")
        else:
            hits = sum(c == k["real_panel"] for c, k in rows)
            tot[f"{task}_hits"] += hits
            tot[f"{task}_n"] += len(rows)
            L.append(f"| {task} | {spec} | {lv} | {len(rows)} | {hits}/{len(rows)} | – | {ch} vs real "
                     f"{[k['real_panel'] for _, k in rows]} |")
    picks = [c for c in choice.values() if c not in (0, None)]
    top = collections.Counter(picks).most_common(3)
    S.update(lineup_hits=f"{tot['lineup_hits']}/{tot['lineup_n']}",
             null_false_alarms=f"{tot['lineup_fa']}/{tot['lineup_null']}",
             top_position=f"{top[0][0]} ({top[0][1]}/{len(picks)})" if top else "–")
    L += ["", f"Most chosen positions: {top} of {len(picks)} picks (uniform would be ~{len(picks) / 20:.1f} each)"]

    # shuffled control: same 20 plots, new positions -> does the pick follow the plot or the position?
    pairs = [(k, keys[k["original"]]) for k in keys.values() if "original" in k]
    same_plot = same_pos = n = 0
    for ks, ko in pairs:
        cs, co = choice.get(ks["item_id"]), choice.get(ko["item_id"])
        if cs in (0, None) or co in (0, None):
            continue
        n += 1
        same_pos += cs == co
        same_plot += ks["panel_runs"][cs - 1] == ko["panel_runs"][co - 1]
    if n:
        S.update(follows_plot=f"{same_plot}/{n}", follows_position=f"{same_pos}/{n}")
        L += [f"Shuffled control ({n} pairs): picked the same PLOT {same_plot}/{n}, the same POSITION {same_pos}/{n} "
              f"(chance ≈ {n / 20:.1f} each)"]

    # --- diagnosis ---
    L += ["", "## Single-plot diagnosis (P(overfitting) as detection score; held-out test items)", "",
          "| spec | level | reader AUC | B1 AUC | B2 AUC (best) | mean P(overfit) failed / healthy | P(other failures) "
          "| cannot_tell | mean top prob |", "|---|---|---|---|---|---|---|---|---|"]
    diag = [r for r in resp if r["task"] == "diagnosis" and isinstance(r["parsed"], dict)
            and probs(r["parsed"]) is not None]
    S["diag_valid"] = f"{len(diag)}/{sum(r['task'] == 'diagnosis' for r in resp)}"
    for spec in meta["specs"]:
        rows = [r for r in diag if r["spec"] == spec]
        healthy = [r for r in rows if r["run_id"] not in level_of]
        for lv in meta["levels"]:
            failed = [r for r in rows if level_of.get(r["run_id"]) == lv["level"]]
            if not healthy or not failed:
                continue
            y = np.r_[np.zeros(len(healthy)), np.ones(len(failed))]
            P = np.array([probs(r["parsed"]) for r in healthy + failed])
            auc = roc_auc_score(y, P[:, 1])
            b1 = res[lv["level"]][f"B1_{spec}"]["test_auc"]
            b2v = [x for x in (res[lv["level"]].get(f"B2_{spec}", {}).get("test_auc"),
                               b2.get((spec, lv["level"]), {}).get("test_auc")) if x is not None]
            ct = np.mean([bool(r["parsed"].get("cannot_tell")) for r in healthy + failed])
            L.append(f"| {spec} | {lv['target']} | {auc:.3f} | {b1:.3f} | {max(b2v) if b2v else float('nan'):.3f} | "
                     f"{P[y == 1, 1].mean():.2f} / {P[y == 0, 1].mean():.2f} | {P[:, 2:].sum(1).mean():.2f} | "
                     f"{ct:.2f} | {P.max(1).mean():.2f} |")
            S[f"auc_{spec}_L{lv['level']}"] = round(auc, 3)
        if healthy:
            Ph = np.array([probs(r["parsed"]) for r in healthy])
            S[f"p_overfit_healthy_{spec}"] = round(float(Ph[:, 1].mean()), 2)
            S[f"cannot_tell_{spec}"] = round(float(np.mean([bool(r["parsed"].get("cannot_tell")) for r in rows])), 2)

    L += ["", "## Prior only (no image)", ""]
    for r in resp:
        if r["task"] == "prior_only":
            L.append(f"- {r['spec']}: {r['parsed']}")

    # --- value probe ---
    truth_f = run / "probe_truth.json"
    probe = [r for r in resp if r["task"] == "probe"]
    if probe and truth_f.exists():
        truth = json.load(open(truth_f))
        L += ["", "## Value probe (can the reader read the plot?)", "",
              "| spec | n valid | median |error| last train | last val | best val | median epoch error | "
              "corr(read rise, true rise) |", "|---|---|---|---|---|---|---|"]
        for spec in meta["specs"]:
            err = collections.defaultdict(list)
            rise_read, rise_true = [], []
            for r in probe:
                if r["spec"] != spec or not isinstance(r["parsed"], dict):
                    continue
                t = truth[r["item_id"]]
                v = {a: _num(r["parsed"].get(a)) for a in ("train_last", "val_last", "val_best", "val_best_epoch")}
                if None in v.values():
                    continue
                for a in ("train_last", "val_last", "val_best"):
                    err[a].append(abs(v[a] - t[a]))
                err["epoch"].append(abs(v["val_best_epoch"] - t["val_best_epoch"]))
                rise_read.append(v["val_last"] - v["val_best"])
                rise_true.append(t["val_last"] - t["val_best"])
            if err:
                c = np.corrcoef(rise_read, rise_true)[0, 1] if np.std(rise_read) > 0 else float("nan")
                L.append(f"| {spec} | {len(err['epoch'])} | {np.median(err['train_last']):.3f} | "
                         f"{np.median(err['val_last']):.3f} | {np.median(err['val_best']):.3f} | "
                         f"{np.median(err['epoch']):.0f} | {c:.2f} |")
                S[f"probe_val_err_{spec}"] = round(float(np.median(err["val_last"])), 3)
                S[f"probe_rise_corr_{spec}"] = round(float(c), 2)
    out = run / f"score_{resp_name.replace('.jsonl', '')}.md"
    out.write_text("\n".join(L) + "\n")
    return S, L


def main(run, names):
    run = pathlib.Path(run)
    names = names or sorted(p.name for p in run.glob("responses_*.jsonl"))
    summaries = []
    for n in names:
        S, L = score_one(run, n)
        summaries.append(S)
        if len(names) == 1:
            print("\n".join(L))
    cols = sorted({k for s in summaries for k in s} - {"model"})
    md = ["# Reader comparison", "", "| metric | " + " | ".join(s["model"].split("/")[-1] for s in summaries) + " |",
          "|---|" + "---|" * len(summaries)]
    md += [f"| {c} | " + " | ".join(str(s.get(c, "–")) for s in summaries) + " |" for c in cols]
    (run / "readers_summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
