"""Score AI readers on the real-world cases (Step 3).

  python score_realworld.py runs/realworld   -> runs/realworld/REALWORLD_RESULTS.md (+ .json)

Per reader × case × plot (verified cases only):
  detection   AUC of P(planted failure): real failure models vs. their healthy controls on the same data;
  named       share of failure models whose top answer is a condition measured as present in that model
              (planted share in brackets);
  false alarm share of control models whose top answer is not a condition present in them (or "healthy").
Includes transfer.json (benchmark on the same plots), if present, so readers and benchmark sit side by side.
"""
import collections
import json
import pathlib
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from make_positive import CANONICAL
from score_confirm import CLASSES, reader_vec


def main(rw):
    rw = pathlib.Path(rw)
    truth = json.load(open(rw / "truth.json"))
    cases = {k: v for k, v in json.load(open(rw / "cases.json")).items() if v["verified"]}
    tr = {(t["case"], t["spec"]): t for t in json.load(open(rw / "transfer.json"))} if (rw / "transfer.json").exists() else {}
    L = ["# Real-world check: AI readers on real failures", "",
         "Verified cases only. Detection AUC uses the probability given to the planted failure, failure vs. control models on the same data. "
         "Named = top answer is a condition measured as present (planted failure in brackets). "
         "Benchmark = the confirmatory B1 detector applied to the same plots.", ""]
    out = {}
    for fp in sorted(rw.glob("responses_*.jsonl")):
        model = fp.stem.replace("responses_", "")
        rows = [json.loads(x) for x in open(fp)]
        by = collections.defaultdict(list)
        for r in rows:
            by[(truth[r["run_id"]]["case"], r["spec"])].append(r)
        L += [f"## {model}", "", "| case | failure | plot | reader AUC | benchmark AUC | named (planted) | control false alarm | no answer |",
              "|---|---|---|---|---|---|---|---|"]
        cells, noans_all = [], 0
        for cid, cv in cases.items():
            for sp in ("lc_loss", "lc_acc", "rel", "rel_n", "cm", "cm_norm", "roc", "pr", "dist"):
                R = by.get((cid, sp), [])
                y, s, named, planted, fa, noans = [], [], [], [], [], 0
                for r in R:
                    t = truth[r["run_id"]]
                    v = reader_vec(r)
                    if v is None:
                        noans += 1
                        continue
                    top = CLASSES[int(np.argmax(v))]
                    fail = t["condition"] == "failure"
                    y.append(int(fail))
                    s.append(v[CLASSES.index(cv["failure"])])
                    if fail:
                        named.append(top in t["present"])
                        planted.append(top == cv["failure"])
                    else:
                        fa.append(top not in t["present"])
                noans_all += noans
                auc = roc_auc_score(y, s) if len(set(y)) == 2 else float("nan")
                b = tr.get((cid, sp), {}).get("real_auc", float("nan"))
                cell = dict(case=cid, failure=cv["failure"], spec=sp, reader_auc=auc, bench_auc=b,
                            named=float(np.mean(named)) if named else float("nan"),
                            planted=float(np.mean(planted)) if planted else float("nan"),
                            false_alarm=float(np.mean(fa)) if fa else float("nan"), no_answer=noans)
                cells.append(cell)
                L.append(f"| {cid} | {cv['failure']} | {sp} | {auc:.2f} | {b:.2f} | {cell['named']:.0%} ({cell['planted']:.0%}) "
                         f"| {cell['false_alarm']:.0%} | {noans} |")
        ok = [c for c in cells if not np.isnan(c["reader_auc"])]
        det = np.mean([c["reader_auc"] >= 0.70 for c in ok]) if ok else float("nan")
        bdet = np.mean([c["bench_auc"] >= 0.70 for c in ok if not np.isnan(c["bench_auc"])]) if tr else float("nan")
        canon = [c for c in ok if c["spec"] == CANONICAL[c["failure"]]]  # each failure's textbook plot (fixed in advance)
        L += ["", f"**Summary:** reader detects (AUC ≥ 0.70) in {det:.0%} of case-plot pairs (benchmark: {bdet:.0%}); "
              f"on each failure's textbook plot the planted failure is named in {np.mean([c['planted'] for c in canon]):.0%} "
              f"of failure models (detection AUC ≥ 0.70 in {np.mean([c['reader_auc'] >= .7 for c in canon]):.0%} of cases); mean control false-alarm rate {np.nanmean([c['false_alarm'] for c in ok]):.0%}; "
              f"no-answer {noans_all}.", ""]
        out[model] = cells
    (rw / "REALWORLD_RESULTS.md").write_text("\n".join(L) + "\n")
    json.dump(out, open(rw / "realworld_results.json", "w"), indent=1, default=float)
    print("\n".join(l for l in L if l.startswith("## ") or l.startswith("**Summary")))


if __name__ == "__main__":
    main(sys.argv[1])
