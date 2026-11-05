"""Which plot for which failure: a one-screen guide derived from the evidential map.

  python plot_guide.py   -> runs/PLOT_GUIDE.md (+ plot_guide.json)

Reads the Explorer data (export_explorer.py) for both confirmatory runs, strongest severity. A plot counts for a
failure only if it has the same label on both data sources:
  diagnoses   identifiable on both (detection AUC >= 0.70 and diagnosis recall >= 0.50); best = highest minimum recall
  detects     at least detect-only on both, but not identifiable on both
  cannot show blind on both; the review grade is given when both runs agree on it (score_review.py)
Plots whose label differs between the runs are listed as "depends on the data".
The last section checks the usual dashboard (loss curves, confusion matrix, ROC) against the map.
"""
import json
import pathlib

EXPLORER = pathlib.Path(__file__).parent.parent / "explorer"
RUNS = pathlib.Path(__file__).parent / "runs"
NAMES = dict(lc_loss="loss curves", lc_acc="accuracy curves", rel="reliability diagram",
             rel_n="reliability diagram + bin sizes", cm="confusion matrix", cm_norm="confusion matrix (row %)",
             roc="ROC curve", pr="precision-recall curve", dist="validation vs test score histogram")
DASHBOARD = ("lc_loss", "cm", "roc")
RANK = {"neither": 0, "detect-only": 1, "identifiable": 2}


def load(run):
    s = (EXPLORER / f"data_{run}.js").read_text()
    js = json.loads(s.split("] = ", 1)[1].rstrip().rstrip(";"))
    top = max(c["level"] for c in js["cells"])
    return js, {(c["failure"], c["plot"]): c for c in js["cells"] if c["level"] == top}


def main():
    (S, cs), (_, cm) = load("confirm_synthetic"), load("confirm_magic")
    guide = {}
    for f in S["failures"]:
        g = dict(diagnoses=[], detects=[], cannot_show=[], depends=[])
        for p in S["plots"]:
            a, b = cs[(f, p)], cm[(f, p)]
            ra, rb = RANK[a["label"]], RANK[b["label"]]
            if ra == rb == 2:
                g["diagnoses"].append((p, min(a["diagnose"], b["diagnose"])))
            elif ra >= 1 and rb >= 1:
                g["detects"].append(p)
            elif ra == rb == 0:
                ev = a.get("evidence") if a.get("evidence") == b.get("evidence") else None
                g["cannot_show"].append((p, ev))
            else:
                g["depends"].append(p)
        g["diagnoses"].sort(key=lambda t: -t[1])
        guide[f] = g
    L = ["# Which plot for which failure", "",
         "From the evidential map at the strongest severity. A plot is listed only where both data sources (synthetic, "
         "MagicTelescope) agree. *Diagnoses* = detects the failure and tells it apart from the others (recall >= 0.50); "
         "*detects* = shows that something is wrong, not what; *cannot show* = no benchmark reaches AUC 0.70.", "",
         "| failure | best plot | also diagnoses | detects only | cannot show | depends on the data |", "|---|---|---|---|---|---|"]
    for f, g in guide.items():
        best = f"{NAMES[g['diagnoses'][0][0]]} (recall {g['diagnoses'][0][1]:.2f})" if g["diagnoses"] else "none diagnoses it"
        also = ", ".join(NAMES[p] for p, _ in g["diagnoses"][1:]) or "–"
        cant = ", ".join(NAMES[p] + (" (proven)" if ev == "invisible (proven)" else "") for p, ev in g["cannot_show"]) or "–"
        L.append(f"| {f.replace('_', ' ')} | {best} | {also} | {', '.join(NAMES[p] for p in g['detects']) or '–'} | "
                 f"{cant} | {', '.join(NAMES[p] for p in g['depends']) or '–'} |")
    L += ["", "## The usual dashboard: loss curves + confusion matrix + ROC", ""]
    for f in S["failures"]:
        best = {r: max(RANK[c[(f, p)]["label"]] for p in DASHBOARD) for r, c in (("synthetic", cs), ("real", cm))}
        if min(best.values()) < 2:
            word = {0: "cannot show it", 1: "detects it but cannot diagnose it", 2: "diagnoses it"}
            v = set(best.values())
            state = word[best["real"]] if len(v) == 1 else "; ".join(f"on {r} data {word[x]}" for r, x in best.items())
            if guide[f]["diagnoses"]:
                add = f" Add the {NAMES[guide[f]['diagnoses'][0][0]]}."
            else:
                det = max(guide[f]["detects"], key=lambda p: min(cs[(f, p)]["detect"], cm[(f, p)]["detect"]))
                add = f" No plot diagnoses it; the {NAMES[det]} at least detects it."
            L.append(f"- **{f.replace('_', ' ')}**: the dashboard {state}.{add}")
    (RUNS / "PLOT_GUIDE.md").write_text("\n".join(L) + "\n")
    json.dump(guide, open(RUNS / "plot_guide.json", "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
