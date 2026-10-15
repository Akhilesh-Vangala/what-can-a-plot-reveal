"""Saturation check from the analysis plan: flag benchmark cells whose held-out detection AUC still gains > 0.03 when
training data grows from 50% to 100% (the benchmark may underestimate what the plot carries). Reports whether any flag
touches a conclusion: an H1 blind-spot cell or an evidential-map label at the strongest level.

  python saturation_report.py runs/confirm_synthetic runs/confirm_magic   -> runs/confirm_synthetic/SATURATION.md
"""
import json
import pathlib
import sys

H1 = {("miscalibration", s) for s in ("roc", "pr", "cm", "lc_acc")} | {(f, s) for f in ("class_imbalance", "distribution_shift")
                                                                       for s in ("lc_loss", "lc_acc")}


def main(paths):
    L = ["# Saturation check (benchmark still gaining > 0.03 AUC from 50% to 100% training data)", ""]
    for p in map(pathlib.Path, paths):
        r = json.load(open(p / "results.json"))
        top = max(x["level"] for x in r["detection"])
        rows = []
        for row in r["detection"]:
            for sp, v in row.items():
                if isinstance(v, dict) and v.get("saturation") and sp not in ("B0", "B0_state_only"):
                    s = dict((round(a, 2), b) for a, b in v["saturation"])
                    gain = s[1.0] - s[0.5]
                    if gain > 0.03:
                        rows.append((row["failure"], row["level"], sp, gain, v["test_auc"]))
        crit = [x for x in rows if x[1] == top and ((x[0], x[2]) in H1 or 0.62 <= x[4] <= 0.78)]
        L += [f"## {p.name}: {len(rows)} flagged plot cells (all levels)", "",
              "| failure | level | plot | gain 50%→100% | test AUC | touches a conclusion? |", "|---|---|---|---|---|---|"]
        for f, j, sp, g, a in sorted(rows):
            touch = ("H1 blind-spot cell" if j == top and (f, sp) in H1 else
                     "near the 0.70 map threshold" if j == top and 0.62 <= a <= 0.78 else "no (not strongest level / far from threshold)")
            L.append(f"| {f} | {j} | {sp} | +{g:.3f} | {a:.3f} | {touch} |")
        L += ["", f"Cells at the strongest level that touch a conclusion: {len(crit)}.", ""]
    out = pathlib.Path(paths[0]) / "SATURATION.md"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(l for l in L if l.startswith("##") or "touch a conclusion" in l or "H1 blind" in l or "near the 0.70" in l))


if __name__ == "__main__":
    main(sys.argv[1:])
