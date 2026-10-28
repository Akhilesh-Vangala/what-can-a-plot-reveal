"""How robust are the evidential-map labels, and the "blind" label in particular? (instructor review, point 4)

  python blind_sensitivity.py runs/confirm_synthetic runs/confirm_magic   -> runs/BLIND_SENSITIVITY.md (+ .json)

(a) Label counts over a grid of detection-AUC cutoffs (0.60–0.80) and diagnosis-recall cutoffs (0.30–0.70).
(b) Every cell that is blind at the default cutoffs (AUC < 0.70), with the evidence behind the label:
    B1* AUC and 95% CI; the best validation AUC of each estimator family on the exact plotted values (logistic,
    gradient boosting, random forest, MLP) and of the image model (ResNet-18, B2); whether the cell is blind by
    construction (certify_blind.py); and the share of grid cutoffs under which the cell stays blind.
    Grade: certified (identical plot to the healthy twin) > robust (CI upper bound < 0.70 and every estimator < 0.70)
    > borderline (otherwise: the label depends on the cutoff or the estimator).
"""
import json
import pathlib
import sys

import numpy as np

from score_confirm import Run

AUCS = [0.60, 0.65, 0.70, 0.75, 0.80]
RECALLS = [0.30, 0.40, 0.50, 0.60, 0.70]


def cells(R):
    j = R.top
    out = []
    for f in R.meta["failures"]:
        for sp in R.meta["specs"]:
            b1, b2 = R.det[(f, j)][sp], R.b2det.get((f, j, sp))
            use_b2 = b2 is not None and b2["val_auc"] > b1["val_auc"]
            auc = (b2 if use_b2 else b1)["test_auc"]
            ci = None if use_b2 else b1.get("ci")
            if ci is None:  # bootstrap the B1* test scores (B2 records carry no CI)
                from score_confirm import auc_of, boot
                sc, _, ids = R.det_scores(f, j, sp)
                ci = boot(lambda u: auc_of(sc, R.truth, u), ids, n=500)
            post = R.dia_post(j, sp)
            fid = [k for k in post if R.truth[k] == f]
            rec = float(np.mean([R.classes[int(np.argmax(post[k]))] == f for k in fid])) if fid else 0.0
            fam = dict(b1["family_val_auc"])
            if b2 is not None:
                fam["resnet18_image"] = b2["val_auc"]
            out.append(dict(failure=f, plot=sp, auc=float(auc), ci=[float(ci[0]), float(ci[1])], recall=rec, families=fam))
    return out


def label(c, a, r):
    return "identifiable" if c["auc"] >= a and c["recall"] >= r else ("detect-only" if c["auc"] >= a else "blind")


def main(paths):
    L = ["# Sensitivity of the evidential map (instructor review, point 4)", "",
         "Strongest severity. B1* = validation-best of the plotted-value benchmark (B1) and the image benchmark (B2).", ""]
    js = {}
    for p in paths:
        R = Run(p)
        C = cells(R)
        cert_f = pathlib.Path(p) / "certified_blind.json"
        cert = {tuple(x) for x in json.load(open(cert_f))["certified"]} if cert_f.exists() else set()
        name = pathlib.Path(p).name
        L += [f"## {name}", "", "### (a) Label counts under alternative cutoffs (identifiable / detect-only / blind)", "",
              "| AUC cutoff \\ recall cutoff | " + " | ".join(f"{r:.2f}" for r in RECALLS) + " |", "|---|" + "---|" * len(RECALLS)]
        grid = {}
        for a in AUCS:
            row = []
            for r in RECALLS:
                labs = [label(c, a, r) for c in C]
                n = (labs.count("identifiable"), labs.count("detect-only"), labs.count("blind"))
                grid[f"{a}|{r}"] = n
                row.append(("**%d/%d/%d**" if (a, r) == (0.70, 0.50) else "%d/%d/%d") % n)
            L.append(f"| {a:.2f} | " + " | ".join(row) + " |")
        stab = {(c["failure"], c["plot"]): np.mean([label(c, a, r) == label(c, 0.70, 0.50) for a in AUCS for r in RECALLS]) for c in C}
        L += ["", f"Cells whose label is the same under all {len(AUCS) * len(RECALLS)} cutoff pairs: "
              f"{sum(v == 1 for v in stab.values())}/{len(C)}; median share of cutoff pairs that keep a cell's label: "
              f"{np.median(list(stab.values())):.2f}.", "",
              "### (b) Every blind cell (AUC < 0.70 at the default cutoffs) and the evidence behind it", "",
              "| failure | plot | B1* AUC [95% CI] | best estimator (val AUC) | blind under cutoffs | grade |", "|---|---|---|---|---|---|"]
        rows = []
        for c in C:
            if label(c, 0.70, 0.50) != "blind":
                continue
            k = (c["failure"], c["plot"])
            best = max(c["families"].items(), key=lambda x: x[1])
            blind_share = np.mean([c["auc"] < a for a in AUCS])
            if k in cert:
                g = "certified"
            elif c["ci"][1] < 0.70 and best[1] < 0.70:
                g = "robust"
            else:
                g = "borderline"
            rows.append(dict(c, grade=g, best_estimator=best[0], best_val=best[1], blind_share=float(blind_share)))
            L.append(f"| {c['failure']} | {c['plot']} | {c['auc']:.2f} [{c['ci'][0]:.2f}, {c['ci'][1]:.2f}] | "
                     f"{best[0]} ({best[1]:.2f}) | {blind_share:.0%} | {g} |")
        n = {g: sum(r["grade"] == g for r in rows) for g in ("certified", "robust", "borderline")}
        L += ["", f"Blind cells: {len(rows)} = {n['certified']} certified + {n['robust']} robust + {n['borderline']} borderline.", ""]
        js[name] = dict(grid=grid, blind=rows, stability={f"{a}|{b}": v for (a, b), v in stab.items()})
    out = pathlib.Path(paths[0]).parent
    (out / "BLIND_SENSITIVITY.md").write_text("\n".join(L) + "\n")
    json.dump(js, open(out / "blind_sensitivity.json", "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1:] or ["runs/confirm_synthetic", "runs/confirm_magic"])
