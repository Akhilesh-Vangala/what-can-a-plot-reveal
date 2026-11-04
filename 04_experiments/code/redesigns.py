"""What each plot redesign changes, for the benchmark and for the AI readers.

  python redesigns.py   -> runs/REDESIGNS.md (+ redesigns.json), ../explorer/data_redesigns.js (Explorer "Redesigns" view)

Collects, from existing result files only (no new runs):
  bin sizes drawn   reliability diagram -> reliability + bin sizes (rel -> rel_n), strongest severity, both runs:
                    benchmark B1* detection AUC, diagnosis recall and cell label per failure; AI readers (synthetic main
                    item set, same models, severities pooled): detection AUC (score = 1 - P(healthy)) and diagnosis recall
  companion plot    benchmark balanced diagnosis accuracy, plot alone vs plot + companion (companion_benchmarks.json);
                    AI readers: balanced accuracy, companion - plain, on the same items (responses_*_iv.jsonl)
  values printed    direct 7-way diagnosis and read-then-reason (reader transcribes values, fixed rule diagnoses) on
                    original vs annotated plots, with the exact-values ceiling (from GAPS_RESULTS.md, score_gaps.py)
  several plots     benchmark: best single plot vs all nine (plotsets.json)
"""
import collections
import json
import pathlib
import re

import numpy as np

from score_confirm import CLASSES, Run, auc_of, bal_acc, reader_vec

RUNS = pathlib.Path(__file__).parent / "runs"


def label(a, r):
    return "identifiable" if a >= 0.7 and r >= 0.5 else ("detect-only" if a >= 0.7 else "blind")


def bench_cell(R, f, sp):
    sc, _, ids = R.det_scores(f, R.top, sp)
    post = R.dia_post(R.top, sp)
    rec = np.mean([R.classes[int(np.argmax(v))] == f for k, v in post.items() if R.truth[k] == f])
    a = auc_of(sc, R.truth, ids)
    return dict(auc=round(a, 3), recall=round(float(rec), 3), label=label(a, rec))


def reader_answers(rows, sp):
    out = {}
    for r in rows:
        if r.get("task") == "diagnosis" and r.get("spec") == sp:
            v = reader_vec(r)
            if v is not None:
                out[r["run_id"]] = v
    return out


def bin_sizes(R, readers):
    bench = {f: {sp: bench_cell(R, f, sp) for sp in ("rel", "rel_n")} for f in R.meta["failures"]}
    rd = {}
    for name, rows in readers.items():
        A = {sp: reader_answers(rows, sp) for sp in ("rel", "rel_n")}
        ids = sorted(set(A["rel"]) & set(A["rel_n"]))
        per = {}
        for f in R.meta["failures"]:
            sub = [i for i in ids if R.truth[i] in ("healthy", f)]
            per[f] = {sp: dict(auc=round(auc_of({i: 1 - A[sp][i][0] for i in sub}, R.truth, sub), 3),
                               recall=round(float(np.mean([CLASSES[int(np.argmax(A[sp][i]))] == f
                                                           for i in sub if R.truth[i] == f])), 3))
                      for sp in ("rel", "rel_n")}
        pred = {sp: {i: CLASSES[int(np.argmax(A[sp][i]))] for i in ids} for sp in A}
        rd[name] = dict(n=len(ids), per_failure=per,
                        balanced={sp: round(bal_acc(pred[sp], R.truth, ids, R.classes), 3) for sp in A})
    return dict(benchmark=bench, readers=rd)


def companions(R):
    cb = json.load(open(R.p / "companion_benchmarks.json"))
    rd = {}
    for fp in sorted(R.p.glob("responses_*_iv.jsonl")):
        model = fp.stem.replace("responses_", "").replace("_iv", "")
        iv = [json.loads(x) for x in open(fp)]
        keys = {(r["run_id"], r["spec"]) for r in iv}
        plain = {(r["run_id"], r["spec"]): r for r in map(json.loads, open(R.p / f"responses_{model}.jsonl"))
                 if r.get("task") == "diagnosis" and (r["run_id"], r["spec"]) in keys}
        comp = {(r["run_id"], r["spec"]): r for r in iv if r["condition"] == "companion"}
        ans = lambda D: {k: CLASSES[int(np.argmax(v))] for k, r in D.items() if (v := reader_vec(r)) is not None}  # noqa: E731
        ap, ac = ans(plain), ans(comp)
        common = [k for k in ap if k in ac]
        tr = {k: R.truth[k[0]] for k in common}
        rd[model] = dict(n=len(common), plain=round(bal_acc(ap, tr, common, R.classes), 3),
                         companion=round(bal_acc(ac, tr, common, R.classes), 3))
    return dict(benchmark={sp: dict(companion=v["companion"], alone=round(v["single_balanced_acc"], 3),
                                    with_companion=round(v["pair_balanced_acc"], 3)) for sp, v in cb.items()},
                readers=rd)


def values_printed(path):
    rows, ceiling = {}, {}
    for line in open(path):
        m = re.match(r"\| (\S+) \| (original|annotated) \|" + r" ([\d.]+) \|" * 7, line)
        if m:
            g = m.groups()
            rows.setdefault(g[0], {})[g[1]] = dict(direct=dict(zip(["rel", "roc", "dist"], map(float, g[5:8]))),
                                                   rtr=dict(zip(["rel", "roc", "dist"], map(float, g[2:5]))))
        m = re.match(r"\| exact values \(ceiling\) \| — \|" + r" ([\d.]+) \|" * 3, line)
        if m:
            ceiling = dict(zip(["rel", "roc", "dist"], map(float, m.groups())))
    return dict(readers=rows, ceiling=ceiling)


def plotsets(R):
    ps = json.load(open(R.p / "plotsets.json"))
    single = max((r for r in ps if len(r["plots"]) == 1), key=lambda r: r["balanced_acc"])
    nine = [r for r in ps if len(r["plots"]) == 9][0]
    return dict(best_single=single["plots"][0], single=round(single["balanced_acc"], 3), all_nine=round(nine["balanced_acc"], 3))


def short(name):
    return name.split("-Instruct")[0].replace("llava-onevision-qwen2-7b-ov-hf", "LLaVA-OV-7B").replace("-AWQ", "")


def main():
    S, M = Run(RUNS / "confirm_synthetic"), Run(RUNS / "confirm_magic")
    readers = {k: v for k, v in S.responses("responses_*.jsonl").items() if not re.search(r"_[a-z0-9]+$", k)}
    js = dict(bin_sizes={"confirm_synthetic": bin_sizes(S, readers), "confirm_magic": bin_sizes(M, {})},
              companion=companions(S), values_printed=values_printed(S.p / "GAPS_RESULTS.md"),
              plotsets={R.p.name: plotsets(R) for R in (S, M)})
    L = ["# What each redesign changes (existing results only)", "",
         "Benchmark = how much evidence the plot carries (B1*, strongest severity). Readers = the open AI readers on "
         "the same models (synthetic run).", "", "## 1. Bin sizes drawn on the reliability diagram (rel -> rel_n)", ""]
    for run, d in js["bin_sizes"].items():
        L += [f"**{run}, benchmark**", "", "| failure | AUC rel -> +sizes | recall rel -> +sizes | label rel -> +sizes |",
              "|---|---|---|---|"]
        for f, c in d["benchmark"].items():
            a, b = c["rel"], c["rel_n"]
            L.append(f"| {f} | {a['auc']:.2f} -> {b['auc']:.2f} | {a['recall']:.2f} -> {b['recall']:.2f} | "
                     f"{a['label']} -> {b['label']}{' **changed**' if a['label'] != b['label'] else ''} |")
        L.append("")
        if d["readers"]:
            L += [f"**{run}, AI readers** (detection AUC for class imbalance; balanced 7-way accuracy, chance 0.14)", "",
                  "| reader | n models | imbalance AUC rel -> +sizes | imbalance recall rel -> +sizes | balanced acc rel -> +sizes |",
                  "|---|---|---|---|---|"]
            for n, r in d["readers"].items():
                ci = r["per_failure"]["class_imbalance"]
                L.append(f"| {short(n)} | {r['n']} | {ci['rel']['auc']:.2f} -> {ci['rel_n']['auc']:.2f} | "
                         f"{ci['rel']['recall']:.2f} -> {ci['rel_n']['recall']:.2f} | "
                         f"{r['balanced']['rel']:.2f} -> {r['balanced']['rel_n']:.2f} |")
            L.append("")
    c = js["companion"]
    L += ["## 2. Companion plot added (synthetic)", "", "| plot | companion | benchmark alone -> with companion |", "|---|---|---|"]
    L += [f"| {sp} | {v['companion']} | {v['alone']:.2f} -> {v['with_companion']:.2f} |" for sp, v in c["benchmark"].items()]
    L += ["", "| reader | items | balanced acc plain -> with companion |", "|---|---|---|"]
    L += [f"| {short(n)} | {r['n']} | {r['plain']:.2f} -> {r['companion']:.2f} |" for n, r in c["readers"].items()]
    L += ["", "## 3. Values printed on the plot (synthetic; direct 7-way diagnosis, chance 0.14)", "",
          "| reader | reliability | ROC | histogram |", "|---|---|---|---|"]
    vp = js["values_printed"]
    for n, r in vp["readers"].items():
        o, a = r["original"]["direct"], r["annotated"]["direct"]
        L.append(f"| {short(n)} | " + " | ".join(f"{o[k]:.2f} -> {a[k]:.2f}" for k in ("rel", "roc", "dist")) + " |")
    L += ["", "Read-then-reason (reader transcribes values, fixed rule diagnoses), ceiling = exact values: "
          + ", ".join(f"{k} {v:.2f}" for k, v in vp["ceiling"].items()), "",
          "| reader | reliability | ROC | histogram |", "|---|---|---|---|"]
    for n, r in vp["readers"].items():
        o, a = r["original"]["rtr"], r["annotated"]["rtr"]
        L.append(f"| {short(n)} | " + " | ".join(f"{o[k]:.2f} -> {a[k]:.2f}" for k in ("rel", "roc", "dist")) + " |")
    L += ["", "## 4. Several plots together (benchmark, balanced diagnosis accuracy)", ""]
    L += [f"- {run}: best single plot ({d['best_single']}) {d['single']:.2f} -> all nine {d['all_nine']:.2f}"
          for run, d in js["plotsets"].items()]
    (RUNS / "REDESIGNS.md").write_text("\n".join(L) + "\n")
    json.dump(js, open(RUNS / "redesigns.json", "w"), indent=1)
    (RUNS.parent.parent / "explorer" / "data_redesigns.js").write_text("window.REDESIGNS = " + json.dumps(js) + ";\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
