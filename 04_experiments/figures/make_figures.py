"""Final figures, computed from the result files (no hand-entered numbers).

  python make_figures.py        -> fig1_evidential_map ... fig7_realworld (.pdf + .png) in this folder
"""
import collections
import json
import pathlib
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

HERE = pathlib.Path(__file__).parent
CODE = HERE.parent / "code"
RUNS = CODE / "runs"
sys.path.insert(0, str(CODE))
from score_confirm import CLASSES, Run, auc_of, is_diagnosis_file, reader_vec  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 8,
                     "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6})
FAIL = ["overfitting", "miscalibration", "label_noise", "class_imbalance", "data_leakage", "distribution_shift"]
FLAB = {"healthy": "Healthy", "overfitting": "Overfitting", "miscalibration": "Miscalibration", "label_noise": "Label noise",
        "class_imbalance": "Class imbalance", "data_leakage": "Data leakage", "distribution_shift": "Distribution shift"}
SPEC = ["lc_loss", "lc_acc", "rel", "rel_n", "cm", "cm_norm", "roc", "pr", "dist"]
SLAB = {"lc_loss": "Loss curves", "lc_acc": "Accuracy curves", "rel": "Reliability", "rel_n": "Reliability\n+ bin sizes",
        "cm": "Confusion\nmatrix", "cm_norm": "Confusion m.\n(row-norm.)", "roc": "ROC", "pr": "Precision–\nrecall",
        "dist": "Score\nhistogram"}
SROT = {"lc_loss": "Loss curves", "lc_acc": "Accuracy curves", "rel": "Reliability", "rel_n": "Reliability + sizes",
        "cm": "Confusion matrix", "cm_norm": "Conf. matrix (norm.)", "roc": "ROC", "pr": "Precision–recall",
        "dist": "Score histogram"}
RLAB = {"InternVL3-38B-AWQ": "InternVL3-38B", "InternVL3-8B": "InternVL3-8B", "Qwen2.5-VL-32B-Instruct-AWQ": "Qwen2.5-VL-32B",
        "Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B", "llava-onevision-qwen2-7b-ov-hf": "LLaVA-OV-7B"}
ORANGE, BLUE, GREY = "#d95f02", "#08306b", "#9e9e9e"
CMAP = LinearSegmentedColormap.from_list("ev", ["#f7f7f7", "#c6dbef", "#4292c6", "#08306b"])


def rl(name):
    for k, v in RLAB.items():
        if name.startswith(k):
            return v
    return name


def save(fig, name):
    fig.savefig(HERE / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(HERE / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def evidential(R):
    j = R.top
    A, Rc = np.zeros((6, 9)), np.zeros((6, 9))
    for a, f in enumerate(FAIL):
        for b, s in enumerate(SPEC):
            sc, _, ids = R.det_scores(f, j, s)
            A[a, b] = auc_of(sc, R.truth, ids)
            post = R.dia_post(j, s)
            pred = {k: R.classes[int(np.argmax(v))] for k, v in post.items()}
            Rc[a, b] = np.mean([pred[k] == f for k in post if R.truth[k] == f])
    return A, Rc


def fig1(runs):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.55), gridspec_kw=dict(wspace=0.08))
    for ax, (title, R) in zip(axes, runs):
        A, Rc = evidential(R)
        im = ax.imshow(A, cmap=CMAP, vmin=.5, vmax=1, aspect="auto")
        for a in range(6):
            for b in range(9):
                if A[a, b] >= .7 and Rc[a, b] < .5:
                    ax.add_patch(Rectangle((b - .46, a - .42), .92, .84, fill=False, ec=ORANGE, lw=1.2, ls=(0, (2, 1.2))))
                elif A[a, b] < .7:
                    ax.plot([b - .42, b + .42], [a + .38, a - .38], color=GREY, lw=.5)
                ax.text(b, a, f"{A[a, b]:.2f}", ha="center", va="center", fontsize=5.6, color="white" if A[a, b] > .82 else "black")
        ax.set_xticks(range(9), [SROT[s] for s in SPEC], fontsize=6, rotation=35, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(6), [FLAB[f] for f in FAIL] if ax is axes[0] else [], fontsize=6.5)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        n_id = int(((A >= .7) & (Rc >= .5)).sum())
        n_do = int(((A >= .7) & (Rc < .5)).sum())
        ax.set_title(f"{title}: {n_id} identifiable, {n_do} detect-only, {54 - n_id - n_do} blind", fontsize=7.5, loc="left")
    cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01)
    cb.ax.tick_params(labelsize=6)
    cb.set_label("benchmark detection AUC", fontsize=6.5)
    save(fig, "fig1_evidential_map")


def fig2(runs):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.3), sharey=True)
    txt = (RUNS / "confirm_synthetic/CONFIRM_RESULTS.md").read_text()
    for ax, (title, R) in zip(axes, runs):
        d = R.dia[R.top]
        b0 = d["B0"]["balanced_acc"]
        vals = [d[s]["balanced_acc"] for s in SPEC]
        order = np.argsort(vals)[::-1]
        ax.bar(range(9), [vals[i] for i in order], color="#6baed6", width=.7, label="single plot (B1)")
        ax.axhline(b0, color=BLUE, lw=1.2, label=f"full model state B0 ({b0:.2f})")
        sec = txt.split(f"## H5 [{R.p.name}]")[1].split("\n## ")[0]
        rd = [float(x) for x in re.findall(r"reader ([0-9.]+) vs B2", sec)]
        ax.axhspan(min(rd) - .004, max(rd) + .004, color=ORANGE, alpha=.45, lw=0, label="AI readers (range)")
        ax.axhline(1 / 7, color=GREY, ls=":", lw=.8, label="chance (1/7)")
        ax.set_xticks(range(9), [SROT[SPEC[i]] for i in order], fontsize=6, rotation=35, ha="right", rotation_mode="anchor")
        ax.set_title(title, fontsize=7.5, loc="left")
        ax.set_ylim(0, .9)
        if ax is axes[0]:
            ax.set_ylabel("balanced diagnosis accuracy (7 states)", fontsize=7)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, [x.split(" (")[0] if x.startswith("full") else x for x in l], fontsize=6.3, frameon=False, ncol=4,
               loc="upper center", bbox_to_anchor=(0.5, 1.06))
    save(fig, "fig2_diagnosis_by_plot")


def fig3():
    txt = (RUNS / "confirm_synthetic/CONFIRM_RESULTS.md").read_text()
    rows = []
    for src, lab in (("confirm_synthetic", "Synthetic"), ("confirm_magic", "Real (MagicTelescope)")):
        sec = txt.split(f"## H4 [{src}]")[1].split("\n## ")[0]
        for pat, name in ((r"imbalance detection rel_n − rel: ([+-][0-9.]+) \[([+-][0-9.]+), ([+-][0-9.]+)\]", "Imbalance detection:\nbin sizes drawn"),
                          (r"diagnosis rel_n − rel: ([+-][0-9.]+) \[([+-][0-9.]+), ([+-][0-9.]+)\]", "Diagnosis:\nbin sizes drawn"),
                          (r"diagnosis cm_norm − cm: ([+-][0-9.]+) \[([+-][0-9.]+), ([+-][0-9.]+)\]", "Diagnosis:\nrow-normalized CM")):
            m = re.search(pat, sec)
            rows.append((lab, name, float(m[1]), float(m[2]), float(m[3])))
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    names = [r[1] for r in rows[:3]]
    for k, (lab, col, off) in enumerate((("Synthetic", BLUE, -.12), ("Real (MagicTelescope)", ORANGE, .12))):
        R_ = [r for r in rows if r[0] == lab]
        y = np.arange(3) + off
        ax.errorbar([r[2] for r in R_], y, xerr=[[r[2] - r[3] for r in R_], [r[4] - r[2] for r in R_]], fmt="o", ms=3.5,
                    color=col, lw=1, capsize=2, label=lab)
    ax.axvline(0, color=GREY, lw=.8)
    ax.set_yticks(range(3), names, fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("change vs. standard design (95% CI)", fontsize=7)
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    save(fig, "fig3_design_variants")


def fig4():
    txt = (RUNS / "confirm_synthetic/CONFIRM_RESULTS.md").read_text()
    sec2 = json.load(open(RUNS / "confirm_synthetic/secondary_results.json"))

    def grab(section, label=None):
        block = txt.split(f"## {section}")[1].split("\n## ")[0]
        return [(rl(m[0]), float(m[2]), float(m[3])) for m in re.findall(r"- (\S+) \(([^)]*)\): reader ([0-9.]+) vs B2 ([0-9.]+)", block)
                if label is None or m[1] == label]
    rows = [("Diagnose the plot", grab("H5 [confirm_synthetic]")), ("…with one example per state", grab("H7a [confirm_synthetic]", "fewshot")),
            ("…with the values as text", grab("H7b [confirm_synthetic]", "numbers")), ("Real data (MagicTelescope)", grab("H5 [confirm_magic]"))]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 2.1), gridspec_kw=dict(width_ratios=[1.6, 1], wspace=.55))
    for y, (lab, R_) in enumerate(rows):
        ax.plot([min(r[1] for r in R_), max(r[2] for r in R_)], [y, y], color="#e0e0e0", lw=3, zorder=0, solid_capstyle="round")
        ax.scatter([r[1] for r in R_], [y] * len(R_), s=12, color=ORANGE, zorder=3, lw=0)
        ax.scatter([np.mean([r[2] for r in R_])], [y], s=22, marker="D", color=BLUE, zorder=3, lw=0)
    ax.axvline(1 / 7, color=GREY, ls=":", lw=.8)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlim(.05, .55)
    ax.set_xlabel("balanced diagnosis accuracy (7 states)", fontsize=7)
    ax.scatter([], [], s=12, color=ORANGE, label="each AI reader")
    ax.scatter([], [], s=22, marker="D", color=BLUE, label="image benchmark B2")
    ax.legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.55, -0.22), ncol=2)
    ax.set_title("(a) Readers stay at chance under every control", fontsize=7.5, loc="left")
    rs = [(rl(k.split("|")[1]), v["unchanged"]) for k, v in sec2.items() if k.startswith("confirm_synthetic") and k.endswith("no_plot")]
    rs.sort(key=lambda r: -r[1])
    bx.barh(range(len(rs)), [r[1] for r in rs], color="#6baed6", height=.6)
    for i, r in enumerate(rs):
        bx.text(r[1] + .01, i, f"{r[1]:.0%}", va="center", fontsize=6)
    bx.set_yticks(range(len(rs)), [r[0] for r in rs], fontsize=6.5)
    bx.invert_yaxis()
    bx.set_xlim(0, .7)
    bx.set_xlabel("same answer as with the plot hidden", fontsize=7)
    bx.set_title("(b) Answers often ignore the plot", fontsize=7.5, loc="left")
    save(fig, "fig4_readers_controls")


def fig5(R):
    """Top answer on HEALTHY plots, per reader and plot type."""
    readers = {k: v for k, v in R.responses("responses_*.jsonl").items() if is_diagnosis_file(k)}
    names = sorted(readers, key=rl)
    M = np.full((len(names), 9), np.nan)
    lab = [[""] * 9 for _ in names]
    short = {"overfitting": "OF", "miscalibration": "MC", "label_noise": "LN", "class_imbalance": "CI", "data_leakage": "DL",
             "distribution_shift": "DS", "healthy": "H"}
    for i, n in enumerate(names):
        cnt = collections.defaultdict(collections.Counter)
        for r in readers[n]:
            if r.get("task") == "diagnosis" and R.truth.get(r["run_id"]) == "healthy":
                v = reader_vec(r)
                if v is not None:
                    cnt[r["spec"]][CLASSES[int(np.argmax(v))]] += 1
        for b, s in enumerate(SPEC):
            if cnt[s]:
                a, c = cnt[s].most_common(1)[0]
                M[i, b] = c / sum(cnt[s].values())
                lab[i][b] = short[a]
    fig, ax = plt.subplots(figsize=(4.6, 1.8))
    im = ax.imshow(M, cmap=LinearSegmentedColormap.from_list("st", ["#fff5eb", "#fd8d3c", "#7f2704"]), vmin=0, vmax=1, aspect="auto")
    for i in range(len(names)):
        for b in range(9):
            if not np.isnan(M[i, b]):
                ax.text(b, i, f"{lab[i][b]}\n{M[i, b]:.0%}", ha="center", va="center", fontsize=5, color="white" if M[i, b] > .6 else "black")
    ax.set_xticks(range(9), [SROT[s] for s in SPEC], fontsize=6, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(names)), [rl(n) for n in names], fontsize=6.3)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Most common answer on HEALTHY models (OF overfitting, MC miscalibration, H healthy, …)", fontsize=6.8, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.ax.tick_params(labelsize=5.5)
    save(fig, "fig5_stereotypes")


def fig6():
    P = json.load(open(RUNS / "positive/positive_results.json"))
    canon = ["overfitting", "miscalibration", "class_imbalance", "data_leakage", "distribution_shift", "label_noise"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 2.3), gridspec_kw=dict(width_ratios=[1.45, 1], wspace=.5))
    names = sorted({k.split("|")[0] for k in P["positive"]}, key=rl)
    A = np.array([[P["positive"][f"{n}|synthetic"]["cells"][f]["auc"] for f in canon] for n in names], float)
    im = ax.imshow(np.nan_to_num(A, nan=.5), cmap=CMAP, vmin=.5, vmax=1, aspect="auto")
    for i, n in enumerate(names):
        for b_, f in enumerate(canon):
            c = P["positive"][f"{n}|synthetic"]["cells"][f]
            auc = "–" if np.isnan(c["auc"]) else f"{c['auc']:.2f}"
            pct = lambda x: "–" if x is None or np.isnan(x) else f"{x:.0%}"  # noqa: E731
            ax.text(b_, i, f"{auc}\n{pct(c['named'])} / {pct(c['false_alarm'])}", ha="center", va="center", fontsize=5,
                    color="white" if (not np.isnan(c["auc"]) and c["auc"] > .85) else "black")
    ax.set_xticks(range(6), [FLAB[f] + ("*" if f == "label_noise" else "") for f in canon], fontsize=6, rotation=25, ha="right",
                  rotation_mode="anchor")
    ax.set_yticks(range(len(names)), [rl(n) for n in names], fontsize=6.3)
    ax.tick_params(length=0)
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    ax.set_title("(a) Blatant failures: detection AUC; named on failure / healthy flagged", fontsize=7, loc="left")
    probe = P["probe"]
    rows = []
    for n, res in probe.items():
        c = [res[f"lc_loss|{q}"]["corr"] for q in ("train_last", "val_last", "val_best")]
        rows.append((rl(n), float(np.mean(c)), res.get("lc_loss|overfit_auc", {}).get("own", np.nan),
                     res.get("lc_loss|overfit_auc", {}).get("true", np.nan)))
    rows.sort(key=lambda r: -r[1])
    y = np.arange(len(rows))
    bx.barh(y, [r[1] for r in rows], color="#6baed6", height=.55, label="reading accuracy (r)")
    bx.scatter([r[2] for r in rows], y, color=ORANGE, s=14, zorder=3, label="overfitting AUC from own readings")
    bx.axvline(rows[0][3], color=BLUE, lw=1, ls="--", label=f"AUC from true values ({rows[0][3]:.2f})")
    bx.set_yticks(y, [r[0] for r in rows], fontsize=6.3)
    bx.invert_yaxis()
    bx.set_xlim(0, 1.05)
    bx.legend(fontsize=5.6, frameon=False, loc="upper center", bbox_to_anchor=(0.4, -0.12), ncol=1)
    bx.set_title("(b) Values are read almost exactly", fontsize=7.2, loc="left")
    save(fig, "fig6_positive_probe")


def fig7():
    T = json.load(open(RUNS / "realworld/transfer.json"))
    W = json.load(open(RUNS / "realworld/realworld_results.json"))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 2.1), gridspec_kw=dict(wspace=.4))
    lab = np.array([o["map_label"] != "blind" for o in T])
    real = np.array([o["real_auc"] >= .7 for o in T])
    M = np.array([[(lab & real).sum(), (lab & ~real).sum()], [(~lab & real).sum(), (~lab & ~real).sum()]])
    ax.imshow(M, cmap="Blues", vmin=0)
    for a in range(2):
        for b in range(2):
            ax.text(b, a, str(M[a, b]), ha="center", va="center", fontsize=9, color="white" if M[a, b] > M.max() / 2 else "black")
    ax.set_xticks([0, 1], ["detectable\non real models", "not detectable"], fontsize=6.5)
    ax.set_yticks([0, 1], ["map: detectable", "map: blind"], fontsize=6.5)
    ax.set_title(f"(a) Evidential map on real models: {np.mean(lab == real):.0%} agreement", fontsize=7.2, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    names = sorted(W, key=rl)
    det = [np.mean([c["reader_auc"] >= .7 for c in W[n] if not np.isnan(c["reader_auc"])]) for n in names]
    bench = np.mean([o["real_auc"] >= .7 for o in T])
    bx.barh(range(len(names)), det, color=ORANGE, height=.55)
    bx.axvline(bench, color=BLUE, lw=1.2, label=f"benchmark ({bench:.0%})")
    bx.set_yticks(range(len(names)), [rl(n) for n in names], fontsize=6.3)
    bx.invert_yaxis()
    bx.set_xlim(0, 1)
    bx.set_xlabel("share of case–plot pairs where the real failure is detected", fontsize=6.6)
    bx.legend(fontsize=6, frameon=False, loc="lower right")
    bx.set_title("(b) AI readers on real failures", fontsize=7.2, loc="left")
    save(fig, "fig7_realworld")




def fig8():
    """Read-then-reason vs direct diagnosis on the same synthetic core models."""
    sys.path.insert(0, str(CODE))
    import rtr
    V = json.load(open(RUNS / "confirm_synthetic/rtr_values.json"))
    R = json.load(open(RUNS / "confirm_synthetic/rtr_results.json"))
    specs = ("lc_loss", "cm", "rel", "roc", "dist")
    core = {json.loads(x)["run_id"] for x in open(RUNS / "confirm_synthetic/responses_InternVL3-8B_extract.jsonl")}

    def bal(y, p):
        return float(np.mean([np.mean(p[y == c] == c) for c in np.unique(y)]))
    direct, rtr_acc = collections.defaultdict(list), collections.defaultdict(list)
    for key, val in R.items():
        if key.startswith("syn|"):
            for sp in specs:
                rtr_acc[sp].append(val["per_plot"][sp])
    files = {"InternVL3-38B-AWQ": "InternVL3-38B-AWQ_core", "InternVL3-8B": "InternVL3-8B", "Qwen2.5-VL-32B-Instruct-AWQ":
             "Qwen2.5-VL-32B-Instruct-AWQ", "Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B-Instruct",
             "llava-onevision-qwen2-7b-ov-hf": "llava-onevision-qwen2-7b-ov-hf"}
    for m, f in files.items():
        rows = [json.loads(x) for x in open(RUNS / f"confirm_synthetic/responses_{f}.jsonl")]
        for sp in specs:
            y, p = [], []
            for r in rows:
                if r.get("task") == "diagnosis" and r["spec"] == sp and r["run_id"] in core:
                    v = reader_vec(r)
                    if v is not None:
                        y.append(CLASSES.index(V[r["run_id"]]["failure"]))
                        p.append(int(np.argmax(v)))
            direct[sp].append(bal(np.array(y), np.array(p)))
    # ceiling: the rule on exact values of the same core models (as in score_rtr)
    ceil = {}
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    for sp in specs:
        tr = [(rtr.features(v["values"][sp], sp), CLASSES.index(v["failure"])) for v in V.values() if v["split"] == "train"]
        mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced")).fit(
            np.nan_to_num(np.array([a for a, _ in tr], float)), [b for _, b in tr])
        ids = [k for k in core]
        y = np.array([CLASSES.index(V[k]["failure"]) for k in ids])
        ceil[sp] = bal(y, mdl.predict(np.nan_to_num(np.array([rtr.features(V[k]["values"][sp], sp) for k in ids], float))))
    fig, ax = plt.subplots(figsize=(4.8, 2.3))
    x = np.arange(len(specs))
    for off, data, col, lab in ((-.13, direct, ORANGE, "readers diagnose directly"), (.13, rtr_acc, BLUE, "readers read values, rule diagnoses")):
        mn = [np.min(data[s]) for s in specs]
        mx = [np.max(data[s]) for s in specs]
        md = [np.median(data[s]) for s in specs]
        ax.vlines(x + off, mn, mx, color=col, lw=2.2, alpha=.45)
        ax.scatter(x + off, md, color=col, s=16, zorder=3, label=lab + " (median, range of 5)")
    ax.scatter(x, [ceil[s] for s in specs], marker="_", s=260, color="black", lw=1.4, zorder=4, label="rule on exact values (ceiling)")
    ax.axhline(1 / 7, color=GREY, ls=":", lw=.8)
    ax.text(len(specs) - .55, 1 / 7 + .008, "chance", fontsize=6, color="#555555")
    ax.set_xticks(x, [SROT[s] for s in specs], fontsize=6.5)
    ax.set_ylabel("balanced diagnosis accuracy", fontsize=7)
    ax.set_ylim(0, .5)
    ax.legend(fontsize=5.8, frameon=False, loc="upper center", bbox_to_anchor=(.5, -.14), ncol=2)
    ax.set_title("Separating reading from reasoning (synthetic core models)", fontsize=7.5, loc="left")
    save(fig, "fig8_read_then_reason")


def fig9():
    """(a) What sets of plots carry (multi-plot benchmark). (b) From chance to near-ceiling: separating reading from
    reasoning, combining plots, reader-specific rules, machine-reader designs (same held-out synthetic core models)."""
    import rtr
    from annotate import BASE
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw=dict(width_ratios=[1, 1.25], wspace=.45))
    for name, col, off in (("confirm_synthetic", BLUE, -.08), ("confirm_magic", ORANGE, .08)):
        S = json.load(open(RUNS / name / "plotsets.json"))
        b0 = json.load(open(RUNS / name / "results.json"))["diagnosis"][2]["B0"]["balanced_acc"]
        groups = [("best single", max(r["balanced_acc"] for r in S if len(r["plots"]) == 1)),
                  ("loss+CM+ROC", [r for r in S if sorted(r["plots"]) == sorted(["lc_loss", "cm", "roc"])][0]["balanced_acc"]),
                  ("best pair", max(r["balanced_acc"] for r in S if len(r["plots"]) == 2)),
                  ("all nine", [r for r in S if len(r["plots"]) == 9][0]["balanced_acc"])]
        x = np.arange(len(groups)) + off
        ax.plot(x, [g[1] for g in groups], "o-", color=col, ms=4, lw=1, label="Synthetic" if "syn" in name else "Real")
        ax.axhline(b0, color=col, lw=.7, ls="--")
    ax.set_xticks(range(4), [g[0] for g in groups], fontsize=6.3, rotation=20, ha="right")
    ax.set_ylim(.5, .9)
    ax.set_ylabel("benchmark balanced accuracy", fontsize=7)
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    ax.text(3.2, .835, "dashed: full\nmodel state", fontsize=5.5, color="#555555", ha="right")
    ax.set_title("(a) Plots together carry what one plot cannot", fontsize=7.3, loc="left")
    V = json.load(open(RUNS / "confirm_synthetic/rtr_values.json"))
    specs = ("lc_loss", "rel", "cm", "roc", "dist")
    dims = {sp: len(rtr.features(next(iter(V.values()))["values"][sp], sp)) for sp in specs}

    def feats(vals, S):
        out = []
        for sp in S:
            f = rtr.features(vals.get(sp), sp) if vals.get(sp) is not None else None
            out += f if f is not None and not any(np.isnan(f)) else [np.nan] * dims[sp]
        return out

    def bal(y, p):
        return float(np.mean([np.mean(p[y == c] == c) for c in np.unique(y)]))
    mk = lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced", C=.5))  # noqa: E731
    exact = {S: mk().fit(np.array([feats(v["values"], S) for v in V.values() if v["split"] == "train"]),
                         [CLASSES.index(v["failure"]) for v in V.values() if v["split"] == "train"]) for S in (("cm",), ("lc_loss", "cm"), specs)}
    steps = ["direct\ndiagnosis", "read → rule\n(one plot)", "read → rule\nloss + CM", "own-reading rule\nall five", "+ annotated\ndesigns"]
    files = {"InternVL3-38B-AWQ": "InternVL3-38B-AWQ_core", "InternVL3-8B": "InternVL3-8B", "Qwen2.5-VL-32B-Instruct-AWQ": "Qwen2.5-VL-32B-Instruct-AWQ",
             "Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B-Instruct", "llava-onevision-qwen2-7b-ov-hf": "llava-onevision-qwen2-7b-ov-hf"}
    for i, (m, f) in enumerate(files.items()):
        O, A = {}, {}
        for line in open(RUNS / f"confirm_synthetic/responses_{m}_extract.jsonl"):
            r = json.loads(line)
            d = rtr.from_reader(r.get("parsed"), r["spec"])
            if d:
                O.setdefault(r["run_id"], {})[r["spec"]] = d
        for line in open(RUNS / f"confirm_synthetic/responses_{m}_ann.jsonl"):
            r = json.loads(line)
            if r["task"] == "extract":
                d = rtr.from_reader(r.get("parsed"), BASE[r["spec"]])
                if d:
                    A.setdefault(r["run_id"], {})[BASE[r["spec"]]] = d
        ks = sorted(O)
        y = np.array([CLASSES.index(V[k]["failure"]) for k in ks])
        dd = {}
        for line in open(RUNS / f"confirm_synthetic/responses_{f}.jsonl"):
            r = json.loads(line)
            if r.get("task") == "diagnosis" and r["spec"] in specs and r["run_id"] in O:
                v = reader_vec(r)
                dd[(r["run_id"], r["spec"])] = int(np.argmax(v)) if v is not None else -1
        dk = list(dd)
        direct = bal(np.array([CLASSES.index(V[k[0]]["failure"]) for k in dk]), np.array([dd[k] for k in dk]))

        def ex(S):
            X = np.array([feats(O[k], S) for k in ks], float)
            return bal(y, exact[S].predict(X))

        def own(src):
            X = np.array([feats(src[k], specs) for k in ks], float)
            p = np.zeros(len(y), int)
            for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
                p[b] = mk().fit(X[a], y[a]).predict(X[b])
            return bal(y, p)
        vals = [direct, ex(("cm",)), ex(("lc_loss", "cm")), own(O), own({k: dict(O[k], **A.get(k, {})) for k in ks})]
        bx.plot(range(5), vals, "o-", ms=3.5, lw=1, color=plt.cm.Blues(.35 + .14 * i) if "Intern" in m else plt.cm.Oranges(.4 + .15 * i), label=rl(m))
    ceil = bal(np.array([CLASSES.index(V[k]["failure"]) for k in V if V[k]["split"] == "test"]),
               exact[specs].predict(np.array([feats(V[k]["values"], specs) for k in V if V[k]["split"] == "test"])))
    bx.axhline(ceil, color="black", lw=.9, ls="--")
    bx.text(4.3, ceil + .01, f"exact values, all five ({ceil:.2f})", fontsize=5.5, ha="right")
    bx.axhline(1 / 7, color=GREY, ls=":", lw=.8)
    bx.set_xticks(range(5), steps, fontsize=5.6)
    bx.set_ylim(0, .85)
    bx.set_ylabel("balanced diagnosis accuracy", fontsize=7)
    bx.legend(fontsize=5.3, frameon=False, loc="lower right", ncol=2)
    bx.set_title("(b) From chance toward the ceiling (same held-out models)", fontsize=7.3, loc="left")
    save(fig, "fig9_sets_and_readers")


if __name__ == "__main__":
    S, M = Run(RUNS / "confirm_synthetic"), Run(RUNS / "confirm_magic")
    runs = [("Synthetic", S), ("Real (MagicTelescope)", M)]
    fig1(runs)
    fig2(runs)
    fig3()
    fig4()
    fig5(S)
    fig6()
    fig7()
    fig8()
    fig9()
