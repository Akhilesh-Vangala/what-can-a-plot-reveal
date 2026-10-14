"""Holm correction within each hypothesis family of CONFIRM_RESULTS.md (as stated in the analysis plan).

  python holm_confirm.py runs/confirm_synthetic/CONFIRM_RESULTS.md   -> prints + writes HOLM.md next to it

One-sided p-values come from the reported bootstrap estimate and 95% CI (normal approximation, se = CI width / 3.92),
in the direction each test predicts. H1 (blind spots) predicts NO effect and H8b is an equivalence test, so Holm
does not apply to them; they are listed as such.
"""
import pathlib
import re
import sys

from scipy.stats import norm


def holm(ps, alpha=0.05):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    rej, m = [False] * len(ps), len(ps)
    for rank, i in enumerate(order):
        if ps[i] <= alpha / (m - rank):
            rej[i] = True
        else:
            break
    return rej


def tests_in(block, title):
    """(label, estimate, lo, hi, sign) for each directional test in a section."""
    out = []
    for line in block.splitlines():
        m = re.search(r"- (.*?): ([+-][0-9.]+) \[([+-][0-9.]+), ([+-][0-9.]+)\]", line)
        if m:
            sign = -1 if "expected < 0" in line else 1
            out.append((m[1], float(m[2]), float(m[3]), float(m[4]), sign))
            continue
        m = re.search(r"- (\S+) \(.*?\): reader ([0-9.]+) vs B2 ([0-9.]+), diff \[([+-][0-9.]+), ([+-][0-9.]+)\]", line)
        if m:  # H5/H7: reader below B2 predicted
            out.append((m[1], float(m[2]) - float(m[3]), float(m[4]), float(m[5]), -1))
    return out


def main(path):
    txt = pathlib.Path(path).read_text()
    L = ["# Holm correction within hypothesis families (α = 0.05, one-sided in the predicted direction)", ""]
    for sec in txt.split("\n## ")[1:]:
        title = sec.splitlines()[0]
        if title.startswith("H1 ") or title.startswith("H6 ") or title.startswith("H8 "):
            L.append(f"- {title.split(' — ')[0]}: not a family of directional tests (H1 predicts no effect; H6 uses "
                     "χ² per reader; H8 combines an equivalence test and descriptive shares) — reported as is")
            continue
        T = tests_in(sec, title)
        if not T:
            continue
        ps = [float(norm.sf(s * est / max((hi - lo) / 3.92, 1e-9))) for _, est, lo, hi, s in T]
        rej = holm(ps)
        L.append(f"- {title.split(' — ')[0]}: {sum(rej)}/{len(T)} tests survive Holm "
                 f"(largest p = {max(ps):.2g})" + ("" if all(rej) else " — FAILING: " +
                                                     ", ".join(t[0] for t, r in zip(T, rej) if not r)))
    out = pathlib.Path(path).with_name("HOLM.md")
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1])
