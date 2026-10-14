"""Run AI readers on lineup items and score them.

  python download_data.py                    # once
  python run_pilot.py build --name pilot     # render items + hidden key
  python run_pilot.py run --name pilot --reader claude --model claude-opus-5-5 --effort medium --limit 10
  python run_pilot.py score --name pilot

Responses are appended to runs/<name>/responses_<reader>_<model>_<effort>.jsonl, so a crash can resume.
"""
import argparse
import json
import pathlib

import pandas as pd

from prompts import ANSWER_SCHEMA, build_prompt

ROOT = pathlib.Path(__file__).parent


def get_reader(name, model, effort):
    if name == "claude":
        from readers.claude_reader import ClaudeReader
        return ClaudeReader(model=model, effort=effort)
    raise SystemExit(f"reader '{name}' not implemented yet (openai / gemini / vllm modules come next)")


def cmd_build(a):
    from lineups import build
    build(a.name)


def cmd_run(a):
    run = ROOT / "runs" / a.name
    items = json.load(open(run / "key.json"))
    out = run / f"responses_{a.reader}_{a.model}_{a.effort}.jsonl"
    done = {json.loads(l)["item_id"] for l in open(out)} if out.exists() else set()
    todo = [it for it in items if it["item_id"] not in done][: a.limit or None]
    print(f"{len(todo)} items to run ({len(done)} already done)")
    if a.dry_run:
        print(build_prompt(todo[0]["family"]) if todo else "nothing to do")
        return
    reader = get_reader(a.reader, a.model, a.effort)
    spent = 0.0
    with open(out, "a") as f:
        for it in todo:
            res = reader.read(run / "img" / f"{it['item_id']}.png", build_prompt(it["family"]), ANSWER_SCHEMA)
            res["item_id"] = it["item_id"]
            f.write(json.dumps(res) + "\n")
            f.flush()
            spent += res.get("cost_usd", 0) or 0
            print(it["item_id"], res.get("answer", {}) and res["answer"].get("selected_panels"),
                  res.get("problem") or res.get("error") or "", f"${spent:.3f}")
            if a.budget and spent >= a.budget:
                print(f"budget ${a.budget} reached, stopping")
                break


def cmd_score(a):
    run = ROOT / "runs" / a.name
    key = {it["item_id"]: it for it in json.load(open(run / "key.json"))}
    rows = []
    for path in sorted(run.glob("responses_*.jsonl")):
        for line in open(path):
            r = json.loads(line)
            k = key[r["item_id"]]
            sel = (r.get("answer") or {}).get("selected_panels")
            row = {"run": path.stem.replace("responses_", ""), **k, "selected": sel,
                   "problem": r.get("problem") or r.get("error"), "cost": r.get("cost_usd", 0)}
            if sel is None:
                row["outcome"] = "no_answer"
            elif k["null_lineup"]:
                row["outcome"] = "correct_none" if len(sel) == 0 else "false_alarm"
            else:
                row["outcome"] = "hit" if k["real_panel"] in sel else ("miss_none" if not sel else "miss_wrong")
            row["detect"] = row["outcome"] == "hit"
            row["n_selected"] = len(sel) if sel is not None else None
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        print("no responses yet")
        return
    print("\n== outcomes by family ==")
    print(df.groupby(["run", "family", "outcome"]).size().unstack(fill_value=0))
    real = df[~df.null_lineup]
    if "human_rate" in real and real.human_rate.notna().any():
        h = real[real.family == "residual"].copy()
        h["band"] = pd.cut(h.human_rate, [-0.01, 0.1, 0.35, 0.6, 0.85, 1.0])
        print("\n== residual lineups: AI detection vs human detection, by human-difficulty band ==")
        print(h.groupby(["run", "band"], observed=True).agg(ai=("detect", "mean"), human=("human_rate", "mean"),
                                                              n=("detect", "size")).round(2))
    sev = real[real.family != "residual"]
    if not sev.empty:
        print("\n== planted flaws: detection by severity ==")
        print(sev.groupby(["run", "family", "severity"]).detect.agg(["mean", "size"]).round(2))
    print(f"\ntotal cost: ${df.cost.sum():.2f}")
    df.to_csv(run / "scored.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(required=True)
    b = sub.add_parser("build"); b.add_argument("--name", default="pilot"); b.set_defaults(f=cmd_build)
    r = sub.add_parser("run")
    r.add_argument("--name", default="pilot"); r.add_argument("--reader", default="claude")
    r.add_argument("--model", default="claude-opus-5-5"); r.add_argument("--effort", default="medium")
    r.add_argument("--limit", type=int, default=0); r.add_argument("--budget", type=float, default=0)
    r.add_argument("--dry-run", action="store_true"); r.set_defaults(f=cmd_run)
    s = sub.add_parser("score"); s.add_argument("--name", default="pilot"); s.set_defaults(f=cmd_score)
    a = p.parse_args()
    a.f(a)
