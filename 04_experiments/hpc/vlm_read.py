"""Open VLM reader (vLLM) on the prototype items.

Tasks
  lineup     : 20-panel lineup -> which panel looks most different (0 = none), confidence 1–5, reason
  diagnosis  : one plot -> probability for healthy + each of the 6 failures, and "cannot tell"
  prior_only : the diagnosis question with no image (what the model guesses from the prompt alone, gap 4.3)
  lineup_shuffled : the same lineups with panels in new positions (position-bias control, gap 3.3)
  probe      : read numbers off a single plot (perception vs reasoning control, gap 4.1)

Same failure definitions for every reader (gap 4.2). Temperature 0. Answer keys are never in prompts.
  python vlm_read.py <run_dir> [model] [items.json] [tag]   -> <run_dir>/responses_<model>[_<tag>].jsonl
  With an items file, the items (task, spec, image path relative to run_dir) come from that file.
"""
import base64
import json
import pathlib
import re
import sys
import time

from vllm import LLM, SamplingParams

RUN = pathlib.Path(sys.argv[1])
MODEL = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen2.5-VL-7B-Instruct"
ITEMS = RUN / sys.argv[3] if len(sys.argv) > 3 else None
TAG = f"_{sys.argv[4]}" if len(sys.argv) > 4 else ""
OUT = RUN / f"responses_{MODEL.split('/')[-1]}{TAG}.jsonl"

CAUSES = ["healthy", "overfitting", "label_noise", "class_imbalance", "data_leakage", "miscalibration",
          "distribution_shift"]
DEFINITIONS = """Possible states of the model:
- healthy: no problem
- overfitting: the model fits the training data much better than new data
- label_noise: some training labels are wrong
- class_imbalance: one class is much rarer than the other
- data_leakage: information about the labels leaked into the training features or the evaluation data
- miscalibration: the model's confidence does not match how often it is right
- distribution_shift: the test data differs from the training data"""
WHAT = {"lc_loss": "the training loss (blue) and validation loss (orange) of a binary classifier over training epochs",
        "lc_acc": "the training accuracy (blue) and validation accuracy (orange) of a binary classifier over training epochs",
        "rel": "a reliability (calibration) diagram of a binary classifier on its test set: mean predicted probability "
               "per bin (x) against the observed fraction of positives (y); the dashed diagonal is perfect calibration",
        "cm": "the confusion matrix (counts) of a binary classifier on its test set, with a decision threshold of 0.5",
        "roc": "the ROC curve of a binary classifier on its test set; the dashed diagonal is a random classifier",
        "pr": "the precision-recall curve of a binary classifier on its test set; the dashed line is the positive-class rate",
        "dist": "histograms of a binary classifier's predicted probabilities on its validation set (orange) and its test "
                "set (green)",
        "rel_n": "a reliability (calibration) diagram of a binary classifier on its test set: mean predicted probability "
                 "per bin (x) against the observed fraction of positives (y); point size shows how many test examples "
                 "fall in each bin; the dashed diagonal is perfect calibration",
        "cm_norm": "the row-normalised confusion matrix of a binary classifier on its test set (each row shows the share "
                   "of that true class predicted as each label), with a decision threshold of 0.5"}
SHORT = {"lc_loss": "loss curves", "lc_acc": "accuracy curves", "rel": "reliability diagram", "cm": "confusion matrix",
         "roc": "ROC curve", "pr": "precision-recall curve", "dist": "validation-vs-test score histogram",
         "rel_n": "reliability diagram with bin sizes", "cm_norm": "row-normalised confusion matrix"}

LINEUP = """You will see a grid of 20 small plots, numbered 1 to 20. Each shows {what}, all on the same axes.
Look at the plots and choose the one that looks most different from the others. If none looks different, answer 0.
Reply with JSON only: {{"choice": <integer 0-20>, "confidence": <integer 1-5>, "reason": "<one short sentence>"}}"""

DIAGNOSIS = """The plot shows {what}. The model is either healthy or has exactly one problem.
{definitions}

Based only on what this plot shows, give the probability of each option (numbers from 0 to 1 that sum to 1),
and say whether the plot is insufficient to tell.
Reply with JSON only: {{{fields}, "cannot_tell": <true or false>, "reason": "<one short sentence>"}}"""

# prompt-sensitivity controls (gap 4.2): same information and answer format, different wording / reasoning order
PARAPHRASE = """Here is a diagnostic plot: {what}. Assume the classifier is either working correctly or suffers from exactly
one of the issues listed below.
{definitions}

Using only the information visible in this plot, estimate how likely each option is (probabilities from 0 to 1 that
add up to 1), and state whether the plot alone is not enough to decide.
Reply with JSON only: {{{fields}, "cannot_tell": <true or false>, "reason": "<one short sentence>"}}"""

DESCRIBE_FIRST = """The plot shows {what}. The model is either healthy or has exactly one problem.
{definitions}

First, describe in 2-4 sentences what you see in the plot: the values, trends, and anything unusual. Then, based only on
that evidence, give the probability of each option (numbers from 0 to 1 that sum to 1), and say whether the plot is
insufficient to tell. End your answer with JSON only on the last line:
{{{fields}, "cannot_tell": <true or false>, "reason": "<one short sentence>"}}"""

FEWSHOT_INTRO = """You will first see {k} labelled example plots of the same type, one for each possible state of the model.
Use them as a reference. Then you will see a new plot to diagnose."""

NUMBERS = """Below are the exact values that are drawn in a plot showing {what}. You get the numbers instead of the image.
The model is either healthy or has exactly one problem.
{definitions}

Values:
{values}

Based only on these values, give the probability of each option (numbers from 0 to 1 that sum to 1),
and say whether the values are insufficient to tell.
Reply with JSON only: {{{fields}, "cannot_tell": <true or false>, "reason": "<one short sentence>"}}"""

PRIOR = """You cannot see any plot. A binary classifier's {what_short} was plotted, but the plot is hidden.
The model is either healthy or has exactly one problem.
{definitions}

Give your best guess of the probability of each option (numbers from 0 to 1 that sum to 1).
Reply with JSON only: {{{fields}, "cannot_tell": <true or false>, "reason": "<one short sentence>"}}"""

PROBE = {
    "lc_loss": """The plot shows {what}. Read these values from the plot as precisely as you can:
the training loss at the last epoch, the validation loss at the last epoch, the lowest validation loss,
and the epoch at which the validation loss is lowest.
Reply with JSON only: {{"train_last": <number>, "val_last": <number>, "val_best": <number>, "val_best_epoch": <integer>}}""",
    "lc_acc": """The plot shows {what}. Read these values from the plot as precisely as you can:
the training accuracy at the last epoch, the validation accuracy at the last epoch, the highest validation accuracy,
and the epoch at which the validation accuracy is highest.
Reply with JSON only: {{"train_last": <number>, "val_last": <number>, "val_best": <number>, "val_best_epoch": <integer>}}""",
}
# read-then-reason: read drawn values only (no diagnosis); the diagnosis is made by a fixed rule (code/rtr.py)
_READ = "The plot shows {what}. Read these values from the plot as precisely as you can. Do not interpret or diagnose.\n"
EXTRACT = {
    "lc_loss": _READ + """the training loss at the last epoch, the validation loss at the last epoch, the lowest validation loss,
and the epoch at which the validation loss is lowest.
Reply with JSON only: {{"train_last": <number>, "val_last": <number>, "val_best": <number>, "val_best_epoch": <integer>}}""",
    "lc_acc": _READ + """the training accuracy at the last epoch, the validation accuracy at the last epoch, the highest validation
accuracy, and the epoch at which the validation accuracy is highest.
Reply with JSON only: {{"train_last": <number>, "val_last": <number>, "val_best": <number>, "val_best_epoch": <integer>}}""",
    "rel": _READ + """the x (mean predicted probability) and y (observed fraction positive) of every plotted point, from left to right.
Reply with JSON only: {{"points": [[<x>, <y>], ...]}}""",
    "rel_n": _READ + """the x (mean predicted probability) and y (observed fraction positive) of every plotted point, from left to
right, and each point's share of all examples (0 to 1, estimated from its size; the shares sum to 1).
Reply with JSON only: {{"points": [[<x>, <y>, <share>], ...]}}""",
    "cm": _READ + """the four numbers in the matrix: true 0 predicted 0 (tn), true 0 predicted 1 (fp), true 1 predicted 0 (fn),
true 1 predicted 1 (tp).
Reply with JSON only: {{"tn": <number>, "fp": <number>, "fn": <number>, "tp": <number>}}""",
    "cm_norm": _READ + """the four shares in the matrix as numbers from 0 to 1: true 0 predicted 0 (tn), true 0 predicted 1 (fp),
true 1 predicted 0 (fn), true 1 predicted 1 (tp).
Reply with JSON only: {{"tn": <number>, "fp": <number>, "fn": <number>, "tp": <number>}}""",
    "roc": _READ + """the true positive rate of the curve at false positive rates 0.05, 0.1, 0.2 and 0.5.
Reply with JSON only: {{"tpr_at_0.05": <number>, "tpr_at_0.1": <number>, "tpr_at_0.2": <number>, "tpr_at_0.5": <number>}}""",
    "pr": _READ + """the precision of the curve at recall 0.2, 0.5 and 0.8, and the height of the dashed positive-rate line.
Reply with JSON only: {{"precision_at_0.2": <number>, "precision_at_0.5": <number>, "precision_at_0.8": <number>, "positive_rate_line": <number>}}""",
    "dist": _READ + """for the validation histogram (orange) and the test histogram (green) separately, the share of its area below
0.1, above 0.9, and above 0.5 (numbers from 0 to 1).
Reply with JSON only: {{"val_below_0_1": <number>, "val_above_0_9": <number>, "val_above_0_5": <number>, "test_below_0_1": <number>, "test_above_0_9": <number>, "test_above_0_5": <number>}}""",
}
# machine-reader designs (gap 5): same data, key values printed on the plot (code/annotate.py)
WHAT["rel_ann"] = WHAT["rel"] + "; each point is labelled with its observed fraction and the x positions are written on the axis"
WHAT["roc_ann"] = WHAT["roc"] + "; a small table on the plot lists the true positive rate at false positive rates 0.05, 0.1, 0.2 and 0.5"
WHAT["dist_ann"] = WHAT["dist"] + "; a small table on the plot lists, for each histogram, the share of predictions below 0.1, above 0.5 and above 0.9"
SHORT.update(rel_ann="annotated reliability diagram", roc_ann="annotated ROC curve", dist_ann="annotated score histogram")
EXTRACT.update(rel_ann=EXTRACT["rel"], roc_ann=EXTRACT["roc"], dist_ann=EXTRACT["dist"])
MODEL_KW = {  # per-model engine settings
    # 32k (model maximum): a few-shot prompt has 8 images x up to 13 tiles x 256 tokens (~27k); 16k failed on controls
    "OpenGVLab/InternVL3-8B": dict(trust_remote_code=True, max_model_len=32768),
    # 38B on one A100-40GB: a 16k context does not fit next to the weights -> 8k (the setting that worked in v3)
    "OpenGVLab/InternVL3-38B-AWQ": dict(trust_remote_code=True, max_model_len=8192, gpu_memory_utilization=0.95),
    "llava-hf/llava-onevision-qwen2-7b-ov-hf": dict(max_model_len=16384),
}

FIELDS = ", ".join(f'"{c}": <p>' for c in CAUSES)


def img_url(path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def parse(text):
    """Take the last parseable JSON object in the answer (describe-first answers put prose before it)."""
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", text, re.S))):
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


def build_items():
    meta = json.load(open(RUN / "meta.json"))
    key = json.load(open(RUN / "lineups" / "key.json"))
    items = []
    for k in key:  # lineups
        items.append(dict(item_id=k["item_id"], task="lineup", spec=k["spec"],
                          image=RUN / "lineups" / f"{k['item_id']}.png", text=LINEUP.format(what=WHAT[k["spec"]])))
    if (RUN / "lineups_shuffled" / "key.json").exists():
        for k in json.load(open(RUN / "lineups_shuffled" / "key.json")):
            items.append(dict(item_id=k["item_id"], task="lineup_shuffled", spec=k["spec"],
                              image=RUN / "lineups_shuffled" / f"{k['item_id']}.png",
                              text=LINEUP.format(what=WHAT[k["spec"]])))
    if (RUN / "probe_items.json").exists():
        for p in json.load(open(RUN / "probe_items.json")):
            items.append(dict(item_id=p["item_id"], run_id=p["run_id"], task="probe", spec=p["spec"],
                              image=RUN / "img" / f"{p['run_id']}_{p['spec']}.png",
                              text=PROBE[p["spec"]].format(what=WHAT[p["spec"]])))
    test_ids = [r for r, s in meta["healthy"].items() if s == "test"]
    for lv in meta["levels"]:
        test_ids += [r for r, s in lv["failed"].items() if s == "test"]
    for spec in meta["specs"]:
        for r in test_ids:  # single-plot diagnosis
            items.append(dict(item_id=f"{r}_{spec}", run_id=r, task="diagnosis", spec=spec,
                              image=RUN / "img" / f"{r}_{spec}.png",
                              text=DIAGNOSIS.format(what=WHAT[spec], definitions=DEFINITIONS, fields=FIELDS)))
        items.append(dict(item_id=f"prior_{spec}", task="prior_only", spec=spec, image=None,
                          text=PRIOR.format(what_short=SHORT[spec], definitions=DEFINITIONS, fields=FIELDS)))
    return items


def items_from_file(path):
    items = []
    for it in json.load(open(path)):
        sp, task = it["spec"], it["task"]
        tmpl = {"paraphrase": PARAPHRASE, "describe_first": DESCRIBE_FIRST}.get(it.get("variant"), DIAGNOSIS)
        what = WHAT[sp]
        if it.get("companion"):  # design intervention: companion view (two plots of the same model)
            what = f"two plots of the same model side by side. Left: {WHAT[sp]}. Right: {WHAT[it['companion']]}"
        if it.get("badge"):  # design intervention: evidence note rendered under the plot
            what += ". A note under the plot states what this type of plot can and cannot reveal"
        if it.get("variant") == "numbers":
            tmpl = None
        text = {"diagnosis": lambda: (NUMBERS.format(what=WHAT[sp], definitions=DEFINITIONS, fields=FIELDS,
                                                     values=it["marks_text"]) if tmpl is None else
                                      tmpl.format(what=what, definitions=DEFINITIONS, fields=FIELDS)),
                "prior_only": lambda: PRIOR.format(what_short=SHORT[sp], definitions=DEFINITIONS, fields=FIELDS),
                "probe": lambda: PROBE[sp].format(what=WHAT[sp]),
                "extract": lambda: EXTRACT[sp].format(what=WHAT[sp])}[task]()
        ex = [dict(e, image=RUN / e["image"]) for e in it.get("examples", [])]
        items.append(dict(it, image=RUN / it["image"] if it.get("image") else None, text=text, examples=ex))
    return items


def main():
    t0 = time.time()
    items = items_from_file(ITEMS) if ITEMS else build_items()
    done = set()
    if OUT.exists():
        done = {json.loads(line)["item_id"] for line in open(OUT)}
    todo = [it for it in items if it["item_id"] not in done]
    print(f"{len(items)} items, {len(todo)} to run, model {MODEL}", flush=True)
    if not todo:
        return
    n_img = 1 + max((len(it.get("examples") or []) for it in todo), default=0)
    kw = dict(max_model_len=8192, limit_mm_per_prompt={"image": n_img}, seed=0)
    kw.update(MODEL_KW.get(MODEL, {}))
    llm = LLM(model=MODEL, **kw)
    params = SamplingParams(temperature=0.0, max_tokens=600 if any(it.get("variant") == "describe_first"
                                                                   for it in todo) else 300)
    for start in range(0, len(todo), 128):
        batch = todo[start:start + 128]
        convs = []
        for it in batch:
            content = [{"type": "text", "text": it["text"]}]
            if it["image"] is not None:
                content.insert(0, {"type": "image_url", "image_url": {"url": img_url(it["image"])}})
            if it.get("examples"):  # few-shot: labelled examples first, then the plot to diagnose
                pre = [{"type": "text", "text": FEWSHOT_INTRO.format(k=len(it["examples"]))}]
                for k, e in enumerate(it["examples"], 1):
                    pre += [{"type": "image_url", "image_url": {"url": img_url(e["image"])}},
                            {"type": "text", "text": f"Example {k}: this model is {e['label'].replace('_', ' ')}."}]
                content = pre + [{"type": "text", "text": "Now the plot to diagnose:"}] + content
            convs.append([{"role": "user", "content": content}])
        outs = llm.chat(convs, params, use_tqdm=False)
        with open(OUT, "a") as fh:
            for it, o in zip(batch, outs):
                text = o.outputs[0].text
                rec = {k: v for k, v in it.items() if k not in ("image", "text", "examples", "marks_text")}
                rec.update(model=MODEL, raw=text, parsed=parse(text), date=time.strftime("%Y-%m-%d"))
                fh.write(json.dumps(rec) + "\n")
        print(f"[{time.time() - t0:5.0f}s] {start + len(batch)}/{len(todo)}", flush=True)
    print("saved", OUT)


if __name__ == "__main__":
    main()
