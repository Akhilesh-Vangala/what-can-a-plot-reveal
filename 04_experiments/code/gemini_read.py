"""Step 4 — frontier reader (Gemini, free tier) on the same items and prompts as the open readers.

  python gemini_read.py <run_dir> <items.json> <model> [tag] [--limit N] [--rpm R]
      -> <run_dir>/responses_<model>[_<tag>].jsonl   (resumes: items already answered are skipped)

Prompts, answer format and parsing are taken verbatim from hpc/vlm_read.py, so answers are scored by the same code.
Temperature 0. The API key is read from ~/.config/visml/gemini_key (never printed). Rate-limited requests are retried
with back-off; a daily quota stop is reported and the run can simply be resumed the next day.
"""
import base64
import ssl
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).parent
SRC = (HERE.parent / "hpc" / "vlm_read.py").read_text()
NS = {"__name__": "prompts"}
exec("CAUSES = " + SRC.split("\nCAUSES = ")[1].split("MODEL_KW = ")[0], NS)  # prompt constants only
exec("import json, re\n" + "def parse" + SRC.split("def parse")[1].split("def build_items")[0], NS)
FIELDS = ", ".join(f'"{c}": <p>' for c in NS["CAUSES"])
try:
    import certifi
    CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    CTX = ssl.create_default_context()
KEY = (pathlib.Path.home() / ".config/visml/gemini_key").read_text().strip()


def text_for(it):
    sp, W = it["spec"], NS["WHAT"]
    if it.get("task") == "probe":
        return NS["PROBE"][sp].format(what=W[sp])
    tmpl = {"paraphrase": NS["PARAPHRASE"], "describe_first": NS["DESCRIBE_FIRST"]}.get(it.get("variant"), NS["DIAGNOSIS"])
    return tmpl.format(what=W[sp], definitions=NS["DEFINITIONS"], fields=FIELDS)


def call(model, img_path, text, max_tries=8):
    parts = [{"text": text}]
    if img_path:
        parts.insert(0, {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(img_path.read_bytes()).decode()}})
    body = json.dumps({"contents": [{"parts": parts}], "generationConfig": {"temperature": 0}}).encode()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    for k in range(max_tries):
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json", "x-goog-api-key": KEY})
        try:
            r = json.load(urllib.request.urlopen(req, timeout=180, context=CTX))
            parts = r.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            return "".join(p.get("text", "") for p in parts if not p.get("thought")), None
        except urllib.error.HTTPError as e:
            msg = json.loads(e.read() or b"{}").get("error", {}).get("message", "")
            if e.code == 429 and ("per day" in msg.lower() or "PerDay" in msg):
                return None, "daily quota reached"
            if e.code in (429, 500, 503):
                time.sleep(min(60, 5 * 2 ** k))
                continue
            return None, f"HTTP {e.code}: {msg[:200]}"
        except (urllib.error.URLError, TimeoutError) as e:
            last = str(e)
            time.sleep(min(60, 5 * 2 ** k))
    return None, f"gave up after retries ({last[:120] if 'last' in dir() else 'rate limited'})"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opt = dict(a.lstrip("-").split("=") for a in sys.argv[1:] if a.startswith("--"))
    run, items_file, model = pathlib.Path(args[0]), args[1], args[2]
    tag = f"_{args[3]}" if len(args) > 3 else ""
    out = run / f"responses_{model}{tag}.jsonl"
    items = json.load(open(run / items_file))
    done = {json.loads(x)["item_id"] for x in open(out)} if out.exists() else set()
    todo = [it for it in items if it["item_id"] not in done][: int(opt.get("limit", 10 ** 9))]
    gap = 60.0 / float(opt.get("rpm", 10))
    print(f"{len(items)} items, {len(done)} done, {len(todo)} to run with {model}", flush=True)
    t0 = time.time()
    for n, it in enumerate(todo, 1):
        t = time.time()
        raw, err = call(model, run / it["image"] if it.get("image") else None, text_for(it))
        if err:
            print(f"stopping after {n - 1}: {err}", flush=True)
            break
        rec = {k: v for k, v in it.items() if k not in ("image", "examples", "marks_text")}
        rec.update(model=model, raw=raw, parsed=NS["parse"](raw), date=time.strftime("%Y-%m-%d"))
        with open(out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        if n % 25 == 0:
            print(f"[{time.time() - t0:5.0f}s] {n}/{len(todo)}", flush=True)
        time.sleep(max(0.0, gap - (time.time() - t)))
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
