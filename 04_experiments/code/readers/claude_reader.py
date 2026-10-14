"""Claude as a lineup reader (Anthropic Python SDK).

Design choices for a measurement study:
- No server-side refusal fallbacks: a fallback would silently change which model answered.
  Refusals are recorded as refusals.
- Effort is set explicitly (Claude Opus 5.5 defaults to "medium"); thinking is always on for this model.
- Structured output via output_config.format so answers are machine-scorable without an LLM judge.
"""
import base64
import json
import pathlib

import anthropic

# $ per million tokens (Anthropic first-party rates, from the claude-api skill, cached 2026-09-25).
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-fable-5-1": (10.00, 50.00),
}


def _image_block(path: pathlib.Path) -> dict:
    data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}


def _params(model, effort, image_path, prompt, schema, max_tokens):
    return dict(
        model=model,
        max_tokens=max_tokens,
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
        messages=[{"role": "user", "content": [_image_block(image_path), {"type": "text", "text": prompt}]}],
    )


def cost_usd(model: str, usage) -> float:
    pin, pout = PRICES.get(model, (float("nan"), float("nan")))
    return (usage.input_tokens * pin + usage.output_tokens * pout) / 1e6


def _parse(message):
    if message.stop_reason == "refusal":
        cat = getattr(message.stop_details, "category", None) if message.stop_details else None
        return None, f"refusal:{cat}"
    if message.stop_reason == "max_tokens":
        return None, "max_tokens"
    text = next((b.text for b in message.content if b.type == "text"), None)
    try:
        return json.loads(text), None
    except (TypeError, json.JSONDecodeError):
        return None, "unparseable"


class ClaudeReader:
    def __init__(self, model="claude-opus-5-5", effort="medium", max_tokens=16000):
        self.model, self.effort, self.max_tokens = model, effort, max_tokens
        self.client = anthropic.Anthropic()  # credentials from env / `ant auth login`

    def read(self, image_path: pathlib.Path, prompt: str, schema: dict) -> dict:
        try:
            msg = self.client.messages.create(
                **_params(self.model, self.effort, image_path, prompt, schema, self.max_tokens))
        except anthropic.BadRequestError as e:
            return {"error": f"bad_request: {e.message}"}
        except anthropic.RateLimitError:
            return {"error": "rate_limited"}  # SDK already retried; caller may resume later
        except anthropic.APIStatusError as e:
            return {"error": f"api_status_{e.status_code}"}
        except anthropic.APIConnectionError:
            return {"error": "connection"}
        answer, problem = _parse(msg)
        return {"answer": answer, "problem": problem, "model": msg.model, "stop_reason": msg.stop_reason,
                "input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens,
                "cost_usd": cost_usd(self.model, msg.usage), "request_id": msg._request_id}

    # ---- Batches API: 50% cheaper, asynchronous (use for the full study) ----
    def submit_batch(self, jobs):
        """jobs: list of (custom_id, image_path, prompt, schema)."""
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        reqs = [Request(custom_id=cid, params=MessageCreateParamsNonStreaming(
                    **_params(self.model, self.effort, img, prompt, schema, self.max_tokens)))
                for cid, img, prompt, schema in jobs]
        return self.client.messages.batches.create(requests=reqs).id

    def batch_status(self, batch_id):
        return self.client.messages.batches.retrieve(batch_id)

    def batch_results(self, batch_id):
        out = {}
        for r in self.client.messages.batches.results(batch_id):
            if r.result.type == "succeeded":
                m = r.result.message
                answer, problem = _parse(m)
                out[r.custom_id] = {"answer": answer, "problem": problem, "model": m.model,
                                    "input_tokens": m.usage.input_tokens, "output_tokens": m.usage.output_tokens,
                                    "cost_usd": cost_usd(self.model, m.usage) * 0.5}
            else:
                out[r.custom_id] = {"error": r.result.type}
        return out
