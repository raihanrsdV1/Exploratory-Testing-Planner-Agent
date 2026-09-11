"""Minimal chat client for the browser agent.

Separate from ``planner.model_client`` on purpose. That one is a single-prompt,
FastAPI-facing helper: it raises ``HTTPException`` and has no notion of a
conversation. The agent loop needs multi-turn chat and needs failures as ordinary
exceptions, so this is its own thin client rather than a widening of that one.

Retry policy matches the planner's, for the same reason: a provider 429 in the
middle of a batch used to end the batch.
"""

from __future__ import annotations

import json
import re
import time

import requests

_RETRY_TOKENS = ("429", "500", "502", "503", "504", "too many requests",
                 "overloaded", "timed out", "timeout",
                 # Transport-level failures. A dropped socket used to be judged
                 # NOT transient (none of the tokens above appear in
                 # "('Connection aborted.', ConnectionResetError(10054, ...))"),
                 # so a blip that one retry would have survived killed the test.
                 "connection aborted", "connection reset", "connection refused",
                 "remote end closed", "max retries", "read timed out",
                 "bad handshake", "ssl", "temporarily unavailable",
                 # A reasoning model that overran its budget on one turn may well
                 # fit on the next; how long it thinks varies per prompt. Worth a
                 # retry rather than declaring the provider dead and ending the
                 # batch on a single occurrence.
                 "returned an empty answer")
_PERMANENT_TOKENS = ("400", "401", "403", "404", "not a valid model", "invalid api key")
_MAX_ATTEMPTS = 4
# Ceiling for the escalating retry above; beyond this the model is the problem.
# Kept modest on purpose: OpenRouter pre-authorises the MAXIMUM cost a request
# could incur, so a large max_tokens can be refused with 402 'would exceed your
# available credits' on a nearly-spent key even when the reply would be tiny.
# A terse model answers a browser step in ~200 tokens; 8000 is already generous.
_MAX_BUDGET = 8000


class LLMError(RuntimeError):
    """Any failure to obtain a usable reply from the executor model.

    Everything that goes wrong talking to the provider must surface as this type.
    The runner catches it and records LLM_UNAVAILABLE (an ENVIRONMENT fault); a
    bare ``requests`` exception instead reaches the generic handler and is
    recorded as CRASH — an APP fault — which files a false defect against the
    site under test. That happened: a ConnectionResetError to openrouter.ai was
    logged as a crash discovered in the application.
    """


class ChatClient:
    """Provider-agnostic chat. ``chat(messages)`` returns the assistant text."""

    def __init__(self, cfg):
        self.provider = (cfg.WEB_LLM_PROVIDER or "openrouter").lower()
        self.model = cfg.WEB_LLM_MODEL
        self.max_tokens = cfg.WEB_LLM_MAX_TOKENS
        if self.provider == "openrouter":
            self.api_key = cfg.OPENROUTER_API_KEY
            self.base_url = cfg.OPENROUTER_BASE_URL
        else:
            self.api_key = cfg.GEMINI_API_KEY
            self.base_url = ""
        if not self.api_key:
            raise LLMError(
                f"No API key for WEB_LLM_PROVIDER='{self.provider}'. Set "
                f"{'OPENROUTER_API_KEY' if self.provider == 'openrouter' else 'GEMINI_API_KEY'}."
            )

    def chat(self, messages: list[dict]) -> str:
        last: Exception | None = None
        budget = self.max_tokens
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                return (self._openrouter(messages, budget) if self.provider == "openrouter"
                        else self._gemini(messages, budget))
            except Exception as exc:
                last = exc
                if attempt >= _MAX_ATTEMPTS or not _is_transient(exc):
                    raise _as_llm_error(exc) from exc
                if "empty answer" in str(exc):
                    # Retrying with the same ceiling reproduces the same overrun:
                    # how much a model thinks depends on the prompt, and the
                    # prompt does not change between attempts. Give it more room.
                    budget = min(budget * 2, _MAX_BUDGET)
                    continue
                time.sleep(2 ** attempt)
        raise _as_llm_error(last)

    def _openrouter(self, messages: list[dict], budget: int | None = None) -> str:
        resp = requests.post(
            f"{self.base_url.rstrip('/')}/chat/completions",
            # OpenRouter asks callers to identify themselves; unidentified
            # traffic is likelier to be treated as a bot.
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json",
                     "HTTP-Referer": "https://github.com/exploratory-testing-planner-agent",
                     "X-Title": "Exploratory Testing Planner Agent"},
            json={"model": self.model, "messages": messages,
                  "max_tokens": budget or self.max_tokens, "temperature": 0.2},
            timeout=180,
        )
        if resp.status_code != 200:
            raise LLMError(f"OpenRouter {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"OpenRouter returned no choices: {str(data)[:300]}")
        choice = choices[0]
        message = choice.get("message") or {}
        content = (message.get("content") or "").strip()
        if content:
            return content

        # A reasoning model that spends its whole allowance on the hidden
        # scratchpad answers HTTP 200 with content=null. Returning "" here made
        # that indistinguishable from a badly formatted reply, so the run logged
        # "model reply was not a JSON action" and nothing recorded the real cause.
        reasoning = (message.get("reasoning") or "").strip()
        if reasoning and choice.get("finish_reason") != "length":
            # The model put its answer in the scratchpad. Usable; the caller's
            # parser will find the JSON object inside it.
            return reasoning
        raise LLMError(
            f"OpenRouter returned an empty answer (finish_reason="
            f"{choice.get('finish_reason')!r}, {len(reasoning)} chars of reasoning). "
            f"The model spent its whole {self.max_tokens}-token budget on its "
            f"scratchpad — raise WEB_LLM_MAX_TOKENS."
        )

    def _gemini(self, messages: list[dict], budget: int | None = None) -> str:
        # Gemini has no "system" role; the system text is prepended to the first
        # user turn, which is how the REST API expects it to be carried.
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = []
        for msg in messages:
            if msg["role"] == "system":
                continue
            role = "model" if msg["role"] == "assistant" else "user"
            text = msg["content"]
            if system and not contents:
                text = f"{system}\n\n{text}"
            contents.append({"role": role, "parts": [{"text": text}]})
        resp = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
            headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
            json={"contents": contents,
                  "generationConfig": {"maxOutputTokens": budget or self.max_tokens, "temperature": 0.2}},
            timeout=180,
        )
        if resp.status_code != 200:
            raise LLMError(f"Gemini {resp.status_code}: {resp.text[:300]}")
        candidates = resp.json().get("candidates") or []
        if not candidates:
            raise LLMError("Gemini returned no candidates")
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts)


def _as_llm_error(exc: Exception | None) -> LLMError:
    """Normalise any provider/transport failure to LLMError, preserving the text."""
    if isinstance(exc, LLMError):
        return exc
    return LLMError(f"{type(exc).__name__}: {exc}")


def _is_transient(exc: Exception) -> bool:
    # Typed check first: a dropped connection is retryable whatever it says.
    # String matching alone missed ConnectionResetError entirely.
    if isinstance(exc, (requests.exceptions.ConnectionError,
                        requests.exceptions.Timeout,
                        requests.exceptions.ChunkedEncodingError)):
        return True
    msg = str(exc).lower()
    if any(tok in msg for tok in _PERMANENT_TOKENS):
        return False
    return any(tok in msg for tok in _RETRY_TOKENS)


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_action(text: str) -> dict:
    """Pull one action object out of a model reply.

    Models wrap JSON in prose and fences no matter what the prompt says, so this
    tries the whole string, then any fenced block, then the first balanced object.
    A parse failure is returned as an ``_error`` action rather than raised: the
    agent can tell the model it produced junk and get on with the run, which is
    far better than aborting a test case over formatting.
    """
    for candidate in _candidates(text):
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        # Models plan several steps ahead and answer with a list. Take the first
        # entry rather than throwing the turn away: it is the action they wanted
        # next, and the loop re-observes before the one after it anyway.
        if isinstance(obj, list) and obj:
            obj = obj[0]
        if isinstance(obj, dict):
            normalised = _normalise_keys(obj)
            if normalised.get("action"):
                return normalised
    # Keep the reply. A 23% parse-failure rate was undiagnosable because the text
    # that failed was discarded, leaving only a generic message in the trace.
    snippet = " ".join((text or "").split())[:200] or "(empty reply)"
    return {"action": "_error",
            "reason": f"no JSON action object found in the reply | reply was: {snippet}"}


# What the models actually emit when they drift off contract, mapped to what the
# dispatcher expects. Observed from real runs: element_id/_element for "ref",
# _action for "action", _text/value for "text".
_KEY_ALIASES = {
    "_action": "action", "action_type": "action", "type": "action", "name": "action",
    "element_id": "ref", "element": "ref", "_element": "ref", "target": "ref",
    "element_ref": "ref", "id": "ref",
    "_text": "text", "value": "text", "input": "text", "content": "text",
    "_reason": "reason", "explanation": "thought", "reasoning": "thought",
}


def _normalise_keys(obj: dict) -> dict:
    """Accept the common near-miss key spellings instead of failing the turn."""
    out = dict(obj)
    for alias, canonical in _KEY_ALIASES.items():
        if alias in out and canonical not in out:
            out[canonical] = out[alias]
    # "[e12]" and "e12" both mean the same element.
    ref = out.get("ref")
    if isinstance(ref, str):
        out["ref"] = ref.strip().strip("[]")
    # A bare "scroll_down"/"scroll up" is the scroll action with a direction.
    act = str(out.get("action", "")).lower().strip()
    if act.startswith("scroll") and act != "scroll":
        direction = act.replace("scroll", "").strip(" _-")
        out["action"] = "scroll"
        out.setdefault("direction", direction or "down")
    else:
        out["action"] = act
    return out


def _candidates(text: str):
    text = (text or "").strip()
    if not text:
        return
    yield text
    for block in _FENCE_RE.findall(text):
        yield block.strip()
    depth, start = 0, -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start >= 0:
                yield text[start:i + 1]
