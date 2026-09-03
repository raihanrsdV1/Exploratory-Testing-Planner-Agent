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
                 "overloaded", "timed out", "timeout")
_PERMANENT_TOKENS = ("400", "401", "403", "404", "not a valid model", "invalid api key")
_MAX_ATTEMPTS = 4


class LLMError(RuntimeError):
    pass


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
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                return (self._openrouter(messages) if self.provider == "openrouter"
                        else self._gemini(messages))
            except Exception as exc:
                last = exc
                if attempt >= _MAX_ATTEMPTS or not _is_transient(exc):
                    raise
                time.sleep(2 ** attempt)
        raise LLMError(str(last))

    def _openrouter(self, messages: list[dict]) -> str:
        resp = requests.post(
            f"{self.base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            json={"model": self.model, "messages": messages,
                  "max_tokens": self.max_tokens, "temperature": 0.2},
            timeout=180,
        )
        if resp.status_code != 200:
            raise LLMError(f"OpenRouter {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"OpenRouter returned no choices: {str(data)[:300]}")
        return choices[0].get("message", {}).get("content", "") or ""

    def _gemini(self, messages: list[dict]) -> str:
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
                  "generationConfig": {"maxOutputTokens": self.max_tokens, "temperature": 0.2}},
            timeout=180,
        )
        if resp.status_code != 200:
            raise LLMError(f"Gemini {resp.status_code}: {resp.text[:300]}")
        candidates = resp.json().get("candidates") or []
        if not candidates:
            raise LLMError("Gemini returned no candidates")
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts)


def _is_transient(exc: Exception) -> bool:
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
        if isinstance(obj, dict) and obj.get("action"):
            return obj
    return {"action": "_error", "reason": "no JSON action object found in the reply"}


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
