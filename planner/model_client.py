"""LLM backend client. Backend chosen by config.MODEL_BACKEND."""

from __future__ import annotations

import requests
from fastapi import HTTPException

from observability import get_logger, inc
from . import config

log = get_logger("model_client")

try:
    from google import genai
    from google.genai import types
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False


# App-agnostic system framing — exploratory testing, JSON-disciplined.
QA_SYSTEM_INSTRUCTION = (
    "You are a senior exploratory QA engineer. You design high-information tests that "
    "discover defects in any application, reasoning only from the context you are given. "
    "Always respond with valid JSON when asked for test cases, and follow the exact output "
    "format specified in the user prompt."
)


# Providers rate-limit (429) and occasionally 5xx; those are worth retrying and an
# unretried one used to kill an entire multi-hour batch. Everything here is
# re-raised as HTTPException(503), so the transient test must look at the
# *upstream* status embedded in the message and explicitly exclude permanent
# failures — otherwise a 400 "bad model id" gets retried three times and still
# fails, just slower.
_RETRY_STATUS = ("429", "500 ", "502", "503 ", "504", "too many requests", "overloaded", "timed out")
_PERMANENT = ("400", "401", "403", "404", "not a valid model", "invalid api key")
_MAX_ATTEMPTS = 4


def _is_transient(exc: Exception) -> bool:
    msg = str(exc).lower()
    if any(tok in msg for tok in _PERMANENT):
        return False
    return any(tok in msg for tok in _RETRY_STATUS)


def call_model(prompt: str, max_new_tokens: int, enable_thinking: bool,
               model: str | None = None, image_b64: str | None = None,
               app_label: str | None = None) -> dict:
    """Call the configured backend. ``model`` overrides the default for this call,
    which is how ingestion can use a stronger model than the planner loop.
    ``image_b64`` (raw base64, no data-URI prefix) attaches a screenshot to the
    call — OpenRouter only for now; other backends silently ignore it.
    ``app_label`` sets OpenRouter's X-Title so different call sites (planner,
    evaluator, ingestion) show up distinctly in OpenRouter's usage dashboard
    instead of all reading "QA Planner Agent" — OpenRouter only, ignored
    elsewhere."""
    import time
    start = time.perf_counter()
    inc("llm_calls_total")

    def _dispatch(use_model: str | None):
        if config.MODEL_BACKEND == "gemini":
            return _call_gemini(prompt, max_new_tokens, enable_thinking)
        if config.MODEL_BACKEND == "openrouter":
            return _call_openrouter(prompt, max_new_tokens, enable_thinking, model=use_model,
                                    image_b64=image_b64, app_label=app_label)
        return _call_ngrok(prompt, max_new_tokens, enable_thinking)

    result = None
    last_exc: Exception | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            result = _dispatch(model)
            break
        except Exception as exc:
            last_exc = exc
            if not _is_transient(exc):
                raise  # a bad request/API key fails identically on any model — no point retrying
            if attempt >= _MAX_ATTEMPTS:
                break  # primary model's retries exhausted — fall through to the fallback below
            delay = 2 ** attempt
            log.warning("llm_retry", backend=config.MODEL_BACKEND, attempt=attempt,
                        delay_s=delay, error=str(exc)[:180])
            inc("llm_retries_total")
            time.sleep(delay)

    if result is None:
        # A 429/unavailable is THIS model's shared capacity, not every model's —
        # one attempt on a differently-provisioned fallback beats giving up
        # outright. Only for OpenRouter (the fallback model id is meaningless to
        # the other backends), and only once — this isn't a second retry loop.
        if config.FALLBACK_MODEL and config.MODEL_BACKEND == "openrouter":
            log.warning("llm_fallback", primary=model or config.OPENROUTER_MODEL,
                        fallback=config.FALLBACK_MODEL, error=str(last_exc)[:180])
            try:
                result = _dispatch(config.FALLBACK_MODEL)
            except Exception as fallback_exc:
                # Report the PRIMARY model's failure as the cause — that's the
                # one a human needs to know about (capacity/rate-limit), not
                # whatever the fallback happened to also fail with.
                raise last_exc from fallback_exc
        else:
            raise last_exc

    duration_ms = round((time.perf_counter() - start) * 1000, 1)
    estimated_tokens = len(prompt) // 4 + len(result.get("answer") or "") // 4

    log.info(
        "llm_call",
        backend=config.MODEL_BACKEND,
        latency_ms=duration_ms,
        estimated_tokens=estimated_tokens,
        has_image=bool(image_b64),
    )
    return result


def _call_gemini(prompt: str, max_new_tokens: int, enable_thinking: bool) -> dict:
    if not HAS_GEMINI:
        raise HTTPException(status_code=500, detail="google-genai package not installed.")
    if not config.GEMINI_API_KEY:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY not set in .env")
    try:
        client = genai.Client(api_key=config.GEMINI_API_KEY)
        response = client.models.generate_content(
            model=config.PLANNER_GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=max_new_tokens,
                temperature=0.7,
                system_instruction=QA_SYSTEM_INSTRUCTION,
            ),
        )
        return {"answer": response.text, "thinking": ""}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model backend (Gemini) unavailable: {e}")


def _call_ngrok(prompt: str, max_new_tokens: int, enable_thinking: bool) -> dict:
    try:
        resp = requests.post(
            f"{config.MODEL_API_URL}/generate",
            json={"prompt": prompt, "max_new_tokens": max_new_tokens, "enable_thinking": enable_thinking},
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Model backend (ngrok) unavailable: {e}")


# OpenRouter's usage dashboard attributes calls by HTTP-Referer, not X-Title —
# the referer is the actual "app" identity; the title is just a mutable display
# name for whatever app is currently registered under that referer. Giving every
# call site the SAME referer and only varying the title meant the *most
# recently sent* title silently became the new display name for that app's
# entire history — old ingestion-run rows retroactively relabelled themselves
# days later just because a newer call reused the same referer with a
# different title. Each call type needs its OWN referer to get a genuinely
# separate, stable identity instead of overwriting a shared one.
#
# These must be distinct ORIGINS (subdomains), not just distinct paths on one
# shared origin — an earlier version of this dict used one domain with a
# different path per app (.../planner, .../evaluator, ...), and evaluator
# calls kept showing up on OpenRouter's dashboard mislabeled with whichever
# title the (far more frequent) executor calls had most recently sent. HTTP
# Referer is conventionally an origin-level signal, and OpenRouter's own docs
# don't specify the comparison granularity, so the safest fix is to remove
# the shared origin entirely rather than rely on path-level separation.
_APP_REFERERS = {
    "QA Planner Agent": "https://planner.qa-planner-agent.local/",
    "QA Evaluator Agent": "https://evaluator.qa-planner-agent.local/",
    "QA SRS Ingestion": "https://srs-ingestion.qa-planner-agent.local/",
}


def _call_openrouter(prompt: str, max_new_tokens: int, enable_thinking: bool,
                     model: str | None = None, image_b64: str | None = None,
                     app_label: str | None = None) -> dict:
    if not config.OPENROUTER_API_KEY:
        raise HTTPException(status_code=503, detail="OPENROUTER_API_KEY not set in .env")

    title = app_label or "QA Planner Agent"
    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": _APP_REFERERS.get(title, _APP_REFERERS["QA Planner Agent"]),
        "X-Title": title,
    }
    # Plain string content everywhere except the one call that carries an image —
    # the multi-part block form is only needed when there's a second part to hold.
    user_content: str | list = prompt
    if image_b64:
        user_content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        ]
    messages = [
        {"role": "system", "content": QA_SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_content},
    ]
    payload = {
        "model": model or config.OPENROUTER_MODEL,
        "messages": messages,
        "temperature": 0.7,
    }
    # max_new_tokens <= 0 means "no cap": omit the field so the provider allows its
    # full output length. Extraction needs this — a reasoning model bills its
    # scratchpad against max_tokens, so any cap we pick can silently truncate the
    # JSON and cost us requirements.
    if max_new_tokens and max_new_tokens > 0:
        payload["max_tokens"] = max_new_tokens
    if not enable_thinking:
        # Reasoning tokens are billed against max_tokens, so on a reasoning model
        # they starve the JSON answer and it gets truncated mid-object. This is a
        # preference, not a requirement: some endpoints reject it outright
        # ("Reasoning is mandatory for this endpoint"), so a 400 naming reasoning
        # is retried once without the flag instead of failing the whole ingest.
        payload["reasoning"] = {"enabled": False}
    try:
        resp = requests.post(
            f"{config.OPENROUTER_BASE_URL}/chat/completions",
            headers=headers, json=payload, timeout=600,
        )
        if resp.status_code == 400 and "reasoning" in (resp.text or "").lower():
            payload.pop("reasoning", None)
            resp = requests.post(
                f"{config.OPENROUTER_BASE_URL}/chat/completions",
                headers=headers, json=payload, timeout=600,
            )
        resp.raise_for_status()
        data = resp.json()
        answer = ""
        thinking = ""
        if data.get("choices"):
            message = data["choices"][0].get("message", {})
            answer = message.get("content") or ""
            thinking = message.get("reasoning") or message.get("thinking") or ""
        return {"answer": answer, "thinking": thinking}
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Model backend (OpenRouter) unavailable: {e}")


def backend_info() -> dict:
    """Diagnostics for the health endpoint."""
    info = {"backend": config.MODEL_BACKEND}
    if config.MODEL_BACKEND == "gemini":
        info["model"] = config.PLANNER_GEMINI_MODEL
        info["api"] = "Google GenAI API"
    elif config.MODEL_BACKEND == "openrouter":
        info["model"] = config.OPENROUTER_MODEL
        info["api"] = config.OPENROUTER_BASE_URL
    else:
        info["api"] = config.MODEL_API_URL
    return info
