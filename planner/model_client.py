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
# When a model is rate-limited upstream it usually stays that way for a while:
# the shared provider pool does not clear between one turn and the next. Without
# a memory of that, every turn of a tool loop re-runs the full 2+4+8s backoff on
# a model that is certainly still limited — measured at ~112s of pure waiting in
# a 115s planning round, i.e. nearly the whole call. After a model exhausts its
# retries, it is skipped in favour of FALLBACK_MODEL for this long.
_RATE_LIMIT_COOLDOWN_S = 180.0
_cooldown_until: dict[str, float] = {}

_RETRY_STATUS = ("429", "500 ", "502", "503 ", "504", "too many requests", "overloaded", "timed out")
_PERMANENT = ("400", "401", "403", "404", "not a valid model", "invalid api key")
_MAX_ATTEMPTS = 4

# Losing the network is not the same failure as a flaky API, and it used to be
# treated as WORSE: a DNS failure matches none of the status codes above, so
# _is_transient() returned False and the call raised immediately — faster than a
# 429 would have. A 40-round campaign died at round 28 that way.
#
# An outage is a pause, not an error. These wait it out on a much longer
# schedule, because the thing being waited for takes minutes to hours, not
# seconds — and there is nothing else to do meanwhile: every backend, including
# FALLBACK_MODEL, is on the far side of the same connection.
_OFFLINE = (
    "connection", "max retries exceeded", "nameresolution", "name resolution",
    "network is unreachable", "connection aborted", "connection reset",
    "failed to establish a new connection", "temporary failure",
)
_OFFLINE_WAITS = (15, 30, 60, 120, 240, 300, 300, 300, 300, 300)  # ~35 min total


def _is_offline(exc: Exception) -> bool:
    """True when the failure looks like an absent network rather than a bad request."""
    return any(tok in str(exc).lower() for tok in _OFFLINE)


def _effective_model(requested: str | None) -> tuple[str, bool]:
    """Resolve which model to actually call, honouring the rate-limit cooldown.

    Returns (model, was_substituted). Falling back BEFORE the request avoids
    re-paying the backoff on a model already known to be limited; the caller
    still gets a working answer, just without the wait.
    """
    import time
    model = requested or config.OPENROUTER_MODEL
    fallback = config.FALLBACK_MODEL
    if (fallback and fallback != model
            and _cooldown_until.get(model, 0.0) > time.time()):
        return fallback, True
    return model, False


def _mark_rate_limited(model: str) -> None:
    import time
    _cooldown_until[model] = time.time() + _RATE_LIMIT_COOLDOWN_S


def _is_transient(exc: Exception) -> bool:
    msg = str(exc).lower()
    if any(tok in msg for tok in _PERMANENT):
        return False
    return _is_offline(exc) or any(tok in msg for tok in _RETRY_STATUS)


def _wait_for_network(send, exc: Exception, label: str):
    """Ride out an outage, retrying `send` on a long schedule. Returns its result.

    Re-raises the original error once the schedule is exhausted, so a genuinely
    long outage still surfaces rather than hanging forever.
    """
    import time
    waited = 0
    for delay in _OFFLINE_WAITS:
        log.warning("network_down", call=label, waiting_s=delay, waited_s=waited,
                    error=str(exc)[:120])
        time.sleep(delay)
        waited += delay
        try:
            result = send()
            log.info("network_back", call=label, offline_for_s=waited)
            return result
        except Exception as again:
            if not _is_offline(again):
                raise          # network is back; this is a different problem
            exc = again
    raise exc


def call_model(prompt: str, max_new_tokens: int, enable_thinking: bool,
               model: str | None = None, image_b64: str | None = None,
               app_label: str | None = None, reasoning_effort: str | None = None) -> dict:
    """Call the configured backend. ``model`` overrides the default for this call,
    which is how ingestion can use a stronger model than the planner loop.
    ``image_b64`` (raw base64, no data-URI prefix) attaches a screenshot to the
    call — OpenRouter only for now; other backends silently ignore it.
    ``app_label`` sets OpenRouter's X-Title so different call sites (planner,
    evaluator, ingestion) show up distinctly in OpenRouter's usage dashboard
    instead of all reading "QA Planner Agent" — OpenRouter only, ignored
    elsewhere.
    ``reasoning_effort`` ("low"/"medium"/"high") caps the model's scratchpad
    rather than its total output. On a reasoning model that refuses
    ``{"enabled": false}`` outright — z-ai/glm-5.3-flash answers 400 "Reasoning
    is mandatory for this endpoint" — the scratchpad is where the time actually
    goes: measured on one real 50-step evaluation prompt, default reasoning took
    113.8s (12,673 chars of reasoning for 5,104 of answer) against 22.7s at
    effort=low for comparable content. Capping ``max_new_tokens`` instead is a
    trap here: reasoning is billed against it, so the budget is consumed before
    any answer is emitted and the call returns empty."""
    import time
    start = time.perf_counter()
    inc("llm_calls_total")

    def _dispatch(use_model: str | None):
        if config.MODEL_BACKEND == "gemini":
            return _call_gemini(prompt, max_new_tokens, enable_thinking)
        if config.MODEL_BACKEND == "openrouter":
            return _call_openrouter(prompt, max_new_tokens, enable_thinking, model=use_model,
                                    image_b64=image_b64, app_label=app_label,
                                    reasoning_effort=reasoning_effort)
        return _call_ngrok(prompt, max_new_tokens, enable_thinking)

    result = None
    last_exc: Exception | None = None
    active_model, substituted = _effective_model(model)
    if substituted:
        log.info("llm_cooldown_skip", skipped=model or config.OPENROUTER_MODEL, using=active_model)
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            result = _dispatch(active_model)
            break
        except Exception as exc:
            last_exc = exc
            if _is_offline(exc):
                # Every model is behind the same connection, so falling back is
                # pointless. Wait for the network instead of failing the round.
                result = _wait_for_network(lambda: _dispatch(active_model), exc, "call_model")
                break
            if not _is_transient(exc):
                raise  # a bad request/API key fails identically on any model — no point retrying
            if attempt >= _MAX_ATTEMPTS:
                _mark_rate_limited(active_model)
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
                     app_label: str | None = None, reasoning_effort: str | None = None) -> dict:
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
        # An explicit effort is the fallback that endpoints DO accept: it keeps
        # the scratchpad small rather than asking for it to be switched off.
        payload["reasoning"] = ({"effort": reasoning_effort} if reasoning_effort
                                else {"enabled": False})
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


# ── Tool-calling transport (OpenRouter only) ─────────────────────────────────

def supports_tools() -> bool:
    """Whether the configured backend can drive a tool loop at all.

    Only the OpenRouter path speaks the tools/tool_calls contract. The gemini and
    ngrok backends take a plain prompt, so a caller must fall back to the
    single-prompt planner rather than silently produce a toolless loop.
    """
    return config.MODEL_BACKEND == "openrouter"


def chat_tools(messages: list[dict], tools: list[dict], model: str | None = None,
               app_label: str | None = None, reasoning_effort: str | None = None,
               temperature: float = 0.7, force_tool: str | None = None) -> dict:
    """One tool-calling turn: send the conversation, return the assistant message.

    Deliberately a transport, not a loop — the loop belongs with the agent that
    knows what its tools mean (planner/agent_loop.py). Returns the raw message
    dict (``content`` and/or ``tool_calls``) so the caller can append it to the
    conversation verbatim, which the API requires.

    ``force_tool`` pins tool_choice to one function, which is how a caller ends
    an open-ended loop: a model left on "auto" will happily keep investigating
    until the turn ceiling and never commit (observed — ten straight turns of
    tool calls with no proposal).

    Shares call_model's retry + FALLBACK_MODEL behaviour: providers rate-limit
    (429) constantly on the shared pool — the planner model returned 429 on six
    consecutive attempts during this feature's own bring-up — and an unretried
    one would kill a whole planning round.
    """
    import time
    if not supports_tools():
        raise HTTPException(status_code=503,
                            detail=f"MODEL_BACKEND={config.MODEL_BACKEND} does not support tool calling")
    if not config.OPENROUTER_API_KEY:
        raise HTTPException(status_code=503, detail="OPENROUTER_API_KEY not set in .env")

    start = time.perf_counter()
    inc("llm_calls_total")
    title = app_label or "QA Planner Agent"
    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": _APP_REFERERS.get(title, _APP_REFERERS["QA Planner Agent"]),
        "X-Title": title,
    }

    def _send(use_model: str):
        choice = ({"type": "function", "function": {"name": force_tool}}
                  if force_tool else "auto")
        payload = {"model": use_model, "messages": messages, "tools": tools,
                   "tool_choice": choice, "temperature": temperature}
        if reasoning_effort:
            payload["reasoning"] = {"effort": reasoning_effort}
        resp = requests.post(f"{config.OPENROUTER_BASE_URL}/chat/completions",
                             headers=headers, json=payload, timeout=600)
        if resp.status_code == 400 and "reasoning" in (resp.text or "").lower():
            payload.pop("reasoning", None)
            resp = requests.post(f"{config.OPENROUTER_BASE_URL}/chat/completions",
                                 headers=headers, json=payload, timeout=600)
        if resp.status_code == 400 and force_tool and "tool" in (resp.text or "").lower():
            # Not every provider honours a forced tool_choice. Falling back to
            # "auto" keeps the round alive; the caller has already appended an
            # explicit "propose now" instruction, so the model is still steered.
            log.warning("tool_choice_force_rejected", model=use_model,
                        error=(resp.text or "")[:200])
            payload["tool_choice"] = "auto"
            resp = requests.post(f"{config.OPENROUTER_BASE_URL}/chat/completions",
                                 headers=headers, json=payload, timeout=600)
        if resp.status_code >= 400:
            # Carry the provider's own message into the exception. A bare
            # "400 Client Error" says nothing about WHICH part of the payload
            # was rejected, and this call has several candidates (tools,
            # tool_choice, reasoning), so the message is the whole diagnosis.
            detail = ""
            try:
                detail = str(resp.json().get("error", {}).get("message", ""))[:300]
            except Exception:
                detail = (resp.text or "")[:300]
            raise RuntimeError(f"{resp.status_code} from OpenRouter: {detail}")
        data = resp.json()
        if not data.get("choices"):
            raise RuntimeError(f"no choices in response: {str(data)[:200]}")
        return data["choices"][0].get("message") or {}

    primary, substituted = _effective_model(model)
    if substituted:
        log.info("llm_cooldown_skip", skipped=model or config.OPENROUTER_MODEL, using=primary)
    last_exc: Exception | None = None
    message = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            message = _send(primary)
            break
        except Exception as exc:
            last_exc = exc
            if _is_offline(exc):
                message = _wait_for_network(lambda: _send(primary), exc, "chat_tools")
                break
            if not _is_transient(exc):
                raise
            if attempt >= _MAX_ATTEMPTS:
                _mark_rate_limited(primary)
                break
            delay = 2 ** attempt
            log.warning("llm_retry", backend="openrouter", attempt=attempt, delay_s=delay,
                        error=str(exc)[:180])
            inc("llm_retries_total")
            time.sleep(delay)

    if message is None:
        if config.FALLBACK_MODEL and config.FALLBACK_MODEL != primary:
            log.warning("llm_fallback", primary=primary, fallback=config.FALLBACK_MODEL,
                        error=str(last_exc)[:180])
            try:
                message = _send(config.FALLBACK_MODEL)
            except Exception as fallback_exc:
                raise last_exc from fallback_exc
        else:
            raise last_exc

    log.info("llm_tool_call", backend="openrouter",
             latency_ms=round((time.perf_counter() - start) * 1000, 1),
             tool_calls=len(message.get("tool_calls") or []),
             has_content=bool(message.get("content")))
    return message
