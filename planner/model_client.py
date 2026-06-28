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


def call_model(prompt: str, max_new_tokens: int, enable_thinking: bool) -> dict:
    import time
    start = time.perf_counter()
    inc("llm_calls_total")

    if config.MODEL_BACKEND == "gemini":
        result = _call_gemini(prompt, max_new_tokens, enable_thinking)
    elif config.MODEL_BACKEND == "openrouter":
        result = _call_openrouter(prompt, max_new_tokens, enable_thinking)
    else:
        result = _call_ngrok(prompt, max_new_tokens, enable_thinking)

    duration_ms = round((time.perf_counter() - start) * 1000, 1)
    estimated_tokens = len(prompt) // 4 + len(result.get("answer") or "") // 4
    
    log.info(
        "llm_call",
        backend=config.MODEL_BACKEND,
        latency_ms=duration_ms,
        estimated_tokens=estimated_tokens,
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


def _call_openrouter(prompt: str, max_new_tokens: int, enable_thinking: bool) -> dict:
    if not config.OPENROUTER_API_KEY:
        raise HTTPException(status_code=503, detail="OPENROUTER_API_KEY not set in .env")

    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://qa-planner-agent.local",
        "X-Title": "QA Planner Agent",
    }
    messages = [
        {"role": "system", "content": QA_SYSTEM_INSTRUCTION},
        {"role": "user", "content": prompt},
    ]
    payload = {
        "model": config.OPENROUTER_MODEL,
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": 0.7,
    }
    try:
        resp = requests.post(
            f"{config.OPENROUTER_BASE_URL}/chat/completions",
            headers=headers, json=payload, timeout=180,
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
