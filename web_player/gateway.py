"""Talking to the planner gateway and the RAG API.

Identical protocol to the Android executor — same endpoints, same payload shapes
— with one addition: every request carries ``platform="web"``, which the graph
already understands as a WP6 dimension (``rag_api/dimensions.py``). That single
field is what keeps one project's Android and web results in one graph without
either polluting the other's retrieval: dimension-tagged content is filtered,
untagged content still matches both.
"""

from __future__ import annotations

import time

import requests

import settings as cfg

GATEWAY_URL = cfg.GATEWAY_URL.rstrip("/")
RAG_URL = cfg.RAG_URL.rstrip("/")

PLATFORM = "web"

_MAX_ATTEMPTS = 4
# A 503 naming one of these is a configuration problem; retrying just fails slower.
_PERMANENT = ("not set in .env", "not installed")


def next_testcase(max_new_tokens: int = 8000) -> dict:
    """Ask the planner for the next test case, scoped to the web platform."""
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        resp = requests.post(
            f"{GATEWAY_URL}/agent/next-testcase",
            json={
                "project": cfg.PROJECT,
                "app_name": cfg.WEB_SITE_NAME,
                "objective": "generate next high-value non-duplicate test case",
                "top_k": cfg.TOP_K,
                "max_new_tokens": max_new_tokens,
                "max_retrieval_rounds": 2,
                "enable_thinking": False,
                "debug_trace": cfg.DEBUG_TRACE,
                "platform": PLATFORM,
            },
            timeout=900,
        )
        if resp.status_code != 503:
            resp.raise_for_status()
            return resp.json()

        if attempt >= _MAX_ATTEMPTS or any(t in resp.text.lower() for t in _PERMANENT):
            resp.raise_for_status()
        delay = 2 ** attempt
        print(f"  ⚠️  gateway 503, retry {attempt}/{_MAX_ATTEMPTS} in {delay}s: {resp.text[:160]}")
        time.sleep(delay)
    return {}


def log_verdict(tc: dict, verdict: str, notes: str) -> dict:
    """Record one executed test case in the knowledge graph."""
    resp = requests.post(
        f"{RAG_URL}/tests/log",
        json={
            "project": cfg.PROJECT,
            "test_case_id": tc.get("test_case_id", "TC-WEB-FALLBACK"),
            "title": tc.get("title", "Web executor test"),
            "verdict": verdict,
            "notes": notes,
            "area": tc.get("area", "general"),
            "requirement_ids": tc.get("requirement_ids", []) or [],
            "platform": PLATFORM,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def log_execution(tc: dict, verdict: str, duration_ms: float, agent_steps: int,
                  urls: list[str], error_type: str = "", error_message: str = "",
                  recovery_action: str = "") -> None:
    """Persist the execution record (WP3): timing, budget, environment, route.

    ``path`` is empty by design. It is meant to hold UIState ids, and the web
    player builds no UIState graph; sending URLs there would look like a state
    map that nothing verified. The route goes in ``path_labels``, which is free
    text, so the trace is still visible in the batch CSV without inventing graph
    nodes.
    """
    payload = {
        "project": cfg.PROJECT,
        "test_case_id": tc.get("test_case_id", ""),
        "title": tc.get("title", ""),
        "verdict": verdict,
        "duration_ms": int(duration_ms),
        "planned_steps": len(tc.get("steps", []) or []),
        "device_steps": int(agent_steps or 0),
        "states_visited": 0,
        "error_type": error_type,
        "error_message": (error_message or "")[:500],
        "recovery_action": (recovery_action or "")[:300],
        "device": f"{cfg.WEB_BROWSER} {cfg.WEB_VIEWPORT}",
        "os_version": "",
        "app_package": cfg.WEB_BASE_URL,
        "path": [],
        "path_labels": urls[:40],
    }
    try:
        requests.post(f"{RAG_URL}/execution/log", json=payload, timeout=30).raise_for_status()
        print(f"   📋 Execution logged: {len(urls)} URLs, verdict={verdict}")
    except Exception as exc:
        print(f"   ⚠️  Execution log failed for {tc.get('test_case_id')}: {exc}")


def recent_tests(limit: int = 10) -> list[dict]:
    try:
        resp = requests.get(f"{RAG_URL}/tests/recent",
                            params={"project": cfg.PROJECT, "limit": limit}, timeout=60)
        resp.raise_for_status()
        return resp.json().get("tests", [])
    except Exception:
        return []
