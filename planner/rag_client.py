"""HTTP client for the local RAG / knowledge-graph API."""

from __future__ import annotations

import requests
from fastapi import HTTPException

from observability import get_logger, inc
from . import config

log = get_logger("rag_client")


def rag_headers() -> dict:
    return {"Authorization": f"Bearer {config.RAG_API_KEY}"} if config.RAG_API_KEY else {}


def rag_get(endpoint: str, params: dict | None = None, timeout: int = 60) -> dict:
    import time
    start = time.perf_counter()
    inc("rag_calls_total")
    try:
        resp = requests.get(
            f"{config.RAG_API_URL.rstrip('/')}{endpoint}",
            params=params, headers=rag_headers(), timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        duration_ms = round((time.perf_counter() - start) * 1000, 1)
        log.info("rag_call", method="GET", endpoint=endpoint, latency_ms=duration_ms)
        return data
    except requests.RequestException as e:
        duration_ms = round((time.perf_counter() - start) * 1000, 1)
        log.error("rag_error", method="GET", endpoint=endpoint, error=str(e), latency_ms=duration_ms)
        raise HTTPException(status_code=503, detail=f"RAG API unavailable ({endpoint}): {e}")


def rag_post(endpoint: str, payload: dict, timeout: int = 120) -> dict:
    import time
    start = time.perf_counter()
    inc("rag_calls_total")
    try:
        resp = requests.post(
            f"{config.RAG_API_URL.rstrip('/')}{endpoint}",
            json=payload, headers=rag_headers(), timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        duration_ms = round((time.perf_counter() - start) * 1000, 1)
        log.info("rag_call", method="POST", endpoint=endpoint, latency_ms=duration_ms)
        return data
    except requests.RequestException as e:
        duration_ms = round((time.perf_counter() - start) * 1000, 1)
        log.error("rag_error", method="POST", endpoint=endpoint, error=str(e), latency_ms=duration_ms)
        raise HTTPException(status_code=503, detail=f"RAG API unavailable ({endpoint}): {e}")


# ── Typed knowledge-graph queries ───────────────────────────────────────────────

def get_srs_and_history(project: str, query: str, top_k: int, dims: dict | None = None) -> dict:
    """
    Hybrid (vector + keyword + graph-hop) SRS retrieval. `query` is sent as
    natural language so the RAG layer can embed it for semantic matching.
    `dims` (WP6) optionally filters retrieval to a profile/platform/application.
    """
    payload = {"project": project, "query": query, "top_k": top_k, "include_history": False}
    payload.update(dims or {})
    return rag_post("/retrieve", payload)


def get_figma_screens(project: str) -> list[dict]:
    return rag_get("/figma/screens", {"project": project}).get("screens", [])


def get_screen_elements(project: str, screen_name: str) -> dict[str, list[str]]:
    data = rag_get("/figma/elements", {"project": project, "screen_name": screen_name, "interactive_only": "true"})
    return data.get("elements", {})


def get_figma_overview(project: str) -> list[dict]:
    return rag_get("/figma/overview", {"project": project, "top_labels": 4}).get("screens", [])


def get_figma_transitions(project: str, screen_name: str | None = None) -> list[dict]:
    params = {"project": project, "limit": 80}
    if screen_name:
        params["screen_name"] = screen_name
    return rag_get("/figma/transitions", params).get("transitions", [])


def get_brief_context(project: str) -> dict:
    return rag_post("/context/brief", {"project": project, "recent_limit": 100})


def semantic_dedup_check(project: str, title: str, threshold: float = 0.9) -> dict:
    """WP8 (307.3): server-side embedding-cosine duplicate check for a candidate title.

    Returns {enabled, is_duplicate, similarity, most_similar_title}. Best-effort — a
    failed/disabled backend degrades to a non-duplicate so generation never blocks."""
    try:
        return rag_post("/tests/dedup-check", {"project": project, "title": title, "threshold": threshold})
    except Exception:
        return {"enabled": False, "is_duplicate": False, "similarity": 0.0, "most_similar_title": ""}


def get_requirement_coverage(project: str) -> dict:
    """Per-requirement coverage, including the real ref_ids of untested requirements."""
    return rag_get("/coverage/requirements", {"project": project})


def get_business_rules(project: str) -> list[dict]:
    """Extracted validation rules with their owning requirement ref_id + confidence."""
    return rag_get("/business-logic/rules", {"project": project}).get("rules", [])
