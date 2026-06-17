"""HTTP client for the local RAG / knowledge-graph API."""

from __future__ import annotations

import requests
from fastapi import HTTPException

from . import config


def rag_headers() -> dict:
    return {"Authorization": f"Bearer {config.RAG_API_KEY}"} if config.RAG_API_KEY else {}


def rag_get(path: str, params: dict | None = None) -> dict:
    try:
        resp = requests.get(f"{config.RAG_API_URL}{path}", params=params, headers=rag_headers(), timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"RAG backend unavailable: {e}")


def rag_post(path: str, body: dict) -> dict:
    try:
        resp = requests.post(f"{config.RAG_API_URL}{path}", json=body, headers=rag_headers(), timeout=60)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"RAG backend unavailable: {e}")


# ── Typed knowledge-graph queries ───────────────────────────────────────────────

def get_srs_and_history(project: str, query: str, top_k: int) -> dict:
    """
    Hybrid (vector + keyword + graph-hop) SRS retrieval. `query` is sent as
    natural language so the RAG layer can embed it for semantic matching.
    """
    return rag_post("/retrieve", {"project": project, "query": query, "top_k": top_k, "include_history": False})


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
