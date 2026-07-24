"""
Knowledge-source registry.

The single place that knows which sources exist. The agent loop asks the registry
which sources are *available* for a project (from the brief) and dispatches
retrieval by source name. Adding a new source (defects, navtree, ...) is a matter
of importing it and appending it to ``_SOURCES``.
"""

from __future__ import annotations

from .base import KnowledgeSource, RetrievalRequest, RetrievedBlock
from .figma_flow import FigmaFlowSource
from .figma_ui import FigmaUISource
from .srs import SRSSource

# Order = the order the retrieval planner sees sources advertised in.
_SOURCES: list[KnowledgeSource] = [SRSSource(), FigmaUISource(), FigmaFlowSource()]


def all_sources() -> list[KnowledgeSource]:
    return list(_SOURCES)


def available_sources(brief: dict) -> list[KnowledgeSource]:
    """Sources that actually have ingested data for this project."""
    return [s for s in _SOURCES if s.is_available(brief or {})]


def get(name: str) -> KnowledgeSource | None:
    name = (name or "").strip().lower()
    for s in _SOURCES:
        if s.name == name:
            return s
    return None


__all__ = [
    "KnowledgeSource",
    "RetrievalRequest",
    "RetrievedBlock",
    "all_sources",
    "available_sources",
    "get",
]
