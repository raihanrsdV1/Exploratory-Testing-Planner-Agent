"""
Knowledge-source registry.

The single place that knows which sources exist. The agent loop asks the registry
which sources are *available* for a project (from the brief) and dispatches
retrieval by source name. Adding a new source (defects, navtree, ...) is a matter
of importing it and appending it to ``_SOURCES``.
"""

from __future__ import annotations

from .base import KnowledgeSource, RetrievalRequest, RetrievedBlock
from .defects import DefectSource
from .figma_flow import FigmaFlowSource
from .figma_ui import FigmaUISource
from .liveui import LiveUISource
from .navtree import NavTreeSource
from .srs import SRSSource

import settings as _settings

# Order = the order the retrieval planner sees sources advertised in.
_SOURCES: list[KnowledgeSource] = [
    SRSSource(), FigmaUISource(), FigmaFlowSource(), LiveUISource(),
    DefectSource(), NavTreeSource(),
]


def all_sources() -> list[KnowledgeSource]:
    return list(_SOURCES)


def enabled_sources() -> list[KnowledgeSource]:
    """Sources permitted by configuration (settings.ENABLED_SOURCES).

    A source is disabled when its data is known to be unreliable — a design file
    that has drifted from the shipped app does not merely fail to help, it
    supplies screens and controls that do not exist and the planner writes tests
    against them.
    """
    allowed = set(getattr(_settings, "ENABLED_SOURCES", ()) or ())
    if not allowed:
        return list(_SOURCES)
    return [s for s in _SOURCES if s.name in allowed]


def available_sources(brief: dict) -> list[KnowledgeSource]:
    """Sources that actually have ingested data for this project."""
    return [s for s in enabled_sources() if s.is_available(brief or {})]


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
