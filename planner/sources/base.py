"""
Pluggable knowledge-source abstraction for the planner.

Each source wraps one retrievable slice of the knowledge graph (SRS, UI, flow,
and — in later work packages — defects, navigation memory, ...). The agent loop
advertises only *available* sources to the retrieval planner and dispatches
retrieval through the registry, so no single source is a hard dependency and new
sources are added by registration alone.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class RetrievalRequest:
    """One retrieval ask from the planner, normalised for a source."""

    source: str
    query: str = ""
    screen: str = ""


@dataclass
class RetrievedBlock:
    """A formatted context block returned by a source."""

    channel: str  # which context bucket this block feeds (srs | figma_ui | figma_flow | ...)
    text: str     # the formatted context block, ready to drop into a prompt
    note: str     # short one-line note for the planner's "retrieved so far" list


class KnowledgeSource(ABC):
    """Interface every retrievable knowledge source implements."""

    #: planner-facing id, used as retrieval_requests[].source
    name: str = ""
    #: context bucket the retrieved text feeds into
    channel: str = ""
    #: one-line guidance shown to the retrieval planner ("Use source=<name> for ...")
    purpose: str = ""

    @abstractmethod
    def is_available(self, brief: dict) -> bool:
        """True when this source has ingested data for the project (checked from the brief)."""

    @abstractmethod
    def retrieve(self, project: str, request: RetrievalRequest, top_k: int) -> RetrievedBlock | None:
        """Fetch a context block for the request, or None if nothing was found."""
