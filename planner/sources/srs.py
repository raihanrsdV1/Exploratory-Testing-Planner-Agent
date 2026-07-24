"""SRS knowledge source — hybrid (vector + keyword + graph-hop) retrieval over requirements."""

from __future__ import annotations

from .. import rag_client, textutil
from .base import KnowledgeSource, RetrievalRequest, RetrievedBlock


class SRSSource(KnowledgeSource):
    name = "srs"
    channel = "srs"
    purpose = "business rules, validation constraints, and error conditions"

    def is_available(self, brief: dict) -> bool:
        return bool((brief.get("srs_summary") or "").strip())

    def retrieve(self, project: str, request: RetrievalRequest, top_k: int) -> RetrievedBlock | None:
        query = (request.query or "").strip()
        if not query:
            return None
        data = rag_client.get_srs_and_history(project, query, top_k=min(top_k, 2))
        block = data.get("context", "")
        if not block:
            return None
        return RetrievedBlock(
            channel=self.channel,
            text=block,
            note=f"srs | query={query} | {textutil.compact_note(block)}",
        )
