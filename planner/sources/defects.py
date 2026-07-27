"""Defect-history knowledge source (ETA-REQ-301.5).

Makes historical defects retrievable by the planner so generation is biased
toward areas and behaviours that have broken before. App-agnostic: all defect
data comes from the ingested graph.
"""

from __future__ import annotations

from .. import rag_client, textutil
from .base import KnowledgeSource, RetrievalRequest, RetrievedBlock


class DefectSource(KnowledgeSource):
    name = "defects"
    channel = "defects"
    purpose = "historical defect reports — the areas and behaviours that have broken before"

    def is_available(self, brief: dict) -> bool:
        return bool(brief.get("defect_count"))

    def retrieve(self, project: str, request: RetrievalRequest, top_k: int) -> RetrievedBlock | None:
        try:
            data = rag_client.rag_get(
                "/defects/context",
                {"project": project, "query": request.query, "area": request.screen, "top_k": top_k},
            )
        except Exception:
            return None
        text = (data.get("context") or "").strip()
        if not text:
            return None
        return RetrievedBlock(
            channel=self.channel,
            text=text,
            note=f"defects | {textutil.compact_note(text)}",
        )
