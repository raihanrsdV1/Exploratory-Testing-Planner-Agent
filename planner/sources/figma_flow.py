"""UI-flow knowledge source — screen-to-screen navigation transitions."""

from __future__ import annotations

from .. import context_builders, rag_client, textutil
from .base import KnowledgeSource, RetrievalRequest, RetrievedBlock


class FigmaFlowSource(KnowledgeSource):
    name = "figma_flow"
    channel = "figma_flow"
    purpose = "navigation / screen-to-screen behaviour"

    def is_available(self, brief: dict) -> bool:
        return bool(brief.get("screen_index"))

    def retrieve(self, project: str, request: RetrievalRequest, top_k: int) -> RetrievedBlock | None:
        screen = (request.screen or "").strip() or None
        transitions = rag_client.get_figma_transitions(project, screen_name=screen)
        if not transitions:
            return None
        block = context_builders.build_figma_flow_context(transitions, top_n=10)
        if not block:
            return None
        return RetrievedBlock(
            channel=self.channel,
            text=block,
            note=f"figma_flow | screen={screen or '*'} | {textutil.compact_note(block)}",
        )
