"""UI-guide knowledge source — interactive elements per screen (exact labels for steps)."""

from __future__ import annotations

from .. import rag_client, textutil
from .base import KnowledgeSource, RetrievalRequest, RetrievedBlock


class FigmaUISource(KnowledgeSource):
    name = "figma_ui"
    channel = "figma_ui"
    purpose = "screen elements and control availability"

    def is_available(self, brief: dict) -> bool:
        return bool(brief.get("screen_index"))

    def retrieve(self, project: str, request: RetrievalRequest, top_k: int) -> RetrievedBlock | None:
        screen = (request.screen or request.query or "").strip()
        if not screen:
            return None
        elements = rag_client.get_screen_elements(project, screen)
        lines = [f"[Screen: {screen}]"]
        for kind, labels in elements.items():
            lines.append(f"  {kind}s: {', '.join(labels[:10])}")
        block = "\n".join(lines)
        if block.strip() == f"[Screen: {screen}]":
            return None
        return RetrievedBlock(
            channel=self.channel,
            text=block,
            note=f"figma_ui | screen={screen} | {textutil.compact_note(block)}",
        )
