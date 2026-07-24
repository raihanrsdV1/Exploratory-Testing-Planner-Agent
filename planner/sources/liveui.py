"""Live App Model knowledge source — the self-built UI state map (WP1).

Makes the observed app map (states the agent has actually reached, and the
transitions between them) retrievable by the planner, so exploration is grounded
in the real app structure even when no Figma/SRS exists.
"""

from __future__ import annotations

from .. import rag_client, textutil
from .base import KnowledgeSource, RetrievalRequest, RetrievedBlock


class LiveUISource(KnowledgeSource):
    name = "live_ui"
    channel = "figma_ui"  # reuse the UI context bucket (same prompt slot as design UI)
    purpose = "the observed live app map — real screens reached and how to navigate between them"

    def is_available(self, brief: dict) -> bool:
        return bool(brief.get("appmodel_state_count"))

    def retrieve(self, project: str, request: RetrievalRequest, top_k: int) -> RetrievedBlock | None:
        try:
            data = rag_client.rag_get("/appmodel/graph", {"project": project})
        except Exception:
            return None
        nodes = data.get("nodes", []) or []
        edges = data.get("edges", []) or []
        if not nodes:
            return None

        by_id = {n["id"]: n for n in nodes}
        lines = ["Observed app map (live states reached during exploration):"]
        for n in nodes[:12]:
            flag = " [dialog]" if n.get("has_dialog") else ""
            lines.append(f"- {n.get('label','?')}{flag} (visited {n.get('visits',0)}x, {n.get('elements',0)} controls)")
        if edges:
            lines.append("Known transitions:")
            for e in edges[:14]:
                a = by_id.get(e["source"], {}).get("label", "?")
                b = by_id.get(e["target"], {}).get("label", "?")
                lines.append(f"- {a} --[{e.get('action','')}]--> {b}")

        text = "\n".join(lines)
        return RetrievedBlock(
            channel=self.channel,
            text=text,
            note=f"live_ui | {len(nodes)} states, {len(edges)} transitions | {textutil.compact_note(text)}",
        )
