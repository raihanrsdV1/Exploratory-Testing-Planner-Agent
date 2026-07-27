"""Navigation-memory knowledge source (ETA-REQ-302.4).

Surfaces the learned shortest path to a target screen so the model writes steps
that follow proven navigation instead of re-exploring. App-agnostic: screens and
actions come from paths the executor actually walked.
"""

from __future__ import annotations

from .. import rag_client, textutil
from .base import KnowledgeSource, RetrievalRequest, RetrievedBlock


def format_path(steps: list[dict]) -> str:
    """Render nav-tree steps as a compact, numbered navigation path."""
    if not steps:
        return ""
    lines = []
    for s in steps:
        depth = s.get("depth", 0)
        action = s.get("action", "") or ""
        screen = s.get("screen", "") or "?"
        if action in ("(entry)", "") and depth == 0:
            lines.append(f"{depth}. start at '{screen}'")
        else:
            lines.append(f"{depth}. {action} → '{screen}'")
    return "\n".join(lines)


class NavTreeSource(KnowledgeSource):
    name = "navtree"
    channel = "navtree"
    purpose = "learned navigation paths — the proven shortest route to reach a target screen"

    def is_available(self, brief: dict) -> bool:
        return bool(brief.get("navtree_node_count"))

    def retrieve(self, project: str, request: RetrievalRequest, top_k: int) -> RetrievedBlock | None:
        if not request.screen:
            return None
        try:
            data = rag_client.rag_get(
                "/navtree/retrieve-path", {"project": project, "screen": request.screen}
            )
        except Exception:
            return None
        steps = data.get("steps", []) or []
        if not steps:
            return None
        body = format_path(steps)
        text = f"Proven shortest path to '{request.screen}':\n{body}"
        return RetrievedBlock(
            channel=self.channel,
            text=text,
            note=f"navtree | {len(steps)} steps to {request.screen}",
        )
