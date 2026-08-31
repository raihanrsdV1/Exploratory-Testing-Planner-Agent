"""Live App Model knowledge source — the self-built UI state map (WP1).

Makes the observed app map (states the agent has actually reached, and the
transitions between them) retrievable by the planner, so exploration is grounded
in the real app structure even when no Figma/SRS exists.
"""

from __future__ import annotations

from .. import rag_client, textutil
from .base import KnowledgeSource, RetrievalRequest, RetrievedBlock

import settings as _settings

_MATCH_THRESHOLD = _settings.LIVE_SCREEN_MATCH_THRESHOLD


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
        # WP1 decay: ignore states that have faded out (stale after an app update),
        # unless every known state is stale (then keep them so we still have a map).
        fresh = [n for n in nodes if not n.get("stale")]
        if fresh:
            nodes = fresh
        if not nodes:
            return None

        wanted = (request.screen or request.query or "").strip()
        if wanted:
            return self._retrieve_one_screen(nodes, wanted)
        return self._retrieve_overview(nodes, edges)

    def _retrieve_one_screen(self, nodes: list[dict], wanted: str) -> RetrievedBlock | None:
        """A specific screen was asked for — find the REAL observed screen it refers
        to (fuzzy name match, since `wanted` may be a Figma design-file name that
        doesn't spell the runtime label identically) and return just that screen's
        real controls, instead of the generic whole-map overview below. Returns
        None on no confident match rather than guessing wrong."""
        best, best_score = None, 0.0
        for n in nodes:
            score = textutil.unicode_jaccard(wanted, n.get("label", ""))
            if score > best_score:
                best, best_score = n, score
        if best is None or best_score < _MATCH_THRESHOLD:
            return None

        flag = " [dialog]" if best.get("has_dialog") else ""
        lines = [f"[Observed screen: {best.get('label', '?')}{flag}] "
                 f"(visited {best.get('visits', 0)}x, {best.get('elements', 0)} controls)"]
        controls = best.get("controls") or []
        if controls:
            lines.append(f"Real controls on this screen: {', '.join(controls[:15])}")
        text = "\n".join(lines)

        resolved_state = {
            "id": best.get("id"), "label": best.get("label"),
            "has_screenshot": bool(best.get("has_shot")), "match_score": round(best_score, 2),
        }
        return RetrievedBlock(
            channel=self.channel,
            text=text,
            note=f"live_ui | matched '{wanted}' -> '{best.get('label', '?')}' "
                 f"({best_score:.2f}) | {len(controls)} controls",
            resolved_state=resolved_state,
        )

    def _retrieve_overview(self, nodes: list[dict], edges: list[dict]) -> RetrievedBlock | None:
        """No specific screen asked for — a general summary of the whole observed
        app map (original behaviour, unchanged)."""
        by_id = {n["id"]: n for n in nodes}
        lines = ["Observed app map — these are the REAL screens and control names seen on the device.",
                 "Prefer these names in test steps over design-file labels; they are what exists at runtime."]
        for n in nodes[:12]:
            flag = " [dialog]" if n.get("has_dialog") else ""
            lines.append(f"- {n.get('label','?')}{flag} (visited {n.get('visits',0)}x, {n.get('elements',0)} controls)")
            controls = n.get("controls") or []
            if controls:
                lines.append(f"    controls: {', '.join(controls[:8])}")
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
