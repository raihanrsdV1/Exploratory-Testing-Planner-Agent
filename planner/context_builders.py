"""
Format knowledge-graph data into compact prompt blocks.

These are app-agnostic: every label/screen/area comes from the ingested graph.
"""

from __future__ import annotations

from . import rag_client


def pick_relevant_screens(screens: list[dict], done_areas: list[str], recent_tests: list[dict]) -> list[str]:
    """Choose up to 2 screens to detail, biased toward untested, interaction-rich screens."""
    if not screens:
        return []
    tested_areas = {str(t.get("area", "")).lower().replace(" ", "_") for t in recent_tests}
    tested_areas.update(a.lower().replace(" ", "_") for a in done_areas)
    untested = [s for s in screens if s.get("purpose", "other") not in tested_areas]
    chosen = untested if untested else screens
    chosen = sorted(chosen, key=lambda s: s.get("interactive_count", 0), reverse=True)
    return [s["screen_name"] for s in chosen[:2]]


def build_figma_context(project: str, screen_names: list[str]) -> str:
    """Fetch + format interactive elements for the chosen screens (exact labels for steps)."""
    if not screen_names:
        return ""
    lines: list[str] = []
    for name in screen_names:
        elements = rag_client.get_screen_elements(project, name)
        if not elements:
            continue
        lines.append(f"[Screen: {name}]")
        for kind, labels in elements.items():
            lines.append(f"  {kind}s: {', '.join(labels[:10])}")
    return "\n".join(lines)


def build_figma_overview_context(figma_overview: list[dict]) -> str:
    if not figma_overview:
        return ""
    lines = ["Available screens and key UI elements:"]
    for s in figma_overview:
        lines.append(
            f"- {s.get('screen_name','?')} (purpose={s.get('purpose','other')}, interactive={s.get('interactive_count',0)})"
        )
        if s.get("buttons"):
            lines.append("  buttons: " + ", ".join(s.get("buttons", [])[:4]))
        if s.get("inputs"):
            lines.append("  inputs: " + ", ".join(s.get("inputs", [])[:4]))
        if s.get("nav"):
            lines.append("  navigation: " + ", ".join(s.get("nav", [])[:3]))
    return "\n".join(lines)


def build_figma_overview_generalized(figma_overview: list[dict]) -> str:
    """Generalized UI context for planning (less label-heavy, less bias)."""
    if not figma_overview:
        return "No screens available"
    lines = [f"Total screens: {len(figma_overview)}", "Screens by purpose:"]
    for s in figma_overview:
        lines.append(
            f"- {s.get('screen_name','?')} (purpose={s.get('purpose','other')}, interactive={s.get('interactive_count',0)})"
        )
    return "\n".join(lines)


def recent_tests_exact(recent_tests: list[dict], limit: int = 50) -> str:
    if not recent_tests:
        return "none"
    return "; ".join(
        f"{t.get('id','?')}|{t.get('verdict','?')}|{t.get('area','general')}|{t.get('title','')[:120]}"
        for t in recent_tests[:limit]
    )


def build_figma_flow_context(transitions: list[dict], top_n: int = 12) -> str:
    if not transitions:
        return ""
    lines = ["Known UI transitions (from prototype links / inferred):"]
    for t in transitions[:top_n]:
        lines.append(f"- {t.get('from_screen','?')} --[{t.get('via_element','?')}]--> {t.get('to_screen','?')}")
    return "\n".join(lines)


def screen_index_compact(screen_index: list[dict], limit: int = 7) -> str:
    if not screen_index:
        return "[]"
    ordered = sorted(
        screen_index,
        key=lambda s: (-int(s.get("interactive_count", 0) or 0), str(s.get("screen_name", "")).lower()),
    )
    parts = [
        f"{s.get('screen_name','?')}|purpose={s.get('purpose','other')}|interactive={s.get('interactive_count',0)}"
        for s in ordered[:limit]
    ]
    return "[" + "; ".join(parts) + "]"
