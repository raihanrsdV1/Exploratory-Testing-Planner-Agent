"""
UI design normaliser.

Turns a UI design export into a canonical UI IR that the graph builder persists,
independent of the design tool. Today it understands Figma file exports; the same
IR shape can be produced from an Android view-hierarchy / accessibility dump or a
vision-model screenshot parse in future, with zero downstream changes.

Canonical UI IR:
    {
      "source_name": str,
      "screens": [
        {
          "screen_name": str,
          "node_id": str,
          "purpose": str,                 # generic slug derived from the name (NOT a fixed app map)
          "elements": [
            {"kind","label","name","node_id","interactive"}
          ],
          "transitions": [               # real prototype links when the export has them
            {"via_node_id","via_label","to_node_id"}
          ],
        }
      ],
    }

Key change vs. the old parser: there is NO hardcoded per-app purpose map.
`purpose` is a generic slug derived from the screen name, and an optional
`purpose_hints` map (e.g. produced by an LLM classification pass in the gateway)
can override it for any app.
"""

from __future__ import annotations

import re


# Generic interactive-element keyword sets (app-agnostic; widget vocabulary, not domain terms).
_BUTTON_KW = ("button", "btn", "fab", "link -", "cta", "submit", "action")
_INPUT_KW = ("input", "field", "textarea", "textfield", "text field", "phone input", "search bar", "form")
_NAV_KW = ("bottom navigation", "bottomnavbar", "bottom nav", "tab bar", "tabbar", "navbar", "nav rail")
_TOGGLE_KW = ("toggle", "switch", "checkbox", "radio")
_DROPDOWN_KW = ("dropdown", "select", "picker", "spinner", "combobox")


def slug(text: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", (text or "").lower())).strip("_") or "screen"


def derive_purpose(screen_name: str, purpose_hints: dict[str, str] | None = None) -> str:
    """
    Generic, app-agnostic purpose slug.

    Priority:
      1. explicit hint from caller (e.g. LLM classification) keyed by screen name
      2. slug of the screen name with cosmetic qualifiers stripped
    """
    if purpose_hints:
        hint = purpose_hints.get(screen_name)
        if hint:
            return slug(hint)
    # Strip common design qualifiers in parentheses, e.g. "Home Screen (Dark Mode)".
    base = re.sub(r"\(.*?\)", "", screen_name or "").strip()
    return slug(base or screen_name)


# ── Figma traversal helpers ─────────────────────────────────────────────────────

def _iter_text_nodes(node: dict, _depth: int = 0):
    if _depth > 6 or not isinstance(node, dict):
        return
    if node.get("type") == "TEXT":
        yield node
    for child in node.get("children", []) or []:
        yield from _iter_text_nodes(child, _depth + 1)


def _all_text_in_subtree(node: dict, max_depth: int = 5, _depth: int = 0) -> list[str]:
    if _depth > max_depth or not isinstance(node, dict):
        return []
    texts = []
    if node.get("type") == "TEXT" and (node.get("characters") or "").strip():
        texts.append(node["characters"].strip())
    for child in node.get("children", []) or []:
        texts.extend(_all_text_in_subtree(child, max_depth, _depth + 1))
    return texts


def _element_label(node: dict) -> str:
    texts = _all_text_in_subtree(node, max_depth=4)
    seen: dict[str, None] = {}
    for t in texts:
        seen[t] = None
    label = " / ".join(seen.keys())
    return label[:120] if label else node.get("name", "")


def _classify_node(node: dict) -> dict | None:
    """Best-effort element classification from Figma structure + layer name."""
    name = node.get("name", "")
    name_lower = name.lower()
    ntype = node.get("type", "")
    component_id = node.get("componentId")  # present in richer exports

    # Components / instances are almost always meaningful interactive widgets.
    is_componentish = ntype in {"INSTANCE", "COMPONENT"} or bool(component_id)

    if ntype in {"FRAME", "INSTANCE", "COMPONENT", "GROUP"}:
        if any(kw in name_lower for kw in _BUTTON_KW):
            return {"kind": "button", "label": _element_label(node), "name": name, "interactive": True}
        if any(kw in name_lower for kw in _INPUT_KW):
            return {"kind": "input", "label": _element_label(node) or name, "name": name, "interactive": True}
        if any(kw in name_lower for kw in _NAV_KW):
            tabs = [c.get("characters", "").strip() for c in _iter_text_nodes(node) if c.get("characters", "").strip()]
            tab_label = " | ".join(dict.fromkeys(tabs)) if tabs else name
            return {"kind": "navigation", "label": tab_label, "name": name, "interactive": True}
        if any(kw in name_lower for kw in _TOGGLE_KW):
            return {"kind": "control", "label": _element_label(node) or name, "name": name, "interactive": True}
        if any(kw in name_lower for kw in _DROPDOWN_KW):
            return {"kind": "dropdown", "label": _element_label(node) or name, "name": name, "interactive": True}
        if "section" in name_lower or "header" in name_lower or "heading" in name_lower:
            texts = _all_text_in_subtree(node, max_depth=2)
            if texts:
                return {"kind": "section", "label": texts[0], "name": name, "interactive": False}
        if is_componentish:
            # Unknown component instance — still a meaningful, likely interactive element.
            return {"kind": "control", "label": _element_label(node) or name, "name": name, "interactive": True}
    return None


def _extract_transitions(node: dict, out: list, _depth: int = 0):
    """
    Pull real prototype navigation links when the export includes them.
    Figma encodes these as `transitionNodeID` and/or `reactions[].action.destinationId`.
    """
    if _depth > 12 or not isinstance(node, dict):
        return
    dest = node.get("transitionNodeID")
    if dest:
        out.append({"via_node_id": node.get("id", ""), "via_label": _element_label(node), "to_node_id": dest})
    for reaction in node.get("reactions", []) or []:
        action = reaction.get("action") or {}
        dest2 = action.get("destinationId")
        if dest2:
            out.append({"via_node_id": node.get("id", ""), "via_label": _element_label(node), "to_node_id": dest2})
    for child in node.get("children", []) or []:
        _extract_transitions(child, out, _depth + 1)


def _walk_for_elements(node: dict, out: list, depth: int, max_depth: int):
    if depth > max_depth or not isinstance(node, dict) or node.get("visible") is False:
        return
    element = _classify_node(node)
    for child in node.get("children", []) or []:
        _walk_for_elements(child, out, depth + 1, max_depth)
    if element:
        element["node_id"] = node.get("id", "")
        out.append(element)


def _collect_frame_candidates(figma_data: dict) -> list[dict]:
    doc = figma_data.get("document", {})
    pages = doc.get("children", []) if isinstance(doc, dict) else []
    frames: list[dict] = []
    if pages:
        for page in pages:
            if page.get("type") == "FRAME":
                frames.append(page)
            frames.extend([c for c in (page.get("children") or []) if c.get("type") == "FRAME"])
    elif figma_data.get("children"):
        frames = [c for c in figma_data.get("children", []) if c.get("type") == "FRAME"]
    if not frames and isinstance(doc, dict) and doc.get("type") == "FRAME":
        frames = [doc]
    return frames


def normalize_figma(figma_data: dict, purpose_hints: dict[str, str] | None = None) -> dict:
    """Parse a Figma file export into the canonical UI IR."""
    frames = _collect_frame_candidates(figma_data)
    screens: list[dict] = []

    for frame in frames:
        screen_name = frame.get("name", "Unknown")
        elements: list[dict] = []
        _walk_for_elements(frame, elements, depth=0, max_depth=9)

        # Deduplicate elements by (kind, label).
        seen: set[tuple] = set()
        deduped: list[dict] = []
        for el in elements:
            key = (el["kind"], el["label"])
            if el["label"] and key not in seen:
                seen.add(key)
                deduped.append(el)

        transitions: list[dict] = []
        _extract_transitions(frame, transitions)

        screens.append({
            "screen_name": screen_name,
            "node_id": frame.get("id", ""),
            "purpose": derive_purpose(screen_name, purpose_hints),
            "elements": deduped,
            "transitions": transitions,
        })

    return {
        "source_name": figma_data.get("name", "figma"),
        "screens": screens,
    }


def screen_label_index(ui_ir: dict, top: int = 6) -> list[dict]:
    """Compact (screen_name -> sample interactive labels) index for an LLM classification pass."""
    out = []
    for s in ui_ir.get("screens", []):
        labels = [e["label"] for e in s.get("elements", []) if e.get("interactive") and e.get("label")][:top]
        out.append({"screen_name": s["screen_name"], "labels": labels})
    return out
