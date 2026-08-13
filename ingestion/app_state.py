"""
Live App Model — UI state abstraction.

Turns a *normalized* UI observation (the shape produced by mobilerun's
``macro.state.normalize_ui_state`` — or any ``{phone_state:{package,activity},
nodes:[...]}`` dict) into a stable STATE SIGNATURE.

Design goal (the hard part of model-based GUI testing): the signature must be
**robust to volatile content** — scrolling a list of different data, or a
light/dark theme switch, must NOT create a new state — while still
**distinguishing genuinely different screens** (a new activity, an open dialog,
an empty vs. populated list).

How it stays robust:
  * Identity is STRUCTURAL, not visual. A theme switch changes pixels, never the
    view hierarchy, so a structural signature is theme-invariant by construction.
  * The per-node key drops the volatile free ``text`` (the contact names that
    change while scrolling) and keeps only *what a control is*:
    (resource_id, class, content_description, clickable).
  * Identical structural keys collapse (50 contact rows sharing one row template
    → one key), so row COUNT and row DATA don't affect the signature.
  * A few salient landmarks that DO define a distinct testable state are kept
    (package, activity, whether a dialog/sheet is open).

This module is pure / dependency-free so the abstraction can be unit-tested
without a device (see ``tests/test_app_state.py``).
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable

# Substrings in a node's class name that mark a modal surface — an open dialog /
# bottom sheet / popup menu is a genuinely different (testable) state.
_MODAL_HINTS = ("dialog", "popupwindow", "bottomsheet", "sheet", "alert", "popupmenu")

# Field separators for building stable string keys (unlikely to appear in ids).
_F = "␟"  # ␟


def _walk(nodes: Iterable[dict]) -> Iterable[dict]:
    """Yield every node, recursing into ``children`` if the input is still nested.

    ``normalize_ui_state`` already flattens, but we recurse defensively so raw
    element trees also work.
    """
    for n in nodes or []:
        if isinstance(n, dict):
            yield n
            children = n.get("children")
            if children:
                yield from _walk(children)


def _structural_key(node: dict) -> str:
    """What a control *is* — stable across content and theme. Drops free text."""
    rid = str(node.get("resource_id") or node.get("resourceId") or "").strip()
    cls = str(node.get("class") or node.get("className") or node.get("type") or "").strip()
    cd = str(node.get("content_description") or node.get("contentDescription") or "").strip()
    clickable = bool(node.get("clickable"))
    return f"{rid}{_F}{cls}{_F}{cd}{_F}{int(clickable)}"


def _has_meaning(key: str) -> bool:
    """Keep only nodes that carry at least one identity-bearing field."""
    rid, cls, cd, _click = key.split(_F)
    return bool(rid or cls or cd)


def abstract_state(normalized: dict) -> dict:
    """Abstract a normalized UI observation into a stable signature + metadata.

    Returns a dict with:
      signature      — hex hash; equal iff two observations are the "same state"
      package        — app package
      activity       — top activity/window
      key_set        — sorted list of structural node keys (for Jaccard re-scoring)
      has_dialog     — a modal surface is open
      element_count  — number of distinct structural controls
      label          — a short human-readable name for the state
    """
    phone = normalized.get("phone_state") or {}
    package = str(phone.get("package") or phone.get("appPackage") or "").strip()
    activity = str(phone.get("activity") or phone.get("appActivity") or "").strip()

    raw_nodes = normalized.get("nodes")
    if raw_nodes is None:
        raw_nodes = normalized.get("elements") or []

    key_set: set[str] = set()
    has_dialog = False
    # Label hints: prefer content-descriptions (stable, e.g. "Create contact")
    # over free text (volatile row data like a contact's name).
    desc_hints: list[str] = []
    text_hints: list[str] = []
    resource_ids: list[str] = []

    for n in _walk(raw_nodes):
        key = _structural_key(n)
        if _has_meaning(key):
            key_set.add(key)
            rid = str(n.get("resource_id") or n.get("resourceId") or "").strip()
            if rid:
                resource_ids.append(rid)
        cls = str(n.get("class") or n.get("className") or "").lower()
        if any(h in cls for h in _MODAL_HINTS):
            has_dialog = True
        # Label hints from any node: content-description (stable icon labels like
        # "Search"/"Create contact") preferred over free text.
        cd = str(n.get("content_description") or n.get("contentDescription") or "").strip()
        if cd:
            desc_hints.append(cd)
        text = str(n.get("text") or "").strip()
        if text:
            text_hints.append(text)

    sorted_keys = sorted(key_set)
    signature = _signature(package, activity, sorted_keys, has_dialog)
    label = _derive_label(activity, desc_hints + text_hints, resource_ids)

    return {
        "signature": signature,
        "package": package,
        "activity": activity,
        "key_set": sorted_keys,
        "has_dialog": has_dialog,
        "element_count": len(sorted_keys),
        "label": label,
    }


def _signature(package: str, activity: str, sorted_keys: list[str], has_dialog: bool) -> str:
    h = hashlib.sha1()
    h.update(package.encode("utf-8"))
    h.update(b"\x00")
    h.update(activity.encode("utf-8"))
    h.update(b"\x00")
    h.update(b"1" if has_dialog else b"0")
    h.update(b"\x00")
    for k in sorted_keys:
        h.update(k.encode("utf-8"))
        h.update(b"\x01")
    return h.hexdigest()[:16]


def control_labels(key_set: Iterable[str], limit: int = 10) -> list[str]:
    """Human-readable names of the controls actually present in a state.

    The planner otherwise writes steps against design-time (Figma) labels, which
    may not exist in the running app. These come from the real accessibility tree:
    a content-description when the app supplies one (e.g. "Create contact"),
    otherwise the resource-id tail. Clickable controls come first, since those are
    what a test step can act on. Keys carrying neither are skipped — a pure-Compose
    screen may legitimately yield none.
    """
    clickable: list[str] = []
    other: list[str] = []
    for key in key_set or []:
        parts = str(key).split(_F)
        if len(parts) < 4:
            continue
        rid, _cls, cd, click = parts[0], parts[1], parts[2], parts[3]
        label = cd.strip() or rid.split("/")[-1].strip()
        # Drop empty and id-shaped labels. The previous form
        # (`not label or _looks_like_resource_id(label) and "/" in label`) parsed
        # as `not label or (id_shaped and "/" in label)`, and `label` is already
        # the tail after split("/"), so the id check could never fire.
        if not label or _looks_like_resource_id(label):
            continue
        bucket = clickable if click == "1" else other
        if label not in bucket:
            bucket.append(label)
    return (clickable + other)[:limit]


def containment(key_set_a: list[str], key_set_b: list[str]) -> float:
    """Overlap coefficient — how completely the SMALLER control set sits inside the larger.

    Jaccard treats a mid-render capture as a different screen because the chrome
    that had not painted yet counts against it. Containment asks the question that
    actually matters for partial observations: is everything I saw also present
    there? It is 1.0 for the same screen caught at two render stages, and stays
    low for screens that genuinely differ.
    """
    a, b = set(key_set_a or []), set(key_set_b or [])
    m = min(len(a), len(b))
    if not m:
        return 0.0
    return len(a & b) / m


def similarity(key_set_a: list[str], key_set_b: list[str]) -> float:
    """Jaccard over structural keys — the near-match score (mirrors mobilerun's
    ``compare_states`` node score, on our content-abstracted keys)."""
    a, b = set(key_set_a or []), set(key_set_b or [])
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 1.0


def is_same_state(
    obs: dict,
    candidate: dict,
    threshold: float = 0.9,
) -> bool:
    """Decide whether an observation is the SAME state as a stored candidate.

    Exact signature match is the fast path; otherwise same package+activity and a
    structural Jaccard at/above ``threshold`` (tolerates minor dynamic chrome).
    """
    if obs.get("signature") and obs["signature"] == candidate.get("signature"):
        return True
    if obs.get("package") != candidate.get("package"):
        return False
    if obs.get("activity") != candidate.get("activity"):
        return False
    if bool(obs.get("has_dialog")) != bool(candidate.get("has_dialog")):
        return False
    return similarity(obs.get("key_set", []), candidate.get("key_set", [])) >= threshold


# Resource-id substrings that mark a screen-defining container (strong first),
# element-level noise to skip, and id-part words to drop when humanizing.
_STRONG_HINTS = ("fragment", "activity", "editor", "screen", "main", "home", "page", "pager")
_WEAK_HINTS = ("list", "detail", "picker", "settings", "search", "directory", "view", "container")
_NOISE_WORDS = ("divider", "separator", "spacer", "icon", "button", "label", "title",
                "header", "footer", "chip", "item", "row", "text", "image", "box", "bar")
# Wrapper ids describe layout, not the screen. "contact_list_detail_container"
# made a contact DETAIL screen read as "Contact List", colliding with the real
# list; the suffix is dropped so the content word wins.
_WRAPPER_SUFFIXES = ("_container", "_wrapper", "_layout", "_root", "_scroller",
                     "_scroll_view", "_view_group", "_anchor", "_host")
_ID_STRIP = {"fragment", "activity", "scroller", "container", "root", "layout",
             "view", "content", "id", "android", "below", "above", "gh", "host",
             "wrapper", "anchor", "group"}


def _label_from_resource_ids(resource_ids: list[str]) -> str:
    """Humanize a screen-defining resource-id, e.g. '.../contact_editor_fragment' -> 'Contact Editor'.

    Ids that merely wrap layout (``*_container``, ``*_root``) are considered only
    after real content ids, because a wrapper often names the parent concept
    rather than the screen — ``contact_list_detail_container`` on a contact detail
    screen produced the label "Contact List", which then collided with the actual
    contacts list.
    """
    def humanize(idpart: str) -> str:
        for suf in _WRAPPER_SUFFIXES:
            if idpart.endswith(suf):
                idpart = idpart[: -len(suf)]
                break
        words = [w for w in re.split(r"[_\-]", idpart) if w and w.lower() not in _ID_STRIP]
        return " ".join(w.capitalize() for w in words) if words else ""

    def candidates(wrappers: bool):
        for rid in resource_ids:
            idpart = rid.split(":id/")[-1].split("/")[-1].lower()
            if not idpart or any(n in idpart for n in _NOISE_WORDS):
                continue
            is_wrapper = any(idpart.endswith(sfx) for sfx in _WRAPPER_SUFFIXES)
            if is_wrapper == wrappers:
                yield idpart

    # Content ids first, wrappers only as a fallback; strong hints before weak.
    for wrappers in (False, True):
        for hints in (_STRONG_HINTS, _WEAK_HINTS):
            for idpart in candidates(wrappers):
                if any(h in idpart for h in hints):
                    lbl = humanize(idpart)
                    if lbl:
                        return lbl
    return ""


def _looks_like_resource_id(text: str) -> bool:
    """True for strings that are really view ids, not human labels.

    Some UI dumps report a resource-id (e.g. 'android:id/content') as a node's
    text/content-description. Such a value is identical on nearly every screen,
    so treating it as a label collapses every state to the same name.
    """
    t = (text or "").strip()
    return ":id/" in t or t.startswith("android:") or ("/" in t and " " not in t)


# Chrome that appears on nearly every screen. Naming a state after one of these
# produced an app map reading "Back #4 -> Back #2 -> Back #4" — technically
# distinct states, but the labels said nothing about WHERE the agent was, which
# is the whole point of the map.
_CHROME_WORDS = {
    "back", "cancel", "close", "navigate up", "up", "ok", "done", "next", "previous",
    "menu", "more options", "overflow", "save", "discard", "delete", "yes", "no",
    "search", "settings", "help", "share", "edit", "add", "new", "open",
}


def _is_chrome(text: str) -> bool:
    return (text or "").strip().lower() in _CHROME_WORDS


def _derive_label(activity: str, texts: list[str], resource_ids: list[str] | None = None) -> str:
    """A short, human-readable name for the state (for the graph + dashboard).

    Structure first: a screen-defining resource-id (``contact_editor_fragment``)
    identifies WHERE we are, whereas the first visible text is usually a button
    that appears on every screen. Ranking text first named states after their
    navigation chrome and made the app map unreadable, so text is now only a
    fallback and generic chrome words are rejected outright.
    """
    rid_label = _label_from_resource_ids(resource_ids or [])
    if rid_label:
        return rid_label

    hint = next(
        (t for t in texts
         if 1 < len(t) <= 24 and not _looks_like_resource_id(t) and not _is_chrome(t)),
        "",
    )
    if activity:
        short = activity.rsplit(".", 1)[-1].rsplit("/", 1)[-1]
        short = short.replace("Activity", "").replace("Fragment", "").strip()
        if short:
            return f"{short} · {hint}" if hint else short
    if hint:
        return hint
    return "Screen"


# Visual-fallback helper: when the accessibility tree is too thin (Compose without
# semantics, games, WebViews, screenshot-only mode), identity falls back to a
# perceptual hash of the screenshot. This is a plain average-hash (aHash) — no
# model required — compared by Hamming distance on the caller side.
def average_hash(image_bytes: bytes, hash_size: int = 8) -> str | None:
    try:
        import io

        from PIL import Image
    except Exception:
        return None
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("L").resize(
            (hash_size, hash_size), Image.BILINEAR
        )
        pixels = list(img.getdata())
        avg = sum(pixels) / len(pixels)
        bits = "".join("1" if p >= avg else "0" for p in pixels)
        return f"{int(bits, 2):0{hash_size * hash_size // 4}x}"
    except Exception:
        return None


def is_thin_tree(normalized: dict, min_controls: int = 3) -> bool:
    """True when the a11y tree is too sparse to trust structurally (use visual fallback)."""
    phone = normalized.get("phone_state") or {}
    if phone.get("observationMode") == "screenshot_only" or phone.get("accessibilityTree") is False:
        return True
    nodes = normalized.get("nodes")
    if nodes is None:
        nodes = normalized.get("elements") or []
    meaningful = sum(1 for n in _walk(nodes) if _has_meaning(_structural_key(n)))
    return meaningful < min_controls


def hamming(a: str, b: str) -> int:
    """Hamming distance between two equal-length hex perceptual hashes."""
    if not a or not b or len(a) != len(b):
        return 10 ** 9
    return bin(int(a, 16) ^ int(b, 16)).count("1")
