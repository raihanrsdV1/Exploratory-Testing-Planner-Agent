"""Deterministic state identity with explicit URL normalization policy."""
import hashlib
import json
import re
from urllib.parse import parse_qsl, urlencode, urlsplit

NAVIGATION_KEYS = ("tab", "view", "mode", "section", "step", "lang", "page", "route")
_SECRET = re.compile(r"token|secret|password|authorization|session|signature|api.?key|credential", re.I)
_ID = re.compile(r"^(?:\d+|[0-9a-f]{8}-[0-9a-f-]{27,}|[0-9a-f]{24,})$", re.I)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()[:32]


def route(url, navigation_keys=NAVIGATION_KEYS):
    """Drop secrets/tracking; template IDs; preserve known navigation query values.

    Unknown query values are placeholders, not concrete record IDs or user input.
    Opaque path slugs cannot reliably be inferred and are retained.
    """
    parsed = urlsplit(str(url))
    if parsed.scheme and parsed.scheme not in ("http", "https"):
        return "[non-http]"
    origin = (parsed.scheme.lower() + "://" + (parsed.hostname or "").lower()
              + (":" + str(parsed.port) if parsed.port else "")) if parsed.netloc else ""
    path = "/".join(":id" if _ID.fullmatch(part) else part for part in parsed.path.split("/")) or "/"
    query = sorted((key, value if key.lower() in navigation_keys else ":value")
                   for key, value in parse_qsl(parsed.query, keep_blank_values=True)
                   if not _SECRET.search(key) and not key.lower().startswith("utm_")
                   and key.lower() not in ("gclid", "fbclid"))
    fragment = ""
    if parsed.fragment.startswith("/"):
        fragment = "#" + route(parsed.fragment, navigation_keys)
    elif parsed.fragment:
        fragment = "#:anchor"
    return origin + path + ("?" + urlencode(query) if query else "") + fragment


def label(value):
    text = str(value or "")
    # URLs can appear in accessible names. Do not preserve their query values.
    text = re.sub(r"https?://[^\s]+", "[url]", text)
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "[email]", text)
    return re.sub(r"\b\d+\b", "#", " ".join(text.split()))[:120]


def control(element, navigation_keys=NAVIGATION_KEYS):
    name = element.get("name")
    if element.get("role") in ("textbox", "password", "searchbox", "combobox") and name == element.get("value"):
        name = "[unlabelled input]"
    result = {"role": element.get("role", "control"), "name": label(name)}
    if element.get("href"):
        result["href"] = route(element["href"], navigation_keys)
    for key in ("disabled", "required", "checked", "selected", "expanded", "invalid"):
        if key in element:
            result[key] = element[key]
    return result


def state(snapshot, navigation_keys=NAVIGATION_KEYS):
    controls = [control(e, navigation_keys) for e in snapshot.get("elements", [])]
    # Deduplicate repeated cards; counts/ref renumbering must not create new screens.
    structure = sorted({json.dumps(c, sort_keys=True) for c in controls})
    result = {
        "route": route(snapshot.get("url", ""), navigation_keys),
        "dialog_open": bool(snapshot.get("dialog_open")),
        "dialog_name": label(snapshot.get("dialog_name")),
        "headings": sorted({label(h) for h in snapshot.get("headings", [])}),
        "controls": [json.loads(c) for c in structure],
    }
    return digest(result), result


def observation_signature(snapshot):
    """Ephemeral only: detect changed values/messages without persisting them."""
    content = {k: v for k, v in snapshot.items() if k not in ("url", "title", "elements")}
    content["elements"] = [{k: v for k, v in e.items() if k != "ref"}
                           for e in snapshot.get("elements", [])]
    return digest(content)


def action_descriptor(action, snapshot, navigation_keys=NAVIGATION_KEYS):
    name = action.get("action", "unknown")
    result = {"action": name}
    elements = snapshot.get("elements", [])
    target = next((e for e in elements if e.get("ref") == action.get("ref")), None)
    if target is not None:
        # A hint, not a guaranteed unique replay selector. Never persist e<N> refs.
        result["target"] = control(target, navigation_keys)
        same = [e for e in elements if control(e, navigation_keys) == result["target"]]
        result["ordinal"] = same.index(target)
    if name == "goto":
        result["url"] = route(action.get("url", ""), navigation_keys)
    if name == "scroll" and action.get("direction") in ("up", "down", "top", "bottom"):
        result["direction"] = action["direction"]
    if name == "press" and action.get("key") in ("Enter", "Tab", "Escape", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "Backspace", "Delete", "Home", "End"):
        result["key"] = action["key"]
    # No typed text, selected values, reasoning, error messages, or key payloads.
    return result
