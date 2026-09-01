"""Pure JSON / similarity / query helpers (no I/O, no app-specific knowledge)."""

from __future__ import annotations

import json
import re


# Reasoning models emit their scratchpad in <think>…</think> before the answer.
# That scratchpad routinely contains braces ("I'll return {"title": ...}"), so the
# naive first-`{` scan below would parse the model's *thinking* instead of its
# answer. Strip the block first. Also handles an unterminated opening tag, which
# happens when the response is cut off mid-thought.
_THINK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"^.*?<think\b[^>]*>", re.DOTALL | re.IGNORECASE)


def strip_reasoning(raw: str) -> str:
    """Remove <think>…</think> scratchpad blocks from a model response."""
    text = raw or ""
    text = _THINK_RE.sub("", text)
    # A closing tag with no opener means the opener was trimmed upstream; drop
    # everything up to it so the answer survives.
    if "</think" in text.lower():
        idx = text.lower().rfind("</think")
        end = text.find(">", idx)
        if end != -1:
            text = text[end + 1:]
    elif "<think" in text.lower():
        text = _THINK_OPEN_RE.sub("", text)
    return text.strip()


def extract_json_text(raw: str) -> str:
    """Strip reasoning blocks + markdown fences, return the outermost {...} object."""
    text = strip_reasoning(raw)
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:] if lines and lines[0].startswith("```") else lines
        lines = lines[:-1] if lines and lines[-1].strip() == "```" else lines
        text = "\n".join(lines).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text


def parse_testcase(raw: str) -> dict:
    try:
        obj = json.loads(extract_json_text(raw))
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # Models often keep talking after the JSON object (or emit several). The
    # outermost {...} slice above then spans that trailing prose and fails to
    # decode, so take just the first complete value and ignore the rest.
    try:
        start = (raw or "").find("{")
        if start != -1:
            obj, _ = json.JSONDecoder().raw_decode(raw[start:])
            if isinstance(obj, dict):
                return obj
    except Exception:
        pass
    return {"raw": raw}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", (text or "").lower())).strip()


def jaccard(a: str, b: str) -> float:
    sa, sb = set(normalize(a).split()), set(normalize(b).split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def is_similar_to_existing(title: str, existing: list[str], threshold: float = 0.72) -> bool:
    return any(jaccard(title, t) >= threshold for t in existing)


def next_testcase_id(recent_tests: list[dict], default_prefix: str = "TC") -> str:
    max_n = 0
    for t in recent_tests or []:
        m = re.search(r"(\d+)$", str(t.get("id", "") or ""))
        if m:
            try:
                max_n = max(max_n, int(m.group(1)))
            except Exception:
                pass
    return f"{default_prefix}-{max_n + 1:03d}"


def compact_note(text: str, limit: int = 260) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    return t[:limit] + (" ..." if len(t) > limit else "")


def to_query_expr(text: str, fallback: str) -> str:
    """
    Lightweight cleanup of a retrieval query. Retrieval is semantic (vector) +
    keyword on the RAG side, so we keep the natural-language phrasing intact and
    only substitute a fallback when empty. (Kept for callers that want it.)
    """
    raw = (text or "").strip()
    if not raw or raw.lower() == "none":
        return (fallback or "").strip()
    return raw


def parse_retrieval_plan(raw: str, fallback_screens: list[str]) -> dict:
    plan = parse_testcase(raw)
    if not isinstance(plan, dict):
        return {"focus_queries": [], "target_screens": fallback_screens[:2], "reason": "fallback"}
    fq = plan.get("focus_queries", [])
    ts = plan.get("target_screens", [])
    fq = fq if isinstance(fq, list) else []
    ts = ts if isinstance(ts, list) else []
    cleaned_fq = [str(x).strip() for x in fq if str(x).strip()][:2]
    cleaned_ts = [str(x).strip() for x in ts if str(x).strip()][:2] or fallback_screens[:2]
    return {"focus_queries": cleaned_fq, "target_screens": cleaned_ts, "reason": str(plan.get("reason", "")).strip()}


def parse_action(raw: str, fallback_screens: list[str]) -> dict:
    data = parse_testcase(raw)
    if not isinstance(data, dict):
        return {"action": "retrieve", "focus_queries": [], "target_screens": fallback_screens[:2], "reason": "fallback"}
    action = str(data.get("action", "retrieve")).strip().lower()
    if action not in {"retrieve", "produce_testcase"}:
        action = "retrieve"
    fq = data.get("focus_queries", [])
    ts = data.get("target_screens", [])
    fq = fq if isinstance(fq, list) else []
    ts = ts if isinstance(ts, list) else []
    cleaned_fq = [str(x).strip() for x in fq if str(x).strip()][:2]
    cleaned_ts = [str(x).strip() for x in ts if str(x).strip()][:2] or fallback_screens[:2]
    reqs = data.get("retrieval_requests", [])
    return {
        "action": action,
        "retrieval_requests": reqs if isinstance(reqs, list) else [],
        "focus_queries": cleaned_fq,
        "target_screens": cleaned_ts,
        "reason": str(data.get("reason", "")).strip(),
    }
