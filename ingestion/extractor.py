"""
Structured requirement extraction.

Turns a free-form requirements document (numbered FR/NFR lists, user stories,
Gherkin, prose, tables, ...) into a typed entity graph payload that the graph
builder can persist directly:

    {
      "document_title": str,
      "requirements": [
        {"id","type","text","actor","action","objects","constraints",
         "acceptance","priority","feature"}
      ],
      "entities": [{"name","description"}],
      "validation_rules": [{"field","rule","requirement_id"}]
    }

The primary path asks the LLM to perform the extraction (format- and
convention-agnostic). If the model is unavailable or returns garbage, a
deterministic rule-based fallback keeps the pipeline running.
"""

from __future__ import annotations

import json
import re
from typing import Callable

# A model caller: (prompt, max_new_tokens, enable_thinking) -> {"answer": str, ...}
ModelCall = Callable[[str, int, bool], dict]


EXTRACTION_PROMPT = """You are a senior requirements analyst building a knowledge graph for QA test planning.

Read the requirements document below (it may be numbered FR/NFR, user stories, Gherkin, prose, tables, or any mix) and extract a STRUCTURED representation.

Output STRICT JSON only — no markdown fences, no prose outside the JSON object — with EXACTLY this schema:
{
  "document_title": "<inferred product/document title>",
  "requirements": [
    {
      "id": "<keep original id if present (e.g. FR-1, US-12); else assign R1, R2, ...>",
      "type": "functional" | "non_functional",
      "text": "<the requirement, one sentence, verbatim or lightly cleaned>",
      "actor": "<who performs/benefits, e.g. user, system, admin>",
      "action": "<core verb phrase, e.g. submit form>",
      "objects": ["<domain nouns acted on>"],
      "constraints": ["<rules/validations/limits this requirement implies>"],
      "acceptance": ["<observable pass conditions, if any>"],
      "priority": "high" | "medium" | "low",
      "feature": "<short feature-area slug, e.g. user_login, search, checkout>"
    }
  ],
  "entities": [
    {"name": "<domain entity, e.g. User, Order>", "description": "<one line>"}
  ],
  "validation_rules": [
    {"field": "<field name>", "rule": "<validation/constraint>", "requirement_id": "<owning requirement id>"}
  ]
}

Rules:
- Preserve original requirement IDs whenever they exist.
- Derive a concise, lowercase, snake_case "feature" slug for every requirement so related requirements cluster naturally. Do NOT invent app-specific categories that aren't supported by the text.
- Keep entities to genuine domain nouns (not UI widgets).
- Extract validation_rules for anything that constrains input/state (formats, required fields, uniqueness, limits).
- Be exhaustive on requirements but compact on wording.

DOCUMENT:
"""


def _extract_json(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text


def _slug(text: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", (text or "").lower())).strip("_") or "general"


def _coerce_list(v) -> list[str]:
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if v is None:
        return []
    s = str(v).strip()
    return [s] if s else []


def _normalize_extraction(data: dict) -> dict:
    """Defensively normalise an LLM extraction into the canonical schema."""
    if not isinstance(data, dict):
        return rule_based_extraction("")

    reqs_out: list[dict] = []
    for i, r in enumerate(data.get("requirements", []) or [], start=1):
        if not isinstance(r, dict):
            continue
        rid = str(r.get("id") or "").strip() or f"R{i}"
        rtype = str(r.get("type") or "functional").strip().lower()
        if rtype not in {"functional", "non_functional"}:
            rtype = "non_functional" if "non" in rtype or "nfr" in rtype else "functional"
        text = str(r.get("text") or "").strip()
        if not text:
            continue
        feature = _slug(r.get("feature") or r.get("action") or "general")
        priority = str(r.get("priority") or "medium").strip().lower()
        if priority not in {"high", "medium", "low"}:
            priority = "medium"
        reqs_out.append({
            "id": rid,
            "type": rtype,
            "text": text,
            "actor": str(r.get("actor") or "").strip(),
            "action": str(r.get("action") or "").strip(),
            "objects": _coerce_list(r.get("objects")),
            "constraints": _coerce_list(r.get("constraints")),
            "acceptance": _coerce_list(r.get("acceptance")),
            "priority": priority,
            "feature": feature,
        })

    entities_out: list[dict] = []
    seen_entities: set[str] = set()
    for e in data.get("entities", []) or []:
        if isinstance(e, dict):
            name = str(e.get("name") or "").strip()
            desc = str(e.get("description") or "").strip()
        else:
            name, desc = str(e).strip(), ""
        key = name.lower()
        if name and key not in seen_entities:
            seen_entities.add(key)
            entities_out.append({"name": name, "description": desc})

    rules_out: list[dict] = []
    for v in data.get("validation_rules", []) or []:
        if not isinstance(v, dict):
            continue
        rule = str(v.get("rule") or "").strip()
        if not rule:
            continue
        rules_out.append({
            "field": str(v.get("field") or "").strip(),
            "rule": rule,
            "requirement_id": str(v.get("requirement_id") or "").strip(),
        })

    return {
        "document_title": str(data.get("document_title") or "").strip() or "Requirements",
        "requirements": reqs_out,
        "entities": entities_out,
        "validation_rules": rules_out,
    }


def extract_with_model(
    text: str,
    model_call: ModelCall,
    max_new_tokens: int = 4000,
) -> dict:
    """
    Run the LLM extraction. Raises on model error so the caller can decide whether
    to fall back (mirrors the gateway's `require_model_summary` pattern).
    """
    prompt = EXTRACTION_PROMPT + (text or "").strip()
    result = model_call(prompt, max_new_tokens, False)
    raw = (result.get("answer", "") or "").strip()
    parsed = json.loads(_extract_json(raw))
    normalized = _normalize_extraction(parsed)
    if not normalized["requirements"]:
        raise ValueError("LLM extraction returned zero requirements")
    return normalized


# ── Deterministic fallback (no model required) ──────────────────────────────────

_FR_RE = re.compile(r"^\s*(FR[-_ ]?\d+|REQ[-_ ]?\d+|US[-_ ]?\d+)\b[:.)-]?\s*(.*)", re.IGNORECASE)
_NFR_RE = re.compile(r"^\s*(NFR[-_ ]?\d+)\b[:.)-]?\s*(.*)", re.IGNORECASE)
_VALIDATION_KW = ("validate", "must", "shall", "required", "format", "constraint", "unique", "limit")


def _guess_feature(text: str) -> str:
    t = text.lower()
    # Pick the most informative verb+noun bigram as a generic feature slug.
    verbs = ("create", "add", "edit", "update", "delete", "remove", "search", "view",
             "display", "import", "export", "sync", "synchronize", "merge", "share",
             "save", "sort", "filter", "backup", "restore", "call", "message", "login")
    for v in verbs:
        if v in t:
            m = re.search(rf"{v}\s+(?:a\s+|an\s+|the\s+)?([a-z]+)", t)
            noun = m.group(1) if m else ""
            return _slug(f"{v}_{noun}") if noun else _slug(v)
    return "general"


def rule_based_extraction(text: str) -> dict:
    """
    Convention-agnostic-ish fallback used when the model is unavailable.
    Recognises FR/NFR/REQ/US prefixes; otherwise treats each non-empty line/sentence
    as a candidate requirement. Never raises.
    """
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    title = lines[0] if lines else "Requirements"

    requirements: list[dict] = []
    validation_rules: list[dict] = []
    auto_n = 0

    def add_req(rid: str | None, body: str, rtype: str):
        nonlocal auto_n
        body = body.strip()
        if not body:
            return
        if not rid:
            auto_n += 1
            rid = f"R{auto_n}"
        feature = _guess_feature(body)
        constraints = [body] if any(k in body.lower() for k in _VALIDATION_KW) else []
        requirements.append({
            "id": rid.upper().replace(" ", "-"),
            "type": rtype,
            "text": body,
            "actor": "user" if "user" in body.lower() else "system",
            "action": "",
            "objects": [],
            "constraints": constraints,
            "acceptance": [],
            "priority": "medium",
            "feature": feature,
        })
        if any(k in body.lower() for k in _VALIDATION_KW):
            validation_rules.append({"field": "", "rule": body, "requirement_id": rid.upper().replace(" ", "-")})

    matched_any = False
    for ln in lines[1:] if len(lines) > 1 else lines:
        m_nfr = _NFR_RE.match(ln)
        m_fr = _FR_RE.match(ln)
        if m_nfr:
            matched_any = True
            add_req(m_nfr.group(1), m_nfr.group(2), "non_functional")
        elif m_fr:
            matched_any = True
            add_req(m_fr.group(1), m_fr.group(2), "functional")

    if not matched_any:
        # No recognised tagging: split into sentence-ish units.
        units = [u.strip() for u in re.split(r"\n\s*\n|(?<=[.!?])\s+", text or "") if u.strip()]
        for u in units:
            if len(u) > 12:
                add_req(None, u, "functional")

    return {
        "document_title": title,
        "requirements": requirements,
        "entities": [],
        "validation_rules": validation_rules,
    }


def extract(
    text: str,
    model_call: ModelCall | None = None,
    require_model: bool = False,
    max_new_tokens: int = 4000,
) -> tuple[dict, str]:
    """
    Convenience wrapper. Returns (extraction, source) where source is
    "model" or "fallback". If require_model is True, model failures propagate.
    """
    if model_call is not None:
        try:
            return extract_with_model(text, model_call, max_new_tokens), "model"
        except Exception:
            if require_model:
                raise
    return rule_based_extraction(text), "fallback"
