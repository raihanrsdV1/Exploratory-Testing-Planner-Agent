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
    max_new_tokens: int = 8092,
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


# ── Multi-pass extraction + self-critique (WP5 / Business-Logic Intelligence) ────

CRITIQUE_PROMPT = """You are a QA lead reviewing a requirements extraction against its SOURCE document.
Judge how well the extraction captured the source's business rules. Output STRICT JSON only:
{
  "confidence": {"<requirement_id>": <0.0-1.0 how well this requirement is captured>, ...},
  "missing": [ {"id","type","text","feature","priority"} , ...],   // rules present in SOURCE but absent from the extraction
  "open_questions": ["<ambiguities, TODOs, or under-specified rules found in the SOURCE>"]
}
Rules:
- Flag anything the SOURCE explicitly defers (e.g. "TODO", "to be defined", "after research") as an open_question, not silence.
- Only list a requirement in "missing" if it is a genuine, testable rule the extraction omitted.
- confidence keys must be requirement ids that exist in the extraction.

SOURCE DOCUMENT:
{source}

EXTRACTION UNDER REVIEW:
{extraction}
"""

# Markers that flag a deferred / ambiguous rule the SRS itself has not settled.
_OPEN_MARKERS = ("todo", "tbd", "to be defined", "to be determined", "after research",
                 "future", "not yet", "pending", "will be defined", "placeholder")


def _chunk_sections(text: str, max_chars: int = 1800) -> list[str]:
    """Split an SRS into section-ish chunks for per-section extraction.

    Prefers numbered headings (3.2.1.1 / FR-n / ALL-CAPS headings); falls back to
    blank-line paragraphs, then a hard character window. App-agnostic."""
    text = (text or "").strip()
    if not text:
        return []
    lines = text.splitlines()
    heading = re.compile(r"^\s*(\d+(\.\d+)+|(FR|NFR|REQ|US)[-_ ]?\d+|[A-Z][A-Z /-]{6,})\b")
    sections: list[list[str]] = []
    current: list[str] = []
    for ln in lines:
        if heading.match(ln) and current and sum(len(x) for x in current) > 200:
            sections.append(current)
            current = [ln]
        else:
            current.append(ln)
    if current:
        sections.append(current)

    # Pack sections up to max_chars so we don't over-call the model on tiny bits.
    chunks: list[str] = []
    buf = ""
    for sec in sections:
        block = "\n".join(sec).strip()
        if not block:
            continue
        if buf and len(buf) + len(block) > max_chars:
            chunks.append(buf)
            buf = block
        else:
            buf = f"{buf}\n{block}".strip() if buf else block
    if buf:
        chunks.append(buf)
    return chunks or [text]


def _merge_extractions(parts: list[dict]) -> dict:
    """Merge per-section extractions: dedup requirements by id, entities by name, rules."""
    title = ""
    reqs: dict[str, dict] = {}
    order: list[str] = []
    entities: dict[str, dict] = {}
    rules: dict[tuple, dict] = {}
    for p in parts:
        title = title or p.get("document_title", "")
        for r in p.get("requirements", []):
            rid = r["id"]
            # If the same id recurs with different text, keep the longer (fuller) text.
            if rid not in reqs:
                reqs[rid] = r
                order.append(rid)
            elif len(r.get("text", "")) > len(reqs[rid].get("text", "")):
                reqs[rid] = {**reqs[rid], **r}
        for e in p.get("entities", []):
            entities.setdefault(e["name"].lower(), e)
        for v in p.get("validation_rules", []):
            key = (v.get("requirement_id", ""), _slug(v.get("rule", ""))[:60])
            rules.setdefault(key, v)
    return {
        "document_title": title or "Requirements",
        "requirements": [reqs[i] for i in order],
        "entities": list(entities.values()),
        "validation_rules": list(rules.values()),
    }


def _attach_confidence(extraction: dict, overrides: dict | None, default: float) -> dict:
    """Stamp confidence + needs_review on each requirement and its rules."""
    overrides = overrides or {}
    for r in extraction.get("requirements", []):
        conf = overrides.get(r["id"])
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = default
        conf = max(0.0, min(1.0, conf))
        # Deferred/ambiguous rules are inherently low-confidence.
        if any(m in r.get("text", "").lower() for m in _OPEN_MARKERS):
            conf = min(conf, 0.4)
        r["confidence"] = round(conf, 2)
        r["needs_review"] = conf < 0.6
    conf_by_req = {r["id"]: r["confidence"] for r in extraction.get("requirements", [])}
    for v in extraction.get("validation_rules", []):
        c = conf_by_req.get(v.get("requirement_id", ""), default)
        v["confidence"] = round(c, 2)
        v["needs_review"] = c < 0.6
    return extraction


def _scan_open_questions(text: str) -> list[str]:
    """Surface deferred/ambiguous lines the SRS itself hasn't settled."""
    out: list[str] = []
    for ln in (text or "").splitlines():
        low = ln.lower()
        if any(m in low for m in _OPEN_MARKERS) and len(ln.strip()) > 12:
            out.append(ln.strip()[:200])
    return out[:20]


def extract_multipass(text: str, model_call: ModelCall, max_new_tokens: int = 4000) -> dict:
    """Per-section extraction → synthesis merge → self-critique (the 'several agent calls').

    Removes truncation/shallowness on large SRS and yields per-rule confidence,
    provenance-ready structure, and surfaced open questions. Raises if no
    requirements survive so the caller can fall back."""
    chunks = _chunk_sections(text)
    parts: list[dict] = []
    for ch in chunks:
        try:
            result = model_call(EXTRACTION_PROMPT + ch, max_new_tokens, False)
            parsed = json.loads(_extract_json(result.get("answer", "") or ""))
            parts.append(_normalize_extraction(parsed))
        except Exception:
            continue  # skip a bad section; other sections still contribute
    if not any(p.get("requirements") for p in parts):
        raise ValueError("multi-pass extraction produced no requirements")

    merged = _merge_extractions(parts)

    # Self-critique pass: model reviews the merge against the source.
    overrides, open_questions = {}, _scan_open_questions(text)
    try:
        critique_prompt = CRITIQUE_PROMPT.replace("{source}", text[:6000]).replace(
            "{extraction}", json.dumps(merged, ensure_ascii=False)[:6000]
        )
        crit = json.loads(_extract_json(model_call(critique_prompt, 2000, False).get("answer", "") or ""))
        overrides = crit.get("confidence", {}) or {}
        for m in (crit.get("missing", []) or []):
            if isinstance(m, dict) and str(m.get("text", "")).strip():
                merged["requirements"].append(_normalize_extraction({"requirements": [m]})["requirements"][0])
        for q in (crit.get("open_questions", []) or []):
            if str(q).strip() and str(q).strip() not in open_questions:
                open_questions.append(str(q).strip())
    except Exception:
        pass  # critique is best-effort; extraction still stands

    merged = _attach_confidence(merged, overrides, default=0.75)
    merged["open_questions"] = open_questions
    merged["extraction_passes"] = len(parts)
    return merged


def extract(
    text: str,
    model_call: ModelCall | None = None,
    require_model: bool = False,
    max_new_tokens: int = 4000,
    multipass: bool = True,
) -> tuple[dict, str]:
    """
    Convenience wrapper. Returns (extraction, source) where source is
    "multipass" | "model" | "fallback". If require_model is True, model failures
    propagate. Multi-pass (per-section + self-critique) is used by default when a
    model is available; set multipass=False for the legacy single-call path.
    """
    if model_call is not None:
        if multipass:
            try:
                return extract_multipass(text, model_call, max_new_tokens), "multipass"
            except Exception:
                if require_model:
                    raise
        try:
            return extract_with_model(text, model_call, max_new_tokens), "model"
        except Exception:
            if require_model:
                raise
    # Rule-based fallback: attach conservative confidence so downstream stays uniform.
    fb = rule_based_extraction(text)
    fb = _attach_confidence(fb, {}, default=0.5)
    fb["open_questions"] = _scan_open_questions(text)
    return fb, "fallback"
