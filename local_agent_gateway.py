import os
import json
import re
from pathlib import Path

import requests
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

# Local services (on your device)
RAG_API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:9010").rstrip("/")
RAG_API_KEY = os.getenv("RAG_API_KEY", "")

# Model API — Gemini local server by default, or set to ngrok URL for Kaggle notebook
MODEL_API_URL = os.getenv("MODEL_API_URL", "https://d7bf-34-90-11-71.ngrok-free.app").rstrip("/")

GATEWAY_API_KEY = os.getenv("GATEWAY_API_KEY", "")

app = FastAPI(title="Local Agent Gateway")


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    project: str = "default"
    top_k: int = 3
    max_new_tokens: int = 1024
    enable_thinking: bool = False


class NextTestCaseRequest(BaseModel):
    project: str = Field(..., min_length=1)
    app_name: str = "contacts app"
    objective: str = "generate the next best test case"
    top_k: int = 5
    max_new_tokens: int = 2048
    enable_thinking: bool = False
    debug_trace: bool = False


class LogVerdictRequest(BaseModel):
    project: str = Field(..., min_length=1)
    app_name: str = "contacts app"
    test_case_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    verdict: str = Field(..., pattern="^(pass|failed)$")
    notes: str = ""
    area: str = "general"
    testcase_payload: dict | None = None
    top_k: int = 5
    max_new_tokens: int = 2048
    enable_thinking: bool = False
    debug_trace: bool = False


class IngestSRSRequest(BaseModel):
    project: str = Field(..., min_length=1)
    source_path: str = Field(..., min_length=1)
    srs_text: str | None = None
    chunk_chars: int = 1200
    use_model_summary: bool = True
    require_model_summary: bool = True


def _summarize_srs_with_model(srs_text: str, max_new_tokens: int = 700) -> str:
    """Create a planner-friendly SRS summary from full source text using the model backend."""
    prompt = (
        "You are an expert software requirements analyst.\n"
        "Read the full SRS below and produce a concise but complete summary for QA test planning.\n\n"
        "Output STRICT plain text (no markdown, no code fences) with this structure:\n"
        "Document: <name or inferred title>\n"
        "Functional requirements summary:\n"
        "- ...\n"
        "- ...\n"
        "Non-functional requirements summary:\n"
        "- ...\n"
        "- ...\n"
        "Validation and constraints:\n"
        "- ...\n"
        "Coverage priorities for next tests:\n"
        "- ...\n\n"
        "Rules:\n"
        "- Preserve critical requirement IDs when available (FR-#, NFR-#).\n"
        "- Keep it compact but include all high-impact behaviors and validations.\n"
        "- If non-functional requirements are not explicitly listed, state that clearly.\n\n"
        "SRS:\n"
        f"{(srs_text or '').strip()}"
    )
    resp = requests.post(
        f"{MODEL_API_URL}/generate",
        json={"prompt": prompt, "max_new_tokens": max_new_tokens, "enable_thinking": False},
        timeout=180,
    )
    resp.raise_for_status()
    model_data = resp.json()
    summary = (model_data.get("answer", "") or "").strip()
    return _extract_json_text(summary) if summary.startswith("{") else summary


class IngestFigmaRequest(BaseModel):
    project: str = Field(..., min_length=1)
    source_path: str = Field(..., min_length=1)
    figma_json: str | None = None


class ResetProjectRequest(BaseModel):
    project: str = Field(..., min_length=1)
    delete_tests: bool = True
    delete_srs: bool = False
    delete_figma: bool = False


# ── Auth & helpers ────────────────────────────────────────────────────────────

def _check_gateway_auth(authorization: str | None):
    if not GATEWAY_API_KEY:
        return
    if authorization != f"Bearer {GATEWAY_API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")


def _rag_headers() -> dict:
    return {"Authorization": f"Bearer {RAG_API_KEY}"} if RAG_API_KEY else {}


def _rag_get(path: str, params: dict | None = None) -> dict:
    try:
        resp = requests.get(
            f"{RAG_API_URL}{path}", params=params,
            headers=_rag_headers(), timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"RAG backend unavailable: {e}")


def _rag_post(path: str, body: dict) -> dict:
    try:
        resp = requests.post(
            f"{RAG_API_URL}{path}", json=body,
            headers=_rag_headers(), timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"RAG backend unavailable: {e}")


def _call_model(prompt: str, max_new_tokens: int, enable_thinking: bool) -> dict:
    try:
        resp = requests.post(
            f"{MODEL_API_URL}/generate",
            json={"prompt": prompt, "max_new_tokens": max_new_tokens, "enable_thinking": enable_thinking},
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Model backend unavailable: {e}")


# ── Knowledge graph queries ───────────────────────────────────────────────────

def _get_srs_and_history(project: str, query: str, top_k: int) -> dict:
    """Fetch SRS chunks + test history from the knowledge graph."""
    return _rag_post("/retrieve", {"project": project, "query": query, "top_k": top_k, "include_history": False})


def _get_figma_screens(project: str) -> list[dict]:
    """List all Figma screens (name, purpose, element counts) — lightweight index."""
    data = _rag_get("/figma/screens", {"project": project})
    return data.get("screens", [])


def _get_screen_elements(project: str, screen_name: str) -> dict[str, list[str]]:
    """Get interactive elements for one screen, grouped by kind. Compact for prompt."""
    data = _rag_get("/figma/elements", {"project": project, "screen_name": screen_name, "interactive_only": "true"})
    return data.get("elements", {})


def _get_figma_overview(project: str) -> list[dict]:
    data = _rag_get("/figma/overview", {"project": project, "top_labels": 4})
    return data.get("screens", [])


def _get_figma_transitions(project: str, screen_name: str | None = None) -> list[dict]:
    params = {"project": project, "limit": 80}
    if screen_name:
        params["screen_name"] = screen_name
    data = _rag_get("/figma/transitions", params)
    return data.get("transitions", [])


def _pick_relevant_screens(screens: list[dict], done_areas: list[str], recent_tests: list[dict]) -> list[str]:
    """
    Choose which Figma screens to pull element detail for.
    Logic: find screens not well-covered by recent tests, bias toward uncovered areas.
    Returns up to 2 screen names.
    """
    if not screens:
        return []

    # Build set of purposes already tested
    tested_areas = set(
        str(t.get("area", "")).lower().replace(" ", "_") for t in recent_tests
    )
    tested_areas.update(a.lower().replace(" ", "_") for a in done_areas)

    # Prefer screens whose purpose hasn't been tested yet
    untested = [s for s in screens if s.get("purpose", "other") not in tested_areas]
    chosen = untested if untested else screens

    # Sort by interactive_count desc — more interactive = more testable
    chosen = sorted(chosen, key=lambda s: s.get("interactive_count", 0), reverse=True)
    return [s["screen_name"] for s in chosen[:2]]


def _build_figma_context(project: str, screen_names: list[str]) -> str:
    """
    Fetch elements for chosen screens and format as compact context block.
    Example output:
      [Screen: Create Contact (No Icons)]
        buttons: Save, Add phone, Email, Birthday
        inputs: First name, Surname, Company, Add phone, Notes
    """
    if not screen_names:
        return ""

    lines = []
    for name in screen_names:
        elements = _get_screen_elements(project, name)
        if not elements:
            continue
        lines.append(f"[Screen: {name}]")
        for kind, labels in elements.items():
            lines.append(f"  {kind}s: {', '.join(labels[:10])}")
    return "\n".join(lines)


def _build_figma_overview_context(figma_overview: list[dict]) -> str:
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


def _build_figma_overview_generalized(figma_overview: list[dict]) -> str:
    """Generalized UI context for planning (less label-heavy, less bias)."""
    if not figma_overview:
        return "No screens available"
    lines = [f"Total screens: {len(figma_overview)}", "Screens by purpose:"]
    for s in figma_overview:
        lines.append(
            f"- {s.get('screen_name','?')} (purpose={s.get('purpose','other')}, interactive={s.get('interactive_count',0)})"
        )
    return "\n".join(lines)


def _recent_tests_exact(recent_tests: list[dict], limit: int = 50) -> str:
    if not recent_tests:
        return "none"
    items = []
    for t in recent_tests[:limit]:
        items.append(
            f"{t.get('id','?')}|{t.get('verdict','?')}|{t.get('area','general')}|{t.get('title','')[:120]}"
        )
    return "; ".join(items)


def _build_figma_flow_context(transitions: list[dict], top_n: int = 12) -> str:
    if not transitions:
        return ""
    lines = ["Known UI transitions (inferred from labels):"]
    for t in transitions[:top_n]:
        lines.append(
            f"- {t.get('from_screen','?')} --[{t.get('via_element','?')}]--> {t.get('to_screen','?')}"
        )
    return "\n".join(lines)


def _screen_index_compact(screen_index: list[dict], limit: int = 7) -> str:
    if not screen_index:
        return "[]"
    ordered = sorted(
        screen_index,
        key=lambda s: (
            0 if str(s.get("purpose", "")).strip().lower() == "create_contact" else 1,
            -int(s.get("interactive_count", 0) or 0),
            str(s.get("screen_name", "")).lower(),
        ),
    )
    parts = []
    for s in ordered[:limit]:
        parts.append(
            f"{s.get('screen_name','?')}|purpose={s.get('purpose','other')}|interactive={s.get('interactive_count',0)}"
        )
    return "[" + "; ".join(parts) + "]"


def _compact_note(text: str, limit: int = 260) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    return t[:limit] + (" ..." if len(t) > limit else "")


def _get_brief_context(project: str) -> dict:
    return _rag_post("/context/brief", {"project": project, "recent_limit": 100})


def _planner_prompt_for_retrieval(brief: dict, app_name: str, objective: str) -> str:
    return (
        f"You are planning retrieval for QA testcase generation in {app_name}.\n"
        "Given compact project context, output retrieval plan JSON only.\n\n"
        "Context summary:\n"
        f"SRS summary:\n{brief.get('srs_summary','')}\n\n"
        f"Figma summary:\n{brief.get('figma_summary','')}\n\n"
        f"Screen index: {brief.get('screen_index', [])}\n\n"
        f"Recent tests: {brief.get('recent_tests', [])}\n\n"
        f"Objective: {objective}\n\n"
        "Return STRICT JSON:\n"
        "{\n"
        "  \"focus_queries\": [\"...\", \"...\"],\n"
        "  \"target_screens\": [\"...\", \"...\"],\n"
        "  \"reason\": \"short reason\"\n"
        "}\n"
        "Constraints: max 2 focus_queries, max 2 target_screens, keep concise."
    )


def _planner_prompt_for_action(
    brief: dict,
    app_name: str,
    objective: str,
    retrieval_round: int,
    max_rounds: int,
    collected_queries: list[str],
    collected_screens: list[str],
    context_chars: int,
    figma_overview: list[dict],
    retrieved_notes: list[str],
) -> str:
    retrieved_notes_str = "\n".join(f"- {n}" for n in retrieved_notes[-6:]) if retrieved_notes else "- none yet"
    srs_summary = brief.get("srs_summary", "") if isinstance(brief, dict) else ""
    srs_full = str(srs_summary).strip() if srs_summary else "(none)"

    figma_summary = brief.get("figma_summary", "") if isinstance(brief, dict) else ""
    figma_full = str(figma_summary).strip() if figma_summary else "(none)"

    figma_overview_general = _build_figma_overview_generalized(figma_overview)
    recent_tests = brief.get("recent_tests", []) if isinstance(brief, dict) else []
    recent_tests_exact = _recent_tests_exact(recent_tests)
    uncovered_frs = brief.get("uncovered_fr_ids", []) if isinstance(brief, dict) else []
    coverage = brief.get("coverage", {}) if isinstance(brief, dict) else {}
    uncovered_frs_text = ", ".join(uncovered_frs[:20]) if uncovered_frs else "none"

    if retrieval_round <= 1:
        global_context_block = (
            "Full global context (round 1):\n"
            f"SRS summary:\n{srs_full}\n\n"
            f"Figma summary:\n{figma_full}\n\n"
            f"Screen index (compact): {_screen_index_compact(brief.get('screen_index', []) if isinstance(brief, dict) else [])}\n"
            f"Figma UI overview (generalized):\n{figma_overview_general}\n"
            f"Recent tests (exact): {recent_tests_exact}\n\n"
            f"FR coverage: total={coverage.get('total_frs', 0)}, covered={coverage.get('covered_frs', 0)}, uncovered={coverage.get('uncovered_frs', 0)}, percent={coverage.get('percent', 0)}\n"
            f"Uncovered FR IDs (prioritize): {uncovered_frs_text}\n\n"
        )
    else:
        global_context_block = (
            "Global memo (do not re-ask same broad context):\n"
            f"Screen index (compact): {_screen_index_compact(brief.get('screen_index', []) if isinstance(brief, dict) else [])}\n"
            f"Recent tests (exact): {recent_tests_exact}\n"
            f"Uncovered FR IDs (prioritize): {uncovered_frs_text}\n"
            "Use Retrieved context so far to refine, not restart.\n\n"
        )

    response_link_line = (
        "This prompt is a response to your previous retrieval request and includes the requested DB context.\n\n"
        if retrieval_round > 1
        else ""
    )

    return (
        f"You are retrieval planner for QA testcase generation in {app_name}.\n"
        "You are interacting with a knowledge database (SRS + Figma KG + test history).\n"
        "Decide your NEXT ACTION only.\n\n"
        f"Objective: {objective}\n"
        f"Retrieval round: {retrieval_round}/{max_rounds}\n"
        f"Collected queries so far: {collected_queries}\n"
        f"Collected screens so far: {collected_screens}\n"
        f"Collected context size (chars): {context_chars}\n\n"
        f"{response_link_line}"
        "Retrieved context so far (continuation from earlier requests):\n"
        f"{retrieved_notes_str}\n\n"
        f"{global_context_block}"
        "Return STRICT JSON only with this schema:\n"
        "{\n"
        "  \"action\": \"retrieve\" | \"produce_testcase\",\n"
        "  \"retrieval_requests\": [{\"source\":\"srs|figma_ui|figma_flow\", \"query\":\"...\", \"screen\":\"optional\"}],\n"
        "  \"focus_queries\": [\"...\", \"...\"],\n"
        "  \"target_screens\": [\"...\", \"...\"],\n"
        "  \"reason\": \"short reason\"\n"
        "}\n"
        "Rules:\n"
        "- If more context is needed, set action=retrieve and provide explicit retrieval_requests (max 3).\n"
        "- Use source=srs for business rules/validation logic.\n"
        "- Use source=figma_ui for screen elements and control availability.\n"
        "- Use source=figma_flow for navigation/button-to-screen behavior.\n"
        "- For source=srs, query must be keyword expression (not question), e.g. \"validation AND (email OR phone)\".\n"
        "- You are replying to the prior retrieval results; avoid asking for the exact same request unless refinement is needed.\n"
        "- If context is sufficient, set action=produce_testcase.\n"
        "- Never output markdown or text outside JSON."
    )


def _to_query_expr(text: str, fallback: str) -> str:
    raw = (text or "").strip()
    if not raw or raw.lower() == "none":
        raw = fallback
    if "?" not in raw and (" and " in raw.lower() or " or " in raw.lower()):
        return raw

    toks = re.findall(r"[a-zA-Z0-9_]+", raw.lower())
    stop = {"what", "how", "when", "where", "why", "is", "are", "the", "a", "an", "for", "to", "of", "in", "on"}
    keep = []
    for t in toks:
        if len(t) < 3 or t in stop:
            continue
        if t not in keep:
            keep.append(t)
    if not keep:
        return fallback
    return " AND ".join(keep[:8])


def _parse_retrieval_plan(raw: str, fallback_screens: list[str]) -> dict:
    plan = _parse_testcase(raw)
    if not isinstance(plan, dict):
        return {"focus_queries": [], "target_screens": fallback_screens[:2], "reason": "fallback"}

    fq = plan.get("focus_queries", [])
    ts = plan.get("target_screens", [])
    if not isinstance(fq, list):
        fq = []
    if not isinstance(ts, list):
        ts = []

    cleaned_fq = [str(x).strip() for x in fq if str(x).strip()][:2]
    cleaned_ts = [str(x).strip() for x in ts if str(x).strip()][:2]
    if not cleaned_ts:
        cleaned_ts = fallback_screens[:2]

    return {
        "focus_queries": cleaned_fq,
        "target_screens": cleaned_ts,
        "reason": str(plan.get("reason", "")).strip(),
    }


def _parse_action(raw: str, fallback_screens: list[str]) -> dict:
    data = _parse_testcase(raw)
    if not isinstance(data, dict):
        return {
            "action": "retrieve",
            "focus_queries": [],
            "target_screens": fallback_screens[:2],
            "reason": "fallback",
        }

    action = str(data.get("action", "retrieve")).strip().lower()
    if action not in {"retrieve", "produce_testcase"}:
        action = "retrieve"

    fq = data.get("focus_queries", [])
    ts = data.get("target_screens", [])
    if not isinstance(fq, list):
        fq = []
    if not isinstance(ts, list):
        ts = []

    cleaned_fq = [str(x).strip() for x in fq if str(x).strip()][:2]
    cleaned_ts = [str(x).strip() for x in ts if str(x).strip()][:2]
    if not cleaned_ts:
        cleaned_ts = fallback_screens[:2]

    return {
        "action": action,
        "retrieval_requests": data.get("retrieval_requests", []) if isinstance(data.get("retrieval_requests", []), list) else [],
        "focus_queries": cleaned_fq,
        "target_screens": cleaned_ts,
        "reason": str(data.get("reason", "")).strip(),
    }


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(
    app_name: str,
    objective: str,
    srs_context: str,
    figma_overview_context: str,
    figma_context: str,
    figma_flow_context: str,
    done_titles: list[str],
    failed_titles: list[str],
    uncovered_frs: list[str],
    coverage_summary: dict,
) -> str:
    parts = [
        f"You are a senior QA test designer for {app_name}.",
        f"Task: {objective}.",
        "Propose exactly ONE test case not already in the executed list.",
        "",
    ]

    if srs_context:
        parts += ["## SRS Requirements (relevant excerpt)", srs_context, ""]

    if figma_overview_context:
        parts += ["## Figma UI overview (all screens, compact)", figma_overview_context, ""]

    if figma_context:
        parts += [
            "## Figma UI — interactive elements on relevant screens",
            "(Use exact button labels / input names in your steps)",
            figma_context,
            "",
        ]

    if figma_flow_context:
        parts += ["## Figma flow hints (button/navigation transitions)", figma_flow_context, ""]

    if coverage_summary or uncovered_frs:
        parts += [
            "## SRS FR coverage status",
            f"Coverage: total={coverage_summary.get('total_frs', 0)}, covered={coverage_summary.get('covered_frs', 0)}, uncovered={coverage_summary.get('uncovered_frs', 0)}, percent={coverage_summary.get('percent', 0)}",
            "Prioritize uncovered FR IDs first:",
            ", ".join(uncovered_frs[:25]) if uncovered_frs else "none",
            "",
        ]

    history_block = "\n".join(f"- {t}" for t in done_titles[:30]) or "- none"
    failed_block = "\n".join(f"- {t}" for t in failed_titles[:20]) or "- none"

    parts += [
        "## Already executed tests (avoid semantic duplicates)",
        history_block,
        "",
        "## Failed tests (prioritise adjacent coverage)",
        failed_block,
        "",
        "## Decision policy",
        "1. Do NOT propose a test semantically similar to executed titles.",
        "2. Prefer tests adjacent to failed behaviours or that close a coverage gap.",
        "2b. Prefer uncovered FR IDs when available.",
        "3. Vary the area — avoid repeating the same feature in consecutive rounds.",
        "4. Validate either business logic (SRS), UI behavior (Figma UI), or navigation flow (Figma flow).",
        "",
        "## Output",
        "Return STRICT JSON only — no markdown, no commentary.",
        "Keys: test_case_id, title, screen, preconditions (array), steps (array),",
        "      expected_result, priority (high/medium/low), area, rationale",
        "Steps must reference actual UI element names from the Figma context above.",
    ]

    return "\n".join(parts)


# ── JSON utils ────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", (text or "").lower())).strip()


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(_normalize(a).split()), set(_normalize(b).split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _is_similar_to_existing(title: str, existing: list[str], threshold: float = 0.72) -> bool:
    return any(_jaccard(title, t) >= threshold for t in existing)


def _next_testcase_id(recent_tests: list[dict], default_prefix: str = "TC") -> str:
    max_n = 0
    for t in recent_tests or []:
        rid = str(t.get("id", "") or "")
        m = re.search(r"(\d+)$", rid)
        if m:
            try:
                max_n = max(max_n, int(m.group(1)))
            except Exception:
                pass
    return f"{default_prefix}-{max_n + 1:03d}"


def _extract_json_text(raw: str) -> str:
    text = (raw or "").strip()
    # Strip markdown fences
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:] if lines[0].startswith("```") else lines
        lines = lines[:-1] if lines and lines[-1].strip() == "```" else lines
        text = "\n".join(lines).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text


def _parse_testcase(raw: str) -> dict:
    try:
        obj = json.loads(_extract_json_text(raw))
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {"raw": raw}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "rag_api": RAG_API_URL, "model_api": MODEL_API_URL}


@app.post("/srs/ingest")
def ingest_srs(req: IngestSRSRequest, authorization: str | None = Header(default=None)):
    _check_gateway_auth(authorization)

    srs_text = req.srs_text
    if not srs_text:
        src = Path(req.source_path)
        if not src.exists():
            raise HTTPException(status_code=404, detail=f"SRS file not found: {req.source_path}")
        srs_text = src.read_text(encoding="utf-8", errors="ignore")

    srs_summary = ""
    summary_source = "fallback"
    if req.use_model_summary and (srs_text or "").strip():
        try:
            srs_summary = _summarize_srs_with_model(srs_text)
            if srs_summary:
                summary_source = "model"
        except Exception as e:
            if req.require_model_summary:
                raise HTTPException(status_code=503, detail=f"SRS summarization failed: {e}")
            srs_summary = ""

    payload = {
        "project": req.project,
        "source_path": req.source_path,
        "srs_text": srs_text,
        "chunk_chars": req.chunk_chars,
        "srs_summary": srs_summary or None,
    }
    out = _rag_post("/ingest/srs", payload)
    out["srs_summary_source"] = summary_source
    out["srs_summary_chars"] = len(srs_summary or "")
    return out


@app.post("/figma/ingest")
def ingest_figma(req: IngestFigmaRequest, authorization: str | None = Header(default=None)):
    _check_gateway_auth(authorization)
    resp = requests.post(
        f"{RAG_API_URL}/ingest/figma",
        json=req.model_dump(),
        headers=_rag_headers(),
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


@app.post("/project/reset")
def reset_project(req: ResetProjectRequest, authorization: str | None = Header(default=None)):
    _check_gateway_auth(authorization)
    return _rag_post("/project/reset", req.model_dump())


@app.post("/agent/next-testcase")
def next_testcase(req: NextTestCaseRequest, authorization: str | None = Header(default=None)):
    _check_gateway_auth(authorization)

    # Stage 1: get compact global context (summaries + recent tests + screen index)
    brief = _get_brief_context(req.project)
    recent_tests = brief.get("recent_tests", []) if isinstance(brief, dict) else []
    done_titles = [str(t.get("title", "")).strip() for t in recent_tests if t.get("title")]
    failed_titles = [
        str(t.get("title", "")).strip()
        for t in recent_tests
        if t.get("title") and str(t.get("verdict", "")).lower() == "failed"
    ]
    done_areas = [str(t.get("area", "")) for t in recent_tests if t.get("area")]
    figma_screens = brief.get("screen_index", []) if isinstance(brief, dict) else []
    uncovered_frs = brief.get("uncovered_fr_ids", []) if isinstance(brief, dict) else []
    coverage_summary = brief.get("coverage", {}) if isinstance(brief, dict) else {}
    figma_overview = _get_figma_overview(req.project)
    fallback_screens = _pick_relevant_screens(figma_screens, done_areas, recent_tests)

    # Stage 2: iterative retrieval planning loop
    max_retrieval_rounds = 3
    planner_trace: list[dict] = []
    collected_queries: list[str] = []
    selected_screens: list[str] = []
    srs_context_blocks: list[str] = []
    figma_ui_blocks: list[str] = []
    flow_context_blocks: list[str] = []
    last_round_retrieved_notes: list[str] = []
    agent_signaled_ready = False
    finalization_mode = "max_retrieval_rounds_fallback"

    debug_trace: dict = {
        "planner_rounds": [],
        "retrieved_blocks": [],
    }

    for round_no in range(1, max_retrieval_rounds + 1):
        action_prompt = _planner_prompt_for_action(
            brief=brief,
            app_name=req.app_name,
            objective=req.objective,
            retrieval_round=round_no,
            max_rounds=max_retrieval_rounds,
            collected_queries=collected_queries,
            collected_screens=selected_screens,
            context_chars=len("\n\n".join(srs_context_blocks)),
            figma_overview=figma_overview,
            retrieved_notes=last_round_retrieved_notes,
        )
        action_model = _call_model(action_prompt, max(320, min(req.max_new_tokens, 700)), False)
        action = _parse_action(action_model.get("answer", ""), fallback_screens)

        if req.debug_trace:
            debug_trace["planner_rounds"].append(
                {
                    "round": round_no,
                    "prompt": action_prompt,
                    "model_answer_raw": action_model.get("answer", ""),
                    "parsed_action": action,
                }
            )

        round_trace = {
            "round": round_no,
            "action": action.get("action", "retrieve"),
            "retrieval_requests": action.get("retrieval_requests", []),
            "focus_queries": action.get("focus_queries", []),
            "target_screens": action.get("target_screens", []),
            "reason": action.get("reason", ""),
        }

        if action["action"] == "produce_testcase":
            planner_trace.append(round_trace)
            agent_signaled_ready = True
            finalization_mode = "agent_signaled_sufficient_context"
            break

        requests_spec = action.get("retrieval_requests", [])
        if not requests_spec:
            # backward-compatible fallback
            requests_spec = [{"source": "srs", "query": q} for q in (action.get("focus_queries", []) or [req.objective])[:2]]
            for s in (action.get("target_screens", []) or fallback_screens[:2])[:2]:
                requests_spec.append({"source": "figma_ui", "screen": s})

        round_retrieved_notes: list[str] = []

        for rr in requests_spec[:3]:
            source = str(rr.get("source", "srs")).strip().lower()
            query_raw = rr.get("query")
            query = str(query_raw).strip() if query_raw is not None else ""
            screen = str(rr.get("screen", "")).strip()

            if source == "srs":
                q = _to_query_expr(query, req.objective)
                if q not in collected_queries:
                    collected_queries.append(q)
                data = _get_srs_and_history(req.project, q, top_k=min(req.top_k, 2))
                block = data.get("context", "")
                if block:
                    srs_context_blocks.append(block)
                    round_retrieved_notes.append(f"srs | query={q} | {_compact_note(block)}")
                    if req.debug_trace:
                        debug_trace["retrieved_blocks"].append({"round": round_no, "source": source, "query": q, "context": block})

            elif source == "figma_ui":
                s = screen or (action.get("target_screens", []) or fallback_screens[:1])[0] if (action.get("target_screens", []) or fallback_screens) else ""
                if s and s not in selected_screens:
                    selected_screens.append(s)
                if s:
                    elements = _get_screen_elements(req.project, s)
                    ui_lines = [f"[Screen: {s}]"]
                    for kind, labels in elements.items():
                        ui_lines.append(f"  {kind}s: {', '.join(labels[:10])}")
                    ui_block = "\n".join(ui_lines)
                    if ui_block.strip() != f"[Screen: {s}]":
                        figma_ui_blocks.append(ui_block)
                        note = f"figma_ui | screen={s} | {_compact_note(ui_block)}"
                        round_retrieved_notes.append(note)
                        if req.debug_trace:
                            debug_trace["retrieved_blocks"].append({"round": round_no, "source": source, "screen": s, "context": ui_block})

            elif source == "figma_flow":
                trans = _get_figma_transitions(req.project, screen_name=screen if screen else None)
                if trans:
                    flow_block = _build_figma_flow_context(trans, top_n=10)
                    if flow_block:
                        flow_context_blocks.append(flow_block)
                        round_retrieved_notes.append(f"figma_flow | screen={screen or '*'} | {_compact_note(flow_block)}")
                        if req.debug_trace:
                            debug_trace["retrieved_blocks"].append({"round": round_no, "source": source, "screen": screen, "context": flow_block})

        round_trace["retrieved_context_chars"] = len("\n\n".join(srs_context_blocks))
        planner_trace.append(round_trace)
        last_round_retrieved_notes = round_retrieved_notes

        if round_no > 1 and not round_retrieved_notes:
            finalization_mode = "no_new_context_early_finalize"
            break

        # safety: avoid overloading prompt size
        if len("\n\n".join(srs_context_blocks)) > 9000:
            break

    # Fallback: ensure at least one retrieval happened
    if not srs_context_blocks:
        fallback_query = req.objective
        data = _get_srs_and_history(req.project, fallback_query, top_k=min(req.top_k, 3))
        block = data.get("context", "")
        if block:
            srs_context_blocks.append(block)
            last_round_retrieved_notes = [f"srs | query={fallback_query} | {_compact_note(block)}"]
        if fallback_screens:
            selected_screens = fallback_screens[:2]

    srs_context = "\n\n".join(dict.fromkeys(srs_context_blocks))[:8000]
    figma_overview_context = _build_figma_overview_context(figma_overview)
    figma_context = "\n\n".join(dict.fromkeys(figma_ui_blocks))[:2400] if figma_ui_blocks else _build_figma_context(req.project, selected_screens[:3])
    figma_flow_context = "\n\n".join(dict.fromkeys(flow_context_blocks))[:2500]

    retrieval_plan = {
        "focus_queries": collected_queries[:2],
        "target_screens": selected_screens[:2],
        "reason": planner_trace[-1].get("reason", "") if planner_trace else "fallback",
    }

    # Stage 4: final testcase generation using targeted context
    prompt = _build_prompt(
        app_name=req.app_name,
        objective=req.objective,
        srs_context=srs_context,
        figma_overview_context=figma_overview_context,
        figma_context=figma_context,
        figma_flow_context=figma_flow_context,
        done_titles=done_titles,
        failed_titles=failed_titles,
        uncovered_frs=uncovered_frs,
        coverage_summary=coverage_summary,
    )

    # 4. Call model
    model_data = _call_model(prompt, req.max_new_tokens, req.enable_thinking)
    raw_answer = model_data.get("answer", "")
    parsed = _parse_testcase(raw_answer)

    if req.debug_trace:
        debug_trace["final_generation"] = {
            "prompt": prompt,
            "model_answer_raw": raw_answer,
            "model_thinking": model_data.get("thinking", ""),
        }

    # 5. Duplicate check — retry with different focus if too similar
    candidate_title = str(parsed.get("title", "")) if isinstance(parsed, dict) else ""
    blocked_titles = list(dict.fromkeys((done_titles or []) + (failed_titles or [])))
    if candidate_title and _is_similar_to_existing(candidate_title, blocked_titles, threshold=0.60):
        # Rotate to next uncovered screens
        already_picked = set(selected_screens)
        alt_screens = [s["screen_name"] for s in figma_screens if s["screen_name"] not in already_picked][:2]
        alt_figma_context = _build_figma_context(req.project, alt_screens) if alt_screens else figma_context

        blocked = "\n".join(f"- {t}" for t in blocked_titles[:20]) or "- none"

        retry_prompt = _build_prompt(
            app_name=req.app_name,
            objective=req.objective + " (choose a DISTINCT testcase, not semantically similar to blocked titles)",
            srs_context=srs_context,
            figma_overview_context=figma_overview_context,
            figma_context=alt_figma_context,
            figma_flow_context=figma_flow_context,
            done_titles=done_titles,
            failed_titles=failed_titles,
            uncovered_frs=uncovered_frs,
            coverage_summary=coverage_summary,
        ) + "\n\nBlocked titles (must avoid semantic overlap):\n" + blocked
        model_data = _call_model(retry_prompt, req.max_new_tokens, req.enable_thinking)
        raw_answer = model_data.get("answer", "")
        parsed = _parse_testcase(raw_answer)
        if req.debug_trace:
            debug_trace["final_retry"] = {
                "prompt": retry_prompt,
                "model_answer_raw": raw_answer,
                "model_thinking": model_data.get("thinking", ""),
            }

    # 6. Enforce stable unique external testcase ID for downstream logging.
    if isinstance(parsed, dict):
        parsed["test_case_id"] = _next_testcase_id(recent_tests)
        raw_answer = json.dumps(parsed, ensure_ascii=False, indent=2)

    out = {
        "project": req.project,
        "retrieval_plan": retrieval_plan,
        "planner_trace": planner_trace,
        "finalization_mode": finalization_mode,
        "agent_signaled_ready": agent_signaled_ready,
        "retrieved_context_stats": {
            "retrieval_rounds_executed": len(planner_trace),
            "queries_used": collected_queries[:2],
            "screens_used": selected_screens[:3],
            "srs_context_chars": len(srs_context),
            "figma_overview_chars": len(figma_overview_context),
            "figma_context_chars": len(figma_context),
            "figma_flow_chars": len(figma_flow_context),
        },
        "target_screens": selected_screens[:3],
        "next_testcase_json": raw_answer,
        "next_testcase": parsed,
        "recent_tests_count": len(recent_tests),
        "failed_tests_count": len(failed_titles),
        "coverage": coverage_summary,
        "uncovered_fr_ids": uncovered_frs[:30],
        "thinking": model_data.get("thinking", ""),
    }

    if req.debug_trace:
        out["debug_trace"] = debug_trace

    return out


@app.post("/agent/log-verdict-and-next")
def log_verdict_and_next(req: LogVerdictRequest, authorization: str | None = Header(default=None)):
    _check_gateway_auth(authorization)

    log_data = _rag_post("/tests/log", {
        "project": req.project,
        "test_case_id": req.test_case_id,
        "title": req.title,
        "verdict": req.verdict,
        "notes": req.notes,
        "area": req.area,
        "testcase_payload": req.testcase_payload,
    })

    next_req = NextTestCaseRequest(
        project=req.project,
        app_name=req.app_name,
        objective="generate the next best test case after latest verdict",
        top_k=req.top_k,
        max_new_tokens=req.max_new_tokens,
        enable_thinking=req.enable_thinking,
        debug_trace=req.debug_trace,
    )
    next_data = next_testcase(next_req, authorization=authorization)

    return {"log": log_data, "next": next_data}


@app.post("/chat")
def chat(req: ChatRequest, authorization: str | None = Header(default=None)):
    _check_gateway_auth(authorization)

    srs_context = ""
    try:
        rag_data = _get_srs_and_history(req.project, req.prompt, req.top_k)
        srs_context = rag_data.get("context", "")
    except Exception:
        pass

    prompt = f"Context:\n{srs_context}\n\nQuestion:\n{req.prompt}" if srs_context else req.prompt
    model_data = _call_model(prompt, req.max_new_tokens, req.enable_thinking)

    return {
        "prompt": req.prompt,
        "context": srs_context,
        "answer": model_data.get("answer", ""),
        "thinking": model_data.get("thinking", ""),
    }
