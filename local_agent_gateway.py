import os
import json
import re

import requests
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

# Local services (on your device)
RAG_API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:9010").rstrip("/")
RAG_API_KEY = os.getenv("RAG_API_KEY", "")

# Model API — Gemini local server by default, or set to ngrok URL for Kaggle notebook
MODEL_API_URL = os.getenv("MODEL_API_URL", "http://127.0.0.1:8000").rstrip("/")

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


class LogVerdictRequest(BaseModel):
    project: str = Field(..., min_length=1)
    app_name: str = "contacts app"
    test_case_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    verdict: str = Field(..., pattern="^(pass|failed)$")
    notes: str = ""
    area: str = "general"
    top_k: int = 5
    max_new_tokens: int = 2048
    enable_thinking: bool = False


class IngestSRSRequest(BaseModel):
    project: str = Field(..., min_length=1)
    source_path: str = Field(..., min_length=1)
    srs_text: str | None = None
    chunk_chars: int = 1200


class IngestFigmaRequest(BaseModel):
    project: str = Field(..., min_length=1)
    source_path: str = Field(..., min_length=1)
    figma_json: str | None = None


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
    return _rag_post("/retrieve", {"project": project, "query": query, "top_k": top_k})


def _get_figma_screens(project: str) -> list[dict]:
    """List all Figma screens (name, purpose, element counts) — lightweight index."""
    data = _rag_get("/figma/screens", {"project": project})
    return data.get("screens", [])


def _get_screen_elements(project: str, screen_name: str) -> dict[str, list[str]]:
    """Get interactive elements for one screen, grouped by kind. Compact for prompt."""
    data = _rag_get("/figma/elements", {"project": project, "screen_name": screen_name, "interactive_only": "true"})
    return data.get("elements", {})


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
            lines.append(f"  {kind}s: {', '.join(labels[:12])}")
    return "\n".join(lines)


def _get_brief_context(project: str) -> dict:
    return _rag_post("/context/brief", {"project": project, "recent_limit": 15})


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


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(
    app_name: str,
    objective: str,
    srs_context: str,
    figma_context: str,
    done_titles: list[str],
    failed_titles: list[str],
) -> str:
    parts = [
        f"You are a senior QA test designer for {app_name}.",
        f"Task: {objective}.",
        "Propose exactly ONE test case not already in the executed list.",
        "",
    ]

    if srs_context:
        parts += ["## SRS Requirements (relevant excerpt)", srs_context, ""]

    if figma_context:
        parts += [
            "## Figma UI — interactive elements on relevant screens",
            "(Use exact button labels / input names in your steps)",
            figma_context,
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
        "3. Vary the area — avoid repeating the same feature in consecutive rounds.",
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
    return _rag_post("/ingest/srs", req.model_dump())


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
    fallback_screens = _pick_relevant_screens(figma_screens, done_areas, recent_tests)

    # Stage 2: ask model to produce retrieval plan (queries + target screens)
    plan_prompt = _planner_prompt_for_retrieval(brief, req.app_name, req.objective)
    plan_model = _call_model(plan_prompt, max(400, min(req.max_new_tokens, 900)), False)
    plan = _parse_retrieval_plan(plan_model.get("answer", ""), fallback_screens)

    # Stage 3: retrieve only targeted slices
    focus_queries = plan.get("focus_queries", []) or [req.objective]
    srs_context_blocks = []
    for q in focus_queries:
        data = _get_srs_and_history(req.project, q, top_k=min(req.top_k, 3))
        block = data.get("context", "")
        if block:
            srs_context_blocks.append(block)
    srs_context = "\n\n".join(dict.fromkeys(srs_context_blocks))[:7000]

    target_screens = plan.get("target_screens", [])
    figma_context = _build_figma_context(req.project, target_screens)

    # Stage 4: final testcase generation using targeted context
    prompt = _build_prompt(
        app_name=req.app_name,
        objective=req.objective,
        srs_context=srs_context,
        figma_context=figma_context,
        done_titles=done_titles,
        failed_titles=failed_titles,
    )

    # 4. Call model
    model_data = _call_model(prompt, req.max_new_tokens, req.enable_thinking)
    raw_answer = model_data.get("answer", "")
    parsed = _parse_testcase(raw_answer)

    # 5. Duplicate check — retry once with different screen if too similar
    candidate_title = str(parsed.get("title", "")) if isinstance(parsed, dict) else ""
    if candidate_title and _is_similar_to_existing(candidate_title, done_titles):
        # Rotate to next uncovered screens
        already_picked = set(target_screens)
        alt_screens = [s["screen_name"] for s in figma_screens if s["screen_name"] not in already_picked][:2]
        alt_figma_context = _build_figma_context(req.project, alt_screens) if alt_screens else figma_context

        retry_prompt = _build_prompt(
            app_name=req.app_name,
            objective=req.objective + " (choose a DIFFERENT feature area)",
            srs_context=srs_context,
            figma_context=alt_figma_context,
            done_titles=done_titles,
            failed_titles=failed_titles,
        )
        model_data = _call_model(retry_prompt, req.max_new_tokens, req.enable_thinking)
        raw_answer = model_data.get("answer", "")
        parsed = _parse_testcase(raw_answer)

    return {
        "project": req.project,
        "retrieval_plan": plan,
        "target_screens": target_screens,
        "next_testcase_json": raw_answer,
        "next_testcase": parsed,
        "recent_tests_count": len(recent_tests),
        "failed_tests_count": len(failed_titles),
        "thinking": model_data.get("thinking", ""),
    }


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
    })

    next_req = NextTestCaseRequest(
        project=req.project,
        app_name=req.app_name,
        objective="generate the next best test case after latest verdict",
        top_k=req.top_k,
        max_new_tokens=req.max_new_tokens,
        enable_thinking=req.enable_thinking,
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
