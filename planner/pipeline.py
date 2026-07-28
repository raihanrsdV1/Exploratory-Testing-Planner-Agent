"""
Orchestration for the exploratory-testing planner.

Pure business logic — no FastAPI routing. Each entry point takes a parsed request
model (+ the raw Authorization header) and returns a plain dict. The thin router
in `gateway/main.py` maps HTTP routes onto these functions.
"""

from __future__ import annotations

import json
from pathlib import Path

import requests
from fastapi import HTTPException

from observability import get_logger
from ingestion import document_loader, extractor, ui_normalizer

log = get_logger("pipeline")

from . import (
    config,
    context_builders,
    coverage,
    model_client,
    prompts,
    rag_client,
    textutil,
)
from .schemas import (
    ChatRequest,
    IngestFigmaRequest,
    IngestSRSRequest,
    LogVerdictRequest,
    NextTestCaseRequest,
    ResetProjectRequest,
)


# ── Ingestion ───────────────────────────────────────────────────────────────────

def ingest_srs(req: IngestSRSRequest, authorization: str | None) -> dict:
    config.check_gateway_auth(authorization)

    if any(part == ".." for part in Path(req.source_path).parts):
        raise HTTPException(status_code=400, detail="Path traversal not allowed in source_path")

    srs_text = req.srs_text
    if srs_text and len(srs_text) > config.MAX_SRS_CHARS:
        raise HTTPException(status_code=413, detail=f"srs_text exceeds {config.MAX_SRS_CHARS:,} character limit")

    # Format-agnostic load: pdf/docx/html/md/txt/... all normalised to text here.
    doc_format = "inline"
    doc_loader = "inline"
    if not srs_text:
        try:
            doc = document_loader.load_document(source_path=req.source_path)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"SRS file not found: {req.source_path}")
        except RuntimeError as e:
            raise HTTPException(status_code=415, detail=str(e))
        srs_text = doc["text"]
        doc_format, doc_loader = doc["format"], doc["loader"]
        if len(srs_text) > config.MAX_SRS_CHARS:
            raise HTTPException(status_code=413, detail=f"SRS document exceeds {config.MAX_SRS_CHARS:,} character limit")

    srs_summary = ""
    summary_source = "fallback"
    if req.use_model_summary and (srs_text or "").strip():
        try:
            srs_summary = prompts.summarize_srs_with_model(srs_text)
            if srs_summary:
                summary_source = "model"
        except Exception as e:
            if req.require_model_summary:
                raise HTTPException(status_code=503, detail=f"SRS summarization failed: {e}")
            srs_summary = ""

    # Structured entity extraction -> requirement knowledge graph.
    extraction = None
    extraction_source = "skipped"
    if req.extract_entities and (srs_text or "").strip():
        model_call = model_client.call_model if req.use_model_summary else None
        extraction, extraction_source = extractor.extract(srs_text, model_call=model_call, require_model=False)

    out = rag_client.rag_post("/ingest/srs", {
        "project": req.project,
        "source_path": req.source_path,
        "srs_text": srs_text,
        "chunk_chars": req.chunk_chars,
        "srs_summary": srs_summary or None,
        "extraction": extraction,
    })
    out["srs_summary_source"] = summary_source
    out["srs_summary_chars"] = len(srs_summary or "")
    out["extraction_source"] = extraction_source
    out["document_format"] = doc_format
    out["document_loader"] = doc_loader
    if extraction:
        out["requirements_extracted"] = len(extraction.get("requirements", []))
    
    log.info(
        "srs_ingested",
        project=req.project,
        summary_source=summary_source,
        extraction_source=extraction_source,
        doc_format=doc_format,
    )
    return out


def ingest_figma(req: IngestFigmaRequest, authorization: str | None) -> dict:
    config.check_gateway_auth(authorization)
    if any(part == ".." for part in Path(req.source_path).parts):
        raise HTTPException(status_code=400, detail="Path traversal not allowed in source_path")

    raw = req.figma_json
    if not raw:
        src = Path(req.source_path)
        if not src.exists():
            raise HTTPException(status_code=404, detail=f"Figma JSON not found: {req.source_path}")
        raw = src.read_text(encoding="utf-8", errors="ignore")

    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        lines = lines[1:] if lines and lines[0].startswith("```") else lines
        lines = lines[:-1] if lines and lines[-1].strip() == "```" else lines
        raw = "\n".join(lines).strip()

    try:
        figma_data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid Figma JSON: {e}")

    ui_ir = ui_normalizer.normalize_figma(figma_data)
    if not ui_ir.get("screens"):
        raise HTTPException(status_code=400, detail="No screens found in Figma JSON")

    # Dynamic, app-agnostic purpose classification (LLM) — skipped at ingest time
    # because it calls the model for each screen and causes timeouts on large JSON files.
    # The name-slug fallback from ui_normalizer is fast and sufficient.
    classification_source = "name_slug"
    if req.use_model_classification:
        try:
            hints = prompts.classify_screen_purposes(ui_ir, req.project)
            if hints:
                classification_source = "model"
                for s in ui_ir["screens"]:
                    s["purpose"] = ui_normalizer.derive_purpose(s["screen_name"], hints)
        except Exception:
            classification_source = "name_slug"  # fall back silently on any LLM error

    resp = requests.post(
        f"{config.RAG_API_URL}/ingest/figma",
        json={"project": req.project, "source_path": req.source_path, "ui_ir": ui_ir},
        headers=rag_client.rag_headers(),
        timeout=600,
    )
    resp.raise_for_status()
    out = resp.json()
    out["classification_source"] = classification_source
    log.info("figma_ingested", project=req.project, screens=len(ui_ir.get("screens", [])))
    return out


def reset_project(req: ResetProjectRequest, authorization: str | None) -> dict:
    config.check_gateway_auth(authorization)
    return rag_client.rag_post("/project/reset", req.model_dump())


def ingest_defects(req, authorization: str | None) -> dict:
    """Proxy defect-history ingestion to the RAG API (ETA-REQ-301.1)."""
    config.check_gateway_auth(authorization)
    if req.source_path and any(part == ".." for part in Path(req.source_path).parts):
        raise HTTPException(status_code=400, detail="Path traversal not allowed in source_path")
    out = rag_client.rag_post("/ingest/defects", req.model_dump())
    log.info("defects_ingested", project=req.project, defects=out.get("defects_written", 0))
    return out


def session_start(req, authorization: str | None) -> dict:
    config.check_gateway_auth(authorization)
    return rag_client.rag_post("/session/start", req.model_dump())


def session_context(project: str, authorization: str | None) -> dict:
    config.check_gateway_auth(authorization)
    return rag_client.rag_get("/session/context", {"project": project})


def session_end(req, authorization: str | None) -> dict:
    config.check_gateway_auth(authorization)
    return rag_client.rag_post("/session/end", req.model_dump())


# ── Core: next test case ─────────────────────────────────────────────────────────

def generate_next_testcase(req: NextTestCaseRequest, authorization: str | None) -> dict:
    config.check_gateway_auth(authorization)
    from .langgraph_agent import run_agent
    return run_agent(req.model_dump())


# ── Verdict logging + adaptive loop ──────────────────────────────────────────────

def log_verdict(req: LogVerdictRequest, authorization: str | None) -> dict:
    config.check_gateway_auth(authorization)
    rag_verdict = req.verdict if req.verdict in {"pass", "failed"} else "failed"
    rag_notes = req.notes
    if req.verdict in {"blocked", "skipped"} and req.notes:
        rag_notes = f"[{req.verdict.upper()}] {req.notes}"
    elif req.verdict in {"blocked", "skipped"}:
        rag_notes = f"[{req.verdict.upper()}] Test was {req.verdict}."

    log.info(
        "verdict_logged",
        project=req.project,
        test_case_id=req.test_case_id,
        verdict=rag_verdict,
        area=req.area,
    )

    return rag_client.rag_post("/tests/log", {
        "project": req.project,
        "test_case_id": req.test_case_id,
        "title": req.title,
        "verdict": rag_verdict,
        "notes": rag_notes,
        "area": req.area,
        "requirement_ids": req.requirement_ids,
        "profile": req.profile,
        "platform": req.platform,
        "application": req.application,
    })


def log_verdict_and_next(req: LogVerdictRequest, authorization: str | None) -> dict:
    config.check_gateway_auth(authorization)
    log_data = log_verdict(req, authorization)

    if req.next_objective.strip():
        next_objective = req.next_objective.strip()
    elif req.verdict == "failed":
        next_objective = (
            f"generate the next best exploratory test case targeting the area '{req.area}' "
            "where the last test failed — map out adjacent edge cases"
        )
    elif req.verdict in {"blocked", "skipped"}:
        next_objective = (
            f"generate an alternative exploratory test case for the area '{req.area}' "
            "that avoids the blocking condition"
        )
    else:
        next_objective = "generate the next best exploratory test case to broaden overall coverage"

    next_req = NextTestCaseRequest(
        project=req.project,
        app_name=req.app_name,
        objective=next_objective,
        top_k=req.top_k,
        max_new_tokens=req.max_new_tokens,
        enable_thinking=req.enable_thinking,
        debug_trace=req.debug_trace,
        profile=req.profile,
        platform=req.platform,
        application=req.application,
    )
    next_data = generate_next_testcase(next_req, authorization)
    return {"log": log_data, "next": next_data}


def agent_coverage(project: str, authorization: str | None) -> dict:
    config.check_gateway_auth(authorization)
    brief = rag_client.get_brief_context(project)
    recent_tests = brief.get("recent_tests", []) if isinstance(brief, dict) else []
    figma_screens = brief.get("screen_index", []) if isinstance(brief, dict) else []
    figma_overview = rag_client.get_figma_overview(project)

    coverage_map = coverage.compute_coverage_map(recent_tests, figma_screens)
    directive = coverage.build_exploration_directive(coverage_map, recent_tests)

    try:
        requirement_coverage = rag_client.rag_get("/coverage/requirements", {"project": project})
    except Exception:
        requirement_coverage = {}

    return {
        "project": project,
        "summary": {
            "total_tests": coverage_map["total_tests"],
            "coverage_pct": coverage_map["coverage_pct"],
            "areas_tested": coverage_map["total_areas_tested"],
            "areas_available": coverage_map["total_areas_available"],
        },
        "area_breakdown": coverage_map["area_stats"],
        "uncovered_areas": coverage_map["uncovered_purposes"],
        "hot_spots": coverage_map["hot_spots"],
        "exhausted_areas": coverage_map["exhausted_areas"],
        "exploration_directive": directive,
        "figma_screen_count": len(figma_overview),
        "requirement_coverage": requirement_coverage,
        "recent_tests": recent_tests[:20],
    }


def dashboard_data(project: str) -> dict:
    """Aggregate everything the operator dashboard renders (read-only, best-effort).

    Combines knowledge-graph stats (requirements / validation rules / entities /
    screens / tests) with the live coverage map, recent tests, and the active
    model backend into a single payload the dashboard polls.
    """
    stats: dict = {}
    try:
        stats = rag_client.rag_get("/graph/stats", {"project": project})
    except Exception:
        stats = {}

    coverage_data: dict = {}
    try:
        coverage_data = agent_coverage(project, None)
    except Exception:
        coverage_data = {}

    appmodel: dict = {}
    try:
        appmodel = rag_client.rag_get("/appmodel/graph", {"project": project})
    except Exception:
        appmodel = {}

    executions: list = []
    try:
        executions = rag_client.rag_get("/execution/logs", {"project": project, "limit": 20}).get("logs", [])
    except Exception:
        executions = []

    # WP7 regression risk + WP2 defect-prone areas (best-effort; drive dashboard panels).
    risk_scores: list = []
    try:
        risk_scores = rag_client.rag_get("/risk/scores", {"project": project}).get("risk_scores", [])
    except Exception:
        risk_scores = []

    defects: dict = {}
    try:
        defects = rag_client.rag_get("/defects/summary", {"project": project})
    except Exception:
        defects = {}

    return {
        "project": project,
        "model": model_client.backend_info(),
        "stats": stats,
        "coverage": coverage_data,
        "appmodel": appmodel,
        "executions": executions,
        "risk_scores": risk_scores,
        "defects": defects,
    }


def chat(req: ChatRequest, authorization: str | None) -> dict:
    config.check_gateway_auth(authorization)
    srs_context = ""
    try:
        srs_context = rag_client.get_srs_and_history(req.project, req.prompt, req.top_k).get("context", "")
    except Exception:
        pass
    prompt = f"Context:\n{srs_context}\n\nQuestion:\n{req.prompt}" if srs_context else req.prompt
    model_data = model_client.call_model(prompt, req.max_new_tokens, req.enable_thinking)
    return {
        "prompt": req.prompt,
        "context": srs_context,
        "answer": model_data.get("answer", ""),
        "thinking": model_data.get("thinking", ""),
    }
