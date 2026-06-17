"""
Orchestration for the exploratory-testing planner.

Pure business logic — no FastAPI routing. Each entry point takes a parsed request
model (+ the raw Authorization header) and returns a plain dict. The thin router
in `local_agent_gateway.py` maps HTTP routes onto these functions.
"""

from __future__ import annotations

import json
from pathlib import Path

import requests
from fastapi import HTTPException

from ingestion import document_loader, extractor, ui_normalizer

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

    # Dynamic, app-agnostic purpose classification (LLM) -> re-derive screen purposes.
    classification_source = "name_slug"
    if req.use_model_classification:
        hints = prompts.classify_screen_purposes(ui_ir, req.project)
        if hints:
            classification_source = "model"
            for s in ui_ir["screens"]:
                s["purpose"] = ui_normalizer.derive_purpose(s["screen_name"], hints)

    resp = requests.post(
        f"{config.RAG_API_URL}/ingest/figma",
        json={"project": req.project, "source_path": req.source_path, "ui_ir": ui_ir},
        headers=rag_client.rag_headers(),
        timeout=120,
    )
    resp.raise_for_status()
    out = resp.json()
    out["classification_source"] = classification_source
    return out


def reset_project(req: ResetProjectRequest, authorization: str | None) -> dict:
    config.check_gateway_auth(authorization)
    return rag_client.rag_post("/project/reset", req.model_dump())


# ── Core: next test case ─────────────────────────────────────────────────────────

def generate_next_testcase(req: NextTestCaseRequest, authorization: str | None) -> dict:
    config.check_gateway_auth(authorization)

    # Stage 1: compact global context (summaries + recent tests + screen index).
    brief = rag_client.get_brief_context(req.project)
    recent_tests = brief.get("recent_tests", []) if isinstance(brief, dict) else []
    done_titles = [str(t.get("title", "")).strip() for t in recent_tests if t.get("title")]
    failed_titles = [
        str(t.get("title", "")).strip()
        for t in recent_tests
        if t.get("title") and str(t.get("verdict", "")).lower() == "failed"
    ]
    done_areas = [str(t.get("area", "")) for t in recent_tests if t.get("area")]
    figma_screens = brief.get("screen_index", []) if isinstance(brief, dict) else []
    figma_overview = rag_client.get_figma_overview(req.project)
    fallback_screens = context_builders.pick_relevant_screens(figma_screens, done_areas, recent_tests)

    coverage_map = coverage.compute_coverage_map(recent_tests, figma_screens)

    # Stage 2: iterative retrieval planning loop.
    max_retrieval_rounds = max(1, min(req.max_retrieval_rounds, 6))
    planner_trace: list[dict] = []
    collected_queries: list[str] = []
    selected_screens: list[str] = []
    srs_context_blocks: list[str] = []
    figma_ui_blocks: list[str] = []
    flow_context_blocks: list[str] = []
    last_round_retrieved_notes: list[str] = []
    agent_signaled_ready = False
    finalization_mode = "max_retrieval_rounds_fallback"
    debug_trace: dict = {"planner_rounds": [], "retrieved_blocks": []}

    for round_no in range(1, max_retrieval_rounds + 1):
        action_prompt = prompts.planner_prompt_for_action(
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
            coverage_map=coverage_map,
        )
        action_model = model_client.call_model(action_prompt, max(320, min(req.max_new_tokens, 700)), False)
        action = textutil.parse_action(action_model.get("answer", ""), fallback_screens)

        if req.debug_trace:
            debug_trace["planner_rounds"].append({
                "round": round_no,
                "prompt": action_prompt,
                "model_answer_raw": action_model.get("answer", ""),
                "parsed_action": action,
            })

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
                # Natural-language query -> semantic (vector) + keyword hybrid retrieval.
                q = query or req.objective
                if q not in collected_queries:
                    collected_queries.append(q)
                data = rag_client.get_srs_and_history(req.project, q, top_k=min(req.top_k, 2))
                block = data.get("context", "")
                if block:
                    srs_context_blocks.append(block)
                    round_retrieved_notes.append(f"srs | query={q} | {textutil.compact_note(block)}")
                    if req.debug_trace:
                        debug_trace["retrieved_blocks"].append({"round": round_no, "source": source, "query": q, "context": block})

            elif source == "figma_ui":
                s = screen or (action.get("target_screens", []) or fallback_screens[:1])[0] if (action.get("target_screens", []) or fallback_screens) else ""
                if s and s not in selected_screens:
                    selected_screens.append(s)
                if s:
                    elements = rag_client.get_screen_elements(req.project, s)
                    ui_lines = [f"[Screen: {s}]"]
                    for kind, labels in elements.items():
                        ui_lines.append(f"  {kind}s: {', '.join(labels[:10])}")
                    ui_block = "\n".join(ui_lines)
                    if ui_block.strip() != f"[Screen: {s}]":
                        figma_ui_blocks.append(ui_block)
                        round_retrieved_notes.append(f"figma_ui | screen={s} | {textutil.compact_note(ui_block)}")
                        if req.debug_trace:
                            debug_trace["retrieved_blocks"].append({"round": round_no, "source": source, "screen": s, "context": ui_block})

            elif source == "figma_flow":
                trans = rag_client.get_figma_transitions(req.project, screen_name=screen if screen else None)
                if trans:
                    flow_block = context_builders.build_figma_flow_context(trans, top_n=10)
                    if flow_block:
                        flow_context_blocks.append(flow_block)
                        round_retrieved_notes.append(f"figma_flow | screen={screen or '*'} | {textutil.compact_note(flow_block)}")
                        if req.debug_trace:
                            debug_trace["retrieved_blocks"].append({"round": round_no, "source": source, "screen": screen, "context": flow_block})

        round_trace["retrieved_context_chars"] = len("\n\n".join(srs_context_blocks))
        planner_trace.append(round_trace)
        last_round_retrieved_notes = round_retrieved_notes

        if round_no > 1 and not round_retrieved_notes:
            finalization_mode = "no_new_context_early_finalize"
            break
        if len("\n\n".join(srs_context_blocks)) > 9000:
            break

    # Fallback: ensure at least one retrieval happened.
    if not srs_context_blocks:
        data = rag_client.get_srs_and_history(req.project, req.objective, top_k=min(req.top_k, 3))
        block = data.get("context", "")
        if block:
            srs_context_blocks.append(block)
            last_round_retrieved_notes = [f"srs | query={req.objective} | {textutil.compact_note(block)}"]
        if fallback_screens:
            selected_screens = fallback_screens[:2]

    srs_context = "\n\n".join(dict.fromkeys(srs_context_blocks))[:8000]
    figma_overview_context = context_builders.build_figma_overview_context(figma_overview)
    figma_context = (
        "\n\n".join(dict.fromkeys(figma_ui_blocks))[:2400]
        if figma_ui_blocks else context_builders.build_figma_context(req.project, selected_screens[:3])
    )
    figma_flow_context = "\n\n".join(dict.fromkeys(flow_context_blocks))[:2500]

    retrieval_plan = {
        "focus_queries": collected_queries[:2],
        "target_screens": selected_screens[:2],
        "reason": planner_trace[-1].get("reason", "") if planner_trace else "fallback",
    }

    # Stage 4: final test case generation.
    prompt = prompts.build_testcase_prompt(
        app_name=req.app_name,
        objective=req.objective,
        srs_context=srs_context,
        figma_overview_context=figma_overview_context,
        figma_context=figma_context,
        figma_flow_context=figma_flow_context,
        done_titles=done_titles,
        failed_titles=failed_titles,
        coverage_map=coverage_map,
        recent_tests=recent_tests,
    )
    model_data = model_client.call_model(prompt, req.max_new_tokens, req.enable_thinking)
    raw_answer = model_data.get("answer", "")
    parsed = textutil.parse_testcase(raw_answer)

    if req.debug_trace:
        debug_trace["final_generation"] = {
            "prompt": prompt,
            "model_answer_raw": raw_answer,
            "model_thinking": model_data.get("thinking", ""),
        }

    # Stage 5: duplicate check — retry with a rotated screen focus if too similar.
    candidate_title = str(parsed.get("title", "")) if isinstance(parsed, dict) else ""
    blocked_titles = list(dict.fromkeys((done_titles or []) + (failed_titles or [])))
    if candidate_title and textutil.is_similar_to_existing(candidate_title, blocked_titles, threshold=0.60):
        already_picked = set(selected_screens)
        alt_screens = [s["screen_name"] for s in figma_screens if s["screen_name"] not in already_picked][:2]
        alt_figma_context = context_builders.build_figma_context(req.project, alt_screens) if alt_screens else figma_context
        blocked = "\n".join(f"- {t}" for t in blocked_titles[:20]) or "- none"
        retry_prompt = prompts.build_testcase_prompt(
            app_name=req.app_name,
            objective=req.objective + " (RETRY — choose a DISTINCT test case; the previous suggestion was too similar to an already-executed test)",
            srs_context=srs_context,
            figma_overview_context=figma_overview_context,
            figma_context=alt_figma_context,
            figma_flow_context=figma_flow_context,
            done_titles=done_titles,
            failed_titles=failed_titles,
            coverage_map=coverage_map,
            recent_tests=recent_tests,
        ) + "\n\nBlocked titles (semantic overlap with any of these is FORBIDDEN):\n" + blocked
        model_data = model_client.call_model(retry_prompt, req.max_new_tokens, req.enable_thinking)
        raw_answer = model_data.get("answer", "")
        parsed = textutil.parse_testcase(raw_answer)
        if req.debug_trace:
            debug_trace["final_retry"] = {
                "prompt": retry_prompt,
                "model_answer_raw": raw_answer,
                "model_thinking": model_data.get("thinking", ""),
            }

    # Stage 6: enforce a stable unique external test case ID.
    if isinstance(parsed, dict):
        parsed["test_case_id"] = textutil.next_testcase_id(recent_tests)
        raw_answer = json.dumps(parsed, ensure_ascii=False, indent=2)

    # Stage 7: auto-log so coverage grows on every call (idempotent on TC-ID).
    if isinstance(parsed, dict) and parsed.get("test_case_id"):
        existing_ids = {str(t.get("id", "")) for t in recent_tests}
        if parsed["test_case_id"] not in existing_ids:
            try:
                rag_client.rag_post("/tests/log", {
                    "project": req.project,
                    "test_case_id": parsed["test_case_id"],
                    "title": parsed.get("title", ""),
                    "verdict": "pass",
                    "notes": "[GENERATED] Awaiting execution.",
                    "area": parsed.get("area", "general"),
                    "requirement_ids": parsed.get("requirement_ids", []) if isinstance(parsed.get("requirement_ids"), list) else [],
                })
            except Exception:
                pass

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
        "thinking": model_data.get("thinking", ""),
        "coverage": {
            "total_tests": coverage_map.get("total_tests", 0),
            "coverage_pct": coverage_map.get("coverage_pct", 0),
            "uncovered_areas": coverage_map.get("uncovered_purposes", []),
            "hot_spots": coverage_map.get("hot_spots", []),
            "exhausted_areas": coverage_map.get("exhausted_areas", []),
            "exploration_directive": coverage.build_exploration_directive(coverage_map, recent_tests),
        },
    }
    if req.debug_trace:
        out["debug_trace"] = debug_trace
    return out


# ── Verdict logging + adaptive loop ──────────────────────────────────────────────

def log_verdict(req: LogVerdictRequest, authorization: str | None) -> dict:
    config.check_gateway_auth(authorization)
    rag_verdict = req.verdict if req.verdict in {"pass", "failed"} else "failed"
    rag_notes = req.notes
    if req.verdict in {"blocked", "skipped"} and req.notes:
        rag_notes = f"[{req.verdict.upper()}] {req.notes}"
    elif req.verdict in {"blocked", "skipped"}:
        rag_notes = f"[{req.verdict.upper()}] Test was {req.verdict}."

    return rag_client.rag_post("/tests/log", {
        "project": req.project,
        "test_case_id": req.test_case_id,
        "title": req.title,
        "verdict": rag_verdict,
        "notes": rag_notes,
        "area": req.area,
        "requirement_ids": req.requirement_ids,
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
