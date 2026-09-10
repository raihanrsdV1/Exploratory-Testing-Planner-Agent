"""
LangGraph implementation of the Exploratory Testing Planner agent.
This replaces the iterative loop in pipeline.py with a formal StateGraph.
"""

from typing import TypedDict, List, Dict, Any, Optional
import json
from datetime import datetime, timezone
from pathlib import Path

import langgraph.graph as lg
from pydantic import BaseModel

from observability import get_logger
from observability import degradations
from observability.tracing import set_trace, timed_node
from . import config, context_builders, coverage, model_client, prompts, rag_client, schemas, textutil
from .sources import registry as sources_registry
from .sources.base import RetrievalRequest

log = get_logger("langgraph_agent")

_PLANNER_LOG_DIR = Path(__file__).resolve().parent.parent / "logs" / "planner"


def _write_planner_log(test_case_id: str, call_log: list) -> None:
    """Write every LLM call made while generating this test case to
    logs/planner/<test_case_id>.txt — input then output, one call after
    another, in the order the calls actually happened. The id isn't known
    until generation finishes, so this flushes what was accumulated in state
    rather than writing incrementally. Best-effort: never breaks generation."""
    if not test_case_id or not call_log:
        return
    try:
        _PLANNER_LOG_DIR.mkdir(parents=True, exist_ok=True)
        lines = []
        for call in call_log:
            lines.append("=" * 80)
            lines.append(f"CALL: {call.get('label', '?')}")
            lines.append(f"TIMESTAMP: {call.get('ts', '?')}")
            lines.append("=" * 80)
            lines.append("--- INPUT ---")
            lines.append(call.get("input", ""))
            lines.append("")
            lines.append("--- OUTPUT ---")
            lines.append(call.get("output", ""))
            lines.append("")
        (_PLANNER_LOG_DIR / f"{test_case_id}.txt").write_text("\n".join(lines), encoding="utf-8")
    except Exception as e:
        log.warning("planner_log_write_failed", test_case_id=test_case_id, error=str(e)[:200])


class AgentState(TypedDict):
    # Request inputs
    project: str
    app_name: str
    objective: str
    top_k: int
    max_new_tokens: int
    enable_thinking: bool
    debug_trace: bool
    
    # Global context (computed once)
    brief: dict
    recent_tests: list
    done_titles: list
    failed_titles: list
    done_areas: list
    figma_screens: list
    figma_overview: list
    fallback_screens: list
    coverage_map: dict
    available_sources: list  # [{"name","purpose"}] — sources with data for this project
    dimensions: dict  # WP6: {profile?, platform?, application?} target env filter
    
    # Iteration state
    round_no: int
    max_retrieval_rounds: int
    collected_queries: list
    selected_screens: list
    selected_live_states: list  # resolved live UIState matches: [{id, label, has_screenshot, match_score}]
    srs_context_blocks: list
    figma_ui_blocks: list
    flow_context_blocks: list
    defect_blocks: list
    nav_blocks: list
    last_round_retrieved_notes: list
    
    # Trace
    planner_trace: list
    debug_trace_data: dict
    
    # Outputs
    finalization_mode: str
    agent_signaled_ready: bool
    next_testcase_json: str
    next_testcase: dict
    retrieval_plan: dict
    model_thinking: str
    failure_context: str
    requirements_context: str
    generation_prompt: str
    generation_answer: str
    # Every LLM call made while generating this one test case, in order — the
    # test_case_id isn't known until the very end (duplicate_check assigns it),
    # so calls accumulate here and get flushed to logs/planner/<id>.txt once
    # it is. Each entry: {"label", "input", "output"}.
    llm_call_log: list


@timed_node("bootstrap_context")
def bootstrap_context(state: AgentState) -> AgentState:
    """Stage 1: Compact global context."""
    project = state["project"]
    brief = rag_client.get_brief_context(project)
    
    recent_tests = brief.get("recent_tests", []) if isinstance(brief, dict) else []
    done_titles = [str(t.get("title", "")).strip() for t in recent_tests if t.get("title")]
    failed_titles = [
        str(t.get("title", "")).strip()
        for t in recent_tests
        if t.get("title") and str(t.get("verdict", "")).lower() == "failed"
    ]
    done_areas = [str(t.get("area", "")) for t in recent_tests if t.get("area")]
    failure_context = context_builders.build_failure_context(project, recent_tests)
    requirements_context = context_builders.build_requirements_context(project)
    # Honour ENABLED_SOURCES here, not just when advertising sources to the
    # retrieval planner. These two feed the generation prompt directly, so
    # reading them unconditionally let a disabled design file describe screens
    # the shipped app does not have — and the planner wrote tests against them.
    _figma_on = sources_registry.is_enabled("figma_ui")
    figma_screens = (brief.get("screen_index", []) if isinstance(brief, dict) else []) if _figma_on else []
    figma_overview = rag_client.get_figma_overview(project) if _figma_on else []
    fallback_screens = context_builders.pick_relevant_screens(figma_screens, done_areas, recent_tests)
    coverage_map = coverage.compute_coverage_map(recent_tests, figma_screens)

    # Graceful degradation: advertise only sources that actually have data for this project.
    available_sources = [
        {"name": s.name, "purpose": s.purpose}
        for s in sources_registry.available_sources(brief if isinstance(brief, dict) else {})
    ]

    return {
        **state,
        "brief": brief,
        "recent_tests": recent_tests,
        "done_titles": done_titles,
        "failed_titles": failed_titles,
        "failure_context": failure_context,
        "requirements_context": requirements_context,
        "done_areas": done_areas,
        "figma_screens": figma_screens,
        "figma_overview": figma_overview,
        "fallback_screens": fallback_screens,
        "coverage_map": coverage_map,
        "available_sources": available_sources,
        "round_no": 1,
    }


@timed_node("planner_step")
def planner_step(state: AgentState) -> AgentState:
    """Stage 2: Ask the LLM what to do next (retrieve or produce)."""
    round_no = state["round_no"]

    # No ingested knowledge sources (zero-doc app): skip retrieval planning entirely and
    # go straight to generation from exploratory heuristics — saves an LLM round.
    if not state.get("available_sources"):
        state["planner_trace"].append({
            "round": round_no,
            "action": "produce_testcase",
            "retrieval_requests": [],
            "focus_queries": [],
            "target_screens": [],
            "reason": "no knowledge sources available — generate from heuristics",
        })
        log.info("planner_retrieval_decision", project=state.get("project", ""), round=round_no,
                  action="produce_testcase", retrieval_requests=[], reason="no knowledge sources available")
        return {**state, "agent_signaled_ready": True, "finalization_mode": "no_sources_available"}

    action_prompt = prompts.planner_prompt_for_action(
        brief=state["brief"],
        app_name=state["app_name"],
        objective=state["objective"],
        retrieval_round=round_no,
        max_rounds=state["max_retrieval_rounds"],
        collected_queries=state["collected_queries"],
        collected_screens=state["selected_screens"],
        context_chars=len("\n\n".join(state["srs_context_blocks"])),
        figma_overview=state["figma_overview"],
        retrieved_notes=state["last_round_retrieved_notes"],
        coverage_map=state["coverage_map"],
        available_sources=state["available_sources"],
    )
    
    action_model = model_client.call_model(action_prompt, max(320, min(state["max_new_tokens"], 4096)), False)
    action = textutil.parse_action(action_model.get("answer", ""), state["fallback_screens"])
    state["llm_call_log"].append({
        "label": f"planner_step (round {round_no})",
        "ts": datetime.now(timezone.utc).isoformat(),
        "input": action_prompt,
        "output": action_model.get("answer", ""),
    })

    if state["debug_trace"]:
        state["debug_trace_data"].setdefault("planner_rounds", []).append({
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
    
    state["planner_trace"].append(round_trace)

    # Always visible in logs/app.jsonl, not just when a caller opts into debug_trace —
    # without this, whether the retrieval loop is asking for anything useful was
    # unauditable after the fact (debug_trace is off by default in real executor calls).
    log.info("planner_retrieval_decision", project=state.get("project", ""), round=round_no,
              action=round_trace["action"], retrieval_requests=round_trace["retrieval_requests"],
              focus_queries=round_trace["focus_queries"], target_screens=round_trace["target_screens"],
              reason=round_trace["reason"])

    # Determine next routing via finalization mode in state if needed, or we just pass action back
    if action["action"] == "produce_testcase":
        return {**state, "agent_signaled_ready": True, "finalization_mode": "agent_signaled_sufficient_context"}
        
    return {**state, "agent_signaled_ready": False}


def _default_requests(state: AgentState, trace: dict, available_names: set) -> list[dict]:
    """Availability-gated default retrieval plan when the planner returns none."""
    reqs: list[dict] = []
    if "srs" in available_names:
        for q in (trace.get("focus_queries", []) or [state["objective"]])[:2]:
            reqs.append({"source": "srs", "query": q})
    if "figma_ui" in available_names:
        for s in (trace.get("target_screens", []) or state["fallback_screens"][:2])[:2]:
            reqs.append({"source": "figma_ui", "screen": s})
    return reqs


@timed_node("execute_retrieval")
def execute_retrieval(state: AgentState) -> AgentState:
    """Stage 3: Execute the planner's retrieval requests via the source registry."""
    trace = state["planner_trace"][-1]
    available_names = {s["name"] for s in state.get("available_sources", [])}

    requests_spec = [
        rr for rr in trace.get("retrieval_requests", [])
        if str(rr.get("source", "")).strip().lower() in available_names
    ]
    if not requests_spec:
        requests_spec = _default_requests(state, trace, available_names)

    # Channel -> the state bucket its retrieved text accumulates into.
    buckets = {
        "srs": state["srs_context_blocks"],
        "figma_ui": state["figma_ui_blocks"],
        "figma_flow": state["flow_context_blocks"],
        "defects": state["defect_blocks"],
        "navtree": state["nav_blocks"],
    }

    round_retrieved_notes = []

    for rr in requests_spec[:3]:
        source_name = str(rr.get("source", "")).strip().lower()
        source = sources_registry.get(source_name)
        if source is None or source_name not in available_names:
            continue

        req = RetrievalRequest(
            source=source_name,
            query=str(rr.get("query", "")).strip(),
            screen=str(rr.get("screen", "")).strip(),
        )

        # Agent-level defaulting + bookkeeping (sources stay pure / objective-agnostic).
        if source_name == "srs":
            req.query = req.query or state["objective"]
            if req.query not in state["collected_queries"]:
                state["collected_queries"].append(req.query)
        elif source_name == "figma_ui":
            if not req.screen:
                fallback = trace.get("target_screens", []) or state["fallback_screens"]
                req.screen = fallback[0] if fallback else ""
            if req.screen and req.screen not in state["selected_screens"]:
                state["selected_screens"].append(req.screen)
        elif source_name == "live_ui":
            if not req.screen:
                fallback = trace.get("target_screens", []) or state["fallback_screens"]
                req.screen = fallback[0] if fallback else ""

        block = source.retrieve(state["project"], req, top_k=state["top_k"])
        if block is None:
            if source_name == "live_ui" and req.screen:
                log.info("live_ui_screen_match", project=state["project"], screen=req.screen, matched=False)
            continue
        bucket = buckets.get(block.channel)
        if bucket is not None:
            bucket.append(block.text)
        round_retrieved_notes.append(block.note)
        if block.resolved_state:
            rs = block.resolved_state
            if not any(s.get("id") == rs.get("id") for s in state["selected_live_states"]):
                state["selected_live_states"].append(rs)
            log.info("live_ui_screen_match", project=state["project"], screen=req.screen,
                      matched=True, resolved_label=rs.get("label"), match_score=rs.get("match_score"),
                      has_screenshot=rs.get("has_screenshot"))

    state["planner_trace"][-1]["retrieved_context_chars"] = len("\n\n".join(state["srs_context_blocks"]))

    # Check early finalize conditions
    finalization_mode = state["finalization_mode"]
    if state["round_no"] > 1 and not round_retrieved_notes:
        finalization_mode = "no_new_context_early_finalize"
    elif len("\n\n".join(state["srs_context_blocks"])) > 9000:
        finalization_mode = "context_limit_reached"

    return {
        **state,
        "last_round_retrieved_notes": round_retrieved_notes,
        "round_no": state["round_no"] + 1,
        "finalization_mode": finalization_mode
    }


def should_continue(state: AgentState):
    """Router logic for the planner loop."""
    if state.get("agent_signaled_ready", False):
        return "generate_testcase"
    if state.get("finalization_mode", "") in ["no_new_context_early_finalize", "context_limit_reached"]:
        return "generate_testcase"
    if state["round_no"] > state["max_retrieval_rounds"]:
        return "generate_testcase"
    return "execute_retrieval"


@timed_node("generate_testcase")
def generate_testcase(state: AgentState) -> AgentState:
    """Stage 4: Generate the final JSON test case."""
    # Ensure best-effort grounding if the retrieval loop gathered nothing — but only from
    # sources that actually exist (no SRS hard-dependency: a UI-only or zero-doc project
    # must still generate).
    available_names = {s["name"] for s in state.get("available_sources", [])}
    if not state["srs_context_blocks"] and "srs" in available_names:
        data = rag_client.get_srs_and_history(
            state["project"], state["objective"], top_k=min(state["top_k"], 3),
            dims=state.get("dimensions") or None,
        )
        block = data.get("context", "")
        if block:
            state["srs_context_blocks"].append(block)
    if not state["figma_ui_blocks"] and "figma_ui" in available_names and state["fallback_screens"]:
        state["selected_screens"] = state["fallback_screens"][:2]
            
    srs_context = "\n\n".join(dict.fromkeys(state["srs_context_blocks"]))[:60000]
    figma_overview_context = context_builders.build_figma_overview_context(state["figma_overview"])
    figma_context = (
        "\n\n".join(dict.fromkeys(state["figma_ui_blocks"]))[:30000]
        if state["figma_ui_blocks"] else context_builders.build_figma_context(state["project"], state["selected_screens"][:3])
    )
    figma_flow_context = "\n\n".join(dict.fromkeys(state["flow_context_blocks"]))[:12000]

    # REQ-301.5 / 302.4 / 303: learned-intelligence context, injected when available.
    available_names = {s["name"] for s in state.get("available_sources", [])}
    defect_context, nav_context, failed_nav, strategy_context, risk_context, anomaly_context = \
        context_builders.build_learned_context(
            state["project"], available_names, state["objective"],
            state["selected_screens"], state["defect_blocks"], state["nav_blocks"],
        )

    target_env = context_builders.target_environment_text(state.get("dimensions") or {})

    prompt = prompts.build_testcase_prompt(
        app_name=state["app_name"],
        objective=state["objective"],
        srs_context=srs_context,
        figma_overview_context=figma_overview_context,
        figma_context=figma_context,
        figma_flow_context=figma_flow_context,
        done_titles=state["done_titles"],
        failed_titles=state["failed_titles"],
        coverage_map=state["coverage_map"],
        recent_tests=state["recent_tests"],
        defect_context=defect_context,
        nav_context=nav_context,
        failed_nav=failed_nav,
        strategy_context=strategy_context,
        target_env=target_env,
        risk_context=risk_context,
        anomaly_context=anomaly_context,
        failure_context=state.get("failure_context", ""),
        requirements_context=state.get("requirements_context", ""),
    )

    # Attach the target screen's real screenshot, if the retrieval loop resolved
    # one with a stored image (Phase 1) — cap at 1, the primary target only, so
    # this never balloons into attaching several screens per call.
    image_b64 = None
    image_state_id = None
    for ls in state.get("selected_live_states") or []:
        if ls.get("has_screenshot"):
            image_b64 = rag_client.get_state_screenshot(state["project"], ls["id"])
            if image_b64:
                image_state_id = ls["id"]
                break
    log.info("generation_screenshot", project=state["project"],
              attached=bool(image_b64), state_id=image_state_id)

    model_data = model_client.call_model(prompt, state["max_new_tokens"], state["enable_thinking"],
                                          image_b64=image_b64)
    raw_answer = model_data.get("answer", "")
    parsed = textutil.parse_testcase(raw_answer)
    state["llm_call_log"].append({
        "label": "generate_testcase", "ts": datetime.now(timezone.utc).isoformat(),
        "input": prompt, "output": raw_answer,
    })

    if state["debug_trace"]:
        state["debug_trace_data"]["final_generation"] = {
            "prompt": prompt,
            "model_answer_raw": raw_answer,
            "model_thinking": model_data.get("thinking", ""),
            "image_attached": bool(image_b64),
            "image_state_id": image_state_id,
        }
        
    retrieval_plan = {
        "focus_queries": state["collected_queries"][:2],
        "target_screens": state["selected_screens"][:2],
        "reason": state["planner_trace"][-1].get("reason", "") if state["planner_trace"] else "fallback",
    }
        
    return {
        **state,
        "next_testcase_json": raw_answer,
        "next_testcase": parsed,
        "model_thinking": model_data.get("thinking", ""),
        "retrieval_plan": retrieval_plan,
        # Full audit trail — unconditional, not gated behind debug_trace, so the
        # dashboard can always show exactly what the planner was given and what
        # it produced. Overwritten by duplicate_check's retry when one fires,
        # so this always reflects the call that produced the FINAL test case.
        "generation_prompt": prompt,
        "generation_answer": raw_answer,
    }


@timed_node("duplicate_check")
def duplicate_check(state: AgentState) -> AgentState:
    """Stage 5 & 6: Check for duplicates, auto-retry if needed, and assign test case ID."""
    parsed = state["next_testcase"]
    candidate_title = str(parsed.get("title", "")) if isinstance(parsed, dict) else ""
    blocked_titles = list(dict.fromkeys((state["done_titles"] or []) + (state["failed_titles"] or [])))

    # WP8 (307.3): Jaccard is a cheap local pre-filter; the embedding-cosine check
    # (server-side) catches semantically identical tests phrased differently.
    jaccard_dupe = bool(candidate_title) and textutil.is_similar_to_existing(
        candidate_title, blocked_titles, threshold=0.60)
    semantic = (rag_client.semantic_dedup_check(state["project"], candidate_title)
                if candidate_title else {"is_duplicate": False})
    if state["debug_trace"]:
        state["debug_trace_data"]["dedup"] = {"jaccard_duplicate": jaccard_dupe, "semantic": semantic}

    # We do a single retry here if similar
    if candidate_title and (jaccard_dupe or semantic.get("is_duplicate")):
        # We re-generate but with different screens (similar to pipeline logic)
        already_picked = set(state["selected_screens"])
        alt_screens = [s["screen_name"] for s in state["figma_screens"] if s["screen_name"] not in already_picked][:2]
        alt_figma_context = context_builders.build_figma_context(state["project"], alt_screens) if alt_screens else ""
        
        blocked = "\n".join(f"- {t}" for t in blocked_titles[:200]) or "- none"
        
        srs_context = "\n\n".join(dict.fromkeys(state["srs_context_blocks"]))[:60000]
        figma_overview_context = context_builders.build_figma_overview_context(state["figma_overview"])
        figma_flow_context = "\n\n".join(dict.fromkeys(state["flow_context_blocks"]))[:12000]
        
        retry_prompt = prompts.build_testcase_prompt(
            app_name=state["app_name"],
            objective=state["objective"] + " (RETRY — choose a DISTINCT test case; the previous suggestion was too similar to an already-executed test)",
            srs_context=srs_context,
            figma_overview_context=figma_overview_context,
            figma_context=alt_figma_context,
            figma_flow_context=figma_flow_context,
            done_titles=state["done_titles"],
            failed_titles=state["failed_titles"],
            coverage_map=state["coverage_map"],
            recent_tests=state["recent_tests"],
            failure_context=state.get("failure_context", ""),
            requirements_context=state.get("requirements_context", ""),
        ) + "\n\nBlocked titles (semantic overlap with any of these is FORBIDDEN):\n" + blocked
        
        model_data = model_client.call_model(retry_prompt, state["max_new_tokens"], state["enable_thinking"])
        raw_answer = model_data.get("answer", "")
        parsed = textutil.parse_testcase(raw_answer)
        state["llm_call_log"].append({
            "label": "duplicate_check retry", "ts": datetime.now(timezone.utc).isoformat(),
            "input": retry_prompt, "output": raw_answer,
        })

        state["next_testcase_json"] = raw_answer
        state["next_testcase"] = parsed
        state["model_thinking"] = model_data.get("thinking", "")
        # This retry is what actually produced the FINAL test case — overwrite
        # the audit trail so it reflects the retry, not the discarded duplicate.
        state["generation_prompt"] = retry_prompt
        state["generation_answer"] = raw_answer

        if state["debug_trace"]:
            state["debug_trace_data"]["final_retry"] = {
                "prompt": retry_prompt,
                "model_answer_raw": raw_answer,
                "model_thinking": model_data.get("thinking", ""),
            }

    # Stage 6: Enforce external test case ID. Require a title so an unparsed
    # ``{"raw": ...}`` blob never gets an id and is never logged as a test.
    if isinstance(parsed, dict) and "title" in parsed:
        parsed["test_case_id"] = textutil.next_testcase_id(state["recent_tests"])
        state["next_testcase_json"] = json.dumps(parsed, ensure_ascii=False, indent=2)
        state["next_testcase"] = parsed

    # Auto-log (Stage 7) is handled outside the graph, or here, but better outside.
    return state


# Build the LangGraph workflow
workflow = lg.StateGraph(AgentState)

workflow.add_node("bootstrap_context", bootstrap_context)
workflow.add_node("planner_step", planner_step)
workflow.add_node("execute_retrieval", execute_retrieval)
workflow.add_node("generate_testcase", generate_testcase)
workflow.add_node("duplicate_check", duplicate_check)

workflow.add_edge(lg.START, "bootstrap_context")
workflow.add_edge("bootstrap_context", "planner_step")
workflow.add_conditional_edges("planner_step", should_continue, {
    "execute_retrieval": "execute_retrieval",
    "generate_testcase": "generate_testcase"
})
workflow.add_edge("execute_retrieval", "planner_step")
workflow.add_edge("generate_testcase", "duplicate_check")
workflow.add_edge("duplicate_check", lg.END)

app = workflow.compile()

def run_agent(req_args: dict) -> dict:
    """Wrapper to initialize state, run the LangGraph, and map back to response dict."""
    project = req_args.get("project", "")
    set_trace(project=project)
    
    initial_state = AgentState(
        project=project,
        app_name=req_args.get("app_name", ""),
        objective=req_args.get("objective", ""),
        top_k=req_args.get("top_k", 5),
        max_new_tokens=req_args.get("max_new_tokens", 8000),
        enable_thinking=req_args.get("enable_thinking", False),
        debug_trace=req_args.get("debug_trace", False),
        
        brief={}, recent_tests=[], done_titles=[], failed_titles=[], done_areas=[],
        figma_screens=[], figma_overview=[], fallback_screens=[], coverage_map={},
        available_sources=[],
        dimensions={k: str(req_args.get(k) or "").strip().lower()
                    for k in ("profile", "platform", "application") if str(req_args.get(k) or "").strip()},

        round_no=1,
        max_retrieval_rounds=max(1, min(req_args.get("max_retrieval_rounds", 6), 6)),
        collected_queries=[],
        selected_screens=[],
        selected_live_states=[],
        srs_context_blocks=[],
        figma_ui_blocks=[],
        flow_context_blocks=[],
        defect_blocks=[],
        nav_blocks=[],
        last_round_retrieved_notes=[],
        
        planner_trace=[],
        debug_trace_data={"planner_rounds": [], "retrieved_blocks": []},
        
        finalization_mode="max_retrieval_rounds_fallback",
        agent_signaled_ready=False,
        next_testcase_json="",
        next_testcase={},
        retrieval_plan={},
        model_thinking="",
        generation_prompt="",
        generation_answer="",
        llm_call_log=[],
    )
    
    # Execute graph
    final_state = app.invoke(initial_state)
    
    # Stage 7: Auto-log
    parsed = final_state.get("next_testcase", {})
    recent_tests = final_state.get("recent_tests", [])
    if isinstance(parsed, dict) and parsed.get("test_case_id"):
        _write_planner_log(parsed["test_case_id"], final_state.get("llm_call_log", []))
        existing_ids = {str(t.get("id", "")) for t in recent_tests}
        if parsed["test_case_id"] not in existing_ids:
            try:
                rag_client.rag_post("/tests/log", {
                    "project": final_state["project"],
                    "test_case_id": parsed["test_case_id"],
                    "title": parsed.get("title") or "Generated Test Case",
                    # Not executed yet — logging "pass" here invents a passing test
                    # that never ran and poisons coverage, risk and effectiveness.
                    "verdict": "planned",
                    "notes": "[GENERATED] Awaiting execution.",
                    "area": parsed.get("area", "general"),
                    "requirement_ids": parsed.get("requirement_ids", []) if isinstance(parsed.get("requirement_ids"), list) else [],
                    "test_type": parsed.get("test_type", ""),
                    "generation_prompt": final_state.get("generation_prompt", ""),
                    "generation_answer": final_state.get("generation_answer", ""),
                    **(final_state.get("dimensions") or {}),
                })
            except Exception as e:
                # This POST is the ONLY place a generated test enters the graph and
                # gets its COVERS edges. Swallowing the error silently made
                # requirement coverage read 0% with no indication why.
                log.warning("auto_log_failed", test_case_id=parsed.get("test_case_id"),
                            error=str(e)[:200])
                degradations.record(
                    "testcase_not_logged", degradations.MAJOR,
                    detail=f"generated test never entered the graph: {e}",
                    test_case_id=str(parsed.get("test_case_id")),
                )

    out = {
        "project": final_state["project"],
        "retrieval_plan": final_state["retrieval_plan"],
        "planner_trace": final_state["planner_trace"],
        "finalization_mode": final_state["finalization_mode"],
        "agent_signaled_ready": final_state["agent_signaled_ready"],
        "retrieved_context_stats": {
            "retrieval_rounds_executed": len(final_state["planner_trace"]),
            "queries_used": final_state["collected_queries"][:2],
            "screens_used": final_state["selected_screens"][:3],
            "srs_context_chars": len("\n\n".join(final_state["srs_context_blocks"])),
            "figma_overview_chars": len(context_builders.build_figma_overview_context(final_state["figma_overview"])),
            "figma_context_chars": len("\n\n".join(final_state["figma_ui_blocks"])),
            "figma_flow_chars": len("\n\n".join(final_state["flow_context_blocks"])),
        },
        "target_screens": final_state["selected_screens"][:3],
        "target_live_states": final_state["selected_live_states"][:3],
        "available_sources": [s["name"] for s in final_state.get("available_sources", [])],
        "next_testcase_json": final_state["next_testcase_json"],
        "next_testcase": parsed,
        "recent_tests_count": len(recent_tests),
        "failed_tests_count": len(final_state["failed_titles"]),
        "thinking": final_state["model_thinking"],
        "coverage": {
            "total_tests": final_state["coverage_map"].get("total_tests", 0),
            "coverage_pct": final_state["coverage_map"].get("coverage_pct", 0),
            "uncovered_areas": final_state["coverage_map"].get("uncovered_purposes", []),
            "hot_spots": final_state["coverage_map"].get("hot_spots", []),
            "exhausted_areas": final_state["coverage_map"].get("exhausted_areas", []),
            "exploration_directive": coverage.build_exploration_directive(final_state["coverage_map"], recent_tests),
        },
    }
    
    if final_state["debug_trace"]:
        out["debug_trace"] = final_state["debug_trace_data"]
        
    return out
