"""
LangGraph implementation of the Exploratory Testing Planner agent.
This replaces the iterative loop in pipeline.py with a formal StateGraph.
"""

from typing import TypedDict, List, Dict, Any, Optional
import json

import langgraph.graph as lg
from pydantic import BaseModel

from observability import get_logger
from observability.tracing import set_trace, timed_node
from . import config, context_builders, coverage, model_client, prompts, rag_client, schemas, textutil

log = get_logger("langgraph_agent")

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
    
    # Iteration state
    round_no: int
    max_retrieval_rounds: int
    collected_queries: list
    selected_screens: list
    srs_context_blocks: list
    figma_ui_blocks: list
    flow_context_blocks: list
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
    figma_screens = brief.get("screen_index", []) if isinstance(brief, dict) else []
    figma_overview = rag_client.get_figma_overview(project)
    fallback_screens = context_builders.pick_relevant_screens(figma_screens, done_areas, recent_tests)
    coverage_map = coverage.compute_coverage_map(recent_tests, figma_screens)
    
    return {
        **state,
        "brief": brief,
        "recent_tests": recent_tests,
        "done_titles": done_titles,
        "failed_titles": failed_titles,
        "done_areas": done_areas,
        "figma_screens": figma_screens,
        "figma_overview": figma_overview,
        "fallback_screens": fallback_screens,
        "coverage_map": coverage_map,
        "round_no": 1,
    }


@timed_node("planner_step")
def planner_step(state: AgentState) -> AgentState:
    """Stage 2: Ask the LLM what to do next (retrieve or produce)."""
    round_no = state["round_no"]
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
    )
    
    action_model = model_client.call_model(action_prompt, max(320, min(state["max_new_tokens"], 4096)), False)
    action = textutil.parse_action(action_model.get("answer", ""), state["fallback_screens"])
    
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
    
    # Determine next routing via finalization mode in state if needed, or we just pass action back
    if action["action"] == "produce_testcase":
        return {**state, "agent_signaled_ready": True, "finalization_mode": "agent_signaled_sufficient_context"}
        
    return {**state, "agent_signaled_ready": False}


@timed_node("execute_retrieval")
def execute_retrieval(state: AgentState) -> AgentState:
    """Stage 3: Execute the retrieval requests from the planner."""
    trace = state["planner_trace"][-1]
    requests_spec = trace.get("retrieval_requests", [])
    if not requests_spec:
        requests_spec = [{"source": "srs", "query": q} for q in (trace.get("focus_queries", []) or [state["objective"]])[:2]]
        for s in (trace.get("target_screens", []) or state["fallback_screens"][:2])[:2]:
            requests_spec.append({"source": "figma_ui", "screen": s})
            
    round_retrieved_notes = []
    
    for rr in requests_spec[:3]:
        source = str(rr.get("source", "srs")).strip().lower()
        query = str(rr.get("query", "")).strip()
        screen = str(rr.get("screen", "")).strip()
        
        if source == "srs":
            q = query or state["objective"]
            if q not in state["collected_queries"]:
                state["collected_queries"].append(q)
            data = rag_client.get_srs_and_history(state["project"], q, top_k=min(state["top_k"], 2))
            block = data.get("context", "")
            if block:
                state["srs_context_blocks"].append(block)
                round_retrieved_notes.append(f"srs | query={q} | {textutil.compact_note(block)}")
                
        elif source == "figma_ui":
            s = screen or (trace.get("target_screens", []) or state["fallback_screens"][:1])[0] if (trace.get("target_screens", []) or state["fallback_screens"]) else ""
            if s and s not in state["selected_screens"]:
                state["selected_screens"].append(s)
            if s:
                elements = rag_client.get_screen_elements(state["project"], s)
                ui_lines = [f"[Screen: {s}]"]
                for kind, labels in elements.items():
                    ui_lines.append(f"  {kind}s: {', '.join(labels[:10])}")
                ui_block = "\n".join(ui_lines)
                if ui_block.strip() != f"[Screen: {s}]":
                    state["figma_ui_blocks"].append(ui_block)
                    round_retrieved_notes.append(f"figma_ui | screen={s} | {textutil.compact_note(ui_block)}")
                    
        elif source == "figma_flow":
            trans = rag_client.get_figma_transitions(state["project"], screen_name=screen if screen else None)
            if trans:
                flow_block = context_builders.build_figma_flow_context(trans, top_n=10)
                if flow_block:
                    state["flow_context_blocks"].append(flow_block)
                    round_retrieved_notes.append(f"figma_flow | screen={screen or '*'} | {textutil.compact_note(flow_block)}")
                    
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
    # Ensure fallback retrieval if none happened
    if not state["srs_context_blocks"]:
        data = rag_client.get_srs_and_history(state["project"], state["objective"], top_k=min(state["top_k"], 3))
        block = data.get("context", "")
        if block:
            state["srs_context_blocks"].append(block)
        if state["fallback_screens"]:
            state["selected_screens"] = state["fallback_screens"][:2]
            
    srs_context = "\n\n".join(dict.fromkeys(state["srs_context_blocks"]))[:8000]
    figma_overview_context = context_builders.build_figma_overview_context(state["figma_overview"])
    figma_context = (
        "\n\n".join(dict.fromkeys(state["figma_ui_blocks"]))[:2400]
        if state["figma_ui_blocks"] else context_builders.build_figma_context(state["project"], state["selected_screens"][:3])
    )
    figma_flow_context = "\n\n".join(dict.fromkeys(state["flow_context_blocks"]))[:2500]
    
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
    )
    
    model_data = model_client.call_model(prompt, state["max_new_tokens"], state["enable_thinking"])
    raw_answer = model_data.get("answer", "")
    parsed = textutil.parse_testcase(raw_answer)
    
    if state["debug_trace"]:
        state["debug_trace_data"]["final_generation"] = {
            "prompt": prompt,
            "model_answer_raw": raw_answer,
            "model_thinking": model_data.get("thinking", ""),
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
        "retrieval_plan": retrieval_plan
    }


@timed_node("duplicate_check")
def duplicate_check(state: AgentState) -> AgentState:
    """Stage 5 & 6: Check for duplicates, auto-retry if needed, and assign test case ID."""
    parsed = state["next_testcase"]
    candidate_title = str(parsed.get("title", "")) if isinstance(parsed, dict) else ""
    blocked_titles = list(dict.fromkeys((state["done_titles"] or []) + (state["failed_titles"] or [])))
    
    # We do a single retry here if similar
    if candidate_title and textutil.is_similar_to_existing(candidate_title, blocked_titles, threshold=0.60):
        # We re-generate but with different screens (similar to pipeline logic)
        already_picked = set(state["selected_screens"])
        alt_screens = [s["screen_name"] for s in state["figma_screens"] if s["screen_name"] not in already_picked][:2]
        alt_figma_context = context_builders.build_figma_context(state["project"], alt_screens) if alt_screens else ""
        
        blocked = "\n".join(f"- {t}" for t in blocked_titles[:20]) or "- none"
        
        srs_context = "\n\n".join(dict.fromkeys(state["srs_context_blocks"]))[:8000]
        figma_overview_context = context_builders.build_figma_overview_context(state["figma_overview"])
        figma_flow_context = "\n\n".join(dict.fromkeys(state["flow_context_blocks"]))[:2500]
        
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
        ) + "\n\nBlocked titles (semantic overlap with any of these is FORBIDDEN):\n" + blocked
        
        model_data = model_client.call_model(retry_prompt, state["max_new_tokens"], state["enable_thinking"])
        raw_answer = model_data.get("answer", "")
        parsed = textutil.parse_testcase(raw_answer)
        
        state["next_testcase_json"] = raw_answer
        state["next_testcase"] = parsed
        state["model_thinking"] = model_data.get("thinking", "")
        
        if state["debug_trace"]:
            state["debug_trace_data"]["final_retry"] = {
                "prompt": retry_prompt,
                "model_answer_raw": raw_answer,
                "model_thinking": model_data.get("thinking", ""),
            }

    # Stage 6: Enforce external test case ID
    if isinstance(parsed, dict):
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
        max_new_tokens=req_args.get("max_new_tokens", 4096),
        enable_thinking=req_args.get("enable_thinking", False),
        debug_trace=req_args.get("debug_trace", False),
        
        brief={}, recent_tests=[], done_titles=[], failed_titles=[], done_areas=[],
        figma_screens=[], figma_overview=[], fallback_screens=[], coverage_map={},
        
        round_no=1,
        max_retrieval_rounds=max(1, min(req_args.get("max_retrieval_rounds", 6), 6)),
        collected_queries=[],
        selected_screens=[],
        srs_context_blocks=[],
        figma_ui_blocks=[],
        flow_context_blocks=[],
        last_round_retrieved_notes=[],
        
        planner_trace=[],
        debug_trace_data={"planner_rounds": [], "retrieved_blocks": []},
        
        finalization_mode="max_retrieval_rounds_fallback",
        agent_signaled_ready=False,
        next_testcase_json="",
        next_testcase={},
        retrieval_plan={},
        model_thinking=""
    )
    
    # Execute graph
    final_state = app.invoke(initial_state)
    
    # Stage 7: Auto-log
    parsed = final_state.get("next_testcase", {})
    recent_tests = final_state.get("recent_tests", [])
    if isinstance(parsed, dict) and parsed.get("test_case_id"):
        existing_ids = {str(t.get("id", "")) for t in recent_tests}
        if parsed["test_case_id"] not in existing_ids:
            try:
                rag_client.rag_post("/tests/log", {
                    "project": final_state["project"],
                    "test_case_id": parsed.get("test_case_id") or f"TC-{int(time.time())}",
                    "title": parsed.get("title") or "Generated Test Case",
                    "verdict": "pass",
                    "notes": "[GENERATED] Awaiting execution.",
                    "area": parsed.get("area", "general"),
                    "requirement_ids": parsed.get("requirement_ids", []) if isinstance(parsed.get("requirement_ids"), list) else [],
                })
            except Exception:
                pass
                
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
