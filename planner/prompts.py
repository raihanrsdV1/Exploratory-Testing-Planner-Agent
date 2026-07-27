"""
LLM prompt builders for the exploratory-testing planner.

Every prompt is app-agnostic: the only domain knowledge comes from the ingested
SRS/UI/history context passed in at call time. Retrieval queries are phrased as
natural language so the RAG layer can match them semantically (vectors).
"""

from __future__ import annotations

import json
import re

from ingestion import ui_normalizer

from . import coverage, context_builders, model_client, textutil


def summarize_srs_with_model(srs_text: str, max_new_tokens: int = 4096) -> str:
    """Planner-friendly SRS summary from full source text via the model backend."""
    prompt = (
        "You are an expert software requirements analyst.\n"
        "Read the full SRS below and produce a concise but complete summary for QA test planning.\n\n"
        "Output STRICT plain text (no markdown, no code fences) with this structure:\n"
        "Document: <name or inferred title>\n"
        "Functional requirements summary:\n- ...\n- ...\n"
        "Non-functional requirements summary:\n- ...\n- ...\n"
        "Validation and constraints:\n- ...\n"
        "Coverage priorities for next tests:\n- ...\n\n"
        "Rules:\n"
        "- Preserve critical requirement IDs when available (FR-#, NFR-#).\n"
        "- Keep it compact but include all high-impact behaviors and validations.\n"
        "- If non-functional requirements are not explicitly listed, state that clearly.\n\n"
        "SRS:\n"
        f"{(srs_text or '').strip()}"
    )
    data = model_client.call_model(prompt, max_new_tokens, False)
    summary = (data.get("answer") or "").strip()
    return textutil.extract_json_text(summary) if summary.startswith("{") else summary


def classify_screen_purposes(ui_ir: dict, app_name: str) -> dict[str, str]:
    """
    LLM-assigned, app-agnostic feature-area slug per screen.
    Returns {screen_name: slug}; {} on any failure (caller falls back to name slug).
    """
    index = ui_normalizer.screen_label_index(ui_ir, top=6)
    if not index:
        return {}
    lines = [
        f"- {s['screen_name']}: " + (", ".join(s["labels"]) if s["labels"] else "(no labels)")
        for s in index
    ]
    prompt = (
        f"You are a UX analyst mapping screens of '{app_name}' to feature areas.\n"
        "For each screen below, assign a SHORT lowercase snake_case feature-area slug that groups "
        "related screens. Infer the slug ONLY from the screen name and its element labels — do NOT "
        "invent categories unsupported by the content, and do not assume any particular kind of app.\n\n"
        "Screens:\n" + "\n".join(lines) + "\n\n"
        "Return STRICT JSON only, mapping each screen name to its slug:\n"
        '{"<screen name>": "<slug>", ...}'
    )
    try:
        data = model_client.call_model(prompt, 600, False)
        parsed = json.loads(textutil.extract_json_text(data.get("answer", "")))
        if not isinstance(parsed, dict):
            return {}
        return {str(k): re.sub(r"[^a-z0-9_]+", "_", str(v).lower()).strip("_") for k, v in parsed.items() if v}
    except Exception:
        return {}


def planner_prompt_for_retrieval(brief: dict, app_name: str, objective: str) -> str:
    return (
        f"You are planning retrieval for exploratory QA test generation in {app_name}.\n"
        "Given compact project context, output a retrieval plan as JSON only.\n\n"
        "Context summary:\n"
        f"SRS summary:\n{brief.get('srs_summary','')}\n\n"
        f"UI summary:\n{brief.get('figma_summary','')}\n\n"
        f"Screen index: {brief.get('screen_index', [])}\n\n"
        f"Recent tests: {brief.get('recent_tests', [])}\n\n"
        f"Objective: {objective}\n\n"
        "Return STRICT JSON:\n"
        '{\n  "focus_queries": ["...", "..."],\n  "target_screens": ["...", "..."],\n  "reason": "short reason"\n}\n'
        "Constraints: max 2 focus_queries, max 2 target_screens, keep concise. "
        "focus_queries are natural-language descriptions (matched semantically), not keyword expressions."
    )


def planner_prompt_for_action(
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
    coverage_map: dict | None = None,
    available_sources: list[dict] | None = None,
) -> str:
    # Advertise ONLY the sources that have data for this project (graceful degradation:
    # no source is a hard dependency). Fall back to all three for backward compatibility
    # when the caller does not pass an explicit availability list.
    avail = available_sources if available_sources is not None else [
        {"name": "srs", "purpose": "business rules, validation constraints, and error conditions"},
        {"name": "figma_ui", "purpose": "screen elements and control availability"},
        {"name": "figma_flow", "purpose": "navigation / screen-to-screen behaviour"},
    ]
    source_ids = "|".join(s["name"] for s in avail) or "none"
    source_guidance = "\n".join(f"- Use source={s['name']} for {s['purpose']}." for s in avail) or (
        "- No external knowledge sources are available; set action=produce_testcase and rely on "
        "general exploratory heuristics."
    )
    srs_available = any(s["name"] == "srs" for s in avail)
    srs_query_note = (
        "- For source=srs, 'query' MUST be a natural-language description of the behaviour or rule you need "
        '(it is matched semantically), e.g. "email format validation before saving".\n'
        if srs_available else ""
    )
    retrieved_notes_str = "\n".join(f"- {n}" for n in retrieved_notes[-6:]) if retrieved_notes else "- none yet"
    srs_summary = brief.get("srs_summary", "") if isinstance(brief, dict) else ""
    srs_full = str(srs_summary).strip() if srs_summary else "(none)"
    figma_summary = brief.get("figma_summary", "") if isinstance(brief, dict) else ""
    figma_full = str(figma_summary).strip() if figma_summary else "(none)"
    figma_overview_general = context_builders.build_figma_overview_generalized(figma_overview)
    recent_tests = brief.get("recent_tests", []) if isinstance(brief, dict) else []
    recent_tests_exact = context_builders.recent_tests_exact(recent_tests)
    screen_index = brief.get("screen_index", []) if isinstance(brief, dict) else []

    cmap = coverage_map or {}
    coverage_hint = (
        f"Coverage hint — hot spots (repeated failures): {cmap.get('hot_spots', [])[:3] or 'none'}; "
        f"unexplored areas: {cmap.get('uncovered_purposes', [])[:4] or 'none'}; "
        f"overall coverage: {cmap.get('coverage_pct', '?')}%"
    )

    if retrieval_round <= 1:
        global_context_block = (
            "Full global context (round 1):\n"
            f"SRS summary:\n{srs_full}\n\n"
            f"UI summary:\n{figma_full}\n\n"
            f"Screen index (compact): {context_builders.screen_index_compact(screen_index)}\n"
            f"UI overview (generalized):\n{figma_overview_general}\n"
            f"Recent tests (exact): {recent_tests_exact}\n"
            f"Exploration coverage: {coverage_hint}\n\n"
        )
    else:
        global_context_block = (
            "Global memo (do not re-ask same broad context):\n"
            f"Screen index (compact): {context_builders.screen_index_compact(screen_index)}\n"
            f"Recent tests (exact): {recent_tests_exact}\n"
            f"Exploration coverage: {coverage_hint}\n"
            "Use Retrieved context so far to refine, not restart.\n\n"
        )

    response_link_line = (
        "This prompt is a response to your previous retrieval request and includes the requested DB context.\n\n"
        if retrieval_round > 1 else ""
    )

    return (
        f"You are a retrieval planner for EXPLORATORY QA test generation in {app_name}.\n"
        "You interact with a knowledge database (SRS + UI knowledge-graph + test history) to gather context.\n"
        "Your goal: retrieve context that helps generate a test case targeting real defects — "
        "prioritise unexplored areas and hot spots with repeated failures.\n"
        "Decide your NEXT ACTION only.\n\n"
        f"Objective: {objective}\n"
        f"Retrieval round: {retrieval_round}/{max_rounds}\n"
        f"Available knowledge sources for this project: {source_ids} (request ONLY these).\n"
        f"Collected queries so far: {collected_queries}\n"
        f"Collected screens so far: {collected_screens}\n"
        f"Collected context size (chars): {context_chars}\n\n"
        f"{response_link_line}"
        "Retrieved context so far (continuation from earlier requests):\n"
        f"{retrieved_notes_str}\n\n"
        f"{global_context_block}"
        "Return STRICT JSON only with this schema:\n"
        "{\n"
        '  "action": "retrieve" | "produce_testcase",\n'
        f'  "retrieval_requests": [{{"source":"{source_ids}", "query":"...", "screen":"optional"}}],\n'
        '  "focus_queries": ["...", "..."],\n'
        '  "target_screens": ["...", "..."],\n'
        '  "reason": "short reason"\n'
        "}\n"
        "Rules:\n"
        "- If more context is needed, set action=retrieve and provide explicit retrieval_requests (max 3).\n"
        "- Prefer retrieving context for HOT SPOTS and UNEXPLORED areas (see coverage hint above).\n"
        f"{source_guidance}\n"
        f"{srs_query_note}"
        "- Avoid re-requesting context you already have unless refining.\n"
        "- If context is sufficient to write a targeted test, set action=produce_testcase.\n"
        "- Never output markdown or text outside JSON."
    )


def build_testcase_prompt(
    app_name: str,
    objective: str,
    srs_context: str,
    figma_overview_context: str,
    figma_context: str,
    figma_flow_context: str,
    done_titles: list[str],
    failed_titles: list[str],
    coverage_map: dict | None = None,
    recent_tests: list[dict] | None = None,
    defect_context: str = "",
    nav_context: str = "",
    failed_nav: str = "",
) -> str:
    cmap = coverage_map or {}
    rtests = recent_tests or []
    coverage_block = coverage.build_coverage_block(cmap)
    directive = coverage.build_exploration_directive(cmap, rtests)

    parts = [
        f"You are a world-class EXPLORATORY software tester running an adaptive, session-based "
        f"testing charter on {app_name}.",
        f"Session objective: {objective}",
        "",
        "## Exploratory testing mindset — apply throughout",
        "- This is EXPLORATORY testing, NOT scripted regression: continuously learn from the SRS, the UI, "
        "and prior results, then steer toward where undiscovered defects most likely hide.",
        "- Your mission is to DISCOVER BUGS, not to confirm expected behaviour.",
        "- Test adversarially: simultaneously design, execute (in intent), and learn — each test should be "
        "the single highest-information probe for THIS moment in the session.",
        "- Stay strictly app-agnostic: rely ONLY on the SRS, UI, and history context provided below. Never "
        "assume features, screens, or rules that are not evidenced in that context.",
        "- Generate exactly ONE test case optimised to uncover a defect not yet found.",
        "",
        "## Live Coverage State",
        coverage_block,
        "",
        "## Exploration Directive — follow this priority order",
        directive,
        "",
    ]

    if srs_context:
        parts += [
            "## Business Rules & Requirements (from SRS)",
            "(Violations of these rules are bugs — verify these constraints are actually enforced by the app)",
            srs_context,
            "",
        ]
    else:
        # Bug-oracle bottom tier: no SRS/requirements available (e.g. a zero-doc app).
        # Derive correctness from UI affordances + universal robustness expectations so the
        # agent still has a notion of "what would be a bug" without a requirements source.
        parts += [
            "## Deriving Expected Behavior (no SRS/requirements available)",
            "No formal requirements were provided. Derive expected behavior from the UI affordances below and "
            "universal UX/robustness expectations: inputs are validated, every action gives feedback, navigation "
            "is reversible, state survives interruption, and the app never crashes or loses data. A violation of "
            "these is a bug. Leave 'requirement_ids' empty ([]).",
            "",
        ]
    if figma_overview_context:
        parts += ["## App Screens & UI Structure", figma_overview_context, ""]
    if figma_context:
        parts += [
            "## Interactive Elements on Relevant Screens",
            "(Use EXACT element names from the list below in your test steps — do not invent labels)",
            figma_context,
            "",
        ]
    if figma_flow_context:
        parts += ["## Screen Navigation Transitions", figma_flow_context, ""]

    # ETA-REQ-301.5 — historical defects steer generation toward fragile areas.
    if defect_context:
        parts += [
            "## Defect History Context",
            "(These areas/behaviours have broken before — prioritise a test that probes one of them "
            "or an adjacent variant. Set 'area' to the defect-prone area when it fits the objective.)",
            defect_context,
            "",
        ]

    # ETA-REQ-302.4 — proven shortest navigation path (follow it, don't re-explore).
    if nav_context:
        parts += [
            "## Learned Navigation Path",
            "(A previously-proven shortest route. Reuse these exact steps to reach the screen instead "
            "of guessing navigation.)",
            nav_context,
            "",
        ]

    # ETA-REQ-302.6 — known dead ends to avoid.
    if failed_nav:
        parts += [
            "## Known Failed Navigation Paths",
            "(These navigation steps have repeatedly failed — do NOT rely on them.)",
            failed_nav,
            "",
        ]

    history_block = "\n".join(f"- {t}" for t in done_titles[:40]) or "- none"
    failed_block = "\n".join(f"- {t}" for t in failed_titles[:20]) or "- none"

    parts += [
        "## Executed Tests — semantic duplicates are FORBIDDEN",
        history_block,
        "",
        "## Known Failures — probe adjacent cases and variants around these",
        failed_block,
        "",
        "## Exploratory Testing Heuristics — apply at least one",
        "- BOUNDARY: Test at the edges of valid input ranges (max length, empty, zero, one-off, overflow).",
        "- INVALID INPUT: Submit malformed, null, or unexpected-type data. Does the app fail gracefully?",
        "- STATE TRANSITION: Perform an action and verify the app reaches the correct subsequent state.",
        "- INTERRUPTION: Start an action, navigate away mid-flow, then return — is data/state preserved?",
        "- COMBINATION: Test interactions between two features that may have unexpected side-effects.",
        "- RECOVERY: After an error or warning, can the user correct and retry without restarting?",
        "- PERMISSION / ACCESS: Test behaviours requiring specific preconditions or data to be present.",
        "",
        "## Strict Decision Policy",
        "1. FORBIDDEN: Any test semantically similar to a title in the executed list above.",
        "2. PRIORITY: Follow the Exploration Directive — hot spots > new areas > breadth > exhausted areas.",
        "3. PREFER negative, boundary, and state-transition tests over happy-path positive tests.",
        "4. Steps MUST reference actual UI element names from the UI context — no invented labels.",
        "5. The 'rationale' field MUST name the specific defect class or risk this test is designed to expose.",
        "6. The 'area' field MUST align with the Exploration Directive — do not default to the easiest area.",
        "7. 'requirement_ids' MUST list the requirement IDs (e.g. FR-5) this test verifies, taken from the "
        "requirements context above. Use [] only if none apply.",
        "",
        "## Output — STRICT JSON only. No markdown fences, no text outside the JSON object.",
        '{"test_case_id":"...","title":"...","screen":"...","preconditions":[...],"steps":[...],'
        '"expected_result":"...","priority":"high|medium|low","area":"...",'
        '"test_type":"positive|negative|boundary|state_transition|recovery|combination",'
        '"requirement_ids":["FR-#"],'
        '"rationale":"what specific bug or risk this test is designed to expose"}',
    ]
    return "\n".join(parts)
