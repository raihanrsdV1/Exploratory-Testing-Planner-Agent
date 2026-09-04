"""
Format knowledge-graph data into compact prompt blocks.

These are app-agnostic: every label/screen/area comes from the ingested graph.
"""

from __future__ import annotations

import settings as _settings
from . import rag_client


def _failure_reason(notes: str) -> str:
    """The actionable part of an execution note.

    The executor writes "Droidrun execution completed in 47.6s. Steps taken: 7.
    Success=False. Reason: <what actually went wrong>" — only the tail matters.
    """
    text = str(notes or "").strip()
    if "Reason: " in text:
        text = text.split("Reason: ", 1)[1]
    for cut in (" | Self-heal:",):
        if cut in text:
            text = text.split(cut, 1)[0]
    return " ".join(text.split())[:1200]


def build_failure_context(project: str, recent_tests: list[dict], objective: str = "") -> str:
    """What previous runs actually discovered, so generation can build on it.

    Without this the planner only sees the *titles* of failed tests and keeps
    re-deriving variants of a defect it already found. Combines each failure's
    real reason with the recurring ErrorPattern signatures mined from execution
    history (REQ-303.2), which were previously computed but never fed back in.

    The findings list prefers SEMANTIC retrieval (ranked by relevance to
    ``objective``, via the testrun_embedding vector index) over the plain
    recency scan ``recent_tests`` already is (a `LIMIT 100` Cypher read, not
    RAG). This is what lets a long execution history steer generation by
    relevance instead of by "was it one of the last 25 to run" — falls back
    to the recency view when embeddings are unavailable or the call fails, so
    behaviour is unchanged for a project with EMBEDDING_BACKEND=none.
    """
    lines: list[str] = []
    if objective:
        try:
            data = rag_client.get_relevant_failure_notes(project, objective, top_k=15)
        except Exception:
            data = {"enabled": False}
        if data.get("enabled"):
            for n in data.get("notes") or []:
                title = str(n.get("title", "")).strip()
                if not title:
                    continue
                reason = _failure_reason(n.get("notes", ""))
                lines.append(f"- {title}\n    → what happened: {reason or 'no reason recorded'}")

    if not lines:
        for t in recent_tests:
            if str(t.get("verdict", "")).lower() != "failed":
                continue
            reason = _failure_reason(t.get("notes", ""))
            title = str(t.get("title", "")).strip()
            if not title:
                continue
            lines.append(f"- {title}\n    → what happened: {reason or 'no reason recorded'}")
            if len(lines) >= 25:
                break

    patterns: list[str] = []
    try:
        data = rag_client.rag_get("/execution/error-patterns", {"project": project})
        for p in (data.get("error_patterns") or [])[:20]:
            freq = p.get("frequency")
            patterns.append(
                f"- {p.get('error_type', '?')} recurring on '{p.get('screen', '?')}'"
                f" ({freq}x) — {p.get('suggested_mitigation', '')}"
            )
    except Exception:
        patterns = []

    out: list[str] = []
    if lines:
        out += ["Confirmed findings from executed tests (do NOT re-test the same defect — "
                "probe a DIFFERENT rule, screen or interaction instead):", *lines]
    if patterns:
        out += ["", "Recurring failure patterns across runs:", *patterns]
    return "\n".join(out)


def build_agent_difficulty_context(project: str) -> str:
    """'Known Agent Difficulty' block (docs/PLANNER_IMPROVEMENTS_FUTURE.md #1/#2).

    Deliberately NOT defect evidence — a screen the agent struggles to finish on
    says nothing about whether the app works. This steers test DESIGN only: write
    a narrower test there, and scope new tests to the step budget an area
    typically needs. Kept as its own block (never merged into build_failure_context
    or defect_context) so it can never be mistaken for "the app is broken here".
    """
    try:
        data = rag_client.get_agent_difficulty(project)
    except Exception:
        return ""

    lines: list[str] = []
    screens = data.get("difficulty_screens") or []
    for s in screens[:5]:
        lines.append(
            f"- {s.get('screen', '?')}: {s.get('occurrences', 0)} recent runs stalled/timed out here "
            f"({', '.join(s.get('error_types', []))}). Prefer a narrower, single-action test on this "
            f"screen rather than a full multi-field flow."
        )

    areas = data.get("step_cost_by_area") or []
    for a in areas[:5]:
        lines.append(
            f"- Typical step cost: '{a.get('area', '?')}' tests average {a.get('avg_steps', '?')} steps "
            f"(median {a.get('median_steps', '?')}, {a.get('sample_count', 0)} runs) — keep new tests "
            f"here scoped to fit the remaining step budget."
        )
    return "\n".join(lines)


def pick_relevant_screens(screens: list[dict], done_areas: list[str], recent_tests: list[dict]) -> list[str]:
    """Choose up to 2 screens to detail, biased toward untested, interaction-rich screens."""
    if not screens:
        return []
    tested_areas = {str(t.get("area", "")).lower().replace(" ", "_") for t in recent_tests}
    tested_areas.update(a.lower().replace(" ", "_") for a in done_areas)
    untested = [s for s in screens if s.get("purpose", "other") not in tested_areas]
    chosen = untested if untested else screens
    chosen = sorted(chosen, key=lambda s: s.get("interactive_count", 0), reverse=True)
    return [s["screen_name"] for s in chosen[:2]]


def build_figma_context(project: str, screen_names: list[str]) -> str:
    """Fetch + format interactive elements for the chosen screens (exact labels for steps)."""
    if not screen_names:
        return ""
    lines: list[str] = []
    for name in screen_names:
        elements = rag_client.get_screen_elements(project, name)
        if not elements:
            continue
        lines.append(f"[Screen: {name}]")
        for kind, labels in elements.items():
            lines.append(f"  {kind}s: {', '.join(labels[:10])}")
    return "\n".join(lines)


def target_environment_text(dims: dict) -> str:
    """`## Target Environment` prompt block content (WP6 / 304.5). Empty when no dims."""
    if not dims:
        return ""
    order = [("application", "Application"), ("platform", "Platform"), ("profile", "Profile")]
    lines = [f"- {label}: {dims[k]}" for k, label in order if dims.get(k)]
    hint = {
        "watch": "small screen, rotary/crown input, terse UI — keep steps minimal.",
        "tv": "10-foot UI, D-pad/remote navigation, no touch.",
        "mobile": "touch gestures on a portrait phone screen.",
        "fhub": "fitness-hub surface — glanceable panels.",
    }.get(dims.get("profile", ""), "")
    if hint:
        lines.append(f"- Interaction model: {hint}")
    return "\n".join(lines)


def build_learned_context(
    project: str,
    available_names: set,
    objective: str,
    selected_screens: list[str],
    defect_blocks: list[str],
    nav_blocks: list[str],
) -> tuple[str, str, str, str, str, str, list[str]]:
    """Assemble defect / navigation / failed-path / strategy / risk / anomaly context.

    Uses whatever the retrieval loop already gathered, and best-effort fills the
    gaps from dedicated endpoints. All app-agnostic — nothing is fetched unless the
    source is available for this project. Returns (defect_context, nav_context,
    failed_nav, strategy_context, risk_context, anomaly_context, risk_areas)."""
    from .sources.navtree import format_path

    # Defects (REQ-301.5): prefer gathered blocks, else fetch a focused block.
    defect_context = "\n\n".join(dict.fromkeys(defect_blocks))[:12000]
    if not defect_context and "defects" in available_names:
        try:
            data = rag_client.rag_get("/defects/context", {"project": project, "query": objective, "top_k": 6})
            defect_context = (data.get("context") or "")[:12000]
        except Exception:
            defect_context = ""

    # Learned navigation path (REQ-302.4) to the screen(s) the planner is targeting.
    nav_context = "\n\n".join(dict.fromkeys(nav_blocks))[:10000]
    failed_nav = ""
    if "navtree" in available_names:
        if not nav_context:
            for screen in (selected_screens or [])[:1]:
                try:
                    data = rag_client.rag_get("/navtree/retrieve-path", {"project": project, "screen": screen})
                    steps = data.get("steps", []) or []
                    if steps:
                        nav_context = f"Proven shortest path to '{screen}':\n{format_path(steps)}"
                        break
                except Exception:
                    pass
        try:
            fp = rag_client.rag_get("/navtree/failed-paths", {"project": project, "limit": 8}).get("failed_paths", [])
            if fp:
                failed_nav = "\n".join(
                    f"- avoid: {s.get('action','?')} → '{s.get('screen','?')}' "
                    f"({s.get('success_count',0)}/{s.get('visit_count',0)} succeeded)"
                    for s in fp
                )
        except Exception:
            failed_nav = ""

    # WP5 (303.3/303.6): bias generation toward strategies that have found defects,
    # decay-weighted so stale wins fade. Best-effort — empty when none learned yet.
    strategy_context = ""
    try:
        strategies = rag_client.rag_get("/strategy/memory", {"project": project}).get("strategies", [])
        effective = [s for s in strategies if s.get("decayed_score", 0) > 0][:15]
        if effective:
            strategy_context = "\n".join(
                f"- {s.get('strategy_type','?')} (effectiveness {round(s.get('decayed_score',0),2)}, "
                f"found defects {s.get('times_effective',0)}/{s.get('times_applied',0)} times)"
                for s in effective
            )
    except Exception:
        strategy_context = ""

    # WP7 (REQ-306.2): bias generation toward the highest regression-risk areas.
    risk_context = ""
    risk_areas: list[str] = []
    try:
        scores = rag_client.rag_get("/risk/scores", {"project": project}).get("risk_scores", [])
        ranked = [s for s in scores if s.get("regression_risk_score", 0) > 0][:20]
        if ranked:
            risk_context = "\n".join(
                f"- {s.get('area','?')} (risk {s.get('regression_risk_score',0)}: "
                f"{s.get('defect_count',0)} defects, {s.get('failed_tests',0)}/{s.get('total_tests',0)} runs failed)"
                for s in ranked
            )
            # Already ranked by regression_risk_score (rag_api/risk.py) — feed the
            # ordered names to the exploration directive so risk actually competes
            # for [PRIORITY] instead of sitting in an informational block only.
            risk_areas = [str(s.get("area", "")).strip() for s in ranked if s.get("area")]
    except Exception:
        risk_context = ""

    # WP8 (REQ-308.2): detect + surface emerging anomalies so generation targets
    # investigation. Detection is triggered here (not only by the dashboard) so a
    # headless agent run still gets fresh anomalies from the latest execution logs.
    anomaly_context = ""
    try:
        alerts = rag_client.rag_post("/anomalies/detect", {"project": project}).get("anomalies", [])
        if alerts:
            anomaly_context = "\n".join(
                f"- [{a.get('severity','?')}] {a.get('description','')}" for a in alerts[:20]
            )
    except Exception:
        anomaly_context = ""

    return defect_context, nav_context, failed_nav, strategy_context, risk_context, anomaly_context, risk_areas


def build_figma_overview_context(figma_overview: list[dict]) -> str:
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


def build_figma_overview_generalized(figma_overview: list[dict]) -> str:
    """Generalized UI context for planning (less label-heavy, less bias)."""
    if not figma_overview:
        return "No screens available"
    lines = [f"Total screens: {len(figma_overview)}", "Screens by purpose:"]
    for s in figma_overview:
        lines.append(
            f"- {s.get('screen_name','?')} (purpose={s.get('purpose','other')}, interactive={s.get('interactive_count',0)})"
        )
    return "\n".join(lines)


def recent_tests_exact(recent_tests: list[dict], limit: int = 50) -> str:
    if not recent_tests:
        return "none"
    return "; ".join(
        f"{t.get('id','?')}|{t.get('verdict','?')}|{t.get('area','general')}|{t.get('title','')[:120]}"
        for t in recent_tests[:limit]
    )


def build_figma_flow_context(transitions: list[dict], top_n: int = 12) -> str:
    if not transitions:
        return ""
    lines = ["Known UI transitions (from prototype links / inferred):"]
    for t in transitions[:top_n]:
        lines.append(f"- {t.get('from_screen','?')} --[{t.get('via_element','?')}]--> {t.get('to_screen','?')}")
    return "\n".join(lines)


def screen_index_compact(screen_index: list[dict], limit: int = 7) -> str:
    if not screen_index:
        return "[]"
    ordered = sorted(
        screen_index,
        key=lambda s: (-int(s.get("interactive_count", 0) or 0), str(s.get("screen_name", "")).lower()),
    )
    parts = [
        f"{s.get('screen_name','?')}|purpose={s.get('purpose','other')}|interactive={s.get('interactive_count',0)}"
        for s in ordered[:limit]
    ]
    return "[" + "; ".join(parts) + "]"


def build_requirements_context(project: str, limit_rules: int = 40) -> str:
    """The requirement ids the model is allowed to cite, plus which are still untested.

    Without this the prompt ships only raw SRS prose, so the model invents
    plausible-looking ids ("FR-64") that match no Requirement node and every
    COVERS edge silently fails — which is why requirement coverage read 1/10.
    Feeding the real ref_ids makes traceability work and lets the planner aim at
    specific uncovered requirements instead of guessing per-area.
    """
    try:
        cov = rag_client.get_requirement_coverage(project)
    except Exception:
        return ""
    try:
        rules = rag_client.get_business_rules(project)
    except Exception:
        rules = []

    uncovered = cov.get("uncovered_requirements") or []
    total = cov.get("total_requirements") or 0
    covered = cov.get("covered_requirements") or 0

    # Never advertise a requirement we have forbidden. This list is framed as
    # "prefer these — each one you cover raises coverage", so an out-of-scope
    # requirement appearing here directly contradicts the session constraints
    # that forbid it — and because nothing is covered at the start of a run, the
    # forbidden ones sit at the very top as the highest-value targets. Coverage
    # pressure beat the constraint: the planner generated an account-registration
    # and a seller-only test while signed in as a buyer, and livelocked trying to
    # reach a screen that role cannot open.
    excluded = 0
    if _settings.OUT_OF_SCOPE:
        terms = [t.lower() for t in _settings.OUT_OF_SCOPE]
        kept = []
        for r in uncovered:
            hay = f"{r.get('ref_id','')} {r.get('feature','')} {r.get('text','')}".lower()
            if any(t in hay for t in terms):
                excluded += 1
            else:
                kept.append(r)
        uncovered = kept

    if not (uncovered or rules):
        return ""

    lines = [
        f"Requirement coverage: {covered}/{total} requirements have at least one test.",
        "Cite ids EXACTLY as written below in 'requirement_ids'. Ids not in this list do not exist.",
    ]
    if uncovered:
        lines.append("")
        note = f" ({excluded} out-of-scope requirement(s) withheld)" if excluded else ""
        lines.append(f"UNTESTED requirements (prefer these — each one you cover raises coverage){note}:")
        for r in uncovered[:20]:
            lines.append(f"- [{r.get('ref_id','?')}] ({r.get('feature','')}) {str(r.get('text',''))[:240]}")

    by_req: dict[str, list[str]] = {}
    for r in rules:
        by_req.setdefault(str(r.get("requirement_id", "")), []).append(str(r.get("rule", "")))
    if by_req:
        lines.append("")
        lines.append("Validation rules extracted from the SRS (violations are bugs):")
        shown = 0
        for ref, rs in by_req.items():
            if not ref or shown >= limit_rules:
                break
            for rule in rs[:4]:
                lines.append(f"- [{ref}] {rule[:200]}")
                shown += 1
    return "\n".join(lines)
