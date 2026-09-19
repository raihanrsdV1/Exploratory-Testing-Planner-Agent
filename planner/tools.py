"""
Planner tools — the knowledge graph, exposed as callable tools.

Replaces `planner/sources/`. A source returned a block of text that was stashed
in a bucket and concatenated into one large prompt at the end; the planner never
read it while deciding, so its "retrieval loop" chose each round from one-line
notes (see docs/PLANNER_REDESIGN.md §1). A tool returns its result **into the
conversation**, so the model reasons over content and fetches only what it needs.

Everything here is pure plumbing over endpoints that already exist — no new
knowledge, no new storage. Two rules hold throughout:

  * Every result is BOUNDED. There is no global prompt budget any more; each
    tool is responsible for not flooding the context on its own.
  * Tool results are DATA, never instructions. They are app content (screen
    labels, requirement text, findings written by a model) and must never be
    able to redirect the planner.

App-agnostic: every value comes from the ingested graph at call time.
"""

from __future__ import annotations

import json

import settings as _settings

from . import context_builders as _ctx, coverage as coverage_mod, rag_client
from .sources import registry as sources_registry

# Per-tool result ceilings (characters). Sized so a full loop of ~12 calls stays
# far inside the model's context while leaving room for the trajectory of the
# conversation itself.
_MAX_REQUIREMENTS = 3000
_MAX_FINDINGS = 2500
_MAX_SCREEN = 1200
_MAX_GENERIC = 2000


def _clip(text: str, limit: int) -> str:
    text = str(text or "")
    return text if len(text) <= limit else text[:limit - 20].rstrip() + "\n…[truncated]"


def _json(obj, limit: int = _MAX_GENERIC) -> str:
    return _clip(json.dumps(obj, ensure_ascii=False), limit)


# ── Tool implementations ─────────────────────────────────────────────────────
# Each takes (project, args) and returns a STRING for the model to read.

def _search_requirements(project: str, args: dict) -> str:
    query = str(args.get("query") or "").strip()
    if not query:
        return "Provide a 'query' describing the behaviour or rule you need."
    data = rag_client.get_srs_and_history(project, query, top_k=int(args.get("top_k") or 5))
    return _clip(data.get("context") or "No requirements matched that query.", _MAX_REQUIREMENTS)


def _get_screen(project: str, args: dict) -> str:
    """Real observed controls for one screen — the grounding for `screen_hint`.

    Returns the observed alternatives on a miss rather than an error, so the
    model can correct itself in the same turn instead of guessing again.
    """
    want = str(args.get("name") or "").strip()
    if not want:
        return "Provide a screen 'name'."
    graph = rag_client.rag_get("/appmodel/graph", {"project": project})
    nodes = [n for n in (graph.get("nodes") or []) if not n.get("stale")] or (graph.get("nodes") or [])
    if not nodes:
        return "The app model is empty — no screens have been observed yet. Do not name a screen_hint; use 'unknown'."

    low = want.lower()
    exact = [n for n in nodes if (n.get("label") or "").lower() == low]
    partial = [n for n in nodes if low in (n.get("label") or "").lower()]
    hit = (exact or partial or [None])[0]
    if hit is None:
        names = [n.get("label") for n in nodes[:15]]
        return (f"No observed screen matches '{want}'. Screens actually observed on this app: "
                f"{json.dumps(names, ensure_ascii=False)}. Use one of these, or 'unknown'.")
    return _json({
        "label": hit.get("label"),
        "visits": hit.get("visits"),
        "control_count": hit.get("elements"),
        "has_dialog": hit.get("has_dialog"),
        "controls": (hit.get("controls") or [])[:15],
    }, _MAX_SCREEN)


def _list_screens(project: str, args: dict) -> str:
    graph = rag_client.rag_get("/appmodel/graph", {"project": project})
    nodes = [n for n in (graph.get("nodes") or []) if not n.get("stale")] or (graph.get("nodes") or [])
    return _json([{"label": n.get("label"), "visits": n.get("visits"),
                   "controls": n.get("elements")} for n in nodes[:25]], _MAX_GENERIC)


def _list_findings(project: str, args: dict) -> str:
    """What previous runs established. `group` is resolved server-side so the
    kind taxonomy has exactly one definition (see rag_api/findings.py)."""
    params = {"project": project, "limit": int(args.get("limit") or 12),
              "group": str(args.get("group") or "oracle")}
    if args.get("screen"):
        params["screens"] = str(args["screen"])
    data = rag_client.rag_get("/findings", params)
    rows = data.get("findings") or []
    if not rows:
        return "No findings recorded yet for that filter — this is early in the campaign."
    # Say how much was NOT shown, and be honest about how much actually fits:
    # each finding serialises to ~320 chars against a 2,500-char cap, so roughly
    # seven reach the model however many the query returned. Reporting the DB
    # limit here would itself be a lie — the count has to be the number that
    # survives truncation, or the model trusts a figure it never saw.
    slim = [{"ref": f.get("ref"), "kind": f.get("kind"), "claim": f.get("claim"),
             "screen": f.get("screen"), "times_seen": f.get("times_seen")} for f in rows]
    shown = slim
    while shown and len(_json(shown, _MAX_FINDINGS * 10)) > _MAX_FINDINGS:
        shown = shown[:-1]
    body = _json(shown or slim[:1], _MAX_FINDINGS)
    total = data.get("total") or len(rows)
    if total > len(shown):
        body += (f"\n[showing {len(shown)} of {total} findings for this project — narrow with "
                 f"screen= or group= to see the rest]")
    return body


def _findings_summary(project: str, args: dict) -> str:
    """The whole finding graph in one bounded view, instead of a truncated list.

    A finding costs ~320 characters, so a list shows about eight however many
    exist — 8 of 28 today, 8 of 300 after one full campaign. This rollup is
    bounded by screen x kind x status combinations rather than by finding count,
    so it stays roughly the same size as the graph grows while covering all of
    it. Use it to choose WHERE to look; use list_findings(screen=...) to read
    the actual claims once you have chosen.
    """
    d = rag_client.rag_get("/findings/stats", {"project": project})
    if not d.get("total"):
        return "No findings recorded yet — this is the start of the campaign."
    screens = d.get("by_screen") or []
    return _json({
        "total_findings": d.get("total"),
        "by_status": {r["status"]: r["n"] for r in (d.get("by_status") or [])},
        "by_kind": {r["kind"]: r["n"] for r in (d.get("by_kind") or [])},
        "by_screen": [{"screen": r["screen"], "total": r["total"], "open": r["open"],
                       "defects": r["defects"], "agent_trouble": r["agent_trouble"]}
                      for r in screens[:15]],
        "screens_shown": min(len(screens), 15), "screens_total": len(screens),
    }, _MAX_FINDINGS)


def _list_open_questions(project: str, args: dict) -> str:
    """Questions previous runs raised but never settled.

    The point of the whole lifecycle: an UNVERIFIED finding ("the run never
    established whether the list loads") has a definite answer, and closing it is
    usually higher-value than opening a brand-new area — a half-finished
    investigation is knowledge the campaign has already partly paid for.
    """
    data = rag_client.rag_get("/findings/open", {"project": project,
                                                 "limit": int(args.get("limit") or 8)})
    rows = data.get("open_questions") or []
    if not rows:
        return "No open questions — every finding so far has reached a conclusion."
    return _json({
        "max_attempts": data.get("max_attempts"),
        "questions": [{"ref": q.get("ref"), "kind": q.get("kind"), "screen": q.get("screen"),
                       "claim": q.get("claim"), "attempts": q.get("attempts"),
                       "attempts_left": q.get("attempts_left"),
                       "evidence": (q.get("evidence") or [])[:1]} for q in rows],
    }, _MAX_FINDINGS)


def _get_coverage(project: str, args: dict) -> str:
    brief = rag_client.get_brief_context(project)
    recent = brief.get("recent_tests", []) if isinstance(brief, dict) else []
    screens = brief.get("screen_index", []) if isinstance(brief, dict) else []
    cmap = coverage_mod.compute_coverage_map(
        recent, screens, _ctx.requirement_feature_areas(project))
    return _json({
        "tests_executed": cmap.get("total_tests"),
        "area_coverage_pct": cmap.get("coverage_pct"),
        "areas_measured_against": cmap.get("area_source"),
        "hot_spots_repeated_app_failures": cmap.get("hot_spots"),
        "dead_ends_we_cannot_reach": cmap.get("dead_ends"),
        "exhausted_areas_stop_testing": cmap.get("exhausted_areas"),
        "untested_areas": cmap.get("uncovered_purposes"),
        "directive": coverage_mod.build_exploration_directive(cmap, recent),
    }, _MAX_GENERIC)


def _list_untested_requirements(project: str, args: dict) -> str:
    """Requirements with no COVERS edge — the coverage frontier, and the only
    ids worth citing on a test meant to broaden coverage."""
    data = rag_client.get_requirement_coverage(project)
    rows = data.get("uncovered_requirements") or []
    feature = str(args.get("feature") or "").strip().lower()
    if feature:
        rows = [r for r in rows if feature in str(r.get("feature", "")).lower()]
    # Never-covered first. A requirement covered in an EARLIER campaign is a
    # re-test, not new ground — the COVERS edge died with that campaign's tests,
    # so without this flag the planner cannot tell the two apart and keeps
    # re-suggesting work it has already done.
    return _json({
        "total_requirements": data.get("total_requirements"),
        "covered_this_campaign": data.get("covered_requirements"),
        "ever_covered": data.get("ever_covered_requirements"),
        "uncovered": [{"id": r.get("ref_id"), "feature": r.get("feature"),
                       "text": r.get("text"),
                       **({"note": f"covered in an earlier campaign ({r.get('covered_count')}x) — "
                                   f"a re-test, not new ground"} if r.get("ever_covered") else {})}
                      for r in rows[:20]],
    }, _MAX_REQUIREMENTS)


def _get_nav_path(project: str, args: dict) -> str:
    screen = str(args.get("screen") or "").strip()
    if not screen:
        return "Provide a 'screen' to route to."
    try:
        data = rag_client.rag_get("/navtree/retrieve-path", {"project": project, "screen": screen})
    except Exception:
        return "No learned route available."
    steps = data.get("steps") or []
    if not steps:
        return f"No proven route to '{screen}' has been walked yet."
    return _json([{"action": s.get("action"), "screen": s.get("screen")} for s in steps], _MAX_GENERIC)


# ── Registry ─────────────────────────────────────────────────────────────────
# name -> (OpenAI-style JSON schema, implementation, required knowledge source)
#
# The `source` field is what enforces settings.ENABLED_SOURCES. In the old design
# a disabled source had to be filtered in two places, because content that
# bypassed the registry still reached the prompt. Here a disabled source's tool
# is simply NOT REGISTERED, so it cannot be called at all — strictly safer.

_TOOLS: dict[str, dict] = {
    "search_requirements": {
        "source": "srs",
        "impl": _search_requirements,
        "schema": {
            "description": "Search the app's ingested requirements (semantic + keyword). Use this to find the rules, validation constraints and error conditions that govern a behaviour, and to get exact requirement ids you may cite.",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string", "description": "Natural-language description of the behaviour or rule you need, e.g. 'validation when saving a farm with an empty name'."},
                "top_k": {"type": "integer", "description": "How many passages to return (default 5)."},
            }, "required": ["query"]},
        },
    },
    "list_untested_requirements": {
        "source": "srs",
        "impl": _list_untested_requirements,
        "schema": {
            "description": "Requirements that no executed test covers yet — the coverage frontier. Use this when the directive says to broaden coverage.",
            "parameters": {"type": "object", "properties": {
                "feature": {"type": "string", "description": "Optional feature-area filter, e.g. 'animal_record'."},
            }, "required": []},
        },
    },
    "get_screen": {
        "source": "live_ui",
        "impl": _get_screen,
        "schema": {
            "description": "The real controls observed on one screen of the running app. ALWAYS call this before naming a screen in a test: a screen you have not confirmed here probably does not exist, and the executor will waste its whole step budget looking for it.",
            "parameters": {"type": "object", "properties": {
                "name": {"type": "string", "description": "Screen name or label to look up."},
            }, "required": ["name"]},
        },
    },
    "list_screens": {
        "source": "live_ui",
        "impl": _list_screens,
        "schema": {
            "description": "Every screen the agent has actually observed on this app, with visit counts. Use it to see what exists before choosing a target.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    "list_findings": {
        "source": "",  # always available: findings come from execution, not ingestion
        "impl": _list_findings,
        "schema": {
            "description": "What previous runs already established about this app. group='oracle' gives confirmed behaviours, suspected defects, spec gaps and unverified items; 'agent' gives places OUR OWN agent struggled (never app defects); 'ui' gives discovered controls. Use this to avoid re-testing something already settled.",
            "parameters": {"type": "object", "properties": {
                "screen": {"type": "string", "description": "Optional: only findings about this screen."},
                "group": {"type": "string", "enum": ["oracle", "defect", "agent", "ui"], "description": "Which family of findings (default 'oracle')."},
                "limit": {"type": "integer"},
            }, "required": []},
        },
    },
    "findings_summary": {
        "source": "",
        "impl": _findings_summary,
        "schema": {
            "description": (
                "A count-only overview of everything previous runs established: totals by kind "
                "and status, and per screen how many findings are open, how many are candidate "
                "defects, and how often OUR OWN agent struggled there. Covers every finding, "
                "unlike list_findings which shows only about eight before its output is capped. "
                "Use this first to decide WHICH screen or area is worth attention, then "
                "list_findings(screen=...) to read the actual claims there."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    "list_open_questions": {
        "source": "",
        "impl": _list_open_questions,
        "schema": {
            "description": (
                "Questions earlier runs raised but never settled — things the app might be "
                "doing wrong, or that a run set out to check and never actually checked. "
                "Closing one of these is usually worth more than opening a brand-new area, "
                "because the campaign has already partly paid for it. Each has an attempts "
                "budget; once it runs out the question is closed as inconclusive for a human. "
                "If you target one, pass its ref as 'addresses' to propose_test_case."
            ),
            "parameters": {"type": "object", "properties": {
                "limit": {"type": "integer"},
            }, "required": []},
        },
    },
    "get_coverage": {
        "source": "",
        "impl": _get_coverage,
        "schema": {
            "description": "Live exploration state: how many tests have run, which areas repeatedly fail (hot spots), which are unreachable dead ends, which are exhausted, and the current exploration directive to follow.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    "get_nav_path": {
        "source": "navtree",
        "impl": _get_nav_path,
        "schema": {
            "description": "A navigation route to a screen that the executor has actually walked before. Use it to confirm a screen is reachable.",
            "parameters": {"type": "object", "properties": {
                "screen": {"type": "string"},
            }, "required": ["screen"]},
        },
    },
}


def available(brief: dict) -> list[str]:
    """Tool names usable for this project: registered, enabled, and with data.

    A tool whose backing source is disabled or un-ingested is not offered at
    all, so the model cannot call it and cannot be misled by an empty result.
    """
    names = []
    for name, spec in _TOOLS.items():
        src = spec["source"]
        if not src:
            names.append(name)
            continue
        if not sources_registry.is_enabled(src):
            continue
        source = sources_registry.get(src)
        if source is not None and source.is_available(brief or {}):
            names.append(name)
    return names


def schemas(names: list[str]) -> list[dict]:
    """OpenAI/OpenRouter tool schemas for the named tools."""
    out = []
    for n in names:
        spec = _TOOLS.get(n)
        if spec:
            out.append({"type": "function", "function": {
                "name": n,
                "description": spec["schema"]["description"],
                "parameters": spec["schema"]["parameters"],
            }})
    return out


def call(name: str, project: str, args: dict) -> str:
    """Run one tool. Never raises — an error is a message the model can act on,
    not an exception that kills the planning round."""
    spec = _TOOLS.get(name)
    if not spec:
        return f"No such tool '{name}'."
    try:
        return spec["impl"](project, args or {})
    except Exception as exc:
        return f"Tool '{name}' failed: {str(exc)[:200]}. Continue without it."
