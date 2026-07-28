"""
Incremental experiential learning (ETA-REQ-303).

Turns the raw ExecutionLog / graph history into reusable intelligence:

  * ErrorPattern mining      (REQ-303.2)  — recurring failure signatures
  * StrategyMemory           (REQ-303.3)  — which test strategies find defects
  * CoverageHeatmap          (REQ-303.4)  — requirement/area/screen/element coverage
  * Session continuity       (REQ-303.5)  — resumable exploration sessions
  * Knowledge decay          (REQ-303.6)  — half-life weighting of stale knowledge

All pure graph logic over a Neo4j ``session``; app-agnostic throughout.
"""

from __future__ import annotations

import math
import os
from datetime import datetime, timezone

KNOWLEDGE_HALF_LIFE_DAYS = float(os.getenv("KNOWLEDGE_HALF_LIFE_DAYS", "90"))

# Generic, app-agnostic mitigations per error class.
_MITIGATION = {
    "navigation_failure": "Try an alternate navigation path from the nav tree; the expected screen was unreachable.",
    "element_not_found": "Wait for the screen to settle and re-locate the element, or search for a similar label.",
    "assertion_failure": "Capture the actual post-action state and log it as a potential defect.",
    "timeout": "Retry with an extended timeout; the app was slow/unresponsive on this step.",
    "crash": "Restart the app and resume from the last stable state before this step.",
    "permission_denied": "Ensure the required permission/precondition is granted before this step.",
}


# ── Knowledge decay (REQ-303.6) ──────────────────────────────────────────────

def decay_weight(iso_timestamp: str, half_life_days: float = KNOWLEDGE_HALF_LIFE_DAYS) -> float:
    """Exponential-decay weight in (0,1]; 1.0 for 'now', 0.5 after one half-life."""
    if not iso_timestamp:
        return 0.5
    try:
        ts = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return 0.5
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    age_days = (datetime.now(timezone.utc) - ts).total_seconds() / 86400.0
    if half_life_days <= 0:
        return 1.0
    return float(2 ** (-max(0.0, age_days) / half_life_days))


# ── Error pattern mining (REQ-303.2) ─────────────────────────────────────────

def mine_error_patterns(session, project, now) -> list[dict]:
    """Aggregate failed execution logs into recurring ErrorPattern nodes."""
    rows = session.run(
        """
        MATCH (p:Project {name:$project})-[:HAS_EXECUTION_LOG]->(e:ExecutionLog)
        WHERE e.verdict = 'failed'
        WITH coalesce(NULLIF(e.error_type,''),'unknown') AS error_type,
             CASE WHEN size(coalesce(e.path_labels,[]))>0 THEN e.path_labels[-1] ELSE 'unknown' END AS screen,
             collect(e.created_at) AS times, count(*) AS freq
        RETURN error_type, screen, freq, times
        ORDER BY freq DESC
        """,
        project=project,
    )
    patterns = []
    for r in rows:
        error_type = r["error_type"]
        screen = r["screen"]
        signature = f"{error_type}@{screen}"
        pat_id = f"{project}::errpat::{signature}"
        # Recency-weighted frequency via knowledge decay.
        weighted = round(sum(decay_weight(t) for t in (r["times"] or [])), 3)
        mitigation = _MITIGATION.get(error_type.lower(), "Investigate this recurring failure with a targeted probe.")
        session.run(
            """
            MERGE (ep:ErrorPattern {id:$id})
            SET ep.project=$project, ep.pattern_signature=$sig, ep.error_type=$et,
                ep.description=$desc, ep.frequency=$freq, ep.weighted_frequency=$wf,
                ep.suggested_mitigation=$mit, ep.updated_at=$now
            WITH ep
            OPTIONAL MATCH (st:UIState {project:$project}) WHERE toLower(st.label)=toLower($screen)
            FOREACH (_ IN CASE WHEN st IS NULL THEN [] ELSE [1] END | MERGE (ep)-[:MANIFESTS_ON]->(st))
            """,
            id=pat_id, project=project, sig=signature, et=error_type,
            desc=f"'{error_type}' failures recurring on '{screen}'", freq=r["freq"],
            wf=weighted, mit=mitigation, now=now, screen=screen,
        )
        patterns.append({
            "pattern_signature": signature, "error_type": error_type, "screen": screen,
            "frequency": r["freq"], "weighted_frequency": weighted, "suggested_mitigation": mitigation,
        })
    return patterns


# ── Strategy memory (REQ-303.3 / 307.2) ──────────────────────────────────────

def record_strategy(session, project, strategy_type, effective: bool, now) -> dict:
    """Reinforce/penalize a test strategy based on whether it found a defect."""
    strategy_type = (strategy_type or "unspecified").strip().lower()
    sid = f"{project}::strategy::{strategy_type}"
    row = session.run(
        """
        MERGE (s:StrategyMemory {id:$id})
        ON CREATE SET s.times_applied=0, s.times_effective=0
        SET s.project=$project, s.strategy_type=$stype, s.updated_at=$now,
            s.times_applied = coalesce(s.times_applied,0) + 1,
            s.times_effective = coalesce(s.times_effective,0) + $eff,
            s.effectiveness_score = toFloat(coalesce(s.times_effective,0) + $eff)
                                    / (coalesce(s.times_applied,0) + 1)
        RETURN s.times_applied AS applied, s.times_effective AS effective,
               s.effectiveness_score AS score
        """,
        id=sid, project=project, stype=strategy_type, eff=1 if effective else 0, now=now,
    ).single()
    return {"strategy_type": strategy_type, **(dict(row) if row else {})}


def top_strategies(session, project, limit: int = 8) -> list[dict]:
    """Strategies ranked by decay-weighted effectiveness (REQ-303.6: stale wins fade)."""
    rows = session.run(
        """
        MATCH (s:StrategyMemory {project:$project})
        RETURN s.strategy_type AS strategy_type, coalesce(s.effectiveness_score,0.0) AS effectiveness_score,
               coalesce(s.times_applied,0) AS times_applied, coalesce(s.times_effective,0) AS times_effective,
               s.updated_at AS updated_at
        """,
        project=project,
    )
    out = []
    for r in rows:
        d = dict(r)
        w = decay_weight(d.pop("updated_at", "") or "")
        d["decayed_score"] = round(d["effectiveness_score"] * w, 4)
        out.append(d)
    out.sort(key=lambda x: (x["decayed_score"], x["times_effective"]), reverse=True)
    return out[:limit]


# ── Coverage heatmap (REQ-303.4) ─────────────────────────────────────────────

def coverage_heatmap(session, project, now) -> dict:
    reqs = session.run(
        """
        MATCH (p:Project {name:$project})-[:HAS_REQUIREMENT]->(req:Requirement)
        OPTIONAL MATCH (req)<-[:COVERS]-(t:TestCase)
        RETURN count(DISTINCT req) AS total,
               count(DISTINCT CASE WHEN t IS NOT NULL THEN req END) AS covered
        """,
        project=project,
    ).single()
    areas = session.run(
        """
        MATCH (p:Project {name:$project})-[:HAS_FEATURE]->(fa:FeatureArea)
        OPTIONAL MATCH (fa)<-[:COVERS_FEATURE]-(t:TestCase)
        RETURN count(DISTINCT fa) AS total,
               count(DISTINCT CASE WHEN t IS NOT NULL THEN fa END) AS covered,
               collect(DISTINCT CASE WHEN t IS NULL THEN fa.label END) AS uncovered
        """,
        project=project,
    ).single()
    screens = session.run(
        """
        MATCH (p:Project {name:$project})
        OPTIONAL MATCH (p)-[:HAS_FIGMA]->(fs:FigmaScreen)
        OPTIONAL MATCH (p)-[:HAS_STATE]->(st:UIState)
        RETURN count(DISTINCT fs) AS total_screens, count(DISTINCT st) AS observed_screens
        """,
        project=project,
    ).single()
    elements = session.run(
        """
        MATCH (p:Project {name:$project})-[:HAS_FIGMA]->(fs:FigmaScreen)
        OPTIONAL MATCH (fs)-[:HAS_ELEMENT]->(el:UIElement)
        WITH fs, count(el) AS ec
        OPTIONAL MATCH (t:TestCase {project:$project}) WHERE toLower(t.area)=toLower(fs.purpose)
        RETURN sum(ec) AS total_elements,
               sum(CASE WHEN t IS NOT NULL THEN ec ELSE 0 END) AS covered_elements
        """,
        project=project,
    ).single()

    total_req = (reqs["total"] if reqs else 0) or 0
    covered_req = (reqs["covered"] if reqs else 0) or 0
    uncovered = [a for a in ((areas["uncovered"] if areas else []) or []) if a]

    heatmap = {
        "project": project,
        "total_requirements": total_req,
        "covered_requirements": covered_req,
        "requirement_coverage_pct": round(100 * covered_req / total_req) if total_req else 0,
        "total_areas": (areas["total"] if areas else 0) or 0,
        "covered_areas": (areas["covered"] if areas else 0) or 0,
        "total_screens": (screens["total_screens"] if screens else 0) or 0,
        "observed_screens": (screens["observed_screens"] if screens else 0) or 0,
        "total_elements": (elements["total_elements"] if elements else 0) or 0,
        "covered_elements": (elements["covered_elements"] if elements else 0) or 0,
        "uncovered_areas": uncovered,
        "last_updated": now,
    }
    session.run(
        """
        MATCH (p:Project {name:$project})
        MERGE (h:CoverageHeatmap {id:$id})
        SET h += $props
        MERGE (p)-[:HAS_HEATMAP]->(h)
        """,
        project=project, id=f"{project}::heatmap", props=heatmap,
    )
    return heatmap


# ── Session continuity (REQ-303.5) ───────────────────────────────────────────

def session_start(session, project, focus_area, strategy, now) -> dict:
    sess_id = f"{project}::session::{now}"
    # Close any previously-active session for this project.
    session.run(
        "MATCH (s:Session {project:$project, status:'active'}) SET s.status='superseded', s.ended_at=$now",
        project=project, now=now,
    )
    session.run(
        """
        MERGE (p:Project {name:$project})
        CREATE (s:Session {id:$id})
        SET s.project=$project, s.started_at=$now, s.focus_area=$focus, s.strategy=$strategy,
            s.status='active', s.tests_generated=0, s.defects_found=0, s.exploration_thread=[]
        MERGE (p)-[:HAS_SESSION]->(s)
        """,
        project=project, id=sess_id, now=now, focus=focus_area or "", strategy=strategy or "",
    )
    return {"session_id": sess_id, "project": project, "focus_area": focus_area, "strategy": strategy,
            "started_at": now, "status": "active"}


def _active_session(session, project) -> dict | None:
    row = session.run(
        """
        MATCH (p:Project {name:$project})-[:HAS_SESSION]->(s:Session {status:'active'})
        RETURN s.id AS id, s.started_at AS started_at, s.focus_area AS focus_area,
               s.strategy AS strategy
        ORDER BY s.started_at DESC LIMIT 1
        """,
        project=project,
    ).single()
    return dict(row) if row else None


def session_context(session, project, recent_limit: int = 10) -> dict:
    sess = _active_session(session, project)
    if not sess:
        return {"project": project, "active": False}
    since = sess["started_at"]
    metrics = session.run(
        """
        MATCH (p:Project {name:$project})-[:HAS_TEST]->(t:TestCase)
        WHERE t.last_run_at >= $since
        RETURN count(t) AS tests_generated,
               sum(CASE WHEN t.last_verdict='failed' THEN 1 ELSE 0 END) AS defects_found,
               collect(t.title)[0..$lim] AS recent_titles
        """,
        project=project, since=since, lim=recent_limit,
    ).single()
    return {
        "project": project, "active": True, "session_id": sess["id"],
        "focus_area": sess.get("focus_area", ""), "strategy": sess.get("strategy", ""),
        "started_at": since,
        "tests_generated": (metrics["tests_generated"] if metrics else 0) or 0,
        "defects_found": (metrics["defects_found"] if metrics else 0) or 0,
        "exploration_thread": (metrics["recent_titles"] if metrics else []) or [],
    }


def _age_seconds(iso_timestamp: str) -> float | None:
    """Seconds between now and an ISO timestamp; None if unparseable."""
    if not iso_timestamp:
        return None
    try:
        ts = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - ts).total_seconds()


# ── Live execution status (WP9 / operator dashboard) ─────────────────────────

def session_live(session, project, running_window_s: int = 120, stream: int = 15) -> dict:
    """What the agent is doing *right now*: the current/most-recent execution, a
    live verdict stream, and the active session. 'executing' is inferred from the
    recency of the latest ExecutionLog (the executor writes one per completed run)."""
    latest = session.run(
        """
        MATCH (p:Project {name:$project})-[:HAS_EXECUTION_LOG]->(e:ExecutionLog)
        RETURN e.test_case_id AS test_case_id, e.title AS title, e.verdict AS verdict,
               e.error_type AS error_type, e.recovery_action AS recovery_action,
               e.duration_ms AS duration_ms, e.device_steps AS device_steps,
               e.created_at AS created_at
        ORDER BY e.created_at DESC LIMIT 1
        """,
        project=project,
    ).single()
    recent = [dict(r) for r in session.run(
        """
        MATCH (p:Project {name:$project})-[:HAS_EXECUTION_LOG]->(e:ExecutionLog)
        RETURN e.test_case_id AS test_case_id, e.title AS title, e.verdict AS verdict,
               e.error_type AS error_type, e.created_at AS created_at
        ORDER BY e.created_at DESC LIMIT $stream
        """,
        project=project, stream=stream,
    )]

    current = dict(latest) if latest else {}
    age = _age_seconds(current.get("created_at")) if current else None
    executing = bool(current) and age is not None and age <= running_window_s

    sess = _active_session(session, project)
    ctx = session_context(session, project) if sess else {"tests_generated": 0, "defects_found": 0}

    return {
        "project": project,
        "executing": executing,
        "status": "executing" if executing else "idle",
        "current": {**current, "age_seconds": round(age) if age is not None else None} if current else {},
        "recent_verdicts": recent,
        "session_active": bool(sess),
        "session_id": (sess or {}).get("id", ""),
        "focus_area": (sess or {}).get("focus_area", ""),
        "tests_run": ctx.get("tests_generated", 0),
        "bugs_found": ctx.get("defects_found", 0),
    }


def session_end(session, project, session_id, now) -> dict:
    ctx = session_context(session, project)
    target = session_id or ctx.get("session_id", "")
    if not target:
        return {"project": project, "ended": False, "reason": "no active session"}
    session.run(
        """
        MATCH (s:Session {id:$id})
        SET s.status='ended', s.ended_at=$now,
            s.tests_generated=$tg, s.defects_found=$df, s.exploration_thread=$thread
        """,
        id=target, now=now, tg=ctx.get("tests_generated", 0), df=ctx.get("defects_found", 0),
        thread=ctx.get("exploration_thread", []),
    )
    return {"project": project, "ended": True, "session_id": target,
            "tests_generated": ctx.get("tests_generated", 0),
            "defects_found": ctx.get("defects_found", 0)}
