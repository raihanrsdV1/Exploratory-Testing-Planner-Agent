"""
Anomaly detection from execution patterns (ETA-REQ-308).

Analyses the ``ExecutionLog`` history for a project and surfaces *emerging*
issues that are not yet captured as formal defects:

  * failure_rate_spike       — a sudden jump in failures for one area
  * execution_time_regression — steps taking longer than the historical average
  * new_error_type            — an error class never seen before on a screen
  * nav_path_instability      — the successful path to a test keeps changing

Detected anomalies are persisted as ``AnomalyAlert`` nodes (MERGEd on a stable
id so re-running refreshes rather than duplicates) and rendered into a prompt
block so generation is steered toward targeted investigation tests (308.2).

Pure graph logic over a Neo4j ``session``; entirely app-agnostic — every area /
screen / error string comes from the ingested execution history.
"""

from __future__ import annotations

# How many of the most-recent logs (per group) count as the "recent" window
# when comparing against the historical baseline.
RECENT_WINDOW = 5

# Tuning knobs (kept conservative so we don't cry wolf on sparse history).
_SPIKE_MIN_RECENT = 2         # need at least this many recent runs to call a spike
_SPIKE_RATE = 0.5             # recent failure rate at/above this ...
_SPIKE_DELTA = 0.3            # ... and this much higher than the baseline
_TIME_REGRESSION_FACTOR = 1.5  # recent avg duration this many x the baseline
_TIME_MIN_SAMPLES = 2          # per side, before a time regression is credible
_NAV_DISTINCT_PATHS = 3        # distinct successful paths => unstable navigation

_SEVERITY = {
    "failure_rate_spike": "high",
    "execution_time_regression": "medium",
    "new_error_type": "medium",
    "nav_path_instability": "low",
}


def _fetch_logs(session, project) -> list[dict]:
    """All execution logs for the project, oldest-first, tagged with an area.

    Area is the covered FeatureArea label when linked, else the last screen the
    run walked, else 'unknown' — so detection works with or without a graph."""
    rows = session.run(
        """
        MATCH (p:Project {name:$project})-[:HAS_EXECUTION_LOG]->(e:ExecutionLog)
        OPTIONAL MATCH (e)-[:FOR_TEST]->(t:TestCase)-[:COVERS_FEATURE]->(fa:FeatureArea)
        WITH e, collect(DISTINCT fa.label)[0] AS fa_label
        RETURN e.test_case_id AS test_case_id, e.title AS title, e.verdict AS verdict,
               e.duration_ms AS duration_ms, e.error_type AS error_type,
               e.path AS path, e.path_labels AS path_labels,
               coalesce(fa_label,
                        CASE WHEN size(coalesce(e.path_labels,[]))>0
                             THEN e.path_labels[-1] ELSE 'unknown' END) AS area,
               e.created_at AS created_at
        ORDER BY e.created_at ASC
        """,
        project=project,
    )
    return [dict(r) for r in rows]


def _group(logs: list[dict], key: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for lg in logs:
        out.setdefault(lg.get(key) or "unknown", []).append(lg)
    return out


def _split(group: list[dict]) -> tuple[list[dict], list[dict]]:
    """(recent, older) for a time-ordered group."""
    if len(group) <= RECENT_WINDOW:
        return group, []
    return group[-RECENT_WINDOW:], group[:-RECENT_WINDOW]


def _fail_rate(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    failed = sum(1 for r in rows if r.get("verdict") == "failed")
    return failed / len(rows)


def _detect(logs: list[dict]) -> list[dict]:
    """Return a list of anomaly dicts (not yet persisted)."""
    found: list[dict] = []

    # 1. Failure-rate spike per area (308.1).
    for area, group in _group(logs, "area").items():
        recent, older = _split(group)
        rr, orr = _fail_rate(recent), _fail_rate(older)
        if len(recent) >= _SPIKE_MIN_RECENT and rr >= _SPIKE_RATE and (rr - orr) >= _SPIKE_DELTA:
            found.append({
                "anomaly_type": "failure_rate_spike",
                "key": area,
                "area": area,
                "description": (
                    f"Failure rate on '{area}' jumped to {round(rr*100)}% "
                    f"over the last {len(recent)} runs (baseline {round(orr*100)}%). "
                    f"Probe this area for an emerging regression."
                ),
                "metric": round(rr, 3),
            })

    # 2. Execution-time regression per area (308.1).
    for area, group in _group(logs, "area").items():
        recent, older = _split(group)
        rd = [r["duration_ms"] for r in recent if r.get("duration_ms")]
        od = [r["duration_ms"] for r in older if r.get("duration_ms")]
        if len(rd) >= _TIME_MIN_SAMPLES and len(od) >= _TIME_MIN_SAMPLES:
            ra, oa = sum(rd) / len(rd), sum(od) / len(od)
            if oa > 0 and ra >= _TIME_REGRESSION_FACTOR * oa:
                found.append({
                    "anomaly_type": "execution_time_regression",
                    "key": area,
                    "area": area,
                    "description": (
                        f"Execution time on '{area}' regressed to {round(ra)}ms "
                        f"(historical {round(oa)}ms, {round(ra/oa, 1)}x slower). "
                        f"Investigate a performance or responsiveness defect."
                    ),
                    "metric": round(ra / oa, 3),
                })

    # 3. New error type per screen (308.1).
    for area, group in _group(logs, "area").items():
        recent, older = _split(group)
        older_errors = {r.get("error_type") for r in older if r.get("error_type")}
        recent_errors = {r.get("error_type") for r in recent if r.get("error_type")}
        # Only meaningful once a screen has some history to be "new" against.
        if older:
            for et in recent_errors - older_errors:
                found.append({
                    "anomaly_type": "new_error_type",
                    "key": f"{area}::{et}",
                    "area": area,
                    "description": (
                        f"New error type '{et}' appeared on '{area}', not seen in its history. "
                        f"Generate a targeted test to reproduce and characterise it."
                    ),
                    "metric": 1.0,
                })

    # 4. Navigation-path instability per test (308.1).
    for tcid, group in _group(logs, "test_case_id").items():
        if tcid == "unknown":
            continue
        passed_paths = {
            "→".join(r.get("path_labels") or []) or "|".join(r.get("path") or [])
            for r in group
            if r.get("verdict") in ("pass", "passed") and (r.get("path_labels") or r.get("path"))
        }
        passed_paths.discard("")
        if len(passed_paths) >= _NAV_DISTINCT_PATHS:
            found.append({
                "anomaly_type": "nav_path_instability",
                "key": tcid,
                "area": (group[-1].get("area") if group else "unknown"),
                "description": (
                    f"Test '{tcid}' reached success via {len(passed_paths)} different paths — "
                    f"the navigation to this feature is unstable; verify the flow is deterministic."
                ),
                "metric": float(len(passed_paths)),
            })

    return found


def detect_anomalies(session, project, now) -> list[dict]:
    """Detect + persist AnomalyAlert nodes; return them ranked by severity.

    Replace-semantics: the persisted set is rebuilt to reflect the *current*
    execution history, so anomalies that have since resolved don't linger as
    stale alerts (they would otherwise MERGE-persist forever)."""
    logs = _fetch_logs(session, project)
    anomalies = _detect(logs)

    # Clear the project's previous alerts so the set mirrors current reality.
    session.run(
        "MATCH (p:Project {name:$project})-[:HAS_ANOMALY]->(al:AnomalyAlert) DETACH DELETE al",
        project=project,
    )

    sev_rank = {"high": 3, "medium": 2, "low": 1}
    out = []
    for a in anomalies:
        severity = _SEVERITY.get(a["anomaly_type"], "low")
        alert_id = f"{project}::anomaly::{a['anomaly_type']}::{a['key']}"
        session.run(
            """
            MERGE (p:Project {name:$project})
            MERGE (al:AnomalyAlert {id:$id})
            ON CREATE SET al.first_detected_at=$now
            SET al.project=$project, al.anomaly_type=$atype, al.area=$area,
                al.description=$desc, al.severity=$sev, al.metric=$metric,
                al.detected_at=$now
            MERGE (p)-[:HAS_ANOMALY]->(al)
            WITH al
            OPTIONAL MATCH (fa:FeatureArea {project:$project})
                WHERE toLower(fa.label)=toLower($area)
            FOREACH (_ IN CASE WHEN fa IS NULL THEN [] ELSE [1] END |
                MERGE (al)-[:CONCERNS_AREA]->(fa))
            """,
            project=project, id=alert_id, now=now, atype=a["anomaly_type"],
            area=a["area"], desc=a["description"], sev=severity, metric=a["metric"],
        )
        out.append({
            "id": alert_id, "anomaly_type": a["anomaly_type"], "area": a["area"],
            "description": a["description"], "severity": severity,
            "metric": a["metric"], "detected_at": now,
        })

    out.sort(key=lambda x: (sev_rank.get(x["severity"], 0), x["metric"]), reverse=True)
    return out


def list_anomalies(session, project, limit: int = 20) -> list[dict]:
    """Persisted anomaly alerts, most-recent / most-severe first."""
    rows = session.run(
        """
        MATCH (p:Project {name:$project})-[:HAS_ANOMALY]->(al:AnomalyAlert)
        RETURN al.id AS id, al.anomaly_type AS anomaly_type, al.area AS area,
               al.description AS description, al.severity AS severity,
               al.metric AS metric, al.detected_at AS detected_at
        """,
        project=project,
    )
    sev_rank = {"high": 3, "medium": 2, "low": 1}
    out = [dict(r) for r in rows]
    out.sort(key=lambda x: (sev_rank.get(x.get("severity"), 0), x.get("detected_at") or ""), reverse=True)
    return out[:limit]


def anomaly_prompt_block(alerts: list[dict], limit: int = 5) -> str:
    """`## Emerging Anomalies` block content (308.2). Empty when none."""
    if not alerts:
        return ""
    return "\n".join(f"- [{a.get('severity','?')}] {a.get('description','')}" for a in alerts[:limit])
