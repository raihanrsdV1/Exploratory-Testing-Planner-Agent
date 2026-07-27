"""
Regression risk scoring (ETA-REQ-306 / WP7).

Computes a 0..1 ``regression_risk_score`` per ``FeatureArea`` from four signals
(the ones the roadmap names):

  * historical defect density in the area          (defects_mod, 0..1)
  * failed/total test-run ratio in the area
  * recency of the newest defect (decay-weighted)  (newer = riskier)
  * navigation instability (share of avoid nav nodes, project-level)

The score is persisted on the ``FeatureArea`` node so it has history and can bias
generation. App-agnostic — every input comes from the ingested graph.
"""

from __future__ import annotations

from . import learning as learning_mod

# Weights for the four risk signals (sum to 1.0).
_W_DENSITY = 0.4
_W_FAILRATIO = 0.3
_W_RECENCY = 0.2
_W_NAV = 0.1


def compute_risk_scores(session, project, now) -> list[dict]:
    """Compute + persist regression_risk_score for every feature area; return ranked."""
    # Project-level navigation instability (shared factor).
    nav = session.run(
        """
        MATCH (n:NavTreeNode {project:$project})
        RETURN count(n) AS total, sum(CASE WHEN coalesce(n.avoid,false) THEN 1 ELSE 0 END) AS avoid
        """,
        project=project,
    ).single()
    nav_total = (nav["total"] if nav else 0) or 0
    nav_instability = round((nav["avoid"] / nav_total), 4) if nav_total else 0.0

    rows = session.run(
        """
        MATCH (p:Project {name:$project})-[:HAS_FEATURE]->(fa:FeatureArea)
        OPTIONAL MATCH (fa)<-[:COVERS_FEATURE]-(t:TestCase)
        WITH fa, count(t) AS total_tests,
             sum(CASE WHEN t.last_verdict='failed' THEN 1 ELSE 0 END) AS failed_tests
        OPTIONAL MATCH (fa)<-[:AFFECTS_AREA]-(d:Defect)
        WITH fa, total_tests, failed_tests, count(d) AS defect_count, max(d.last_seen) AS newest_defect
        RETURN fa.key AS key, fa.label AS area, coalesce(fa.defect_density,0.0) AS density,
               total_tests, failed_tests, defect_count, newest_defect
        """,
        project=project,
    )

    scored = []
    for r in rows:
        total = r["total_tests"] or 0
        failed = r["failed_tests"] or 0
        fail_ratio = (failed / total) if total else 0.0
        recency = learning_mod.decay_weight(r["newest_defect"]) if r["newest_defect"] else 0.0
        density = float(r["density"] or 0.0)
        risk = (_W_DENSITY * density + _W_FAILRATIO * fail_ratio
                + _W_RECENCY * recency + _W_NAV * nav_instability)
        risk = round(max(0.0, min(1.0, risk)), 4)
        session.run(
            "MATCH (fa:FeatureArea {key:$key}) SET fa.regression_risk_score=$risk, fa.risk_updated_at=$now",
            key=r["key"], risk=risk, now=now,
        )
        scored.append({
            "area": r["area"], "regression_risk_score": risk,
            "defect_density": round(density, 4), "fail_ratio": round(fail_ratio, 4),
            "defect_recency": round(recency, 4), "nav_instability": nav_instability,
            "defect_count": r["defect_count"] or 0, "total_tests": total, "failed_tests": failed,
        })
    scored.sort(key=lambda x: x["regression_risk_score"], reverse=True)
    return scored


def risk_prompt_block(scored: list[dict], top: int = 6) -> str:
    """`## Regression Risk Assessment` block content (306.2). Empty when no signal."""
    ranked = [s for s in scored if s["regression_risk_score"] > 0][:top]
    if not ranked:
        return ""
    return "\n".join(
        f"- {s['area']} (risk {s['regression_risk_score']}: "
        f"{s['defect_count']} defects, {s['failed_tests']}/{s['total_tests']} runs failed)"
        for s in ranked
    )
