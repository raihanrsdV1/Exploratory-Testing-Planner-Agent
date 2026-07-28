"""
Test-effectiveness metrics + semantic duplicate detection (ETA-REQ-307).

  * 307.1 — per-test-case effectiveness, computed from real run history and
            persisted on the ``TestCase`` node:
              - defect_discovery_rate  (share of runs that exposed a defect)
              - execution_stability    (verdict consistency; low == flaky)
              - coverage_contribution  (requirements + areas the test exercises)
  * 307.3 — embedding-cosine duplicate detection, augmenting the planner's
            Jaccard pre-filter with true semantic similarity.

Pure graph logic over a Neo4j ``session``; app-agnostic. Strategy-score feedback
(307.2) already lives in ``learning.record_strategy`` (reinforced on defect-
finding runs), so it is not duplicated here.
"""

from __future__ import annotations

import math
from collections import Counter

from . import embeddings


# ── 307.1 test effectiveness ────────────────────────────────────────────────

def compute_test_effectiveness(session, project, now) -> list[dict]:
    """Compute + persist effectiveness metrics for every test case; return ranked."""
    rows = session.run(
        """
        MATCH (p:Project {name:$project})-[:HAS_TEST]->(t:TestCase)
        OPTIONAL MATCH (t)<-[:FOR_TEST]-(e:ExecutionLog)
        WITH t, collect(e.verdict) AS verdicts
        OPTIONAL MATCH (t)-[:COVERS]->(req:Requirement)
        WITH t, verdicts, count(DISTINCT req) AS reqs
        OPTIONAL MATCH (t)-[:COVERS_FEATURE]->(fa:FeatureArea)
        RETURN t.id AS id, t.title AS title, t.area AS area, t.external_id AS external_id,
               verdicts, reqs, count(DISTINCT fa) AS areas
        """,
        project=project,
    )
    out = []
    for r in rows:
        verdicts = [v for v in (r["verdicts"] or []) if v]
        total = len(verdicts)
        counts = Counter(verdicts)
        failed = counts.get("failed", 0)

        # A 'failed' verdict in this system means the test exposed a defect.
        discovery = round(failed / total, 3) if total else 0.0
        # Stability = verdict consistency (dominant verdict share); a test that both
        # passes and fails across runs is flaky. Robust to verdict spelling (pass/passed).
        stability = round(max(counts.values()) / total, 3) if total else 1.0
        contribution = int((r["reqs"] or 0) + (r["areas"] or 0))

        session.run(
            """
            MATCH (t:TestCase {id:$id})
            SET t.defect_discovery_rate=$disc, t.execution_stability=$stab,
                t.coverage_contribution=$contrib, t.metrics_updated_at=$now
            """,
            id=r["id"], disc=discovery, stab=stability, contrib=contribution, now=now,
        )
        out.append({
            "test_case_id": r["external_id"] or r["id"], "title": r["title"], "area": r["area"],
            "runs": total, "defect_discovery_rate": discovery,
            "execution_stability": stability, "coverage_contribution": contribution,
        })
    out.sort(key=lambda x: (x["defect_discovery_rate"], x["coverage_contribution"]), reverse=True)
    return out


# ── 307.3 semantic duplicate detection ──────────────────────────────────────

def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def dedup_check(session, project, title, threshold: float = 0.9) -> dict:
    """Embed ``title`` and compare against stored TestCase embeddings.

    Returns the most-similar existing test and whether it is a semantic duplicate.
    Degrades to ``enabled=False`` (never a false positive) when embeddings are off."""
    title = (title or "").strip()
    if not title or not embeddings.is_enabled():
        return {"enabled": False, "is_duplicate": False, "similarity": 0.0, "most_similar_title": ""}

    vec = embeddings.embed_query(title)
    if not vec:
        return {"enabled": False, "is_duplicate": False, "similarity": 0.0, "most_similar_title": ""}

    rows = session.run(
        """
        MATCH (p:Project {name:$project})-[:HAS_TEST]->(t:TestCase)
        WHERE t.embedding IS NOT NULL AND t.title <> $title
        RETURN t.title AS title, t.embedding AS embedding
        """,
        project=project, title=title,
    )
    best_title, best_sim = "", 0.0
    for r in rows:
        sim = _cosine(vec, r["embedding"])
        if sim > best_sim:
            best_sim, best_title = sim, r["title"]

    return {
        "enabled": True,
        "is_duplicate": best_sim >= threshold,
        "similarity": round(best_sim, 4),
        "most_similar_title": best_title,
        "threshold": threshold,
    }
