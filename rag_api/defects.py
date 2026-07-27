"""
Defect-history knowledge layer (ETA-REQ-301).

Pure graph logic over a Neo4j ``session`` handed in by ``main.py`` (so there is no
import cycle and the driver/embeddings helpers stay in one place). Everything here
is app-agnostic — areas, screens and requirements are resolved from the ingested
graph at write time, never hardcoded.

Graph model written:
    (:Project)-[:HAS_DEFECT]->(:Defect)
    (:Defect)-[:AFFECTS_AREA]->(:FeatureArea)
    (:Defect)-[:AFFECTS_SCREEN]->(:UIState|:FigmaScreen)   # best-effort by label
    (:Defect)-[:SIMILAR_TO]->(:Defect)                     # embedding cosine
    (:Defect)-[:DERIVED_FROM]->(:Chunk)                    # SRS rule it violates
    (:FeatureArea).defect_density  (0..1)                  # REQ-301.3
    (:Summary {kind:'defects'})                            # REQ-301.4
"""

from __future__ import annotations

import math

# Severity term → weight (higher = more fragile). App-agnostic vocabulary.
_SEVERITY_WEIGHT = {
    "blocker": 4.0, "critical": 4.0, "sev1": 4.0, "s1": 4.0, "p1": 4.0,
    "high": 3.0, "major": 3.0, "sev2": 3.0, "s2": 3.0, "p2": 3.0,
    "medium": 2.0, "normal": 2.0, "moderate": 2.0, "sev3": 2.0, "s3": 2.0, "p3": 2.0,
    "low": 1.0, "minor": 1.0, "trivial": 1.0, "sev4": 1.0, "s4": 1.0, "p4": 1.0,
}
_RESOLVED_STATES = {"closed", "resolved", "fixed", "done", "verified", "completed", "wontfix"}

_SIMILAR_THRESHOLD = 0.80  # cosine over defect embeddings to draw a SIMILAR_TO edge


def severity_weight(severity: str) -> float:
    return _SEVERITY_WEIGHT.get(str(severity or "").strip().lower(), 2.0)


def is_unresolved(status: str) -> bool:
    return str(status or "").strip().lower() not in _RESOLVED_STATES


def _cosine(a, b) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def write_defects(session, project, defects, embed_texts, slug, now) -> dict:
    """Persist a batch of canonical defect dicts and derive their relationships."""
    if not defects:
        return {"defects_written": 0}

    # Embed "title. description" for SIMILAR_TO + DERIVED_FROM.
    embed_texts_in = [f"{d.get('title','')}. {d.get('description','')}".strip() for d in defects]
    vecs = embed_texts(embed_texts_in)

    area_weight: dict[str, float] = {}
    written = 0
    for i, d in enumerate(defects):
        ref_id = str(d.get("id") or f"D{i+1}")
        defect_uid = f"{project}::defect::{slug(ref_id)}"
        area = str(d.get("area") or "general")
        area_slug = slug(area)
        feature_key = f"{project}::{area_slug}"
        vec = vecs[i] if i < len(vecs) else None
        w = severity_weight(d.get("severity"))
        freq = int(d.get("frequency") or 1)
        area_weight[area_slug] = area_weight.get(area_slug, 0.0) + w * max(1, freq)

        session.run(
            """
            MERGE (p:Project {name:$project})
            ON CREATE SET p.created_at=$now
            SET p.updated_at=$now
            MERGE (dfct:Defect {id:$uid})
            SET dfct.project=$project, dfct.ref_id=$ref_id, dfct.title=$title,
                dfct.description=$description, dfct.severity=$severity, dfct.status=$status,
                dfct.area=$area, dfct.root_cause_category=$rcc, dfct.frequency=$freq,
                dfct.first_seen=$first_seen, dfct.last_seen=$last_seen, dfct.resolution=$resolution,
                dfct.affected_screen=$affected_screen, dfct.severity_weight=$weight,
                dfct.unresolved=$unresolved, dfct.embedding=$embedding, dfct.updated_at=$now
            MERGE (p)-[:HAS_DEFECT]->(dfct)
            MERGE (fa:FeatureArea {key:$feature_key})
            SET fa.project=$project, fa.label=$area_label, fa.updated_at=$now
            MERGE (p)-[:HAS_FEATURE]->(fa)
            MERGE (dfct)-[:AFFECTS_AREA]->(fa)
            """,
            project=project, uid=defect_uid, ref_id=ref_id,
            title=str(d.get("title", "")), description=str(d.get("description", "")),
            severity=str(d.get("severity", "")), status=str(d.get("status", "")),
            area=area, rcc=str(d.get("root_cause_category", "")), freq=freq,
            first_seen=str(d.get("first_seen", "")), last_seen=str(d.get("last_seen", "")),
            resolution=str(d.get("resolution", "")),
            affected_screen=str(d.get("affected_screen", "")),
            weight=w, unresolved=is_unresolved(d.get("status")), embedding=vec,
            feature_key=feature_key, area_label=area.replace("_", " "), now=now,
        )
        written += 1

        # AFFECTS_SCREEN — best-effort match to an observed UIState or Figma screen by label.
        screen = str(d.get("affected_screen") or "").strip()
        if screen:
            session.run(
                """
                MATCH (dfct:Defect {id:$uid})
                OPTIONAL MATCH (fs:FigmaScreen {project:$project})
                  WHERE toLower(fs.screen_name) = toLower($screen)
                OPTIONAL MATCH (st:UIState {project:$project})
                  WHERE toLower(st.label) = toLower($screen)
                FOREACH (_ IN CASE WHEN fs IS NULL THEN [] ELSE [1] END | MERGE (dfct)-[:AFFECTS_SCREEN]->(fs))
                FOREACH (_ IN CASE WHEN st IS NULL THEN [] ELSE [1] END | MERGE (dfct)-[:AFFECTS_SCREEN]->(st))
                """,
                uid=defect_uid, project=project, screen=screen,
            )

        # DERIVED_FROM — link to the SRS chunk this defect most likely relates to.
        if vec:
            try:
                session.run(
                    """
                    MATCH (dfct:Defect {id:$uid})
                    CALL db.index.vector.queryNodes('chunk_embedding', 1, $vec)
                    YIELD node, score
                    WITH dfct, node, score WHERE score >= 0.75 AND node.project = $project
                    MERGE (dfct)-[r:DERIVED_FROM]->(node)
                    SET r.score = score
                    """,
                    uid=defect_uid, vec=vec, project=project,
                )
            except Exception:
                pass  # no vector index / older Neo4j / embeddings disabled

        # Explicit requirement links (if the export named requirement ids).
        for ref in (d.get("requirement_ids") or []):
            session.run(
                """
                MATCH (dfct:Defect {id:$uid})
                MATCH (req:Requirement {id:$req_uid})
                MERGE (dfct)-[:DERIVED_FROM]->(req)
                """,
                uid=defect_uid, req_uid=f"{project}::req::{str(ref).strip()}",
            )

    # SIMILAR_TO — cluster semantically-close defects (cosine over embeddings, in-process).
    sim_edges = 0
    idx_vecs = [(i, vecs[i]) for i in range(len(defects)) if i < len(vecs) and vecs[i]]
    for a in range(len(idx_vecs)):
        i, vi = idx_vecs[a]
        for b in range(a + 1, len(idx_vecs)):
            j, vj = idx_vecs[b]
            if _cosine(vi, vj) >= _SIMILAR_THRESHOLD:
                session.run(
                    """
                    MATCH (d1:Defect {id:$id1}), (d2:Defect {id:$id2})
                    MERGE (d1)-[:SIMILAR_TO]->(d2)
                    MERGE (d2)-[:SIMILAR_TO]->(d1)
                    """,
                    id1=f"{project}::defect::{slug(str(defects[i].get('id') or f'D{i+1}'))}",
                    id2=f"{project}::defect::{slug(str(defects[j].get('id') or f'D{j+1}'))}",
                )
                sim_edges += 1

    # Defect density per feature area (REQ-301.3): normalized 0..1 by the hottest area.
    max_w = max(area_weight.values()) if area_weight else 0.0
    for area_slug, w in area_weight.items():
        density = round(w / max_w, 4) if max_w else 0.0
        session.run(
            "MATCH (fa:FeatureArea {key:$key}) SET fa.defect_density=$density, fa.defect_weight=$w",
            key=f"{project}::{area_slug}", density=density, w=w,
        )

    # Project-level defect summary (REQ-301.4).
    summary_text = _build_summary_text(session, project)
    session.run(
        """
        MATCH (p:Project {name:$project})
        SET p.defect_summary=$text
        MERGE (sum:Summary {id:$sid})
        SET sum.project=$project, sum.kind='defects', sum.text=$text, sum.updated_at=$now
        MERGE (p)-[:HAS_SUMMARY]->(sum)
        """,
        project=project, sid=f"{project}::summary::defects", text=summary_text, now=now,
    )

    return {
        "defects_written": written,
        "similar_edges": sim_edges,
        "areas_scored": len(area_weight),
        "defect_summary_chars": len(summary_text),
    }


def prone_areas(session, project, limit: int = 10) -> list[dict]:
    """Top defect-prone feature areas by weighted density (REQ-301, /defects/prone-areas)."""
    rows = session.run(
        """
        MATCH (p:Project {name:$project})-[:HAS_DEFECT]->(d:Defect)-[:AFFECTS_AREA]->(fa:FeatureArea)
        WITH fa, count(d) AS defect_count,
             sum(d.severity_weight * d.frequency) AS weight,
             sum(CASE WHEN d.unresolved THEN 1 ELSE 0 END) AS unresolved
        RETURN fa.label AS area, coalesce(fa.defect_density, 0.0) AS density,
               defect_count, weight, unresolved
        ORDER BY weight DESC, defect_count DESC
        LIMIT $limit
        """,
        project=project, limit=limit,
    )
    return [dict(r) for r in rows]


def defect_summary(session, project) -> dict:
    """Structured defect summary for /defects/summary and the brief (REQ-301.4)."""
    totals = session.run(
        """
        MATCH (p:Project {name:$project})-[:HAS_DEFECT]->(d:Defect)
        RETURN count(d) AS total,
               sum(CASE WHEN d.unresolved THEN 1 ELSE 0 END) AS unresolved
        """,
        project=project,
    ).single()
    total = (totals["total"] if totals else 0) or 0

    severity_dist = {
        r["severity"]: r["c"]
        for r in session.run(
            """
            MATCH (p:Project {name:$project})-[:HAS_DEFECT]->(d:Defect)
            WITH coalesce(d.severity,'unspecified') AS severity, count(*) AS c
            RETURN severity, c ORDER BY c DESC
            """,
            project=project,
        )
    }
    root_causes = {
        r["rcc"]: r["c"]
        for r in session.run(
            """
            MATCH (p:Project {name:$project})-[:HAS_DEFECT]->(d:Defect)
            WHERE d.root_cause_category IS NOT NULL AND d.root_cause_category <> ''
            WITH d.root_cause_category AS rcc, count(*) AS c
            RETURN rcc, c ORDER BY c DESC LIMIT 8
            """,
            project=project,
        )
    }
    text_row = session.run(
        "MATCH (p:Project {name:$project}) RETURN p.defect_summary AS text", project=project
    ).single()

    return {
        "project": project,
        "total_defects": total,
        "unresolved_defects": (totals["unresolved"] if totals else 0) or 0,
        "severity_distribution": severity_dist,
        "root_cause_categories": root_causes,
        "prone_areas": prone_areas(session, project, limit=5),
        "summary_text": (text_row["text"] if text_row else "") or "",
    }


def _build_summary_text(session, project) -> str:
    """Rule-based project defect summary (no LLM needed) for the planner brief."""
    s = defect_summary_lite(session, project)
    lines = [f"Defect history: {s['total_defects']} defects ({s['unresolved']} unresolved)."]
    if s["areas"]:
        top = ", ".join(f"{a['area']} ({a['defect_count']})" for a in s["areas"][:5])
        lines.append(f"Top defect-prone areas: {top}.")
    if s["root_causes"]:
        rc = ", ".join(f"{k} ({v})" for k, v in list(s["root_causes"].items())[:5])
        lines.append(f"Recurring root causes: {rc}.")
    if s["severity"]:
        sv = ", ".join(f"{k}: {v}" for k, v in s["severity"].items())
        lines.append(f"Severity distribution: {sv}.")
    lines.append("Bias test generation toward the highest-density areas above — they historically break most.")
    return "\n".join(lines)


def defect_summary_lite(session, project) -> dict:
    """Cheap counts used by _build_summary_text (avoids recomputing the full summary)."""
    totals = session.run(
        """
        MATCH (p:Project {name:$project})-[:HAS_DEFECT]->(d:Defect)
        RETURN count(d) AS total, sum(CASE WHEN d.unresolved THEN 1 ELSE 0 END) AS unresolved
        """,
        project=project,
    ).single()
    severity = {
        r["severity"]: r["c"]
        for r in session.run(
            """
            MATCH (p:Project {name:$project})-[:HAS_DEFECT]->(d:Defect)
            WITH coalesce(d.severity,'unspecified') AS severity, count(*) AS c
            RETURN severity, c ORDER BY c DESC
            """,
            project=project,
        )
    }
    root_causes = {
        r["rcc"]: r["c"]
        for r in session.run(
            """
            MATCH (p:Project {name:$project})-[:HAS_DEFECT]->(d:Defect)
            WHERE d.root_cause_category IS NOT NULL AND d.root_cause_category <> ''
            WITH d.root_cause_category AS rcc, count(*) AS c
            RETURN rcc, c ORDER BY c DESC LIMIT 8
            """,
            project=project,
        )
    }
    return {
        "total_defects": (totals["total"] if totals else 0) or 0,
        "unresolved": (totals["unresolved"] if totals else 0) or 0,
        "areas": prone_areas(session, project, limit=6),
        "root_causes": root_causes,
        "severity": severity,
    }


def retrieve_defect_context(session, project, query: str, area: str, top_k: int = 6) -> str:
    """Formatted defect context block for the planner's DefectSource (REQ-301.5)."""
    rows = session.run(
        """
        MATCH (p:Project {name:$project})-[:HAS_DEFECT]->(d:Defect)-[:AFFECTS_AREA]->(fa:FeatureArea)
        WITH d, fa,
             (CASE WHEN $area <> '' AND toLower(fa.label) CONTAINS toLower($area) THEN 2 ELSE 0 END)
           + (CASE WHEN $q <> '' AND (toLower(d.title) CONTAINS toLower($q)
                    OR toLower(d.description) CONTAINS toLower($q)) THEN 2 ELSE 0 END)
           + (CASE WHEN d.unresolved THEN 1 ELSE 0 END) AS relevance
        RETURN d.ref_id AS ref_id, d.title AS title, d.severity AS severity,
               d.status AS status, fa.label AS area, d.root_cause_category AS rcc,
               coalesce(fa.defect_density,0.0) AS density, relevance
        ORDER BY relevance DESC, density DESC, d.severity_weight DESC
        LIMIT $limit
        """,
        project=project, q=(query or ""), area=(area or ""), limit=max(1, min(top_k, 20)),
    )
    defects = [dict(r) for r in rows]
    if not defects:
        return ""
    lines = ["Historical defects (probe these fragile behaviours and adjacent variants):"]
    for d in defects:
        lines.append(
            f"- [{d['ref_id']}] {d['title']} (severity={d.get('severity') or '?'}, "
            f"status={d.get('status') or '?'}, area={d.get('area')}"
            + (f", root_cause={d['rcc']}" if d.get("rcc") else "") + ")"
        )
    return "\n".join(lines)
