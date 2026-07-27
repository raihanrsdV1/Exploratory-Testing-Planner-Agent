"""
Multi-dimensional knowledge-graph partitioning (ETA-REQ-304 / WP6).

Partitions and filters graph content along three dimensions — **profile**
(mobile/tv/fhub/watch), **platform** (android/windows/tizen) and **application**
(contacts/settings/...). All optional and additive: content with no dimension
tag is dimension-agnostic and matches every filter, so existing single-dimension
behaviour is unchanged (B6 backward-compat).

Design:
  * Dimensions are stored as node **properties** (`n.profile/platform/application`)
    for cheap equality filtering, AND as explicit dimension nodes
    (`Profile`/`Platform`/`Application`) linked from the Project for the graph
    model (304.2). App-agnostic — the taxonomy is open, values come from callers.
  * `where_clause` implements graceful degradation: untagged content (property IS
    NULL) always matches.
  * Cross-dimensional transfer (304.4): a test valid for one (profile, platform)
    combo can transfer to another combo of the **same application** with a
    confidence that falls as more dimensions differ.
"""

from __future__ import annotations

DIMENSIONS = ("profile", "platform", "application")
_DIM_LABEL = {"profile": "Profile", "platform": "Platform", "application": "Application"}
_DIM_REL = {"profile": "TARGETS_PROFILE", "platform": "TARGETS_PLATFORM", "application": "TARGETS_APPLICATION"}


def clean_dims(profile: str = "", platform: str = "", application: str = "") -> dict:
    """Normalise dimension inputs into a dict of only the non-empty ones."""
    raw = {"profile": profile, "platform": platform, "application": application}
    return {k: str(v).strip().lower() for k, v in raw.items() if v and str(v).strip()}


def register(session, project, dims: dict, now) -> dict:
    """MERGE the dimension nodes and TARGETS_* edges from the project (304.2)."""
    if not dims:
        return {}
    for dim, value in dims.items():
        label, rel = _DIM_LABEL[dim], _DIM_REL[dim]
        session.run(
            f"""
            MERGE (p:Project {{name:$project}})
            ON CREATE SET p.created_at=$now
            MERGE (d:{label} {{name:$value}})
            SET d.updated_at=$now
            MERGE (p)-[:{rel}]->(d)
            """,
            project=project, value=value, now=now,
        )
    return dims


def tag_props(dims: dict) -> dict:
    """Property map to SET dimension tags directly on a content node."""
    return dict(dims)


def tag_project_content(session, project, label: str, dims: dict) -> int:
    """Tag every node of a label in a project with the given dimensions.

    Safe for content that is fully replaced on ingest (Chunk/Requirement/
    FigmaScreen). Do NOT use for additive content (Defect/TestCase) — tag those
    per-node as they are written."""
    if not dims:
        return 0
    row = session.run(
        f"MATCH (n:{label} {{project:$project}}) SET n += $props RETURN count(n) AS c",
        project=project, props=dims,
    ).single()
    return (row["c"] if row else 0) or 0


def where_clause(alias: str, dims: dict) -> tuple[str, dict]:
    """Cypher fragment filtering `alias` to matching dimensions (untagged matches all).

    Returns ('', {}) when no dimensions are requested (no filtering)."""
    parts, params = [], {}
    for dim in DIMENSIONS:
        v = dims.get(dim)
        if v:
            parts.append(f"({alias}.{dim} IS NULL OR {alias}.{dim} = $dim_{dim})")
            params[f"dim_{dim}"] = v
    return (" AND ".join(parts), params)


def list_dimensions(session, project) -> dict:
    """Dimensions a project targets + how much content carries each tag."""
    out: dict = {"project": project}
    for dim in DIMENSIONS:
        rel = _DIM_REL[dim]
        rows = session.run(
            f"""
            MATCH (p:Project {{name:$project}})-[:{rel}]->(d)
            RETURN d.name AS name
            ORDER BY name
            """,
            project=project,
        )
        out[dim] = [r["name"] for r in rows]
    return out


def target_environment_text(dims: dict) -> str:
    """`## Target Environment` prompt block content (304.5). Empty when no dims."""
    if not dims:
        return ""
    order = [("application", "Application"), ("platform", "Platform"), ("profile", "Profile")]
    lines = [f"- {label}: {dims[k]}" for k, label in order if dims.get(k)]
    hint = {
        "watch": "small round/rectangular screen, rotary/crown input, terse UI.",
        "tv": "10-foot UI, D-pad/remote navigation, no touch.",
        "mobile": "touch gestures, portrait phone screen.",
        "fhub": "fitness-hub surface — glanceable panels.",
    }.get(dims.get("profile", ""), "")
    if hint:
        lines.append(f"- Interaction model: {hint}")
    return "\n".join(lines)


def transfer_suggestions(session, project, target: dict, limit: int = 10, materialize: bool = True, now: str = "") -> list[dict]:
    """Cross-dimensional transfer (304.4): tests from a different (profile/platform)
    of the SAME application that could apply to the target environment.

    A suggestion means the target combo has no equivalent test yet (untested combo).
    When `materialize`, a `MAY_APPLY_TO {transfer_confidence}` edge is created from
    the source test to the target Application node (annotated with the target env)."""
    if not target:
        return []
    rows = session.run(
        """
        MATCH (p:Project {name:$project})-[:HAS_TEST]->(t:TestCase)
        WHERE t.profile IS NOT NULL OR t.platform IS NOT NULL OR t.application IS NOT NULL
        RETURN coalesce(t.external_id, t.id) AS id, t.id AS uid, t.title AS title, t.area AS area,
               t.profile AS profile, t.platform AS platform, t.application AS application
        """,
        project=project,
    )
    tests = [dict(r) for r in rows]

    # Titles already covered in the exact target combo → not "untested".
    covered_titles = {
        t["title"] for t in tests
        if all((not target.get(d)) or t.get(d) == target.get(d) for d in DIMENSIONS)
    }

    suggestions = []
    for t in tests:
        # Same application (or target app unspecified) — never transfer across apps.
        if target.get("application") and t.get("application") != target.get("application"):
            continue
        # Which profile/platform dims differ from the target?
        differs = [d for d in ("profile", "platform")
                   if target.get(d) and t.get(d) and t.get(d) != target.get(d)]
        if not differs:
            continue  # same env — not a transfer
        if t["title"] in covered_titles:
            continue  # target combo already has this test
        confidence = round(max(0.3, 0.9 - 0.25 * len(differs)), 2)
        suggestions.append({
            "test_id": t["id"], "title": t["title"], "area": t["area"],
            "source_env": {d: t.get(d) for d in DIMENSIONS if t.get(d)},
            "target_env": dict(target), "differs_on": differs,
            "transfer_confidence": confidence,
        })
        if materialize and target.get("application"):
            session.run(
                """
                MATCH (t:TestCase {id:$uid})
                MERGE (a:Application {name:$app})
                MERGE (t)-[m:MAY_APPLY_TO]->(a)
                SET m.transfer_confidence=$conf, m.target_platform=$plat,
                    m.target_profile=$prof, m.updated_at=$now
                """,
                uid=t["uid"], app=target["application"], conf=confidence,
                plat=target.get("platform", ""), prof=target.get("profile", ""), now=now,
            )
    suggestions.sort(key=lambda s: s["transfer_confidence"], reverse=True)
    return suggestions[:limit]
