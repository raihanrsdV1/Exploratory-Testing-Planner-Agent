"""
Execution Path Memory — the navigation tree (ETA-REQ-302).

A ``NavTreeNode`` prefix-tree built from the ordered UIState paths that test
executions actually walk (recorded via /execution/log or /navtree/record-path).
Shared prefixes collapse into shared nodes, so:

  * the tree grows incrementally as new screens are reached (REQ-302.5),
  * sub-paths are reused across tests for free (REQ-302.7),
  * the shortest successful path to complete a test wins (REQ-302.3),
  * repeatedly-failing steps get ``avoid=true`` (REQ-302.6).

All app-agnostic: a "screen" is whatever UIState label the live app model
recorded; actions come from the TRANSITIONS_TO edges the observer already stored.

Graph model:
    (:Project)-[:HAS_NAV_TREE]->(:NavTreeNode)         # depth-0 entry nodes
    (:NavTreeNode)-[:CHILD]->(:NavTreeNode)
    (:NavTreeNode)-[:LEADS_TO_SCREEN]->(:UIState)
    (:NavTreeNode)-[:RESOLVES_TEST]->(:TestCase)        # terminal of shortest pass path
"""

from __future__ import annotations

import hashlib

# success_count/visit_count below this marks a node as a dead end to avoid.
_AVOID_THRESHOLD = 0.3


def _path_hash(state_ids: list[str]) -> str:
    h = hashlib.sha1("|".join(state_ids).encode("utf-8")).hexdigest()
    return h[:16]


def _reconstruct_actions(session, path: list[str]) -> list[str]:
    """Fill action labels from TRANSITIONS_TO edges when the caller didn't supply them."""
    actions = ["(entry)"]
    for k in range(1, len(path)):
        row = session.run(
            """
            MATCH (a:UIState {id:$a})-[e:TRANSITIONS_TO]->(b:UIState {id:$b})
            RETURN e.action AS action, e.element AS element
            ORDER BY e.visit_count DESC LIMIT 1
            """,
            a=path[k - 1], b=path[k],
        ).single()
        actions.append((row["action"] if row and row["action"] else "step"))
    return actions


def record_path(session, project, test_case_id, title, verdict, path, path_labels, actions, slug, now) -> dict:
    """Record one navigation path; extend/reinforce the tree; keep the shortest pass path."""
    path = [p for p in (path or []) if p]
    if not path:
        return {"nodes_touched": 0, "recorded": False}

    labels = list(path_labels or [])
    if not actions or len(actions) < len(path):
        actions = _reconstruct_actions(session, path)

    is_pass = str(verdict or "").lower() == "pass"
    internal_tc = f"{project}::tc::{slug(title)}" if title else ""

    parent_id = None
    touched = 0
    for k in range(len(path)):
        node_hash = _path_hash(path[: k + 1])
        node_id = f"{project}::nav::{node_hash}"
        screen = labels[k] if k < len(labels) else path[k]
        action = actions[k] if k < len(actions) else "step"

        session.run(
            """
            MERGE (p:Project {name:$project})
            MERGE (n:NavTreeNode {id:$node_id})
            ON CREATE SET n.first_seen=$now, n.visit_count=0, n.success_count=0
            SET n.project=$project, n.path_hash=$node_hash, n.screen_name=$screen,
                n.action=$action, n.element_label=$action, n.depth=$depth,
                n.state_id=$state_id, n.last_visited_at=$now,
                n.visit_count = coalesce(n.visit_count,0) + 1,
                n.success_count = coalesce(n.success_count,0) + $succ
            WITH p, n
            OPTIONAL MATCH (st:UIState {id:$state_id})
            FOREACH (_ IN CASE WHEN st IS NULL THEN [] ELSE [1] END | MERGE (n)-[:LEADS_TO_SCREEN]->(st))
            FOREACH (_ IN CASE WHEN $is_root THEN [1] ELSE [] END | MERGE (p)-[:HAS_NAV_TREE]->(n))
            """,
            project=project, node_id=node_id, now=now, node_hash=node_hash,
            screen=screen, action=action, depth=k, state_id=path[k],
            succ=1 if is_pass else 0, is_root=(k == 0),
        )
        if parent_id:
            session.run(
                "MATCH (a:NavTreeNode {id:$a}), (b:NavTreeNode {id:$b}) MERGE (a)-[:CHILD]->(b)",
                a=parent_id, b=node_id,
            )
        # Failed-path avoidance flag (REQ-302.6).
        session.run(
            """
            MATCH (n:NavTreeNode {id:$node_id})
            SET n.avoid = (toFloat(coalesce(n.success_count,0)) / n.visit_count) < $thr
            """,
            node_id=node_id, thr=_AVOID_THRESHOLD,
        )
        parent_id = node_id
        touched += 1

    # Shortest successful path wins for this test (REQ-302.3).
    terminal_id = f"{project}::nav::{_path_hash(path)}"
    resolved = False
    if is_pass and internal_tc:
        row = session.run(
            "MATCH (t:TestCase {id:$tc}) RETURN t.nav_path_len AS len", tc=internal_tc
        ).single()
        prev_len = (row["len"] if row and row["len"] is not None else None)
        new_len = len(path)
        if prev_len is None or new_len <= prev_len:
            session.run(
                """
                MATCH (t:TestCase {id:$tc})
                OPTIONAL MATCH (:NavTreeNode)-[old:RESOLVES_TEST]->(t)
                DELETE old
                WITH t
                MATCH (n:NavTreeNode {id:$terminal})
                MERGE (n)-[:RESOLVES_TEST]->(t)
                SET t.nav_path_len=$len
                """,
                tc=internal_tc, terminal=terminal_id, len=new_len,
            )
            resolved = True

    return {"nodes_touched": touched, "recorded": True, "resolved_test": resolved,
            "terminal_id": terminal_id, "is_pass": is_pass}


def retrieve_path(session, project, screen: str = "", test_id: str = "", slug=None) -> dict:
    """Return the shortest known-good path to a target screen or that completes a test."""
    steps: list[dict] = []
    terminal_id = None

    if test_id and slug is not None:
        internal_tc = f"{project}::tc::{slug(test_id)}"
        row = session.run(
            """
            MATCH (n:NavTreeNode)-[:RESOLVES_TEST]->(t:TestCase)
            WHERE t.id=$tc OR t.external_id=$ext
            RETURN n.id AS id ORDER BY n.depth ASC LIMIT 1
            """,
            tc=internal_tc, ext=test_id,
        ).single()
        terminal_id = row["id"] if row else None

    if terminal_id is None and screen:
        # Shortest successful node whose screen matches the target label.
        row = session.run(
            """
            MATCH (p:Project {name:$project})-[:HAS_NAV_TREE]->(:NavTreeNode)
            WITH p
            MATCH (n:NavTreeNode {project:$project})
            WHERE toLower(n.screen_name) CONTAINS toLower($screen)
              AND coalesce(n.success_count,0) > 0
            RETURN n.id AS id ORDER BY n.depth ASC, n.success_count DESC LIMIT 1
            """,
            project=project, screen=screen,
        ).single()
        terminal_id = row["id"] if row else None

    if terminal_id is None:
        return {"found": False, "steps": []}

    # Walk parents back to the root to reconstruct the ordered path.
    rows = session.run(
        """
        MATCH path = (root:NavTreeNode)-[:CHILD*0..]->(target:NavTreeNode {id:$terminal})
        WHERE (:Project {name:$project})-[:HAS_NAV_TREE]->(root)
        WITH nodes(path) AS ns ORDER BY length(path) DESC LIMIT 1
        UNWIND ns AS n
        RETURN n.depth AS depth, n.screen_name AS screen, n.action AS action,
               coalesce(n.avoid,false) AS avoid,
               coalesce(n.success_count,0) AS success_count, coalesce(n.visit_count,0) AS visit_count
        ORDER BY n.depth ASC
        """,
        project=project, terminal=terminal_id,
    )
    steps = [dict(r) for r in rows]
    return {"found": bool(steps), "steps": steps, "length": len(steps)}


def failed_paths(session, project, limit: int = 15) -> list[dict]:
    """Navigation steps to avoid — repeatedly-failing nodes (REQ-302.6)."""
    rows = session.run(
        """
        MATCH (p:Project {name:$project})
        MATCH (n:NavTreeNode {project:$project})
        WHERE coalesce(n.avoid,false) = true AND n.visit_count >= 2
        RETURN n.screen_name AS screen, n.action AS action, n.depth AS depth,
               n.success_count AS success_count, n.visit_count AS visit_count
        ORDER BY (toFloat(n.success_count)/n.visit_count) ASC, n.visit_count DESC
        LIMIT $limit
        """,
        project=project, limit=limit,
    )
    return [dict(r) for r in rows]


def stats(session, project) -> dict:
    row = session.run(
        """
        MATCH (n:NavTreeNode {project:$project})
        RETURN count(n) AS nodes,
               sum(CASE WHEN coalesce(n.avoid,false) THEN 1 ELSE 0 END) AS avoid_nodes,
               max(n.depth) AS max_depth
        """,
        project=project,
    ).single()
    return {
        "nav_nodes": (row["nodes"] if row else 0) or 0,
        "avoid_nodes": (row["avoid_nodes"] if row else 0) or 0,
        "max_depth": (row["max_depth"] if row else 0) or 0,
    }
