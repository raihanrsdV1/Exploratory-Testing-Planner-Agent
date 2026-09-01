"""
Restore a Neo4j knowledge graph from a JSON backup made by backup_neo4j.py.

Nodes are recreated first (grouped by their exact label-set, since Cypher needs
labels as literal text, not parameters), tagged with a temporary `_restore_id`
matching the backup's `element_id`. Relationships are recreated second (grouped
by type, for the same reason), matched on that temporary id, then the temporary
id is dropped from every node.

Usage:
    python scripts/restore_neo4j.py <backup.json> [--database NAME] [--wipe-first]

`--database` targets a specific Neo4j database (Enterprise multi-database) —
use this to restore into a THROWAWAY database for verification before ever
pointing this at production data. `--wipe-first` clears the target database
before restoring (required for a clean restore; refused without the flag so a
restore can never silently overwrite something by accident).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD  # noqa: E402
from neo4j import GraphDatabase  # noqa: E402


def restore(backup_path: str, database: str | None, wipe_first: bool) -> None:
    data = json.loads(Path(backup_path).read_text(encoding="utf-8"))
    nodes = data["nodes"]
    rels = data["relationships"]
    print(f"Backup: {data.get('timestamp')} — {len(nodes)} nodes, {len(rels)} relationships")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    db_kwargs = {"database": database} if database else {}

    with driver.session(**db_kwargs) as session:
        existing = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        if existing:
            if not wipe_first:
                print(f"REFUSING: target already has {existing} node(s). "
                      f"Pass --wipe-first to clear it before restoring.")
                sys.exit(1)
            print(f"Wiping {existing} existing node(s) in target...")
            session.run("MATCH (n) DETACH DELETE n")

        # ── Nodes, grouped by exact label-set (Cypher labels are not parameterizable) ──
        by_labels: dict[tuple, list] = defaultdict(list)
        for n in nodes:
            by_labels[tuple(sorted(n["labels"]))].append(n)

        created_nodes = 0
        for labels, group in by_labels.items():
            label_str = ":".join(labels) if labels else "Node"
            rows = [{"rid": n["element_id"], "props": n["properties"]} for n in group]
            res = session.run(
                f"""
                UNWIND $rows AS row
                CREATE (n:{label_str})
                SET n = row.props, n._restore_id = row.rid
                """,
                rows=rows,
            )
            res.consume()
            created_nodes += len(rows)
        print(f"Created {created_nodes} nodes across {len(by_labels)} label-set group(s)")

        # ── Relationships, grouped by type ──
        by_type: dict[str, list] = defaultdict(list)
        for r in rels:
            by_type[r["type"]].append(r)

        created_rels = 0
        skipped = 0
        for rtype, group in by_type.items():
            rows = [
                {"start": r["start_element_id"], "end": r["end_element_id"], "props": r["properties"]}
                for r in group
            ]
            res = session.run(
                f"""
                UNWIND $rows AS row
                MATCH (a {{_restore_id: row.start}})
                MATCH (b {{_restore_id: row.end}})
                CREATE (a)-[rel:{rtype}]->(b)
                SET rel = row.props
                RETURN count(rel) AS c
                """,
                rows=rows,
            )
            c = res.single()["c"]
            created_rels += c
            skipped += len(rows) - c
        print(f"Created {created_rels} relationships across {len(by_type)} type(s)"
              + (f"  ({skipped} skipped — endpoint node missing)" if skipped else ""))

        # ── Cleanup: drop the temporary id now that relationships are wired ──
        session.run("MATCH (n) WHERE n._restore_id IS NOT NULL REMOVE n._restore_id")

        # ── Self-check ──
        final_nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        final_rels = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        print(f"\nFinal: {final_nodes} nodes, {final_rels} relationships "
              f"(backup declared {data.get('total_nodes')} / {data.get('total_relationships')})")
        ok = final_nodes == data.get("total_nodes") and created_rels == len(rels)
        print("MATCH — restore looks complete" if ok else "MISMATCH — inspect before trusting this restore")

    driver.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("backup_path")
    ap.add_argument("--database", default=None, help="Target database name (Enterprise multi-db)")
    ap.add_argument("--wipe-first", action="store_true")
    args = ap.parse_args()
    restore(args.backup_path, args.database, args.wipe_first)
