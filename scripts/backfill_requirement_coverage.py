#!/usr/bin/env python3
"""One-time migration: give already-covered Requirements their coverage history.

Coverage used to live only on the ``(TestCase)-[:COVERS]->(Requirement)`` edge,
which CLEAN_SLATE destroys along with the test at the start of every campaign.
Requirements now carry ``covered_count`` / ``first_covered_at`` themselves, so
"has this ever been exercised?" survives the wipe — but only for coverage
recorded AFTER that change. This backfills the requirements that were already
covered when it landed.

Idempotent: only requirements with a live COVERS edge and no covered_count are
touched, so re-running never inflates a count.

Usage:  ./venv/bin/python scripts/backfill_requirement_coverage.py [--project NAME] [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from neo4j import GraphDatabase  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=os.getenv("PROJECT", ""),
                    help="project to migrate (default: $PROJECT; empty = all)")
    ap.add_argument("--dry-run", action="store_true", help="report without writing")
    args = ap.parse_args()

    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI") or "neo4j://127.0.0.1:7687",
        auth=(os.getenv("NEO4J_USER") or "neo4j", os.getenv("NEO4J_PASSWORD") or "hihi"))

    where = "WHERE coalesce(r.covered_count, 0) = 0"
    params = {"now": datetime.now(timezone.utc).isoformat()}
    if args.project:
        where += " AND r.project = $project"
        params["project"] = args.project

    with driver.session() as session:
        rows = [dict(x) for x in session.run(
            f"""
            MATCH (r:Requirement)<-[:COVERS]-(t:TestCase)
            {where}
            RETURN r.ref_id AS ref_id, count(t) AS tests
            ORDER BY r.ref_id
            """, **params)]
        if not rows:
            print("Nothing to migrate — every covered requirement already has its history.")
            return 0

        if not args.dry_run:
            session.run(
                f"""
                MATCH (r:Requirement)<-[:COVERS]-(t:TestCase)
                {where}
                WITH r, count(t) AS n
                SET r.covered_count = n,
                    r.first_covered_at = coalesce(r.first_covered_at, $now),
                    r.last_covered_at = $now
                """, **params)

        print(f"{'Would migrate' if args.dry_run else 'Migrated'} {len(rows)} requirement(s)"
              f"{' in ' + args.project if args.project else ''}:")
        for r in rows[:20]:
            print(f"  {r['ref_id']:16} covered by {r['tests']} test(s)")
        if len(rows) > 20:
            print(f"  … and {len(rows) - 20} more")
    driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
