#!/usr/bin/env python3
"""One-time migration: give existing Findings a lifecycle status.

Findings created before the lifecycle existed have no `status`, so they never
appear in the open-question queue and the whole "reach a conclusion" mechanism
would be dead on the knowledge already in the graph.

Status is derived from `kind` exactly as `findings.initial_status()` does for new
findings, so a backfilled graph is indistinguishable from one built after the
change. Idempotent: only findings with no status are touched, so re-running
never resets a question that has since been resolved.

Usage:  ./venv/bin/python scripts/backfill_finding_status.py [--project NAME] [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from neo4j import GraphDatabase  # noqa: E402

from rag_api import findings as F  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=os.getenv("PROJECT", ""),
                    help="project to migrate (default: $PROJECT; empty = all)")
    ap.add_argument("--dry-run", action="store_true", help="report without writing")
    args = ap.parse_args()

    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI") or "neo4j://127.0.0.1:7687",
        auth=(os.getenv("NEO4J_USER") or "neo4j", os.getenv("NEO4J_PASSWORD") or "hihi"))

    where = "WHERE coalesce(f.status,'') = ''"
    params = {}
    if args.project:
        where += " AND f.project = $project"
        params["project"] = args.project

    with driver.session() as session:
        rows = [dict(r) for r in session.run(
            f"MATCH (f:Finding) {where} RETURN f.id AS id, f.kind AS kind, f.claim AS claim",
            **params)]
        if not rows:
            print("Nothing to migrate — every finding already has a status.")
            return 0

        # Kinds that legitimately carry no status (AGENT_DIFFICULTY — not an app
        # question) are SKIPPED, not written with an empty status. Writing them
        # would leave them matching the "no status" filter forever, so every
        # re-run would report migrating them again while changing nothing.
        counts: dict[str, int] = {}
        skipped = 0
        for r in rows:
            status = F.initial_status(r["kind"])
            if not status:
                skipped += 1
                continue
            counts[status] = counts.get(status, 0) + 1
            if not args.dry_run:
                session.run(
                    "MATCH (f:Finding {id:$id}) SET f.status = $status, f.attempts = 0",
                    id=r["id"], status=status)

        migrated = sum(counts.values())
        if not migrated:
            print(f"Nothing to migrate — every finding that takes a status already has one"
                  f"{f' ({skipped} status-less by kind)' if skipped else ''}.")
            return 0

        print(f"{'Would migrate' if args.dry_run else 'Migrated'} {migrated} finding(s)"
              f"{' in ' + args.project if args.project else ''}:")
        for status, n in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {n:>3}  -> {status}")
        if skipped:
            print(f"  {skipped:>3}  skipped (kind carries no lifecycle status)")
        if not args.dry_run:
            open_now = session.run(
                "MATCH (f:Finding) WHERE f.status = $open" +
                (" AND f.project = $project" if args.project else "") +
                " RETURN count(f) AS c", open=F.OPEN, **params).single()["c"]
            print(f"\n{open_now} open question(s) now in the queue "
                  f"(cap {F.MAX_FINDING_ATTEMPTS} attempts each).")
    driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
