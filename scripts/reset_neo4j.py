"""
Quick project reset for iterative dev/testing.

By default, clears ONLY test/execution history (TestCase, ExecutionLog, and the
learning artifacts derived from runs) — the ingested SRS/Figma knowledge and the
Live App Model are left untouched, so you don't have to re-ingest between runs.
Backs up the whole Neo4j instance first regardless (cheap insurance — a few
seconds even for a few thousand nodes), then calls rag_api's /project/reset
with the requested slices. rag_api must be running.

Usage:
    python3 scripts/reset_neo4j.py                  # tests only (default) — SRS/Figma/app model kept
    python3 scripts/reset_neo4j.py --wipe-appmodel   # also clear the Live App Model
    python3 scripts/reset_neo4j.py --full            # also clear SRS + Figma (re-ingest required after)
    python3 scripts/reset_neo4j.py --project other-app
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from settings import PROJECT  # noqa: E402

import requests  # noqa: E402
from backup_neo4j import backup_neo4j  # noqa: E402

BASE_RAG = os.getenv("RAG_URL", "http://127.0.0.1:9010").rstrip("/")


def main(project: str, wipe_appmodel: bool, full: bool) -> None:
    slices = ["tests"]
    if full:
        slices += ["SRS", "figma"]
    if wipe_appmodel:
        slices.append("app model")
    print(f"Project: {project}")
    print(f"Slices to clear: {', '.join(slices)}")

    print("\n[1/2] Backing up the full Neo4j instance (cheap insurance)...")
    backup_path = backup_neo4j()

    print("\n[2/2] Resetting via rag_api /project/reset...")
    try:
        resp = requests.post(f"{BASE_RAG}/project/reset", json={
            "project": project,
            "delete_tests": True,
            "delete_srs": full,
            "delete_figma": full,
            "delete_appmodel": wipe_appmodel,
        }, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"\nFAILED: {e}\nIs rag_api running on {BASE_RAG}? Backup is safe at: {backup_path}")
        sys.exit(1)

    print(resp.json())
    print(f"\nDone. Backup (whole instance, pre-reset): {backup_path}")
    if not full:
        print("SRS/Figma knowledge kept — no re-ingest needed, just start the executor.")
    else:
        print("SRS/Figma cleared — run scripts/ingest_all.py before testing again.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=PROJECT, help=f"Project to reset (default: {PROJECT!r} from settings)")
    ap.add_argument("--wipe-appmodel", action="store_true",
                     help="Also clear the Live App Model (learned screens/navigation). Expensive to rebuild — off by default.")
    ap.add_argument("--full", action="store_true",
                     help="Also clear SRS + Figma knowledge. You will need to re-ingest before testing again.")
    args = ap.parse_args()
    main(project=args.project, wipe_appmodel=args.wipe_appmodel, full=args.full)
