import os
import requests
from dotenv import load_dotenv
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from settings import PROJECT, SRS_PATH, FIGMA_PATH  # noqa: E402

load_dotenv()

BASE_GATEWAY = os.getenv("GATEWAY_URL", "http://127.0.0.1:9100").rstrip("/")
BASE_RAG = os.getenv("RAG_URL", "http://127.0.0.1:9010").rstrip("/")


def post(url: str, payload: dict):
    r = requests.post(url, json=payload, timeout=5400)
    r.raise_for_status()
    return r.json()


def get(url: str, params: dict | None = None):
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def main():
    print("[1] Resetting project slices (tests, srs, figma)")
    print(post(f"{BASE_RAG}/project/reset", {
        "project": PROJECT,
        "delete_tests": True,
        "delete_srs": True,
        "delete_figma": True,
    }))

    print("\n[2] Ingesting SRS")
    print(post(f"{BASE_GATEWAY}/srs/ingest", {
        "project": PROJECT,
        "source_path": SRS_PATH,
    }))

    # Only ingest a design file when one is configured AND the source is enabled.
    # Ingesting another app's design file supplies screens the app under test does
    # not have, and the planner then writes tests against them — which is exactly
    # what happened when the Contacts Figma landed in a livestock-marketplace
    # project and the first generated test was about a Contacts List.
    from planner.sources import registry as _registry  # noqa: E402
    if not FIGMA_PATH:
        print("\n[3] Figma: skipped — FIGMA_PATH not set")
    elif not _registry.is_enabled("figma_ui"):
        print("\n[3] Figma: skipped — 'figma_ui' not in ENABLED_SOURCES")
    elif not os.path.exists(FIGMA_PATH):
        print(f"\n[3] Figma: skipped — {FIGMA_PATH} does not exist")
    else:
        print("\n[3] Ingesting Figma")
        print(post(f"{BASE_GATEWAY}/figma/ingest", {
            "project": PROJECT,
            "source_path": FIGMA_PATH,
        }))

    print("\n[4] Graph stats")
    print(get(f"{BASE_RAG}/graph/stats", {"project": PROJECT}))

    print("\n[5] Brief context")
    brief = post(f"{BASE_RAG}/context/brief", {"project": PROJECT, "recent_limit": 10})
    print({
        "project": brief.get("project"),
        "has_srs_summary": bool(brief.get("srs_summary")),
        "has_figma_summary": bool(brief.get("figma_summary")),
        "screen_index_count": len(brief.get("screen_index", [])),
        "recent_tests_count": len(brief.get("recent_tests", [])),
    })


if __name__ == "__main__":
    main()
