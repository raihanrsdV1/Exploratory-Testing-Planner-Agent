import os
import requests

BASE_GATEWAY = os.getenv("GATEWAY_URL", "http://127.0.0.1:9100").rstrip("/")
BASE_RAG = os.getenv("RAG_URL", "http://127.0.0.1:9010").rstrip("/")
PROJECT = os.getenv("PROJECT", "contacts-app")
SRS_PATH = os.getenv("SRS_PATH", "./data/inputs/SRS1.txt")
FIGMA_PATH = os.getenv("FIGMA_PATH", "./data/inputs/GENERATED_JSON.json")


def post(url: str, payload: dict):
    r = requests.post(url, json=payload, timeout=240)
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
