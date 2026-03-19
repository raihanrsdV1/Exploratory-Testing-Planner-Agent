import json
import os
import requests

GATEWAY_URL = os.getenv("GATEWAY_URL", "http://127.0.0.1:9100").rstrip("/")
PROJECT = os.getenv("PROJECT_NAME", "contacts-app")
APP_NAME = os.getenv("APP_NAME", "contacts app")
SRS_PATH = os.getenv("SRS_PATH", "./SRS1.txt")
GATEWAY_API_KEY = os.getenv("GATEWAY_API_KEY", "")


def _headers():
    if not GATEWAY_API_KEY:
        return {}
    return {"Authorization": f"Bearer {GATEWAY_API_KEY}"}


def ingest_srs():
    payload = {"project": PROJECT, "source_path": SRS_PATH}
    r = requests.post(f"{GATEWAY_URL}/srs/ingest", json=payload, headers=_headers(), timeout=60)
    r.raise_for_status()
    print("SRS ingested:", r.json())


def get_next_testcase():
    payload = {
        "project": PROJECT,
        "app_name": APP_NAME,
        "objective": "give the next high-value non-duplicate test case",
        "top_k": 8,
        "max_new_tokens": 700,
        "enable_thinking": False,
    }
    r = requests.post(f"{GATEWAY_URL}/agent/next-testcase", json=payload, headers=_headers(), timeout=180)
    r.raise_for_status()
    data = r.json()
    print("\n=== NEXT TEST CASE (JSON string from model) ===")
    print(data.get("next_testcase_json", ""))
    return data


def log_verdict_and_get_next(current_test_json: str):
    try:
        tc = json.loads(current_test_json)
    except Exception:
        tc = {
            "test_case_id": "manual-unknown",
            "title": "manual test",
            "area": "general",
        }

    verdict = input("Verdict for this test? (pass/failed/quit): ").strip().lower()
    if verdict == "quit":
        return None
    if verdict not in {"pass", "failed"}:
        print("Invalid verdict. Use pass or failed.")
        return current_test_json

    notes = input("Optional notes: ").strip()

    payload = {
        "project": PROJECT,
        "app_name": APP_NAME,
        "test_case_id": str(tc.get("test_case_id", "manual-unknown")),
        "title": str(tc.get("title", "manual test")),
        "verdict": verdict,
        "notes": notes,
        "area": str(tc.get("area", "general")),
        "top_k": 8,
        "max_new_tokens": 700,
        "enable_thinking": False,
    }

    r = requests.post(
        f"{GATEWAY_URL}/agent/log-verdict-and-next",
        json=payload,
        headers=_headers(),
        timeout=240,
    )
    r.raise_for_status()
    data = r.json()

    next_json = data.get("next", {}).get("next_testcase_json", "")
    print("\n=== NEXT TEST CASE (JSON string from model) ===")
    print(next_json)
    return next_json


def main():
    print("Starting QA loop for", PROJECT)
    ingest_srs()
    first = get_next_testcase().get("next_testcase_json", "")

    current = first
    while True:
        nxt = log_verdict_and_get_next(current)
        if nxt is None:
            print("Stopped.")
            break
        current = nxt


if __name__ == "__main__":
    main()
