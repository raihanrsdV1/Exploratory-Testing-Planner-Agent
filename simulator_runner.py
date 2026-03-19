import requests

BASE = "http://127.0.0.1:9100"
PROJECT = "contacts-app"
APP_NAME = "contacts app"


def next_case(max_new_tokens: int = 600):
    resp = requests.post(
        f"{BASE}/agent/next-testcase",
        json={
            "project": PROJECT,
            "app_name": APP_NAME,
            "objective": "generate next high-value non-duplicate test case",
            "top_k": 8,
            "max_new_tokens": max_new_tokens,
            "enable_thinking": False,
        },
        timeout=240,
    )
    resp.raise_for_status()
    return resp.json()


def log_and_next(tc: dict, verdict: str, notes: str, max_new_tokens: int = 420):
    payload = {
        "project": PROJECT,
        "app_name": APP_NAME,
        "test_case_id": tc.get("test_case_id", "TC-MANUAL-FALLBACK"),
        "title": tc.get("title", "Manual fallback test"),
        "verdict": verdict,
        "notes": notes,
        "area": tc.get("area", "general"),
        "top_k": 8,
        "max_new_tokens": max_new_tokens,
        "enable_thinking": False,
    }
    resp = requests.post(f"{BASE}/agent/log-verdict-and-next", json=payload, timeout=240)
    resp.raise_for_status()
    return resp.json()


def main(rounds: int = 10):
    data = next_case()
    tc = data.get("next_testcase", {})
    print("ROUND 0 NEXT:", tc.get("test_case_id"), "|", tc.get("title"))

    for i in range(1, rounds + 1):
        verdict = "pass" if i % 2 == 1 else "failed"
        notes = f"simulated verdict round {i}: {verdict}"
        out = log_and_next(tc, verdict, notes)
        log = out.get("log", {})
        nxt = out.get("next", {}).get("next_testcase", {})
        print(f"ROUND {i} LOG:", log.get("test_case_id"), log.get("run_id"), verdict)
        print(f"ROUND {i} NEXT:", nxt.get("test_case_id"), "|", nxt.get("title"))
        tc = nxt

    recent = requests.get(
        "http://127.0.0.1:9000/tests/recent",
        params={"project": PROJECT, "limit": 10},
        timeout=60,
    )
    recent.raise_for_status()
    tests = recent.json().get("tests", [])
    print("\nRECENT TESTS IN NEO4J:")
    for t in tests:
        print("-", t.get("id"), "|", t.get("verdict"), "|", t.get("title"))


if __name__ == "__main__":
    main(rounds=3)
