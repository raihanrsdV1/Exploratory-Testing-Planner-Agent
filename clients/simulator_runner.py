import os
import sys
import requests
from dotenv import load_dotenv

# ── Force UTF-8 console output ────────────────────────────────────────────────
# Planner/model output can contain emoji and arrows; the Windows console
# defaults to cp1252 and raises UnicodeEncodeError on such characters.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# ── Configuration (single source of truth: settings.py) ──────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from settings import (  # noqa: E402
    SIM_OUTPUT_FILE as OUTPUT_FILE, GATEWAY_URL as BASE, RAG_URL, PROJECT, APP_NAME,
    TOP_K, DEBUG_TRACE, RESET_TESTS_FIRST, RESET_ALL_FIRST, SIM_FAIL_EVERY, SIM_ROUNDS,
)

BASE = BASE.rstrip("/")
RAG_URL = RAG_URL.rstrip("/")


def _print_stage_header(text: str):
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def _show_reasoning_flow(data: dict):
    plan = data.get("retrieval_plan", {}) or {}
    trace = data.get("planner_trace", []) or []
    finalization_mode = data.get("finalization_mode", "")
    ctx_stats = data.get("retrieved_context_stats", {}) or {}
    focus_queries = plan.get("focus_queries", []) or []
    target_screens = data.get("target_screens", []) or plan.get("target_screens", []) or []

    _print_stage_header("STAGE FLOW: SUMMARY CONTEXT -> RETRIEVAL PLAN -> NEXT TEST")
    print("1) Global summary context loaded from RAG:")
    print("   - SRS summary")
    print("   - Figma summary + screen index")
    print("   - Recent test history")

    print("2) Planner decided retrieval plan:")
    print("   focus_queries:", focus_queries if focus_queries else ["(fallback objective query)"])
    print("   target_screens:", target_screens if target_screens else ["(fallback uncovered screens)"])
    if plan.get("reason"):
        print("   reason:", plan.get("reason"))

    if trace:
        print("   retrieval rounds:")
        for step in trace:
            print(
                "    - round",
                step.get("round"),
                "| action=",
                step.get("action"),
                "| queries=",
                step.get("focus_queries", []),
                "| screens=",
                step.get("target_screens", []),
            )
    if finalization_mode:
        print("   finalization_mode:", finalization_mode)
    if ctx_stats:
        print(
            "   context_sent: srs_chars=",
            ctx_stats.get("srs_context_chars", 0),
            "figma_chars=",
            ctx_stats.get("figma_context_chars", 0),
        )

    tc = data.get("next_testcase", {}) or {}
    print("3) Generated next testcase:")
    print("   ", tc.get("test_case_id", "(missing id)"), "|", tc.get("title", "(missing title)"))
    if tc.get("area"):
        print("   area:", tc.get("area"))

    dbg = data.get("debug_trace", {}) or {}
    if dbg:
        _print_stage_header("CONVERSATION VIEW (ROUND)")
        planner_rounds = dbg.get("planner_rounds", []) or []
        retrieved = dbg.get("retrieved_blocks", []) or []

        for pr in planner_rounds:
            pa = pr.get("parsed_action", {}) or {}
            print(f"Agent[{pr.get('round')}]: action={pa.get('action')} | reason={pa.get('reason','')}")
            for rr in (pa.get("retrieval_requests", []) or []):
                src = rr.get("source", "srs")
                q = rr.get("query", "")
                sc = rr.get("screen", "")
                print(f"  asks -> source={src} query={q} screen={sc}")

            round_blocks = [b for b in retrieved if b.get("round") == pr.get("round")]
            for b in round_blocks:
                ctx = (b.get("context", "") or "").strip().replace("\n", " ")
                if len(ctx) > 220:
                    ctx = ctx[:220] + " ..."
                print(f"  db -> source={b.get('source','srs')} summary={ctx}")

        print("System protocol:")
        print("  1) Agent asks retrieval requests in strict JSON.")
        print("  2) Gateway retrieves from SRS/Figma UI/Figma flow DB.")
        print("  3) Agent gets additional summarized context and decides continue or finalize.")
        print("  4) Final output is one strict testcase JSON.")

        _print_stage_header("DEBUG TRACE: PROMPTS, CONTEXT, RAW MODEL OUTPUT")
        for pr in dbg.get("planner_rounds", []):
            print(f"[Planner round {pr.get('round')}] Prompt:")
            print(pr.get("prompt", ""))
            print("\n[Planner round raw answer]:")
            print(pr.get("model_answer_raw", ""))
            print("-" * 72)

        for rb in dbg.get("retrieved_blocks", []):
            print(f"[Retrieved context round {rb.get('round')} query={rb.get('query')}]\n")
            print(rb.get("context", ""))
            print("-" * 72)

        final_gen = dbg.get("final_generation", {})
        if final_gen:
            print("[Final generation prompt]:")
            print(final_gen.get("prompt", ""))
            print("\n[Final generation raw answer]:")
            print(final_gen.get("model_answer_raw", ""))
            if final_gen.get("model_thinking"):
                print("\n[Final generation thinking]:")
                print(final_gen.get("model_thinking", ""))
            print("-" * 72)

        final_retry = dbg.get("final_retry", {})
        if final_retry:
            print("[Final retry prompt]:")
            print(final_retry.get("prompt", ""))
            print("\n[Final retry raw answer]:")
            print(final_retry.get("model_answer_raw", ""))
            if final_retry.get("model_thinking"):
                print("\n[Final retry thinking]:")
                print(final_retry.get("model_thinking", ""))
            print("-" * 72)


def next_case(max_new_tokens: int = 4096):
    resp = requests.post(
        f"{BASE}/agent/next-testcase",
        json={
            "project": PROJECT,
            "app_name": APP_NAME,
            "objective": "generate next high-value non-duplicate test case",
            "top_k": TOP_K,
            "max_new_tokens": max_new_tokens,
            "enable_thinking": False,
            "debug_trace": DEBUG_TRACE,
        },
        timeout=240,
    )
    resp.raise_for_status()
    return resp.json()


def log_and_next(tc: dict, verdict: str, notes: str, max_new_tokens: int = 4096):
    payload = {
        "project": PROJECT,
        "app_name": APP_NAME,
        "test_case_id": tc.get("test_case_id", "TC-MANUAL-FALLBACK"),
        "title": tc.get("title", "Manual fallback test"),
        "verdict": verdict,
        "notes": notes,
        "area": tc.get("area", "general"),
        "top_k": TOP_K,
        "max_new_tokens": max_new_tokens,
        "enable_thinking": False,
        "debug_trace": DEBUG_TRACE,
    }
    resp = requests.post(f"{BASE}/agent/log-verdict-and-next", json=payload, timeout=240)
    resp.raise_for_status()
    return resp.json()


def _preflight():
    _print_stage_header("PREFLIGHT CHECK")
    gw = requests.get(f"{BASE}/health", timeout=30)
    gw.raise_for_status()
    gw_data = gw.json()
    print("Gateway health:", gw_data)

    rag = requests.get(f"{RAG_URL}/health", timeout=30)
    rag.raise_for_status()
    print("RAG health:", rag.json())

    backend = ((gw_data or {}).get("model") or {}).get("backend", "")
    if backend != "ngrok":
        # openrouter/gemini call the cloud API directly; there is no local
        # model server to health-check.
        print(f"Model backend: {backend} (cloud API, skipping local /health check)")
        return

    model_api = (gw_data or {}).get("model_api", "")
    if not model_api:
        raise RuntimeError("Gateway health did not return model_api URL")

    try:
        mh = requests.get(f"{model_api.rstrip('/')}/health", timeout=30)
        mh.raise_for_status()
        print("Model health:", mh.json())
    except Exception as e:
        raise RuntimeError(
            "Model API is unreachable from simulator. "
            "Restart gateway with the active Kaggle/ngrok URL, e.g. "
            "MODEL_API_URL='https://<active-ngrok>.ngrok-free.app' ./start.sh --no-ingest. "
            f"Details: {e}"
        )


def _reset_if_requested():
    if not RESET_TESTS_FIRST and not RESET_ALL_FIRST:
        return
    _print_stage_header("RESET PROJECT DATA")
    payload = {
        "project": PROJECT,
        "delete_tests": True,
        "delete_srs": bool(RESET_ALL_FIRST),
        "delete_figma": bool(RESET_ALL_FIRST),
    }
    r = requests.post(f"{BASE}/project/reset", json=payload, timeout=60)
    r.raise_for_status()
    print("Reset response:", r.json())


def main(rounds: int = 3):
    _preflight()
    _reset_if_requested()

    data = next_case()
    _show_reasoning_flow(data)
    tc = data.get("next_testcase", {})

    for i in range(1, rounds + 1):
        verdict = "failed" if SIM_FAIL_EVERY > 0 and i % SIM_FAIL_EVERY == 0 else "pass"
        notes = f"simulated verdict round {i}: {verdict}"
        out = log_and_next(tc, verdict, notes)
        log = out.get("log", {})
        nxt_data = out.get("next", {})
        nxt = nxt_data.get("next_testcase", {})

        _print_stage_header(f"ROUND {i}")
        print("Logged:", log.get("test_case_id"), "|", verdict, "|", log.get("run_id"))
        _show_reasoning_flow(nxt_data)
        tc = nxt

    recent = requests.get(
        f"{RAG_URL}/tests/recent",
        params={"project": PROJECT, "limit": 10},
        timeout=60,
    )
    recent.raise_for_status()
    tests = recent.json().get("tests", [])

    _print_stage_header("RECENT TESTS SAVED IN NEO4J")
    for t in tests:
        print("-", t.get("id"), "|", t.get("verdict"), "|", t.get("title"))


class _Tee:
    """Mirror stdout to a file so the full run is saved as well as printed."""

    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh

    def write(self, data):
        self._stream.write(data)
        self._fh.write(data)

    def flush(self):
        self._stream.flush()
        self._fh.flush()


if __name__ == "__main__":
    rounds = SIM_ROUNDS
    if OUTPUT_FILE:
        os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_FILE)), exist_ok=True)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
            sys.stdout = _Tee(sys.stdout, fh)
            try:
                main(rounds=rounds)
            finally:
                sys.stdout = sys.stdout._stream
        print(f"\nSimulator output saved to: {OUTPUT_FILE}")
    else:
        main(rounds=rounds)
