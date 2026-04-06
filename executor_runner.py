#!/usr/bin/env python3
"""
executor_runner.py — Droidrun-based Test Executor

Replaces simulator_runner.py with REAL device execution.
Workflow:
  1. Ask Gateway for the next test case
  2. Translate planner JSON → Droidrun natural-language goal
  3. Execute on connected Android device via Droidrun
  4. Interpret pass/fail from Droidrun result
  5. Log verdict back to Gateway → triggers next test case generation
  6. Repeat
"""

import asyncio
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

# ── Load .env from project root ──────────────────────────────────────────────
load_dotenv()

# ── Planner Gateway config ───────────────────────────────────────────────────
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://127.0.0.1:9100").rstrip("/")
RAG_URL = os.getenv("RAG_URL", "http://127.0.0.1:9010").rstrip("/")
PROJECT = os.getenv("PROJECT", "contacts-app")
APP_NAME = os.getenv("APP_NAME", "contacts app")
TOP_K = int(os.getenv("TOP_K", "8"))
DEBUG_TRACE = os.getenv("DEBUG_TRACE", "1").strip().lower() not in {"0", "false", "no"}

# ── Executor-specific config ─────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
EXECUTOR_LLM_PROVIDER = os.getenv("EXECUTOR_LLM_PROVIDER", "GoogleGenAI")
EXECUTOR_LLM_MODEL = os.getenv("EXECUTOR_LLM_MODEL", "gemini-2.5-pro")
EXECUTOR_TIMEOUT = int(os.getenv("EXECUTOR_TIMEOUT", "120"))
EXECUTOR_ROUNDS = int(os.getenv("EXECUTOR_ROUNDS", "2"))
TARGET_APP_PACKAGE = os.getenv("TARGET_APP_PACKAGE", "com.android.contacts")

# ── Logtail / Better Stack live logging ──────────────────────────────────────
LOGTAIL_SOURCE_TOKEN = os.getenv("LOGTAIL_SOURCE_TOKEN", "")

logger = logging.getLogger("executor")
logger.setLevel(logging.INFO)

# Console handler (always active)
_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_console_handler)

# Cloud handler (only if token is configured)
if LOGTAIL_SOURCE_TOKEN:
    try:
        from logtail import LogtailHandler
        _cloud_handler = LogtailHandler(source_token=LOGTAIL_SOURCE_TOKEN)
        logger.addHandler(_cloud_handler)
    except ImportError:
        pass  # logtail not installed, skip silently


def cloud_log(level: str, message: str, **extra):
    """
    Log to both console and Better Stack cloud.
    Extra kwargs become structured metadata visible in the Better Stack dashboard.
    """
    log_fn = getattr(logger, level.lower(), logger.info)
    log_fn(message, extra=extra if extra else {})


def _print_header(text: str):
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)
    cloud_log("info", text)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


# ──────────────────────────────────────────────────────────────────────────────
# PLANNER TRACE LOGGING (RAG + Neo4j interaction)
# ──────────────────────────────────────────────────────────────────────────────

def _log_planner_trace(planner_data: dict, label: str = ""):
    """
    Extract and log the planner's RAG retrieval trace and context stats
    to Better Stack for full visibility into the planning pipeline.
    """
    ctx = planner_data.get("retrieved_context_stats", {})
    tc = planner_data.get("next_testcase", {})
    retrieval_plan = planner_data.get("retrieval_plan", "")
    debug_trace = planner_data.get("debug_trace", {})

    # Summarize retrieved blocks from debug trace
    retrieved_blocks = debug_trace.get("retrieved_blocks", [])
    block_summary = []
    for blk in retrieved_blocks[:10]:
        block_summary.append({
            "round": blk.get("round"),
            "source": blk.get("source"),
            "query": blk.get("query", ""),
            "screen": blk.get("screen", ""),
            "context_preview": blk.get("context", "")[:200],
        })

    # Summarize planner reasoning rounds
    planner_rounds = debug_trace.get("planner_rounds", [])
    rounds_summary = []
    for pr in planner_rounds:
        rounds_summary.append({
            "round": pr.get("round"),
            "action": pr.get("parsed_action", {}).get("action", ""),
            "queries": pr.get("parsed_action", {}).get("queries", []),
            "screens": pr.get("parsed_action", {}).get("screens", []),
        })

    cloud_log(
        "info",
        f"Planner RAG interaction [{label}]: {tc.get('test_case_id', '?')} generated",
        phase=label,
        test_case_id=tc.get("test_case_id", "?"),
        title=tc.get("title", "?"),
        retrieval_plan=str(retrieval_plan)[:500] if retrieval_plan else "",
        retrieval_stats={
            "rounds_executed": ctx.get("retrieval_rounds_executed", 0),
            "queries_used": ctx.get("queries_used", []),
            "screens_used": ctx.get("screens_used", []),
            "srs_context_chars": ctx.get("srs_context_chars", 0),
            "figma_overview_chars": ctx.get("figma_overview_chars", 0),
            "figma_context_chars": ctx.get("figma_context_chars", 0),
            "figma_flow_chars": ctx.get("figma_flow_chars", 0),
        },
        target_screens=planner_data.get("target_screens", []),
        recent_tests_count=planner_data.get("recent_tests_count", 0),
        failed_tests_count=planner_data.get("failed_tests_count", 0),
        finalization_mode=planner_data.get("finalization_mode", ""),
        agent_signaled_ready=planner_data.get("agent_signaled_ready", False),
        planner_rounds=rounds_summary,
        retrieved_blocks=block_summary,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 1. PLANNER GATEWAY COMMUNICATION
# ──────────────────────────────────────────────────────────────────────────────

def get_next_testcase(max_new_tokens: int = 4096) -> dict:
    """Ask the planner gateway for the next test case."""
    resp = requests.post(
        f"{GATEWAY_URL}/agent/next-testcase",
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


def log_verdict_and_get_next(
    tc: dict, verdict: str, notes: str, max_new_tokens: int = 600
) -> dict:
    """Log the execution verdict for a test case and get the next one."""
    payload = {
        "project": PROJECT,
        "app_name": APP_NAME,
        "test_case_id": tc.get("test_case_id", "TC-EXECUTOR-FALLBACK"),
        "title": tc.get("title", "Executor fallback test"),
        "verdict": verdict,
        "notes": notes,
        "area": tc.get("area", "general"),
        "top_k": TOP_K,
        "max_new_tokens": max_new_tokens,
        "enable_thinking": False,
        "debug_trace": DEBUG_TRACE,
    }
    resp = requests.post(
        f"{GATEWAY_URL}/agent/log-verdict-and-next", json=payload, timeout=240
    )
    resp.raise_for_status()
    return resp.json()


def log_verdict_only(tc: dict, verdict: str, notes: str) -> dict:
    """Log the verdict directly to RAG without generating a next test case."""
    payload = {
        "project": PROJECT,
        "test_case_id": tc.get("test_case_id", "TC-EXECUTOR-FALLBACK"),
        "title": tc.get("title", "Executor fallback test"),
        "verdict": verdict,
        "notes": notes,
        "area": tc.get("area", "general"),
    }
    resp = requests.post(
        f"{RAG_URL}/tests/log", json=payload, timeout=60
    )
    resp.raise_for_status()
    return resp.json()


# ──────────────────────────────────────────────────────────────────────────────
# 2. TEST CASE → DROIDRUN GOAL TRANSLATION
# ──────────────────────────────────────────────────────────────────────────────

def build_droidrun_goal(test_case: dict) -> str:
    """
    Convert the planner's structured JSON test case into a natural language
    goal string that Droidrun's LLM agent can interpret and execute.

    Example output:
      "Open the Contacts app. Navigate to 'Create Contact' screen.
       Step 1: Enter 'John' in the 'First name' field.
       Step 2: Click the 'Save' button.
       Expected: The contact is saved successfully and appears in the list."
    """
    screen = test_case.get("screen", "")
    preconditions = test_case.get("preconditions", [])
    steps = test_case.get("steps", [])
    expected = test_case.get("expected_result", "")

    # Build the goal instruction
    goal_parts = []

    # Opening instruction — tell Droidrun which app to work with
    goal_parts.append("Open the Contacts app on this device.")

    # Screen navigation
    if screen:
        goal_parts.append(f"Navigate to the '{screen}' screen if not already there.")

    # Preconditions as context
    if preconditions:
        pre_text = " ".join(preconditions)
        goal_parts.append(f"Preconditions: {pre_text}")

    # Steps — numbered imperatives
    if steps:
        goal_parts.append("")  # blank line for readability
        for i, step in enumerate(steps, 1):
            goal_parts.append(f"Step {i}: {step}")

    # Expected outcome
    if expected:
        goal_parts.append(f"\nExpected result: {expected}")

    # Final instruction to Droidrun
    goal_parts.append(
        "\nAfter performing all steps, report whether the expected result was achieved. "
        "If any step fails or the app crashes, report the failure."
    )

    return "\n".join(goal_parts)


# ──────────────────────────────────────────────────────────────────────────────
# 3. DROIDRUN EXECUTION
# ──────────────────────────────────────────────────────────────────────────────

async def execute_test_on_device(test_case: dict) -> dict:
    """
    Run a single test case on the connected Android device using Droidrun.

    Droidrun v0.5.7 API:
      - agent.run() returns a WorkflowHandler (awaitable)
      - Awaiting it yields a ResultEvent with:
          .success (bool)  — did the agent achieve the goal?
          .reason  (str)   — explanation of success or failure
          .steps   (int)   — number of steps the agent took

    Returns:
        {
            "verdict": "pass" | "failed",
            "notes": "Execution details or error message",
            "duration_seconds": float,
        }
    """
    # Lazy import so the script doesn't crash during --help / preflight
    from droidrun import DroidAgent, AndroidDriver, load_llm

    goal = build_droidrun_goal(test_case)
    tc_id = test_case.get("test_case_id", "?")
    title = test_case.get("title", "?")

    _print_header(f"EXECUTING ON DEVICE: {tc_id}")
    print(f"Title: {title}")
    print(f"Goal sent to Droidrun:\n{goal}")
    print("-" * 72)

    # Log the goal instruction sent to Droidrun
    cloud_log(
        "info",
        f"Droidrun goal dispatched for {tc_id}",
        test_case_id=tc_id,
        title=title,
        droidrun_goal=goal,
        screen=test_case.get("screen", ""),
        steps=test_case.get("steps", []),
        expected_result=test_case.get("expected_result", ""),
    )

    start_time = time.time()

    try:
        # Set up device driver (connects to default adb device)
        driver = AndroidDriver()

        # Set up LLM for Droidrun
        llm = load_llm(
            EXECUTOR_LLM_PROVIDER,
            model=EXECUTOR_LLM_MODEL,
            api_key=GEMINI_API_KEY,
        )

        # Create and run the agent
        agent = DroidAgent(
            goal=goal,
            llms=llm,
            driver=driver,
            timeout=EXECUTOR_TIMEOUT,
        )

        # agent.run() returns a WorkflowHandler; await it to get ResultEvent
        result = await agent.run()

        duration = time.time() - start_time

        # ResultEvent has: .success (bool), .reason (str), .steps (int)
        success = result.success
        reason = result.reason or "No reason provided by Droidrun"
        steps_taken = result.steps

        verdict = "pass" if success else "failed"
        notes = (
            f"Droidrun execution completed in {duration:.1f}s. "
            f"Steps taken: {steps_taken}. "
            f"Success={success}. Reason: {reason}"
        )

        status_icon = "✅" if success else "❌"
        print(f"\n{status_icon} Droidrun result: success={success}")
        print(f"   Steps taken: {steps_taken}")
        print(f"   Reason: {reason[:300]}")

        cloud_log(
            "info" if success else "warning",
            f"Test {tc_id} execution: {'PASS' if success else 'FAILED'}",
            test_case_id=tc_id,
            title=title,
            verdict=verdict,
            steps_taken=steps_taken,
            duration_seconds=round(duration, 1),
            reason=reason[:500],
        )

        return {"verdict": verdict, "notes": notes, "duration_seconds": duration}

    except asyncio.TimeoutError:
        duration = time.time() - start_time
        notes = (
            f"Droidrun execution TIMED OUT after {EXECUTOR_TIMEOUT}s. "
            f"The test case may be too complex or the device is unresponsive."
        )
        print(f"\n⏰ TIMEOUT after {EXECUTOR_TIMEOUT}s")
        cloud_log("error", f"Test {tc_id} TIMED OUT", test_case_id=tc_id, title=title, timeout=EXECUTOR_TIMEOUT)
        return {"verdict": "failed", "notes": notes, "duration_seconds": duration}

    except Exception as e:
        duration = time.time() - start_time
        tb = traceback.format_exc()
        notes = (
            f"Droidrun execution CRASHED after {duration:.1f}s. "
            f"Error: {type(e).__name__}: {e}\n{tb[-500:]}"
        )
        print(f"\n❌ CRASH: {e}")
        cloud_log("error", f"Test {tc_id} CRASHED: {e}", test_case_id=tc_id, title=title, error=str(e))
        return {"verdict": "failed", "notes": notes, "duration_seconds": duration}


# ──────────────────────────────────────────────────────────────────────────────
# 4. DISPLAY HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _show_testcase(tc: dict):
    """Pretty-print a generated test case."""
    print(f"  ID:       {tc.get('test_case_id', '?')}")
    print(f"  Title:    {tc.get('title', '?')}")
    print(f"  Screen:   {tc.get('screen', '?')}")
    print(f"  Area:     {tc.get('area', '?')}")
    print(f"  Priority: {tc.get('priority', '?')}")
    steps = tc.get("steps", [])
    if steps:
        print(f"  Steps:")
        for i, s in enumerate(steps, 1):
            print(f"    {i}. {s}")
    print(f"  Expected: {tc.get('expected_result', '?')}")


def _show_round_summary(
    round_num: int, tc: dict, verdict: str, notes: str, duration: float
):
    """Print a compact round summary."""
    _print_header(f"ROUND {round_num} RESULT")
    print(f"  Test Case: {tc.get('test_case_id', '?')} | {tc.get('title', '?')[:80]}")
    print(f"  Verdict:   {'✅ PASS' if verdict == 'pass' else '❌ FAILED'}")
    print(f"  Duration:  {duration:.1f}s")
    if verdict == "failed":
        print(f"  Notes:     {notes[:200]}")


# ──────────────────────────────────────────────────────────────────────────────
# 5. PREFLIGHT CHECKS
# ──────────────────────────────────────────────────────────────────────────────

def preflight():
    """Verify all services and the device are ready before starting."""
    _print_header("PREFLIGHT CHECK")

    # 1. Check Gateway health
    print("[1/4] Checking Gateway...")
    gw = requests.get(f"{GATEWAY_URL}/health", timeout=30)
    gw.raise_for_status()
    gw_data = gw.json()
    print(f"  ✅ Gateway: {gw_data}")

    # 2. Check RAG health
    print("[2/4] Checking RAG API...")
    rag = requests.get(f"{RAG_URL}/health", timeout=30)
    rag.raise_for_status()
    print(f"  ✅ RAG API: {rag.json()}")

    # 3. Check Model API
    print("[3/4] Checking Model API...")
    model_api = gw_data.get("model_api", "")
    if model_api:
        try:
            mh = requests.get(f"{model_api.rstrip('/')}/health", timeout=30)
            mh.raise_for_status()
            print(f"  ✅ Model API: {mh.json()}")
        except Exception as e:
            print(f"  ⚠️  Model API unreachable: {e}")
            print("     (This is okay if the gateway can still reach it internally)")

    # 4. Check ADB device
    print("[4/4] Checking ADB device connection...")
    import subprocess
    try:
        result = subprocess.run(
            ["adb", "devices"], capture_output=True, text=True, timeout=10
        )
        lines = [
            l.strip()
            for l in result.stdout.strip().split("\n")[1:]
            if l.strip() and "device" in l
        ]
        if lines:
            print(f"  ✅ ADB devices found: {lines}")
        else:
            print("  ❌ No ADB devices found! Start your emulator first.")
            sys.exit(1)
    except FileNotFoundError:
        print("  ❌ ADB not found! Install it: brew install android-platform-tools")
        sys.exit(1)

    # 5. Check Gemini API key
    if not GEMINI_API_KEY:
        print("  ❌ GEMINI_API_KEY not set in .env!")
        sys.exit(1)
    print(f"  ✅ Gemini API key: ...{GEMINI_API_KEY[-6:]}")

    print("\n🚀 All preflight checks passed. Starting executor loop.\n")


# ──────────────────────────────────────────────────────────────────────────────
# 6. MAIN LOOP
# ──────────────────────────────────────────────────────────────────────────────

async def main(rounds: int = EXECUTOR_ROUNDS):
    """
    Main executor loop:
      1. Get first test case from planner
      2. Execute on device
      3. Log verdict, get next test case
      4. Repeat for N rounds
    """
    preflight()

    results_log = []

    # ── Get the first test case ──────────────────────────────────────────
    _print_header("PLANNER → GENERATING FIRST TEST CASE")
    planner_data = get_next_testcase()
    tc = planner_data.get("next_testcase", {})

    # Log planner's RAG interaction to cloud
    _log_planner_trace(planner_data, "first")

    if not tc or not tc.get("steps"):
        print("❌ Planner returned empty test case. Aborting.")
        return

    print("Generated test case:")
    _show_testcase(tc)

    # ── Execute loop ─────────────────────────────────────────────────────
    for i in range(1, rounds + 1):
        _print_header(f"ROUND {i}/{rounds}")

        # Execute on device
        exec_result = await execute_test_on_device(tc)
        verdict = exec_result["verdict"]
        notes = exec_result["notes"]
        duration = exec_result["duration_seconds"]

        # Show round results
        _show_round_summary(i, tc, verdict, notes, duration)

        results_log.append({
            "round": i,
            "test_case_id": tc.get("test_case_id"),
            "title": tc.get("title"),
            "screen": tc.get("screen", "?"),
            "area": tc.get("area", "?"),
            "priority": tc.get("priority", "?"),
            "steps": tc.get("steps", []),
            "expected_result": tc.get("expected_result", ""),
            "verdict": verdict,
            "duration": duration,
            "notes": notes,
        })

        # Log verdict and get next test case (if not last round)
        if i < rounds:
            _print_header(f"PLANNER → LOGGING VERDICT & GENERATING NEXT TEST CASE")
            response = log_verdict_and_get_next(tc, verdict, notes)
            log_info = response.get("log", {})
            next_data = response.get("next", {})
            tc = next_data.get("next_testcase", {})

            print(f"  Logged: {log_info.get('test_case_id')} | {verdict} | {log_info.get('run_id', '?')}")

            # Log the planner's RAG interaction for next test case
            _log_planner_trace(next_data, f"round-{i}")

            # Log the verdict to cloud
            cloud_log(
                "info",
                f"Verdict logged: {log_info.get('test_case_id')} → {verdict}",
                test_case_id=log_info.get("test_case_id"),
                run_id=log_info.get("run_id", "?"),
                verdict=verdict,
            )

            if tc and tc.get("steps"):
                print("\n  Next test case generated:")
                _show_testcase(tc)
            else:
                print("  ⚠️  Planner returned empty next test case. Ending loop.")
                break
        else:
            # Last round — just log the verdict, no need for next test case
            _print_header("LOGGING FINAL VERDICT")
            try:
                log_info = log_verdict_only(tc, verdict, notes)
                print(f"  Logged: {log_info.get('test_case_id', tc.get('test_case_id'))} | {verdict}")
            except Exception as e:
                print(f"  ⚠️  Failed to log final verdict: {e}")
                print(f"  (Results are still recorded in the summary below)")

    # ── Summary ────────────────────────────────────────────────────────────
    _print_header("EXECUTION SUMMARY")
    total = len(results_log)
    passed = sum(1 for r in results_log if r["verdict"] == "pass")
    failed = total - passed
    total_duration = sum(r["duration"] for r in results_log)

    print(f"  Total Rounds:    {total}")
    print(f"  Passed:          {passed} ✅")
    print(f"  Failed:          {failed} ❌")
    print(f"  Total Duration:  {total_duration:.1f}s")
    print(f"  Pass Rate:       {(passed/total*100) if total else 0:.0f}%")

    cloud_log(
        "info",
        f"Execution complete: {passed}/{total} passed ({(passed/total*100) if total else 0:.0f}%)",
        total_rounds=total,
        passed=passed,
        failed=failed,
        total_duration_seconds=round(total_duration, 1),
        pass_rate=round((passed/total*100) if total else 0, 1),
        test_results=[
            {
                "test_case_id": r["test_case_id"],
                "title": r["title"],
                "verdict": r["verdict"],
                "duration": round(r["duration"], 1),
                "area": r.get("area", "?"),
            }
            for r in results_log
        ],
    )

    # ── Detailed per-test report ─────────────────────────────────────────────
    for r in results_log:
        status = "✅ PASS" if r["verdict"] == "pass" else "❌ FAILED"
        print(f"\n  {'─' * 66}")
        print(f"  Round {r['round']}: {r['test_case_id']} | {status} | {r['duration']:.1f}s")
        print(f"  {'─' * 66}")
        print(f"    Title:    {r['title']}")
        print(f"    Screen:   {r['screen']}")
        print(f"    Area:     {r['area']}")
        print(f"    Priority: {r['priority']}")

        steps = r.get("steps", [])
        if steps:
            print(f"    Steps ({len(steps)}):")
            for j, s in enumerate(steps, 1):
                print(f"      {j}. {s}")

        print(f"    Expected: {r.get('expected_result', '?')}")
        print(f"    Verdict:  {status}")
        print(f"    Duration: {r['duration']:.1f}s")

        # Extract the reason from notes (after "Reason: ")
        notes_str = r.get("notes", "")
        if "Reason: " in notes_str:
            reason = notes_str.split("Reason: ", 1)[1]
        else:
            reason = notes_str
        print(f"    Reason:   {reason[:300]}")

    # ── Check recent tests in Neo4j ─────────────────────────────────────────
    try:
        recent = requests.get(
            f"{RAG_URL}/tests/recent",
            params={"project": PROJECT, "limit": 10},
            timeout=60,
        )
        recent.raise_for_status()
        tests = recent.json().get("tests", [])
        _print_header("RECENT TESTS IN NEO4J")
        for t in tests:
            print(f"  - {t.get('id')} | {t.get('verdict')} | {t.get('title')}")
    except Exception as e:
        print(f"  ⚠️  Could not fetch recent tests: {e}")


if __name__ == "__main__":
    asyncio.run(main(rounds=EXECUTOR_ROUNDS))
