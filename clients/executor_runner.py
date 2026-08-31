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
import base64
import logging
import os
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

# ── Force UTF-8 console output ────────────────────────────────────────────────
# droidrun/mobilerun logs emoji and arrows; the Windows console defaults to
# cp1252 and raises UnicodeEncodeError on every such line. Reconfiguring the
# existing stdout/stderr objects in place fixes it for all logging handlers.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# ── Configuration ────────────────────────────────────────────────────────────
# Every tunable lives in settings.py (single source of truth); nothing in this
# file reads the environment directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from settings import (  # noqa: E402
    GATEWAY_URL, RAG_URL, PROJECT, APP_NAME, TOP_K, DEBUG_TRACE,
    GEMINI_API_KEY, OPENROUTER_API_KEY,
    EXECUTOR_LLM_PROVIDER, EXECUTOR_LLM_MODEL, EXECUTOR_TIMEOUT, EXECUTOR_ROUNDS,
    EXECUTOR_MAX_STEPS, EXECUTOR_MAX_TOKENS, EXECUTOR_CONTEXT_WINDOW,
    SELF_HEAL, TARGET_APP_PACKAGE, LOGTAIL_SOURCE_TOKEN, EXECUTOR_VISION, CLEAN_SLATE,
    DEVICE_RESET, DATA_PROVIDER_PACKAGES, DEVICE_RESET_SCOPE, CLEAN_SLATE_APPMODEL,
    TARGET_APP_ONLY, DEVICE_FIXTURE_DIR, DEVICE_FIXTURE_DEST, COMPANION_PACKAGES,
    UNACHIEVABLE_PRECONDITIONS,
    app_login_block, app_identity_block, device_input_block, verification_block,
)
from observability import degradations  # noqa: E402

GATEWAY_URL = GATEWAY_URL.rstrip("/")
RAG_URL = RAG_URL.rstrip("/")

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

# ── Live mobilerun log file ───────────────────────────────────────────────────
# mobilerun logs its agent "thinking"/actions to the "mobilerun" logger (Rich
# console, propagate=False). We ALSO tee it to logs/mobilerun.log so the run is
# observable live and the dashboard can stream it. Attached after the mobilerun
# import (below) so it survives mobilerun's own logging configuration.
_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(_LOG_DIR, exist_ok=True)
MOBILERUN_LOG = os.path.join(_LOG_DIR, "mobilerun.log")
_mobilerun_file_attached = False


class _StreamReassemblingHandler(logging.FileHandler):
    """Join mobilerun's token-by-token stream back into readable lines.

    mobilerun emits each generated token as its own log record (with
    ``extra={"stream": True}``), which turned the log into a column of single
    words and made the agent's reasoning impossible to follow. Streamed records
    are buffered and flushed as one line when the stream ends or a normal record
    arrives.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._buf: list[str] = []

    def _flush_buffer(self) -> None:
        if not self._buf:
            return
        text = "".join(self._buf).strip()
        self._buf.clear()
        if text:
            rec = logging.LogRecord("mobilerun", logging.INFO, "", 0, "%s", (text,), None)
            super().emit(rec)

    def emit(self, record: logging.LogRecord) -> None:
        if getattr(record, "stream", False):
            self._buf.append(record.getMessage())
            return
        if getattr(record, "stream_end", False):
            self._flush_buffer()
            return
        self._flush_buffer()
        super().emit(record)


def _attach_mobilerun_file_log():
    """Idempotently tee the mobilerun + executor loggers to logs/mobilerun.log."""
    global _mobilerun_file_attached
    if _mobilerun_file_attached:
        return
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s", "%H:%M:%S")
    # UTF-8 explicitly: mobilerun logs emoji (📁 🚀 🔄 💡) and on Windows a
    # FileHandler defaults to the locale encoding (cp1252), which raises
    # UnicodeEncodeError on every such line ("--- Logging error ---" spam).
    fh = logging.FileHandler(MOBILERUN_LOG, encoding="utf-8", errors="replace")
    fh.setFormatter(fmt)
    for name in ("mobilerun", "executor"):
        lg = logging.getLogger(name)
        lg.addHandler(fh)
        if lg.level == logging.NOTSET:
            lg.setLevel(logging.INFO)
    logging.getLogger("mobilerun").propagate = False
    _mobilerun_file_attached = True


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

def get_next_testcase(max_new_tokens: int = 8000) -> dict:
    """Ask the planner gateway for the next test case."""
    resp = requests.post(
        f"{GATEWAY_URL}/agent/next-testcase",
        json={
            "project": PROJECT,
            "app_name": APP_NAME,
            "objective": "generate next high-value non-duplicate test case",
            "top_k": TOP_K,
            "max_new_tokens": max_new_tokens,
            "max_retrieval_rounds": 2,   # 2 rounds keeps latency under ~60s
            "enable_thinking": False,
            "debug_trace": DEBUG_TRACE,
        },
        timeout=900,
    )
    resp.raise_for_status()
    return resp.json()


def log_verdict_and_get_next(
    tc: dict, verdict: str, notes: str, max_new_tokens: int = 8000
) -> dict:
    """Log the execution verdict then separately fetch the next test case.

    Splitting into two calls gives each its own timeout budget and prevents
    the combined gateway call from timing out mid-generation when the LLM
    is slow (each LLM round can take 30-60s on OpenRouter free tier).
    """
    # Step 1: Log verdict only (fast — just a Neo4j write, no LLM call)
    log_info = log_verdict_only(tc, verdict, notes)

    # Step 2: Fetch next test case as a fully independent request with its own timeout
    next_data = get_next_testcase(max_new_tokens=max_new_tokens)

    return {"log": log_info, "next": next_data}


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

# Unachievable preconditions are configured (settings.UNACHIEVABLE_PRECONDITIONS):
# a phrase that is impossible in one app is ordinary setup in another.


def filter_preconditions(preconditions: list[str]) -> tuple[list[str], list[str]]:
    """Split preconditions into (keep, dropped-as-unachievable)."""
    keep, dropped = [], []
    for p in preconditions or []:
        (dropped if any(k in str(p).lower() for k in UNACHIEVABLE_PRECONDITIONS) else keep).append(p)
    return keep, dropped


def build_droidrun_goal(test_case: dict) -> str:
    """
    Convert the planner's structured JSON test case into a natural language
    goal string that Droidrun's LLM agent can interpret and execute.

    The planner names a LIKELY screen and states an OBJECTIVE — it does not
    prescribe exact taps. It has no live view of the app, so a literal step
    script is often simply wrong: measured across one real campaign, the
    planner's named screen matched a real observed screen only ~16% of the
    time. The executor has live vision and the real accessibility tree, so it
    is better placed to work out HOW than the planner is to dictate it. This
    goal states WHAT to verify and trusts the executor to find its own way,
    correcting course when the screen_hint doesn't match what it actually finds.

    Example output:
      "Open the Contacts app. I believe the relevant screen is 'Create Contact'
       — that is a lead, not a fact.
       Your objective: create a new contact and verify it is saved and appears
       in the list."
    """
    screen_hint = test_case.get("screen_hint", "")
    preconditions = test_case.get("preconditions", [])
    objective = test_case.get("objective", "")
    expected = test_case.get("expected_result", "")

    # Build the goal instruction
    goal_parts = []

    # Which app to work with, and how to open it deterministically. Defined once
    # in settings.app_identity_block() so every caller says the same thing.
    goal_parts.append(app_identity_block())

    # Credentials, when the app is gated behind a login the agent cannot create
    # for itself (OTP to a phone we do not own, NID upload, admin approval).
    login = app_login_block()
    if login:
        goal_parts.append(login)

    # Screen hint — a lead, not an instruction to obey against contrary evidence.
    if screen_hint:
        goal_parts.append(
            f"I believe the relevant screen is '{screen_hint}' — that is a LEAD to check, not "
            f"a fact; I cannot see the live app. If it does not exist or isn't right, explore and "
            f"find the real screen for this objective yourself. Do not stall trying to match that "
            f"name exactly."
        )

    # Preconditions as context
    kept_pre, dropped_pre = filter_preconditions(preconditions)
    if dropped_pre:
        print(f"   ✂️  Dropped unachievable precondition(s): {dropped_pre}")
    if kept_pre:
        goal_parts.append(f"Preconditions (create these yourself if missing): {' '.join(kept_pre)}")
    # Only claim a clean slate when one was actually taken. Telling the agent the
    # app was reset when it was not sends it hunting for absent data.
    if DEVICE_RESET == "pm_clear":
        goal_parts.append(
            "The app has just been reset to a clean state. If the test needs data to exist, "
            "CREATE it as your first steps. Do not abandon the test because data is missing."
        )
    else:
        goal_parts.append(
            "The app keeps state from earlier tests. If the test needs data that is missing, "
            "CREATE it as your first steps. Do not abandon the test because data is missing."
        )

    goal_parts.append(device_input_block())
    goal_parts.append(verification_block())

    # Objective — state WHAT to verify; deciding HOW (which screens, which taps) is the
    # executor's job, since it has live information the planner does not.
    if objective:
        goal_parts.append(f"\nYour objective: {objective}")

    # Expected outcome
    if expected:
        goal_parts.append(f"\nExpected result: {expected}")

    # Final instruction to Droidrun
    goal_parts.append(
        "\nDecide the concrete steps yourself from what you actually see on screen — you have "
        "live access to the app that the objective above does not. After acting, report whether "
        "the expected result was achieved. If the screen or feature described above genuinely "
        "does not seem to exist after a reasonable search, say so explicitly rather than "
        "continuing to search indefinitely. If any step fails or the app crashes, report the "
        "failure."
    )

    return "\n".join(goal_parts)


# ──────────────────────────────────────────────────────────────────────────────
# 2c. SELF-HEALING (WP7 / ETA-REQ-305): classify failures + adaptive recovery
# ──────────────────────────────────────────────────────────────────────────────

# Failure category -> recovery strategy (REQ-305.1 / 305.2). Pure + app-agnostic.
_RECOVERY = {
    "NAVIGATION_FAILURE": {"action": "try an alternate navigation path from the learned nav tree", "retry": True},
    "ELEMENT_NOT_FOUND": {"action": "wait for the screen to settle and re-locate the element, or use a similar label", "retry": True},
    "ASSERTION_FAILURE": {"action": "capture the actual post-action state and log it as a potential defect", "retry": False},
    "TIMEOUT": {"action": "retry with an extended timeout", "retry": True},
    "CRASH": {"action": "restart the app and resume from the last stable screen", "retry": True},
    "PERMISSION_DENIED": {"action": "grant the required permission/precondition, then retry", "retry": True},
    # Budget exhaustion, not app misbehaviour: the agent simply ran out of steps.
    # No retry — the retry gets the same budget and would exhaust it again.
    "STEP_LIMIT_EXCEEDED": {"action": "ran out of steps before finishing; raise EXECUTOR_MAX_STEPS or simplify the test", "retry": False},
}


def classify_failure(reason: str, success: bool = False) -> str:
    """Classify a Droidrun failure reason into a category (REQ-305.1). Pure."""
    if success:
        return ""
    r = (reason or "").lower()
    # Checked FIRST: the run never exercised the app, so this is a test-data /
    # environment problem, NOT app misbehaviour. Keeping it out of the defect
    # categories stops it from inflating defect-discovery and strategy scores.
    if any(k in r for k in ("precondition not met", "preconditions not met",
                            "precondition failed", "preconditions are not met")):
        return "PRECONDITION_NOT_MET"
    # Also not app misbehaviour: the step budget ran out mid-test. Without this it
    # falls through to ASSERTION_FAILURE and is counted as a discovered defect.
    if any(k in r for k in ("max step count", "max steps", "step limit")):
        return "STEP_LIMIT_EXCEEDED"
    if any(k in r for k in ("permission", "denied", "not granted")):
        return "PERMISSION_DENIED"
    if any(k in r for k in ("crash", "terminated", "closed unexpectedly", "anr", "has stopped", "force close")):
        return "CRASH"
    if any(k in r for k in ("timeout", "timed out", "unresponsive", "no response")):
        return "TIMEOUT"
    if any(k in r for k in ("not found", "no such element", "could not find", "couldn't find",
                            "element missing", "no element", "unable to locate")):
        return "ELEMENT_NOT_FOUND"
    if any(k in r for k in ("could not reach", "navigat", "wrong screen", "unable to open",
                            "screen not", "did not reach")):
        return "NAVIGATION_FAILURE"
    return "ASSERTION_FAILURE"


def recovery_strategy(category: str) -> dict:
    """Recovery strategy descriptor for a failure category (REQ-305.2). Pure."""
    return _RECOVERY.get(category, {"action": "re-attempt with a fresh observation", "retry": False})


def build_retry_goal(test_case: dict, category: str, reason: str, strategy: dict, extra: str = "") -> str:
    """Retry goal with a `## Previous Failure Context` block (REQ-305.3). Pure."""
    base = build_droidrun_goal(test_case)
    block = [
        "",
        "## Previous Failure Context",
        f"The previous attempt FAILED (classified as {category}).",
        f"What went wrong: {(reason or 'unknown')[:300]}",
        f"Recovery approach to apply now: {strategy.get('action', 're-attempt')}.",
    ]
    if extra:
        block.append(extra)
    block.append("Adjust your steps accordingly and re-attempt the goal.")
    return base + "\n" + "\n".join(block)


def _learned_nav_hint(test_case: dict) -> str:
    """Best-effort learned shortest path to the target screen (for NAVIGATION_FAILURE recovery)."""
    screen = test_case.get("screen_hint", "")
    if not screen:
        return ""
    try:
        data = requests.get(f"{RAG_URL}/navtree/retrieve-path",
                            params={"project": PROJECT, "screen": screen}, timeout=15).json()
        steps = data.get("steps", []) or []
        if steps:
            path = "; ".join(f"{s.get('action','')}->{s.get('screen','')}" for s in steps)
            return f"Proven path to '{screen}': {path}"
    except Exception as e:
        # Losing the learned route is REQ-302 quietly ceasing to pay off. The
        # only symptom is a step count that stops improving, which is invisible
        # unless it is said out loud.
        _degrade("navtree_retrieval_failed", degradations.MAJOR,
                      f"learned navigation paths unavailable ({e}); the executor "
                      f"re-explores routes it has already proven")
    return ""


# ──────────────────────────────────────────────────────────────────────────────
# 2b. LIVE APP MODEL — feed the executor's real trajectory into the graph (WP1)
# ──────────────────────────────────────────────────────────────────────────────


_DEGRADED_COUNTS: dict[str, int] = {}
# Emit an event on these occurrence numbers only. Recording every occurrence
# floods the report from a per-observation hot path; recording just the first
# tells you something degraded but not whether it hit 3 observations or 300 —
# which is the difference between a footnote and a void run.
_DEGRADE_REPORT_AT = (1, 2, 5, 10, 25, 50, 100, 250, 500, 1000)


def _degrade(kind: str, severity: str, detail: str) -> None:
    """Count every occurrence; emit an event at widening intervals."""
    n = _DEGRADED_COUNTS.get(kind, 0) + 1
    _DEGRADED_COUNTS[kind] = n
    if n not in _DEGRADE_REPORT_AT:
        return
    suffix = "" if n == 1 else f" [occurrence {n}]"
    try:
        degradations.record(kind, severity, detail=f"{detail}{suffix}", occurrences=n)
    except Exception:
        pass
    cloud_log("warning", f"DEGRADED [{severity}] {kind} (x{n}): {detail}")


def degradation_counts() -> dict[str, int]:
    """Final per-kind occurrence counts, for the end-of-run report."""
    return dict(_DEGRADED_COUNTS)


def adapt_driver_tree(tree) -> dict:
    """Reshape ``AndroidDriver.get_ui_tree()`` output for ``normalize_ui_state``.

    The driver returns ``{'a11y_tree': <root node dict>, 'phone_state':
    {'packageName': ...}}`` but the normalizer expects a LIST of elements and a
    phone_state keyed ``package``. Handed the raw output it walks a dict as if it
    were a list, yields zero nodes, and reports package=None — so every observed
    screen arrived with no controls, no content-descriptions and no package.

    That is what made the app model useless on this app: states were labelled
    ``android.view.View #2``, identity had nothing but class names to compare,
    the package/activity bucket that keeps launcher and system UI apart was
    empty, and every mined error pattern and anomaly was keyed to a meaningless
    name. Wrapping the root node in a list and mapping ``packageName`` took the
    same screen from 0 usable nodes to 24, with 7 content-descriptions and 12
    clickable controls.
    """
    if not isinstance(tree, dict):
        return tree
    root = tree.get("a11y_tree")
    if root is None and "elements" not in tree and "nodes" not in tree:
        return tree                      # not driver output; pass through
    ps = tree.get("phone_state") or {}
    dc = tree.get("device_context") or {}
    root_pkg = root.get("packageName") if isinstance(root, dict) else None
    return {
        "elements": [root] if isinstance(root, dict) else (root or []),
        "phone_state": {
            "package": (ps.get("packageName") or ps.get("package")
                        or dc.get("packageName") or root_pkg or ""),
            "activity": (ps.get("activityName") or ps.get("activity")
                         or dc.get("activity") or ""),
        },
    }


# uiautomator dump is racy while the UI animates; a couple of quick retries
# turn a transient miss into a clean read instead of a degraded observation.
_UI_TREE_ATTEMPTS = 3
_UI_TREE_RETRY_DELAY = 0.6


async def _assert_portal(driver) -> bool:
    """Check the on-device portal is actually in use, and say so when it is not.

    mobilerun defaults to portal_mode="auto": it tries the portal and silently
    drops to ADB when it is missing. That fallback is not equivalent —
    ``adb input text`` races the IME and drops characters, its clear-before-type
    cannot verify itself and reports success having cleared nothing, and
    ``uiautomator dump`` returns no XML while the screen animates. Every run we
    had done was on that path, and nothing said so: a field holding
    "Trust Dairy Farm 1" became "Trust Dairy  FarTest FTest Farm 1rm 1" while the
    tool logged success, and the agent looked incompetent for it.

    `portal_available` was a property sitting there returning False the whole
    time. Nothing read it. Now a degraded device is a first-class, visible fact.
    """
    try:
        await driver.ensure_connected()
        ok = bool(getattr(driver, "portal_available", False))
        keyboard = bool(getattr(driver, "_portal_keyboard_available", False))
    except Exception as e:
        _degrade("device_portal_check_failed", degradations.MAJOR,
                      f"could not determine whether the device portal is active ({e})")
        return False

    if ok and keyboard:
        print("   🔌 Device portal active (reliable text input + state feed)")
        return True

    # Try once to repair it before giving up. The portal probe can miss while the
    # accessibility service is still starting, and a driver reconnected between
    # rounds then spends the rest of the suite on the ADB path — which is how
    # round 1 ran with the portal and round 2 silently did not.
    try:
        from mobilerun_core_cli.portal import ensure_portal_ready
        await ensure_portal_ready(driver.device)
        driver._connected = False           # force a fresh probe
        await driver.ensure_connected()
        ok = bool(getattr(driver, "portal_available", False))
        keyboard = bool(getattr(driver, "_portal_keyboard_available", False))
        if ok and keyboard:
            print("   🔌 Device portal recovered after re-setup")
            _degrade("device_portal_recovered", degradations.MINOR,
                          "portal was inactive at connect and was re-established")
            return True
    except Exception as e:
        cloud_log("warning", f"Portal re-setup failed: {e}")

    _degrade(
        "device_portal_unavailable", degradations.CRITICAL,
        f"portal_available={ok} keyboard={keyboard}; falling back to ADB input, "
        f"which drops characters, cannot reliably clear a field, and races "
        f"uiautomator — text-entry results from this run are not trustworthy. "
        f"Install it with mobilerun_core_cli.portal.ensure_portal_ready(driver.device).",
    )
    return False


async def _observe_device(driver, fallback):
    """Current screen from the driver, falling back to the streamed event payload.

    The event's ``ui_state`` is far poorer than what the driver can report, so we
    ask the device directly and only fall back if that fails.
    """
    if driver is not None:
        # `uiautomator dump` returns no XML while the screen is mid-animation,
        # which during an active run is most of the time, and get_ui_tree() does
        # not retry (mobilerun's own get_state does). One attempt therefore drops
        # us onto the sparse event payload for transient reasons, and the app
        # model silently loses that observation's controls and package.
        last = None
        for attempt in range(_UI_TREE_ATTEMPTS):
            try:
                return adapt_driver_tree(await driver.get_ui_tree())
            except Exception as e:
                last = e
                if attempt + 1 < _UI_TREE_ATTEMPTS:
                    await asyncio.sleep(_UI_TREE_RETRY_DELAY)
        _degrade("driver_ui_tree_failed", degradations.MAJOR,
                      f"cannot read the accessibility tree from the driver after "
                      f"{_UI_TREE_ATTEMPTS} attempts ({last}); falling back to the "
                      f"sparser event payload, which carries no content-descriptions "
                      f"and no package")
    return fallback


def reset_device_app() -> None:
    """Put the app back to a known state before a test runs.

    Tests that create data (contacts, drafts) leave it behind, so by the sixth run
    the app held 30 contacts and any test whose setup assumed an empty app could
    never satisfy it. Resetting also clears half-filled forms, which previously
    leaked into the next test.
    """
    import subprocess
    if DEVICE_RESET == "none" or not TARGET_APP_PACKAGE:
        return
    try:
        if DEVICE_RESET == "pm_clear":
            # The UI package first, then the providers that actually hold the data.
            for pkg in (TARGET_APP_PACKAGE, *DATA_PROVIDER_PACKAGES):
                subprocess.run(["adb", "shell", "pm", "clear", pkg],
                               capture_output=True, timeout=60)
        else:
            subprocess.run(["adb", "shell", "am", "force-stop", TARGET_APP_PACKAGE],
                           capture_output=True, timeout=30)
        subprocess.run(["adb", "shell", "monkey", "-p", TARGET_APP_PACKAGE,
                        "-c", "android.intent.category.LAUNCHER", "1"],
                       capture_output=True, timeout=60)
        time.sleep(2)  # let the launcher settle before the first observation
        print(f"   🧹 Device reset ({DEVICE_RESET}) for {TARGET_APP_PACKAGE}")
        seed_device_fixtures()
    except Exception as e:
        cloud_log("warning", f"Device reset failed: {e}")


def seed_device_fixtures() -> None:
    """Put test media back on the device after a reset.

    `pm clear` removes the app's MediaStore contributions, and /sdcard/Pictures
    is owned by the app's uid — so clearing the app deletes any images seeded
    there. Every image-dependent test then fails for want of a file to pick,
    which looks like an app defect and is actually the reset eating its own
    fixtures. Re-seeding is part of resetting, not a separate manual step.
    """
    import subprocess
    if not DEVICE_FIXTURE_DIR or not os.path.isdir(DEVICE_FIXTURE_DIR):
        return
    files = sorted(
        f for f in os.listdir(DEVICE_FIXTURE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    )
    if not files:
        return
    pushed = 0
    for name in files:
        src = os.path.join(DEVICE_FIXTURE_DIR, name)
        dst = f"{DEVICE_FIXTURE_DEST.rstrip('/')}/{name}"
        r = subprocess.run(["adb", "shell", "mkdir", "-p", DEVICE_FIXTURE_DEST],
                           capture_output=True, timeout=30)
        r = subprocess.run(["adb", "push", src, dst], capture_output=True, timeout=60)
        if r.returncode != 0:
            continue
        # Without a media scan the file exists on disk but no picker can see it.
        subprocess.run(["adb", "shell", "am", "broadcast",
                        "-a", "android.intent.action.MEDIA_SCANNER_SCAN_FILE",
                        "-d", f"file://{dst}"], capture_output=True, timeout=30)
        pushed += 1
    if pushed:
        print(f"   🖼️  Seeded {pushed} test image(s) into {DEVICE_FIXTURE_DEST}")
    else:
        _degrade("device_fixtures_missing", degradations.MAJOR,
                      f"could not seed test media from {DEVICE_FIXTURE_DIR}; "
                      f"any test needing an image will fail for lack of a file")


async def _safe_screenshot(driver):
    """Best-effort current-device screenshot (bytes), or None."""
    if driver is None:
        return None
    try:
        return await driver.screenshot()
    except Exception:
        return None


def _record_observations(observations, tc_id: str = "", driver_shot: bytes | None = None) -> int:
    """Post each observed (ui_state, screenshot) into the Live App Model.

    ``observations`` is a list of ``(elements_list, screenshot_bytes)`` captured
    per UI state from mobilerun's event stream during the run. Each is normalized
    and POSTed to ``/liveui/observe`` — which dedupes into the graph (structural
    signature + visual fallback) and links the transition. This is the
    execution->learning loop: exploring builds a reusable, screenshotted map.
    """
    if not observations:
        return []
    try:
        from mobilerun.macro.state import normalize_ui_state
    except Exception as e:
        # This one handler disables the ENTIRE Live App Model: no UIState nodes,
        # no transitions, no navigation memory — i.e. the data source for
        # REQ-302 and REQ-303 — while the run still reports success.
        _degrade("live_app_model_disabled", degradations.CRITICAL,
                      f"cannot import normalize_ui_state ({e}); no UI states, "
                      f"transitions or navigation memory will be recorded")
        return []

    prev_id = None
    path = []  # ordered [{id, label}] — the route this test walked through the app model
    last_i = len(observations) - 1
    for i, (elements, shot) in enumerate(observations):
        if not elements:
            continue
        try:
            normalized = normalize_ui_state(elements)
        except Exception as e:
            # A dropped observation leaves a hole in the transition graph that
            # is indistinguishable from a screen the agent never reached.
            _degrade("observation_normalise_failed_x", degradations.MAJOR,
                          f"dropping observations the app model cannot normalise ({e}); "
                          f"the transition graph will have gaps")
            continue

        # The app model must contain the APP, not whatever was on screen. Without
        # this the launcher, the status bar and permission dialogs became "app
        # states" — 3 of 10 in the first real run — and, because they share the
        # (empty) package/activity bucket with real screens, they competed to
        # match them on similarity too.
        obs_pkg = ((normalized.get("phone_state") or {}).get("package") or "").strip()
        if TARGET_APP_PACKAGE and obs_pkg and obs_pkg != TARGET_APP_PACKAGE:
            if obs_pkg not in COMPANION_PACKAGES:
                continue

        raw_shot = shot if shot else (driver_shot if i == last_i else None)
        shot_b64 = None
        if raw_shot:
            try:
                shot_b64 = base64.b64encode(raw_shot).decode("ascii")
            except Exception as e:
                _degrade("screenshot_encode_failed", degradations.MINOR,
                              f"state screenshots unavailable ({e}); the app model "
                              f"will have nodes with no image")
                shot_b64 = None
        try:
            resp = requests.post(
                f"{RAG_URL}/liveui/observe",
                json={
                    "project": PROJECT,
                    "normalized": normalized,
                    "screenshot_b64": shot_b64,
                    "from_state_id": prev_id,
                    "action": f"step {i + 1}",
                },
                timeout=30,
            )
            resp.raise_for_status()
            j = resp.json()
            prev_id = j.get("state_id")
            path.append({"id": prev_id, "label": j.get("label", "")})
        except Exception as e:
            cloud_log("warning", f"App model observe failed for {tc_id} step {i}: {e}")
            break

    if path:
        shots = sum(1 for _e, s in observations if s)
        print(f"   🗺  App model: recorded {len(path)} UI state(s), {shots} with screenshots")
        cloud_log("info", f"App model updated for {tc_id}", test_case_id=tc_id, states_recorded=len(path))
    return path


# Cached device environment (adb serial + OS version) for execution logs.
_DEVICE_ENV = None


def _device_env() -> dict:
    global _DEVICE_ENV
    if _DEVICE_ENV is not None:
        return _DEVICE_ENV
    import subprocess

    def sh(args):
        try:
            return subprocess.run(args, capture_output=True, text=True, timeout=8).stdout.strip()
        except Exception:
            return ""
    serial = ""
    for line in sh(["adb", "devices"]).splitlines()[1:]:
        if "\tdevice" in line:
            serial = line.split("\t")[0]
            break
    _DEVICE_ENV = {"device": serial, "os": sh(["adb", "shell", "getprop", "ro.build.version.release"])}
    return _DEVICE_ENV


def _log_execution(tc: dict, verdict: str, duration_ms: float, device_steps: int,
                   path: list, error_type: str = "", error_message: str = "",
                   recovery_action: str = "") -> str:
    """Persist one execution record (WP3/WP7): timing, steps, environment, walked
    path, classified failure category, and any self-healing recovery outcome.

    Returns the created ExecutionLog's exact id (empty string on failure) — used
    to attach the trajectory evaluator's assessment to precisely this record,
    not a 'most recent' guess."""
    env = _device_env()
    payload = {
        "project": PROJECT,
        "test_case_id": tc.get("test_case_id", ""),
        "title": tc.get("title", ""),
        "verdict": verdict,
        "duration_ms": int(duration_ms),
        "planned_steps": len(tc.get("steps", []) or []),
        "device_steps": int(device_steps or 0),
        "states_visited": len(path),
        "error_type": error_type,
        "error_message": (error_message or "")[:500],
        "recovery_action": (recovery_action or "")[:300],
        "device": env.get("device", ""),
        "os_version": env.get("os", ""),
        "app_package": TARGET_APP_PACKAGE,
        "path": [s["id"] for s in path],
        "path_labels": [s["label"] for s in path],
    }
    try:
        resp = requests.post(f"{RAG_URL}/execution/log", json=payload, timeout=30)
        resp.raise_for_status()
        print(f"   📋 Execution logged: {len(path)} states, verdict={verdict}")
        return str(resp.json().get("log_id") or "")
    except Exception as e:
        cloud_log("warning", f"Execution log failed for {tc.get('test_case_id')}: {e}")
        return ""


_TRAJECTORY_DIR = os.path.join("logs", "trajectories")


def _trajectory_snapshot() -> set:
    """Folder names under logs/trajectories right now — call before agent.run()
    so the run's own folder can be identified by diffing afterward, instead of
    guessing by timestamp."""
    try:
        return set(os.listdir(_TRAJECTORY_DIR))
    except OSError:
        return set()


def _evaluate_run(tc: dict, log_id: str, exec_path: list, traj_before: set | None) -> None:
    """Best-effort: ask the gateway to evaluate this run's trajectory against the
    test's own objective, so the planner learns what actually happened instead of
    a terse verdict. Fire-and-check, never blocks or fails the primary pipeline —
    a failure here is a missed learning opportunity, not a broken test run."""
    if not log_id or traj_before is None:
        return  # nothing to attach to, or agent.run() was never reached
    try:
        after = _trajectory_snapshot()
        new_folders = after - traj_before
        if not new_folders:
            return
        # The common case is exactly one new folder; if self-heal ran a second
        # agent, the most recently modified one is the one worth evaluating.
        folder = max(new_folders, key=lambda n: os.path.getmtime(os.path.join(_TRAJECTORY_DIR, n)))
        # /execution/evaluate is a GATEWAY endpoint (it calls the model), not a
        # rag_api one — posting to RAG_URL here silently hit a nonexistent route.
        resp = requests.post(f"{GATEWAY_URL}/execution/evaluate", json={
            "project": PROJECT,
            "log_id": log_id,
            "test_case_id": tc.get("test_case_id", ""),
            "title": tc.get("title", ""),
            "objective": tc.get("objective", ""),
            "screen_hint": tc.get("screen_hint", ""),
            "expected_result": tc.get("expected_result", ""),
            "path_labels": [s.get("label", "") for s in exec_path],
            "trajectory_folder": folder,
        }, timeout=60)
        resp.raise_for_status()
    except Exception as e:
        _degrade("trajectory_evaluation_failed", degradations.MINOR,
                 detail=f"could not evaluate run for {tc.get('test_case_id')}: {e}")


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
    from mobilerun import MobileAgent, AndroidDriver, load_llm, MobileConfig, AgentConfig
    from mobilerun.config_manager.config_manager import (
        LoggingConfig, FastAgentConfig, ManagerConfig, ExecutorConfig,
    )
    from workflows.errors import WorkflowTimeoutError
    _attach_mobilerun_file_log()  # tee mobilerun's live logs to logs/mobilerun.log

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
        screen_hint=test_case.get("screen_hint", ""),
        objective=test_case.get("objective", ""),
        expected_result=test_case.get("expected_result", ""),
    )

    if DEVICE_RESET_SCOPE == "test":
        reset_device_app()

    start_time = time.time()
    agent = None
    driver = None
    observations = []  # (elements_list, screenshot_bytes) captured per observed UI state
    # None until agent.run() is actually reached — a setup failure before then means
    # no trajectory folder exists for this run at all, so _evaluate_run must skip
    # rather than diff against an empty set and pick up an unrelated older folder.
    traj_before = None

    try:
        # Set up device driver (connects to default adb device)
        driver = AndroidDriver()
        await _assert_portal(driver)

        # Determine which API key to use based on the provider
        if EXECUTOR_LLM_PROVIDER.lower() == "openrouter":
            api_key = OPENROUTER_API_KEY
        else:
            api_key = GEMINI_API_KEY

        # Set up LLM for Droidrun
        provider = "OpenRouter" if EXECUTOR_LLM_PROVIDER.lower() == "openrouter" else EXECUTOR_LLM_PROVIDER
        llm_kwargs = dict(
            model=EXECUTOR_LLM_MODEL,
            api_key=api_key,
            max_tokens=EXECUTOR_MAX_TOKENS,
            context_window=EXECUTOR_CONTEXT_WINDOW,
        )
        if provider == "OpenRouter":
            # Without this, these calls show up as "Unknown" in OpenRouter's usage
            # dashboard — indistinguishable from every other app on the account.
            # Only OpenRouter's underlying client accepts this kwarg. The referer
            # path (not just the title) must be distinct from the planner-side
            # referers in model_client.py's _APP_REFERERS — OpenRouter identifies
            # an "app" by referer, and a shared referer with a different title
            # would just overwrite this app's display name for its whole history.
            llm_kwargs["default_headers"] = {
                "X-Title": "QA Executor Agent",
                "HTTP-Referer": "https://qa-planner-agent.local/executor",
            }
        llm = load_llm(provider, **llm_kwargs)

        # Create and run the agent. Trajectory capture is enabled so we can feed the
        # real per-step UI states + screenshots into the Live App Model (WP1).
        config = MobileConfig(
            agent=AgentConfig(
                max_steps=EXECUTOR_MAX_STEPS,
                # Screenshots go to the sub-agents that decide and act. Costs image
                # tokens per step, but it is the only signal on screens whose
                # accessibility tree exposes no usable control names.
                fast_agent=FastAgentConfig(vision=EXECUTOR_VISION),
                manager=ManagerConfig(vision=EXECUTOR_VISION),
                executor=ExecutorConfig(vision=EXECUTOR_VISION),
            ),
            logging=LoggingConfig(
                save_trajectory="all",
                trajectory_path="logs/trajectories",
                trajectory_gifs=False,
            ),
        )
        # Pass a single LLM instance so it is used for ALL agent roles
        # (manager/executor/fast_agent/...). Passing a dict makes mobilerun
        # fill missing roles from config defaults (GoogleGenAI) and crash
        # when GOOGLE_API_KEY is not set.
        agent = MobileAgent(
            goal=goal,
            llms=llm,
            driver=driver,
            timeout=EXECUTOR_TIMEOUT,
            config=config,
        )

        # Snapshot before agent.run() so its trajectory folder can be identified by
        # diffing afterward (see _evaluate_run), not guessed from a timestamp.
        traj_before = _trajectory_snapshot()

        # Run the agent, STREAMING its events so we can grab a screenshot for each
        # observed UI state (mobilerun only screenshots itself in vision mode; our
        # text model doesn't, so we capture per-state screenshots ourselves here).
        handler = agent.run()
        try:
            from mobilerun.agent.common.events import RecordUIStateEvent
            async for ev in handler.stream_events():
                if isinstance(ev, RecordUIStateEvent):
                    # Ask the device rather than trusting the event payload: the
                    # streamed ui_state carries no content-descriptions, no
                    # package and clickable=0 for everything.
                    elements = await _observe_device(driver, getattr(ev, "ui_state", None))
                    shot = await _safe_screenshot(driver)
                    observations.append((elements, shot))
        except Exception as e:
            cloud_log("warning", f"Event streaming issue for {tc_id}: {e}")
        result = await handler

        duration = time.time() - start_time

        # Execution -> learning loop: fold observed states into the Live App Model.
        exec_path = []
        try:
            exec_path = _record_observations(observations, tc_id, driver_shot=await _safe_screenshot(driver))
        except Exception as e:
            cloud_log("warning", f"App model recording skipped for {tc_id}: {e}")

        # ResultEvent has: .success (bool), .reason (str), .steps (int)
        success = result.success
        reason = result.reason or "No reason provided by Droidrun"
        steps_taken = result.steps

        # WP7 self-healing (REQ-305): classify the failure and attempt one adaptive
        # recovery before giving up.
        error_type = classify_failure(reason, success)
        recovery_action = ""
        if not success and SELF_HEAL:
            strat = recovery_strategy(error_type)
            if strat["retry"]:
                extra = _learned_nav_hint(test_case) if error_type == "NAVIGATION_FAILURE" else ""
                retry_goal = build_retry_goal(test_case, error_type, reason, strat, extra)
                print(f"\n🔧 Self-heal: {error_type} → {strat['action']} (retrying once)")
                cloud_log("info", f"Self-healing retry for {tc_id}",
                          test_case_id=tc_id, category=error_type, strategy=strat["action"])
                try:
                    retry_timeout = EXECUTOR_TIMEOUT * 2 if error_type == "TIMEOUT" else EXECUTOR_TIMEOUT
                    rec_agent = MobileAgent(goal=retry_goal, llms=llm, driver=driver,
                                            timeout=retry_timeout, config=config)
                    rec_handler = rec_agent.run()
                    try:
                        from mobilerun.agent.common.events import RecordUIStateEvent
                        async for ev in rec_handler.stream_events():
                            if isinstance(ev, RecordUIStateEvent):
                                shot = await _safe_screenshot(driver)
                                observations.append((getattr(ev, "ui_state", None), shot))
                    except Exception:
                        pass
                    rec_result = await rec_handler
                    if rec_result.success:
                        success = True
                        steps_taken += rec_result.steps
                        reason = f"Recovered via self-heal: {rec_result.reason or ''}".strip()
                        recovery_action = f"{error_type}: {strat['action']} -> RECOVERED"
                    else:
                        recovery_action = f"{error_type}: {strat['action']} -> still failed"
                    try:  # fold recovery observations into the app model
                        exec_path = _record_observations(observations, tc_id, driver_shot=await _safe_screenshot(driver))
                    except Exception:
                        pass
                except Exception as e:
                    recovery_action = f"{error_type}: recovery attempt errored ({e})"
            else:
                recovery_action = f"{error_type}: {strat['action']} (no retry — logged for investigation)"

        duration = time.time() - start_time
        verdict = "pass" if success else "failed"
        logged_error_type = "" if success else error_type
        notes = (
            f"Droidrun execution completed in {duration:.1f}s. "
            f"Steps taken: {steps_taken}. "
            f"Success={success}. Reason: {reason}"
            + (f" | Self-heal: {recovery_action}" if recovery_action else "")
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
            error_type=logged_error_type,
            recovery_action=recovery_action,
        )

        log_id = _log_execution(test_case, verdict, duration * 1000, steps_taken, exec_path,  # WP3 + WP7
                                error_type=logged_error_type, error_message=("" if success else reason[:500]),
                                recovery_action=recovery_action)
        _evaluate_run(test_case, log_id, exec_path, traj_before)
        return {"verdict": verdict, "notes": notes, "duration_seconds": duration}

    except asyncio.TimeoutError:
        duration = time.time() - start_time
        notes = (
            f"Droidrun execution TIMED OUT after {EXECUTOR_TIMEOUT}s. "
            f"The test case may be too complex or the device is unresponsive."
        )
        print(f"\n⏰ TIMEOUT after {EXECUTOR_TIMEOUT}s")
        cloud_log("error", f"Test {tc_id} TIMED OUT", test_case_id=tc_id, title=title, timeout=EXECUTOR_TIMEOUT)
        exec_path = []
        try:
            exec_path = _record_observations(observations, tc_id, driver_shot=await _safe_screenshot(driver))  # partial exploration still enriches the map
        except Exception:
            pass
        log_id = _log_execution(test_case, "failed", duration * 1000, len(exec_path), exec_path,
                                error_type="TIMEOUT", error_message=notes)
        _evaluate_run(test_case, log_id, exec_path, traj_before)
        return {"verdict": "failed", "notes": notes, "duration_seconds": duration}

    except WorkflowTimeoutError:
        # mobilerun's own MobileAgent(timeout=EXECUTOR_TIMEOUT) enforces its budget
        # by raising THIS, not asyncio.TimeoutError — so it fell through to the
        # generic crash handler below and was recorded as error_type="CRASH" with
        # no message, silently counting a budget timeout as a candidate app defect.
        # By decision: don't raise EXECUTOR_TIMEOUT/EXECUTOR_MAX_STEPS to chase
        # this — just tag it honestly so a human decides whether to look closer.
        # STEP_LIMIT_EXCEEDED if it genuinely used the full step budget (an
        # environment/budget issue, excluded from defect metrics); otherwise
        # ASSERTION_FAILURE, since timing out with steps still available is not
        # explained by the budget and is worth a manual look.
        duration = time.time() - start_time
        exec_path = []
        try:
            exec_path = _record_observations(observations, tc_id, driver_shot=await _safe_screenshot(driver))
        except Exception:
            pass
        ran_full_budget = len(exec_path) >= EXECUTOR_MAX_STEPS
        error_type = "STEP_LIMIT_EXCEEDED" if ran_full_budget else "ASSERTION_FAILURE"
        notes = (
            f"Workflow timed out after {EXECUTOR_TIMEOUT}s "
            f"({len(exec_path)}/{EXECUTOR_MAX_STEPS} steps observed). "
            + ("Ran the full step budget without finishing."
               if ran_full_budget else
               "Timed out with steps still available - not a step-budget issue; "
               "flagged for manual review of what it was doing when it stalled.")
        )
        print(f"\n⏰ Workflow timeout after {EXECUTOR_TIMEOUT}s -> tagged {error_type}")
        cloud_log("error", f"Test {tc_id} workflow-timed-out -> {error_type}",
                  test_case_id=tc_id, title=title, timeout=EXECUTOR_TIMEOUT, steps=len(exec_path))
        log_id = _log_execution(test_case, "failed", duration * 1000, len(exec_path), exec_path,
                                error_type=error_type, error_message=notes)
        _evaluate_run(test_case, log_id, exec_path, traj_before)
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
        exec_path = []
        try:
            exec_path = _record_observations(observations, tc_id, driver_shot=await _safe_screenshot(driver))  # partial exploration still enriches the map
        except Exception:
            pass
        log_id = _log_execution(test_case, "failed", duration * 1000, len(exec_path), exec_path,
                                error_type="CRASH", error_message=str(e))
        _evaluate_run(test_case, log_id, exec_path, traj_before)
        return {"verdict": "failed", "notes": notes, "duration_seconds": duration}


# ──────────────────────────────────────────────────────────────────────────────
# 4. DISPLAY HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _show_testcase(tc: dict):
    """Pretty-print a generated test case."""
    print(f"  ID:          {tc.get('test_case_id', '?')}")
    print(f"  Title:       {tc.get('title', '?')}")
    print(f"  Screen hint: {tc.get('screen_hint', '?')}")
    print(f"  Area:        {tc.get('area', '?')}")
    print(f"  Priority:    {tc.get('priority', '?')}")
    print(f"  Objective:   {tc.get('objective', '?')}")
    print(f"  Expected:    {tc.get('expected_result', '?')}")


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


def _export_batch_csv(path: str | None = None) -> str:
    """Write one row per executed test: outcome, attribution, and the path walked.

    Batch results otherwise live only in Neo4j and are wiped by the next clean
    slate, so every run's evidence is lost as soon as the following run starts.
    """
    import csv
    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = path or os.path.join(_LOG_DIR, f"batch_{PROJECT}_{stamp}.csv")
    try:
        logs = requests.get(f"{RAG_URL}/execution/logs",
                            params={"project": PROJECT, "limit": 500}, timeout=60).json().get("logs", [])
        graph = requests.get(f"{RAG_URL}/appmodel/graph",
                             params={"project": PROJECT}, timeout=60).json()
        tests = requests.get(f"{RAG_URL}/tests/recent",
                             params={"project": PROJECT, "limit": 500}, timeout=60).json().get("tests", [])
    except Exception as e:
        print(f"  ⚠️  CSV export failed: {e}")
        return ""

    label = {n["id"]: n.get("label", "?") for n in graph.get("nodes", [])}
    meta = {t.get("id"): t for t in tests}
    APP_FAULT = {"ASSERTION_FAILURE", "CRASH"}
    from settings import AGENT_FAULT  # single source of truth

    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["test_id", "title", "area", "verdict", "error_type", "attribution",
                    "device_steps", "states_visited", "distinct_states", "duration_s",
                    "requirement_ids", "route", "error_message", "created_at"])
        for e in sorted(logs, key=lambda x: x.get("created_at") or ""):
            tid = e.get("test_case_id", "")
            t = meta.get(tid, {})
            et = e.get("error_type") or ""
            attribution = ("pass" if e.get("verdict") in ("pass", "passed")
                           else "app" if (et in APP_FAULT or not et)
                           else "agent" if et in AGENT_FAULT else "environment")
            ids = e.get("path") or []
            route, prev = [], None
            for pid in ids:
                lbl = label.get(pid, "?")
                if lbl != prev:
                    route.append(lbl)
                    prev = lbl
            w.writerow([
                tid, t.get("title") or e.get("title", ""), t.get("area", ""),
                e.get("verdict", ""), et, attribution,
                e.get("device_steps", 0), e.get("states_visited", 0), len(set(ids)),
                round((e.get("duration_ms") or 0) / 1000, 1),
                "|".join(t.get("requirement_ids") or []) if isinstance(t.get("requirement_ids"), list) else "",
                " -> ".join(route), (e.get("error_message") or "")[:300],
                e.get("created_at", ""),
            ])
    print(f"\n  📄 Batch CSV: {out}  ({len(logs)} runs)")
    return out


async def main(rounds: int = EXECUTOR_ROUNDS):
    """
    Main executor loop:
      1. Get first test case from planner
      2. Execute on device
      3. Log verdict, get next test case
      4. Repeat for N rounds
    """
    preflight()

    if CLEAN_SLATE:
        _print_header("CLEAN SLATE — resetting execution history")
        # Every batch must start from the same graph state or runs are not
        # comparable: leftover tests skew dedup, coverage and risk, and leftover
        # UIStates make the app model look richer than this run earned.
        try:
            r = requests.post(f"{RAG_URL}/project/reset", json={
                "project": PROJECT, "delete_tests": True,
                # The app model MUST be cleared too. Leaving it credits this run
                # with screens an earlier run discovered, so "states mapped" and
                # every path in the report describe a graph this batch did not
                # build — and the dashboard shows a map before any test has run.
                "delete_appmodel": CLEAN_SLATE_APPMODEL,
                # Knowledge (SRS/Figma) is ingested separately and is not part of
                # what a run earns, so it is preserved.
                "delete_srs": False, "delete_figma": False,
            }, timeout=120)
            r.raise_for_status()
            print(f"  reset: {r.json().get('deleted')}")
        except Exception as e:
            print(f"  ⚠️  reset failed: {e}")

    if DEVICE_RESET_SCOPE == "suite":
        # One wipe for the whole run: every suite starts from an identical device,
        # while state still accumulates between tests within the run.
        reset_device_app()

    results_log = []

    # ── Get the first test case ──────────────────────────────────────────
    _print_header("PLANNER → GENERATING FIRST TEST CASE")
    planner_data = get_next_testcase()
    tc = planner_data.get("next_testcase", {})

    # Log planner's RAG interaction to cloud
    _log_planner_trace(planner_data, "first")

    if not tc or not tc.get("objective"):
        print(f"DEBUG: tc = {tc}")
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
            "screen_hint": tc.get("screen_hint", "?"),
            "area": tc.get("area", "?"),
            "priority": tc.get("priority", "?"),
            "objective": tc.get("objective", ""),
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

            if tc and tc.get("objective"):
                print("\n  Next test case generated:")
                _show_testcase(tc)
            else:
                # Retry fetching the next test case — sometimes the LLM response
                # doesn't parse cleanly on the first attempt.
                print("  ⚠️  Planner returned empty next test case. Retrying...")
                retried = False
                for _retry in range(2):
                    try:
                        retry_data = get_next_testcase()
                        tc = retry_data.get("next_testcase", {})
                        if tc and tc.get("objective"):
                            print(f"  ✅ Retry {_retry+1} succeeded — next test case generated:")
                            _show_testcase(tc)
                            retried = True
                            break
                        else:
                            print(f"  ⚠️  Retry {_retry+1} also returned empty. Trying again...")
                    except Exception as retry_err:
                        print(f"  ⚠️  Retry {_retry+1} failed: {retry_err}")
                if not retried:
                    print("  ❌ All retries exhausted. Ending loop.")
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
        print(f"    Title:       {r['title']}")
        print(f"    Screen hint: {r['screen_hint']}")
        print(f"    Area:        {r['area']}")
        print(f"    Priority:    {r['priority']}")
        print(f"    Objective:   {r.get('objective', '?')}")
        print(f"    Expected:    {r.get('expected_result', '?')}")
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

    _export_batch_csv()


if __name__ == "__main__":
    asyncio.run(main(rounds=EXECUTOR_ROUNDS))
