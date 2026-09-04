"""Web failure taxonomy and self-healing recovery strategies.

Mirrors ``clients.executor_runner.classify_failure`` in shape and shares its
category *names* through ``settings.APP_FAULT`` / ``AGENT_FAULT`` / ``ENV_FAULT``
— attribution must mean the same thing on both platforms or the batch report
cannot add them up.

The keyword sets differ because the failures differ. A browser does not crash the
way an app does; it throws, it 500s, it re-renders the node you were about to
click. Reusing Android's regexes here would classify most web failures as
ASSERTION_FAILURE, which is the bucket that counts as a discovered defect — the
exact mistake the Android side already had to fix twice.
"""

from __future__ import annotations

# Category -> recovery descriptor. ``retry`` decides whether one adaptive
# re-attempt is worth the budget; a category whose cause will not change on a
# second identical run must not retry.
_RECOVERY = {
    "NAVIGATION_FAILURE": {
        "action": "navigate by URL instead of by clicking through the menus",
        "retry": True,
    },
    "ELEMENT_NOT_FOUND": {
        "action": "re-observe after waiting for the page to settle, then use the closest matching label",
        "retry": True,
    },
    "STALE_ELEMENT": {
        "action": "re-observe the page and address the element by its new ref",
        "retry": True,
    },
    "TIMEOUT": {
        "action": "retry with a longer timeout and fewer steps per attempt",
        "retry": True,
    },
    "PAGE_ERROR": {
        "action": "capture the exception and the state that produced it; do not retry a reproducible crash",
        "retry": False,
    },
    "HTTP_ERROR": {
        "action": "record the failing request; a server error is not fixed by clicking again",
        "retry": False,
    },
    "ASSERTION_FAILURE": {
        "action": "capture the actual page state and log it as a potential defect",
        "retry": False,
    },
    "NAVIGATION_LIVELOCK": {
        "action": "the page did not respond to the repeated action; a re-run would repeat it",
        "retry": False,
    },
    "STEP_LIMIT_EXCEEDED": {
        "action": "ran out of steps before finishing; raise WEB_MAX_STEPS or simplify the test",
        "retry": False,
    },
    "PRECONDITION_NOT_MET": {
        "action": "the test needs data or a state the site does not have; report rather than retry",
        "retry": False,
    },
    "LLM_UNAVAILABLE": {
        "action": "the executor model was unreachable; nothing was learned about the app — rerun when it is back",
        "retry": False,
    },
    "BLOCKED_BY_GUARDRAIL": {
        "action": "the test requires a control the guardrails forbid; report as blocked",
        "retry": False,
    },
}


def classify(reason: str, success: bool = False) -> str:
    """Classify an agent's failure reason into a category. Pure.

    Order matters: the categories that are NOT evidence about the application are
    checked first, so a budget or environment problem never lands in
    ASSERTION_FAILURE and inflates the defect count.
    """
    if success:
        return ""
    r = (reason or "").lower()

    # 0. Our own toolchain failed. Checked FIRST because these messages carry
    #    HTTP statuses ("403", "503") that every later rule would misread as the
    #    site's own failure — an OpenRouter outage was recorded as a site CRASH.
    if any(k in r for k in ("openrouter", "gemini", "llmerror", "api key",
                            "model backend", "no choices", "no candidates")):
        return "LLM_UNAVAILABLE"

    # 1. Not the app's fault, and not ours either — the run was refused or the
    #    test was impossible as written.
    if any(k in r for k in ("refused to", "blocked control", "blocked url",
                            "off-origin", "guardrail")):
        return "BLOCKED_BY_GUARDRAIL"
    if any(k in r for k in ("precondition not met", "preconditions not met",
                            "precondition failed", "cannot be completed as specified",
                            "task cannot be completed", "not achievable as specified",
                            "no such page", "requires data that does not exist")):
        return "PRECONDITION_NOT_MET"
    if any(k in r for k in ("step limit", "max steps", "step budget", "out of steps",
                            "steps without reaching")):
        return "STEP_LIMIT_EXCEEDED"
    if "livelock" in r or "unchanged page" in r:
        return "NAVIGATION_LIVELOCK"

    # 2. Our agent's problem: it could not drive the page.
    if any(k in r for k in ("not attached", "detached", "element is not stable",
                            "stale", "node is detached")):
        return "STALE_ELEMENT"
    if any(k in r for k in ("timeout", "timed out", "exceeded", "waiting for")):
        return "TIMEOUT"
    if any(k in r for k in ("no element", "not found", "could not find", "couldn't find",
                            "no such element", "unable to locate", "resolved to 0 elements")):
        return "ELEMENT_NOT_FOUND"
    if any(k in r for k in ("could not reach", "navigat", "wrong page", "did not reach",
                            "err_", "net::", "unable to open")):
        return "NAVIGATION_FAILURE"

    # 3. The app's fault, reported by the browser itself.
    if any(k in r for k in ("uncaught", "page exception", "unhandled rejection",
                            "typeerror", "referenceerror")):
        return "PAGE_ERROR"
    if any(k in r for k in ("500", "502", "503", "504", "server error", "internal error")):
        return "HTTP_ERROR"

    # 4. Everything left is the agent saying the expected result did not happen.
    return "ASSERTION_FAILURE"


def recovery_strategy(category: str) -> dict:
    """Recovery descriptor for a category. Pure."""
    return _RECOVERY.get(category, {"action": "re-attempt with a fresh observation", "retry": False})
