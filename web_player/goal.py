"""Planner test case -> a goal the browser agent can execute.

The web counterpart of ``clients.executor_runner.build_droidrun_goal``, and
deliberately the same shape: identity, credentials, preconditions, input
guidance, verification discipline, numbered steps, expected result. The planner
emits one test-case JSON for both platforms, so the only thing that may differ
between the two builders is the platform-specific advice each agent needs.

``filter_preconditions`` is shared with the Android executor rather than
re-implemented — ``UNACHIEVABLE_PRECONDITIONS`` is a property of the *project*
(an OTP nobody can receive is unreachable in a browser too), not of the driver.
"""

from __future__ import annotations

import settings as cfg
from clients.executor_runner import filter_preconditions

from . import fixtures


def build_goal(test_case: dict) -> str:
    """Render a planner test case as the goal text handed to the agent."""
    parts: list[str] = [cfg.site_identity_block()]

    login = cfg.web_login_block()
    if login:
        parts.append(login)

    safety = cfg.web_safety_block()
    if safety:
        parts.append(safety)

    # A LEAD, not an instruction. The planner cannot see the live site, so a
    # screen name it invented must never become something the agent stalls on
    # trying to match exactly (mirrors build_droidrun_goal's wording).
    screen = test_case.get("screen_hint", "") or test_case.get("screen", "")
    if screen:
        parts.append(
            f"I believe the relevant page is '{screen}' — that is a LEAD to check, not a "
            f"fact; I cannot see the live site. If it does not exist or isn't right, "
            f"explore and find the real page for this objective yourself. Do not stall "
            f"trying to match that name exactly."
        )

    kept, dropped = filter_preconditions(test_case.get("preconditions", []))
    if dropped:
        print(f"   ✂️  Dropped unachievable precondition(s): {dropped}")
    if kept:
        parts.append(f"Preconditions (create these yourself if missing): {' '.join(kept)}")

    # The browser context is reused across test cases in a batch, exactly like
    # the Android default. Say so, so the agent does not abandon a test because
    # data it expected to be absent is present, or vice versa.
    parts.append(
        "The browser keeps cookies and state from earlier tests in this batch. If "
        "the test needs data that is missing, create it only when the available UI "
        "actions and guardrails permit it within this test's budget. Otherwise finish "
        "with success=false and 'Precondition not met', naming the missing data. "
        "You CAN upload files with the 'upload' action — a file input counts as usable "
        "even when the page hides it behind a styled drop zone, so look for role 'file' "
        "in the observation rather than clicking the drop zone. You cannot drag-and-drop, "
        "inspect downloaded files, or solve CAPTCHAs. "
        "Do not search repeatedly for a workaround for an unsupported action."
    )
    parts.append(fixtures.prompt_block())

    parts.append(cfg.web_input_block())
    parts.append(cfg.verification_block())

    # Objective — WHAT to verify. Deciding HOW (which pages, which controls) is
    # the agent's job: it has live access to the site that the planner does not.
    objective = test_case.get("objective", "")
    if objective:
        parts.append(f"\nYour objective: {objective}")

    # Pre-redesign test cases carried an explicit step list. Still honoured when
    # present so an older/hand-written case keeps working, but it is no longer
    # required — the planner now emits screen_hint + objective instead.
    steps = test_case.get("steps", [])
    if steps:
        parts.append("")
        for i, step in enumerate(steps, 1):
            parts.append(f"Step {i}: {step}")

    expected = test_case.get("expected_result", "")
    if expected:
        parts.append(f"\nExpected result: {expected}")

    parts.append(
        "\nDecide the concrete actions yourself from what you actually see on the page — "
        "you have live access to the site that the objective above does not. After acting, "
        "report whether the expected result was achieved. If the page or feature described "
        "above genuinely does not seem to exist after a reasonable search, say so explicitly "
        "rather than searching indefinitely. If any action fails or the page misbehaves, "
        "report the failure."
    )
    return "\n".join(parts)


def build_retry_goal(test_case: dict, category: str, reason: str, strategy: dict) -> str:
    """Retry goal carrying a `## Previous Failure Context` block (WP7 shape)."""
    block = [
        "",
        "## Previous Failure Context",
        f"The previous attempt FAILED (classified as {category}).",
        f"What went wrong: {(reason or 'unknown')[:300]}",
        f"Recovery approach to apply now: {strategy.get('action', 're-attempt')}.",
        "Adjust your approach accordingly and re-attempt the goal.",
    ]
    return build_goal(test_case) + "\n" + "\n".join(block)
