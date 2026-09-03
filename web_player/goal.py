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


def build_goal(test_case: dict) -> str:
    """Render a planner test case as the goal text handed to the agent."""
    parts: list[str] = [cfg.site_identity_block()]

    login = cfg.web_login_block()
    if login:
        parts.append(login)

    safety = cfg.web_safety_block()
    if safety:
        parts.append(safety)

    screen = test_case.get("screen", "")
    if screen:
        parts.append(f"Navigate to the '{screen}' page if you are not already there.")

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
        "the test needs data that is missing, CREATE it as your first steps. Do not "
        "abandon the test because data is missing."
    )

    parts.append(cfg.web_input_block())
    parts.append(cfg.verification_block())

    steps = test_case.get("steps", [])
    if steps:
        parts.append("")
        for i, step in enumerate(steps, 1):
            parts.append(f"Step {i}: {step}")

    expected = test_case.get("expected_result", "")
    if expected:
        parts.append(f"\nExpected result: {expected}")

    parts.append(
        "\nAfter performing all steps, report whether the expected result was "
        "achieved. If any step fails or the page misbehaves, report the failure."
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
