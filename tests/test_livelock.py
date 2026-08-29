#!/usr/bin/env python3
"""Livelock detection — the four cases that matter.

Case 4 is a regression guard. `_state_signature` used to return "" on failure;
empty strings all compare equal, so one import error made every observation look
identical, `is_livelocked` returned True for EVERY test in a suite, and autonomy
fell to zero with nothing logged. An unusable signal must never be
representable as a value that can accidentally equal another one.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import settings as st  # noqa: E402
from clients.executor_runner import (  # noqa: E402
    is_livelocked, _unavailable, _is_cyclic, classify_stuck, _dominant_action,
)

_passed = _failed = 0


def check(label, got, want):
    global _passed, _failed
    ok = got == want
    _passed, _failed = _passed + ok, _failed + (not ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + ("" if ok else f"  (got {got}, want {want})"))


def main():
    print("livelock detection")
    check("same screen + static content -> livelocked",
          is_livelocked(["A"] * 12, ["t"] * 12), True)
    check("same screen + changing content -> working, not stuck",
          is_livelocked(["A"] * 12, [f"t{i}" for i in range(12)]), False)
    check("varied screens -> not stuck",
          is_livelocked([f"S{i}" for i in range(12)], ["t"] * 12), False)
    check("below the window -> undecided",
          is_livelocked(["A"] * 3, ["t"] * 3), False)

    print("regression: unusable signals must not read as a livelock")
    sigs = [_unavailable("sig") for _ in range(12)]
    conts = [_unavailable("content") for _ in range(12)]
    check("all signatures unavailable -> cannot conclude",
          is_livelocked(sigs, conts), False)
    check("unavailable values are mutually distinct",
          len(set(sigs)), 12)
    check("unavailable never equals empty string",
          any(s == "" for s in sigs), False)

    print("regression: a repeating cycle is a loop however varied its content")
    # A->B->C->A->B->C produces 3 distinct screens AND 3 distinct contents, so
    # distinct-value counting called it healthy exploration. A real run cycled
    # FrameLayout -> Button -> Menu five times and burned all 50 steps.
    check("A-B-C repeated 4x -> livelocked",
          is_livelocked((["A", "B", "C"] * 4)[:12], [f"c{i}" for i in range(12)]), True)
    check("A-B repeated 6x -> livelocked",
          is_livelocked((["A", "B"] * 6)[:12], [f"c{i}" for i in range(12)]), True)
    check("period-1 left to the content check (add-ten-records stays legal)",
          _is_cyclic(["A"] * 12), False)
    check("genuinely varied exploration is not cyclic",
          _is_cyclic(list("ABCDEFGHIJKL")), False)
    # A period-5 loop needs 15 observations to show three passes, so it cannot
    # fit in the 12-observation stationary window at all. A real run cycled
    # Farm Profile -> Button -> My Products -> Farmer Market -> Menu and ran to
    # the 50-step wall because the first version of this check stopped at period 4.
    check("period-5 cycle -> livelocked",
          is_livelocked(["a", "b", "c", "d", "e"] * 4,
                        [f"c{i}" for i in range(20)]), True)
    check("period-6 cycle -> livelocked",
          is_livelocked(["a", "b", "c", "d", "e", "f"] * 4,
                        [f"c{i}" for i in range(24)]), True)

    print("whose fault is 'no progress'?")
    # Two independently generated tests filled the Add-Cattle form then tapped
    # 'Next Step' seven times against a screen that never changed. Both were
    # scored as OUR agent failing; both were the app silently refusing.
    # An app fault is claimed ONLY when the repeated action should have changed
    # something. Screen/content counts alone are not evidence: an agent that
    # scrolls eleven times at the bottom of a form also freezes the screen, and
    # that is the agent failing to look elsewhere, not the app refusing.
    check("frozen screen + repeated TAP -> app fault",
          classify_stuck(1, 1, 146.4, _dominant_action(["click"] * 7))[0], "APP_UNRESPONSIVE")
    check("frozen screen + repeated SWIPE -> our fault (scrolling past the end)",
          classify_stuck(1, 1, 146.4, _dominant_action(["swipe"] * 11))[0], "NAVIGATION_LIVELOCK")
    check("no action evidence -> blame ourselves, not the app",
          classify_stuck(1, 1, 146.4, None)[0], "NAVIGATION_LIVELOCK")
    check("mixed actions are not 'repeating one thing'",
          _dominant_action(["click", "swipe", "click"]), None)
    check("agent cycling several screens -> our fault",
          classify_stuck(3, 1, 150.1, _dominant_action(["click"] * 7))[0], "NAVIGATION_LIVELOCK")
    check("app-unresponsive is counted as evidence about the APP",
          "APP_UNRESPONSIVE" in st.APP_FAULT, True)
    check("...and not against our autonomy",
          "APP_UNRESPONSIVE" in st.AGENT_FAULT, False)
    check("the message says what the app did, not what we did",
          "did not respond" in classify_stuck(1, 1, 1.0, _dominant_action(["click"] * 7))[1], True)

    print(f"\n{_passed}/{_passed + _failed} checks passed")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
