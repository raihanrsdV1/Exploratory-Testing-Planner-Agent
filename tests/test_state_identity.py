#!/usr/bin/env python3
"""State identity: same widgets with different values are the same screen.

A control's content-description is its LABEL on a button and its VALUE on a form
field — a dropdown reads "Select the type of animal" before use and "Cow" after.
Treating that as identity made every step of a wizard a new screen: one
Add-Cattle form produced nine states in a real run. These checks pin the fix,
and equally pin the guard that stops it over-merging.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import settings as st  # noqa: E402
from ingestion import app_state as A  # noqa: E402

_passed = _failed = 0


def check(label, got, want):
    global _passed, _failed
    ok = got == want
    _passed, _failed = _passed + ok, _failed + (not ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + ("" if ok else f"  (got {got!r})"))


def main():
    print("the real pair that fragmented (dropdown placeholder -> value)")
    a = ["rid␟Button␟Select the type of animal␟1", "␟View␟Height␟0", "␟View␟Width␟0"]
    b = ["rid␟Button␟Cow␟1", "␟View␟Height␟0", "␟View␟Width␟0"]
    check("full containment alone misses it",
          A.containment(a, b) < st.STATE_CONTAINMENT_THRESHOLD, True)
    check("skeleton containment recognises the same widgets",
          A.skeleton_containment(a, b) >= st.STATE_SKELETON_THRESHOLD, True)
    check("full key still overlaps enough to be one screen",
          A.containment(a, b) >= st.STATE_SKELETON_MIN_FULL, True)

    print("guards against over-merging (finding A-5)")
    # a genuine subset: the blank screen's one control also exists on the big one
    tiny = ["␟FrameLayout␟␟0"]
    big = ["␟FrameLayout␟␟0"] + [f"␟View␟L{i}␟0" for i in range(19)]
    check("a 1-control blank screen is 'contained' in everything",
          A.containment(tiny, big), 1.0)
    check("...but size_ratio blocks the merge",
          A.size_ratio(tiny, big) > st.STATE_CONTAINMENT_MAX_RATIO, True)
    check("comparable screens are still allowed",
          A.size_ratio(["a", "b", "c"], ["a", "b", "c", "d"]) <= st.STATE_CONTAINMENT_MAX_RATIO,
          True)

    print("different screens built from similar widget types stay apart")
    x = ["␟Button␟Save␟1", "␟Button␟Cancel␟1", "␟View␟Name␟0"]
    y = ["␟Button␟Delete␟1", "␟Button␟Share␟1", "␟View␟Price␟0"]
    check("skeletons match (same widget types)",
          A.skeleton_containment(x, y) >= st.STATE_SKELETON_THRESHOLD, True)
    check("but the full-key floor rejects the merge",
          A.containment(x, y) < st.STATE_SKELETON_MIN_FULL, True)

    print("skeleton strips only the value, not the structure")
    check("skeleton keeps resource_id and class",
          A.skeleton(["rid␟Button␟Cow␟1"]), ["rid␟Button"])

    print(f"\n{_passed}/{_passed + _failed} checks passed")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
