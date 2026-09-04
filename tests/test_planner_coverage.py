#!/usr/bin/env python3
"""Exploration-directive prioritisation: hot spots, risk scores, agent difficulty.

Pure logic, no Neo4j/LLM needed. Each check corresponds to a way the planner
could silently misprioritise: a real regression risk never surfacing because it
has no failure yet THIS session, or an agent-capability gap (our own step-budget
limit) getting mistaken for evidence the app is broken.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from planner import coverage  # noqa: E402
from planner import context_builders  # noqa: E402

_passed = _failed = 0


def check(label, got, want):
    global _passed, _failed
    ok = got == want
    _passed, _failed = _passed + ok, _failed + (not ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + ("" if ok else f"  (got {got!r}, want {want!r})"))


def main():
    print("regression risk scores compete for [PRIORITY], not just an FYI block")
    cmap = {"hot_spots": [], "dead_ends": [], "uncovered_purposes": [], "exhausted_areas": []}
    directive = coverage.build_exploration_directive(cmap, [], mode="balanced", risk_areas=["checkout", "payments"])
    check("a risk area with no session failures still gets a [RISK] priority line",
          "[RISK]" in directive, True)
    check("the risk line names the risky area", "checkout" in directive, True)

    print("a risk area already surfaced as a hot spot is not repeated")
    cmap2 = {"hot_spots": ["checkout"], "dead_ends": [], "uncovered_purposes": [], "exhausted_areas": []}
    directive2 = coverage.build_exploration_directive(cmap2, [], mode="balanced", risk_areas=["checkout"])
    check("no [RISK] line when every risk area is already a hot spot",
          "[RISK]" in directive2, False)
    check("the hot spot itself is still investigated", "[INVESTIGATE]" in directive2, True)

    print("no risk data degrades cleanly (most projects, most of the time)")
    directive3 = coverage.build_exploration_directive(cmap, [], mode="balanced", risk_areas=None)
    check("no crash, no stray [RISK] line", "[RISK]" in directive3, False)

    print("known agent difficulty never reads as defect evidence")
    check("no project configured -> empty, not a crash",
          context_builders.build_agent_difficulty_context(""), "")

    print("failure context degrades to the recency view when semantic retrieval is unavailable")
    recent = [
        {"verdict": "failed", "title": "Checkout with empty cart", "notes": "Reason: total showed $NaN"},
        {"verdict": "pass", "title": "Checkout with one item"},
    ]
    ctx_no_objective = context_builders.build_failure_context("demo-project", recent)
    check("no objective -> recency path used directly",
          "Checkout with empty cart" in ctx_no_objective, True)
    ctx_with_objective = context_builders.build_failure_context("demo-project", recent, objective="checkout flow")
    check("semantic call unavailable/unconfigured -> falls back to the same recency findings",
          "Checkout with empty cart" in ctx_with_objective, True)

    print(f"\n{_passed}/{_passed + _failed} checks passed")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
