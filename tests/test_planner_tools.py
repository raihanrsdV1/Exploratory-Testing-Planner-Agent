#!/usr/bin/env python3
"""Tool planner: the proposal gate must stop what the prompt could only ask for.

The old planner asked the model, in prose, not to invent requirement ids and to
prefer real screen names. Nothing enforced either. Measured result: the named
screen matched a real observed screen ~16% of the time, and the executor's goal
text had to call it "a LEAD, not a fact" while the agent burned its step budget
hunting for screens that do not exist.

These checks pin the gate that replaces those unenforceable rules, and the
availability rule that stops a disabled knowledge source being callable at all.

Graph-backed checks need Neo4j and skip cleanly without it.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

import settings as st  # noqa: E402
from planner import proposal as P, tools  # noqa: E402

_passed = _failed = 0


def check(label, got, want):
    global _passed, _failed
    ok = got == want
    _passed, _failed = _passed + ok, _failed + (not ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + ("" if ok else f"  (got {got!r}, want {want!r})"))


def _has(errors, needle):
    return any(needle.lower() in e.lower() for e in errors)


def main():
    print("a proposal is normalised into exactly what the executor consumes")
    n = P.normalize({"title": " T ", "objective": "o", "expected_result": "e",
                     "screen_hint": "unknown", "requirement_ids": ["FR-1", "", "  "],
                     "preconditions": ["a"] * 20})
    check("'unknown' screen becomes empty, not the literal word", n["screen_hint"], "")
    check("blank requirement ids dropped", n["requirement_ids"], ["FR-1"])
    check("preconditions capped", len(n["preconditions"]), 6)
    check("area defaults rather than going missing", n["area"], "general")
    check("steps are NOT invented by the planner", "steps" in n, False)

    print("\nrequired fields are enforced")
    ok, errs = P.validate("__no_such_project__", {"title": "x"}, [])
    check("missing objective and expected_result both reported", ok, False)
    check("objective named", _has(errs, "'objective' is required"), True)
    check("expected_result named", _has(errs, "'expected_result' is required"), True)

    print("\nout-of-scope areas override coverage pressure")
    if st.OUT_OF_SCOPE:
        area = st.OUT_OF_SCOPE[0]
        ok, errs = P.validate("__no_such_project__", {
            "title": "unique title for scope check", "objective": f"Do {area} and verify it",
            "expected_result": "e"}, [])
        check(f"a test requiring '{area[:24]}' is rejected", _has(errs, "out of scope"), True)
    else:
        print("  [SKIP] OUT_OF_SCOPE is not configured")

    print("\nunachievable preconditions are rejected before the run, not after")
    if st.UNACHIEVABLE_PRECONDITIONS:
        phrase = st.UNACHIEVABLE_PRECONDITIONS[0]
        ok, errs = P.validate("__no_such_project__", {
            "title": "unique title for precondition check", "objective": "o", "expected_result": "e",
            "preconditions": [f"The app starts from an {phrase}"]}, [])
        check("a precondition the tester can only observe is rejected",
              _has(errs, "cannot be established"), True)
    else:
        print("  [SKIP] UNACHIEVABLE_PRECONDITIONS is not configured")

    print("\nduplicates are caught at proposal time, not after generation")
    ok, errs = P.validate("__no_such_project__", {
        "title": "Verify the farm profile update shows feedback",
        "objective": "o", "expected_result": "e"},
        ["Verify the farm profile update shows feedback and persists"])
    check("a near-identical title is rejected", _has(errs, "already been executed"), True)

    # ── graph-backed checks ──────────────────────────────────────────────────
    project = os.getenv("PROJECT") or ""
    observed = P._observed_screens(project) if project else []
    if not observed:
        print(f"\n  [SKIP] no observed screens for PROJECT={project!r} — graph checks skipped")
    else:
        print("\nscreen_hint must name a screen the app has actually been observed to have")
        real = observed[0]
        ok, errs = P.validate(project, {"title": "a unique title for screen grounding",
                                        "objective": "o", "expected_result": "e",
                                        "screen_hint": real}, [])
        check(f"an observed screen is accepted ({real[:26]})", _has(errs, "not a screen"), False)

        ok, errs = P.validate(project, {"title": "another unique title for grounding",
                                        "objective": "o", "expected_result": "e",
                                        "screen_hint": "Animal Registration Multi-step Flow"}, [])
        check("an invented screen is rejected", _has(errs, "not a screen"), True)
        check("the rejection names the real alternatives",
              any("observed" in e.lower() and real[:8] in e for e in errs), True)

        ok, errs = P.validate(project, {"title": "a third unique title for grounding",
                                        "objective": "o", "expected_result": "e",
                                        "screen_hint": "unknown"}, [])
        check("'unknown' is accepted — honest beats a confident guess",
              _has(errs, "not a screen"), False)

        print("\ninvented requirement ids are rejected (a COVERS edge that silently fails)")
        citable = P._citable_ids(project)
        if citable:
            good = sorted(citable)[0]
            ok, errs = P.validate(project, {"title": "unique title for id check",
                                            "objective": "o", "expected_result": "e",
                                            "screen_hint": "unknown",
                                            "requirement_ids": [good, "FR-64"]}, [])
            check("the invented id is named", _has(errs, "FR-64"), True)
            check("the real id is not flagged", any(good in e for e in errs), False)
        else:
            print("  [SKIP] no requirements ingested for this project")

        print("\ntools are bounded and never raise into the planning round")
        check("an unknown tool returns a message, not an exception",
              tools.call("no_such_tool", project, {}).startswith("No such tool"), True)
        check("a miss returns the real alternatives so the model can self-correct",
              "observed" in tools.call("get_screen", project, {"name": "Zzz Nonexistent"}).lower(), True)
        check("get_screen with no name asks for one",
              tools.call("get_screen", project, {}).startswith("Provide"), True)

    print("\nrecent runs come back as evidence, interpreted")
    # A raw error_type means nothing to a planner deciding what to write next.
    # What it needs is what that outcome implies about the test IT wrote — above
    # all, that a budget-exhausted run produced no evidence at all.
    from planner import agent_loop as _al
    from planner import rag_client as _rc

    def _with_logs(logs, total=None):
        orig = _rc.rag_get
        _rc.rag_get = lambda ep, params=None, **kw: {"logs": logs, "total": total or len(logs)}
        try:
            return _al._recent_runs_block("p", 50)
        finally:
            _rc.rag_get = orig

    def _run(tc, verdict, steps, err=""):
        return {"test_case_id": tc, "title": f"title for {tc}", "verdict": verdict,
                "device_steps": steps, "error_type": err}

    blk = _with_logs([_run("TC-9", "failed", 50, "STEP_LIMIT_EXCEEDED")])
    check("budget exhaustion is reported as producing NO evidence", "NO evidence" in blk, True)
    check("and it names over-scoping as the cause", "over-scoped" in blk, True)
    check("steps are shown against the budget", "50/50" in blk, True)

    check("an unreachable target points back at get_screen",
          "get_screen" in _with_logs([_run("TC-8", "failed", 34, "NAVIGATION_FAILURE")]), True)
    check("a real app failure is framed as a result, not a waste",
          "not a waste" in _with_logs([_run("TC-7", "failed", 21, "ASSERTION_FAILURE")]), True)
    blk_pass = _with_logs([_run("TC-6", "pass", 12)])
    check("a pass says do not re-verify", "re-verify" in blk_pass, True)
    check("no history yields no block at all (campaign start)", _with_logs([]), "")

    print("\ntwo runs get reasoning; the rest exist to make a pattern visible")
    many = [_run("TC-5", "failed", 50, "STEP_LIMIT_EXCEEDED"),
            _run("TC-4", "failed", 50, "STEP_LIMIT_EXCEEDED"),
            _run("TC-3", "pass", 9), _run("TC-2", "pass", 11)]
    blk = _with_logs(many, total=12)
    check("only the newest two are interpreted",
          blk.count("produced NO evidence"), 2)
    check("but every run in the window is listed",
          all(t in blk for t in ("TC-5", "TC-4", "TC-3", "TC-2")), True)
    check("the campaign total is shown, not just the window", "of 12 executed" in blk, True)

    print("\npatterns are stated outright, not left to be inferred")
    check("repeated budget exhaustion is called out",
          "consistently too large" in blk, True)
    nav = _with_logs([_run("TC-1", "failed", 31, "NAVIGATION_FAILURE"),
                      _run("TC-0", "failed", 28, "NAVIGATION_FAILURE"),
                      _run("TC-x", "pass", 11)])
    check("a repeated error type is called out as a property, not bad luck",
          "not bad luck" in nav, True)
    quiet = _with_logs([_run("TC-a", "pass", 12), _run("TC-b", "failed", 20, "ASSERTION_FAILURE")])
    check("no warning fires when there is no pattern", "WARNING" in quiet, False)

    print("\nthe recent-runs block ends on guidance, not a hedge")
    blk = _with_logs([_run("TC-1", "failed", 50, "STEP_LIMIT_EXCEEDED")])
    check("no trailing 'information, not an instruction' hedge",
          "not an instruction" in blk, False)
    check("it still ends on something actionable", blk.rstrip().endswith("sharply.")
          or "narrower" in blk.rstrip().split("\n")[-1] or "scope" in blk.rstrip().lower(), True)

    print("\nthe evaluator is told when a run had a mission")
    import gateway.main as _gw
    check("the contract allows leaving a question open",
          "legitimate outcome" in _gw._EVALUATOR_CONTRACT or
          "genuinely do not settle it" in _gw._EVALUATOR_CONTRACT, True)
    check("and forbids guessing a resolution",
          "Do not guess" in _gw._EVALUATOR_CONTRACT, True)
    from planner.schemas import ExecutionEvaluateRequest as _R
    check("the evaluate request carries the mission ref",
          "addresses" in _R.model_fields, True)
    check("and it defaults to empty for ordinary runs",
          _R.model_fields["addresses"].default, "")

    print("\nthe step budget is stated where the decision is made")
    prop = _al._PROPOSE_TOOL["function"]
    check("the proposal tool warns about scope", "yields NOTHING" in prop["description"], True)
    check("the objective field asks for ONE behaviour",
          "ONE behaviour" in prop["parameters"]["properties"]["objective"]["description"], True)

    print("\na failing area must be able to saturate — it was a one-way door")
    # hot_spots promoted an area on failures; the only exit required failed == 0,
    # which a failing area can never reach. Measured: 9 of 13 tests landed on one
    # screen, and 20 of 74 findings described a single text field.
    from planner import coverage as _c
    fail = {"area": "farm_management", "verdict": "failed", "error_type": "ASSERTION_FAILURE"}
    few = _c.compute_coverage_map([fail] * 2, [], ["farm_management", "chat", "orders"])
    check("a freshly-failing area IS a hot spot", few["hot_spots"], ["farm_management"])
    check("and is not yet saturated", few["saturated_areas"], [])

    many = _c.compute_coverage_map([fail] * _c.AREA_SATURATION, [], ["farm_management", "chat", "orders"])
    check("past the spend cap it saturates", many["saturated_areas"], ["farm_management"])
    check("and stops being promoted as a hot spot", many["hot_spots"], [])
    check("even though it is still failing",
          many["area_stats"]["farm_management"]["failed"] >= 2, True)

    d = _c.build_exploration_directive(many, [fail] * _c.AREA_SATURATION)
    check("the directive says the area is spent, with a reason", "[SPENT]" in d, True)
    check("and untested areas are now priority 1",
          d.index("[EXPAND]") < d.index("[SPENT]"), True)

    mixed = _c.compute_coverage_map([fail] * _c.AREA_SATURATION +
                                    [{"area": "chat", "verdict": "failed", "error_type": "ASSERTION_FAILURE"}] * 2,
                                    [], ["farm_management", "chat", "orders"])
    check("a different failing area is still promoted", mixed["hot_spots"], ["chat"])

    print("\nthe recent-runs note no longer pushes the same behaviour")
    from planner import agent_loop as _al2
    note = _al2._interpret("failed", "ASSERTION_FAILURE", 37, 50)
    check("it no longer says to probe the same behaviour",
          "same behaviour" in note, False)
    check("it points at the coverage tools instead",
          "get_coverage" in note or "findings_summary" in note, True)

    print("\ncoverage measures against something real, and says what")
    # The area universe used to come only from Figma. With no design file the
    # map collapsed: 0% and an empty uncovered list, both of which read as true
    # rather than as "no basis to measure" — and the empty list silently removed
    # the planner's only breadth signal.
    from planner import coverage as _cov
    runs = [{"area": "animal_record", "verdict": "pass"},
            {"area": "chat", "verdict": "failed", "error_type": "ASSERTION_FAILURE"}]
    feats = ["animal_record", "chat", "location", "seller_reviews", "notifications"]

    none_ = _cov.compute_coverage_map(runs, [], [])
    check("with no basis at all, the source says so", none_["area_source"], "none")
    check("and the percentage is 0 because it is unmeasurable", none_["coverage_pct"], 0)

    reqs = _cov.compute_coverage_map(runs, [], feats)
    check("requirement features become the universe", reqs["area_source"], "requirements")
    check("the percentage is real", reqs["coverage_pct"], 40)
    check("and the breadth signal comes back to life",
          reqs["uncovered_purposes"], ["location", "notifications", "seller_reviews"])

    figma = _cov.compute_coverage_map(runs, [{"purpose": "contacts"}, {"purpose": "settings"}], feats)
    check("a design file still takes precedence", figma["area_source"], "figma")
    check("features do not leak in when Figma exists", figma["total_areas_available"], 2)

    check("hot spots and dead ends are unaffected by the fallback",
          (reqs["hot_spots"], reqs["dead_ends"]), (none_["hot_spots"], none_["dead_ends"]))

    print("\ncold start: an empty app model has nothing to validate against")
    # On a brand-new project ANY screen name is a guess, and there is no map to
    # catch it — the worst case, on the run where the agent knows least. Opt-in
    # via REQUIRE_GROUNDED_SCREEN_HINT, default OFF, so enabling it is a
    # deliberate decision rather than a silent behaviour change.
    import importlib
    import planner.proposal as _P

    def _validate_cold(flag, hint, title):
        os.environ["REQUIRE_GROUNDED_SCREEN_HINT"] = flag
        for m in ("settings", "planner.proposal"):
            sys.modules.pop(m, None)
        import settings as _s  # noqa: F401
        import planner.proposal as _p
        importlib.reload(_p)
        return _p.validate("__cold_start_selftest__",
                           {"title": title, "objective": "o", "expected_result": "e",
                            "screen_hint": hint}, [])[1]

    _orig = os.environ.get("REQUIRE_GROUNDED_SCREEN_HINT", "")
    try:
        off = _validate_cold("0", "Animal Registration Multi-step Flow",
                             "cold start check with the flag off")
        check("OFF: an unverifiable screen still passes (previous behaviour)",
              any("guess" in e for e in off), False)

        on = _validate_cold("1", "Animal Registration Multi-step Flow",
                            "cold start check with the flag on")
        check("ON: an unverifiable screen is rejected", any("guess" in e for e in on), True)
        check("ON: the rejection tells the model what to do instead",
              any("'unknown'" in e for e in on), True)

        unk = _validate_cold("1", "unknown", "cold start check using unknown")
        check("ON: 'unknown' is always accepted — honest beats a guess",
              any("guess" in e for e in unk), False)
    finally:
        if _orig:
            os.environ["REQUIRE_GROUNDED_SCREEN_HINT"] = _orig
        else:
            os.environ.pop("REQUIRE_GROUNDED_SCREEN_HINT", None)
        for m in ("settings", "planner.proposal"):
            sys.modules.pop(m, None)

    print("\na disabled knowledge source is not callable at all")
    names = tools.available({"appmodel_state_count": 5, "srs_summary": "x",
                             "defect_count": 0, "navtree_node_count": 0})
    check("findings and coverage are always available",
          {"list_findings", "get_coverage"} <= set(names), True)
    check("every advertised tool has a schema", len(tools.schemas(names)), len(names))
    if st.ENABLED_SOURCES and "figma_ui" not in st.ENABLED_SOURCES:
        check("a disabled source contributes no tools",
              any(n.startswith("get_figma") for n in names), False)

    print(f"\n{_passed}/{_passed + _failed} checks passed")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
