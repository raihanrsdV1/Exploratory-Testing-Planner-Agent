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
from planner import proposal as P, textutil, tools  # noqa: E402

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

    print("\na disabled knowledge source is not callable at all")
    names = tools.available({"appmodel_state_count": 5, "srs_summary": "x",
                             "defect_count": 0, "navtree_node_count": 0})
    check("findings and coverage are always available",
          {"list_findings", "get_coverage"} <= set(names), True)
    check("every advertised tool has a schema", len(tools.schemas(names)), len(names))
    if st.ENABLED_SOURCES and "figma_ui" not in st.ENABLED_SOURCES:
        check("a disabled source contributes no tools",
              any(n.startswith("get_figma") for n in names), False)

    # -- requirement ids survive the model\'s decoration --------------
    # The planner emitted ['[FR-PROJ-02]', '[FR-PROJ-06]'] - brackets copied out
    # of the bracketed list the prompt shows. Graph nodes carry bare ids, so every
    # COVERS edge silently failed to match and requirement coverage sat at 2%
    # however many tests ran. Nothing errored, which is why it went unnoticed.
    check("a bracketed id is cleaned", P._clean_req_id("[FR-PROJ-02]"), "FR-PROJ-02")
    check("a bare id is untouched", P._clean_req_id("FR-RESP-01"), "FR-RESP-01")
    check("quotes and padding go too", P._clean_req_id('  "FR-AUTH-09" '), "FR-AUTH-09")
    check("parens are decoration as well", P._clean_req_id("(NFR-03)"), "NFR-03")
    check("a trailing comma is stripped", P._clean_req_id("FR-SURV-02,"), "FR-SURV-02")
    check("nothing survives as empty", P._clean_req_id(None), "")
    check("an internal hyphen is not eaten", P._clean_req_id("[FR-I18N-02]"), "FR-I18N-02")

    norm = P.normalize({"title": "t", "requirement_ids": ["[FR-PROJ-02]", " FR-RESP-01 ", ""]})
    check("normalize emits ids the graph can match",
          norm["requirement_ids"], ["FR-PROJ-02", "FR-RESP-01"])

    # validate() compared the RAW ids against the citable set, so a bracketed id
    # was rejected as non-existent and the model burned a retry re-sending it.
    prop = {"title": "x", "requirement_ids": ["[FR-PROJ-02]"]}
    P.validate("__no_such_project__", prop)
    check("validate writes the cleaned ids back", prop["requirement_ids"], ["FR-PROJ-02"])

    # The gateway path never calls normalize(): langgraph_agent builds the test
    # case straight from the parsed model output, and that dict is what both the
    # response AND the /tests/log POST (the only place COVERS edges are written)
    # carry. So the cleaning has to happen in the parser, not just the validator.
    tc = textutil.parse_testcase(
        '{"title":"t","requirement_ids":["[FR-PROJ-01]","[FR-PROJ-06]",""]}')
    check("the parser cleans ids on the gateway path",
          tc["requirement_ids"], ["FR-PROJ-01", "FR-PROJ-06"])

    # Models often keep talking after the JSON; that second parse path needs it too.
    tc2 = textutil.parse_testcase('{"requirement_ids":["[FR-X-9]"]} and then I said more')
    check("the trailing-prose parse path cleans too",
          tc2["requirement_ids"], ["FR-X-9"])

    check("a test case with no ids is left alone",
          textutil.parse_testcase('{"title":"t"}'), {"title": "t"})

    # strip_reasoning must keep stripping only whitespace - it shares the phrase
    # "text.strip(" with the cleaner and is easy to clobber by accident.
    check("strip_reasoning still strips only whitespace",
          textutil.strip_reasoning('  <think>x</think> {"a":1}  '), '{"a":1}')

    print(f"\n{_passed}/{_passed + _failed} checks passed")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
