#!/usr/bin/env python3
"""The downloadable run report: the document a stakeholder acts on.

The failure this guards against is the one settings.py already documents in
prose: a failure that belongs to OUR agent being counted as a defect in the
application. In a dashboard that mistake is a wrong number; in a PDF headed
"Test Report" it is a wrong claim about someone's software. Every check below
is about that boundary holding — through classification, through aggregation,
and through the model being unavailable.
"""
import os
import sys
import tempfile

# Must happen BEFORE the first project import: degradations.py resolves its sink
# path once, at import time, and that sink is shared across processes and read
# by the operator dashboard. Without this, the deliberate "model is down" case
# below posts a real-looking incident to the real dashboard.
os.environ["DEGRADATION_SINK"] = os.path.join(tempfile.gettempdir(),
                                              "test_report_degradations.jsonl")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import settings  # noqa: E402
from gateway import report_api  # noqa: E402
from observability import degradations  # noqa: E402

_passed = _failed = 0


def check(label, got, want):
    global _passed, _failed
    ok = got == want
    _passed, _failed = _passed + ok, _failed + (not ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + ("" if ok else f"  (got {got!r}, want {want!r})"))


def _log(tc, verdict, error_type="", created_at="2026-09-11T10:00:00+00:00", **extra):
    return {"test_case_id": tc, "title": f"Test {tc}", "verdict": verdict,
            "error_type": error_type, "created_at": created_at,
            "device_steps": 5, "duration_ms": 1000, "path_labels": ["Home"], **extra}


def main():
    print("this module never writes to the real degradation sink")
    check("the sink is redirected before the first project import",
          degradations._SINK, os.environ["DEGRADATION_SINK"])

    print("attribution is decided by the recorded type, never by the verdict alone")
    for error_type in sorted(settings.APP_FAULT):
        check(f"{error_type} is the app's fault",
              settings.classify_run("failed", error_type), "app")
    for error_type in sorted(settings.AGENT_FAULT):
        check(f"{error_type} is our agent's fault",
              settings.classify_run("failed", error_type), "agent")
    for error_type in sorted(settings.ENV_FAULT):
        check(f"{error_type} is the environment's fault",
              settings.classify_run("failed", error_type), "environment")

    print("the specific confusions that have burned this project before")
    check("a model-provider outage is NOT a defect in the site under test",
          settings.classify_run("failed", "LLM_UNAVAILABLE"), "environment")
    check("a livelock is our agent flailing, not the app breaking",
          settings.classify_run("failed", "NAVIGATION_LIVELOCK"), "agent")
    check("running out of steps is a budget limit, not a defect",
          settings.classify_run("failed", "STEP_LIMIT_EXCEEDED"), "environment")
    check("a guardrail stopping the agent is not a defect",
          settings.classify_run("failed", "BLOCKED_BY_GUARDRAIL"), "environment")
    check("an untyped failure is an unmet assertion about the app",
          settings.classify_run("failed", ""), "app")
    check("a pass is a pass regardless of a stale type",
          settings.classify_run("passed", "TIMEOUT"), "pass")
    check("an unknown type is admitted, not guessed into a bucket",
          settings.classify_run("failed", "SOMETHING_NEW"), "unclassified")

    print("a batch is the group of runs that ran together")
    logs = [_log("TC-3", "passed", created_at="2026-09-11T10:20:00+00:00"),
            _log("TC-2", "failed", created_at="2026-09-11T10:10:00+00:00"),
            _log("TC-1", "passed", created_at="2026-09-11T10:00:00+00:00"),
            _log("OLD", "passed", created_at="2026-09-10T08:00:00+00:00")]
    batch = report_api.latest_batch(logs)
    check("yesterday's run is excluded", [r["test_case_id"] for r in batch],
          ["TC-1", "TC-2", "TC-3"])
    check("the batch reads oldest-first, as a reader expects",
          batch[0]["test_case_id"], "TC-1")
    check("an unparseable timestamp ends the walk rather than joining the batch",
          len(report_api.latest_batch([_log("A", "passed", created_at="not-a-date")])), 0)
    check("no logs is an empty batch, not an error", report_api.latest_batch([]), [])

    print("aggregation keeps defects and tool limits apart")
    captured = {
        "/execution/logs": {"logs": [
            _log("TC-1", "passed"),
            _log("TC-2", "failed", "ASSERTION_FAILURE"),
            _log("TC-3", "failed", "NAVIGATION_LIVELOCK"),
            _log("TC-4", "failed", "STEP_LIMIT_EXCEEDED"),
        ]},
        "/findings": {"findings": [
            {"kind": "SPEC_VIOLATION", "claim": "Survey slug 404s", "screen": "Survey"},
            {"kind": "AGENT_DIFFICULTY", "claim": "could not find the menu", "screen": "Home"},
            {"kind": "CONTROL_DISCOVERED", "claim": "a Save button exists", "screen": "Home"},
        ]},
        "/graph/stats": {"test_case_count": 4, "requirement_count": 10,
                         "covered_requirement_count": 3},
        "/appmodel/graph": {"state_count": 7},
    }
    original_fetch = report_api._fetch
    report_api._fetch = lambda path, required, **kw: captured.get(path, {})
    try:
        facts = gathered = report_api.gather("demo")
    finally:
        report_api._fetch = original_fetch

    check("exactly one run is counted as a defect", facts["counts"]["app"], 1)
    check("the livelock is counted as our agent's limit", facts["counts"]["agent"], 1)
    check("the step budget is counted as environment", facts["counts"]["environment"], 1)
    check("the pass is counted as a pass", facts["counts"]["pass"], 1)
    check("only runs that said something about the app are informative",
          facts["informative"], 2)
    check("agent difficulty is never listed as a defect",
          [f["kind"] for f in facts["defect_findings"]], ["SPEC_VIOLATION"])
    check("a discovered control is not a defect either",
          any(f.get("kind") == "CONTROL_DISCOVERED" for f in facts["defect_findings"]), False)
    check("agent difficulty is still reported, in its own section",
          len(facts["agent_difficulty"]), 1)
    check("coverage is carried through", facts["coverage"]["ui_states"], 7)

    print("prose is parsed by markers, which quotes and apostrophes cannot break")
    # The real failure this replaced: the model emitted prose containing a
    # quote inside a JSON string, and the whole narrative was lost.
    quoted = ('[EXECUTIVE_SUMMARY]\nThe page said "not found", which is correct; '
              "it didn't leak data.\n\n[RECOMMENDATION]\nShip it.\n")
    parsed = report_api.parse_sections(quoted)
    check("a section survives embedded double quotes",
          parsed["executive_summary"].startswith('The page said "not found"'), True)
    check("an apostrophe does not truncate the section",
          parsed["executive_summary"].endswith("it didn't leak data."), True)
    check("a later section is still found", parsed["recommendation"], "Ship it.")
    check("omitted sections come back empty, not missing",
          parsed["limitations"], "")
    check("reasoning blocks are stripped before parsing",
          report_api.parse_sections("<think>hmm</think>\n[RECOMMENDATION]\nDo it.")["recommendation"],
          "Do it.")
    check("a reply with no markers at all yields nothing usable",
          any(report_api.parse_sections("Here is your report, sir.").values()), False)

    print("the model writes prose, and never gets to re-judge the verdicts")
    prompt_seen = {}

    def fake_model(prompt, **kwargs):
        prompt_seen["text"] = prompt
        return {"answer": "[EXECUTIVE_SUMMARY]\nFour tests ran.\n\n"
                          "[SCOPE_AND_METHOD]\ns\n\n[WHY_THESE_TESTS]\nw\n\n"
                          "[TEST_NARRATIVE]\nt\n\n[FINDINGS_DISCUSSION]\nf\n\n"
                          "[LIMITATIONS]\nl\n\n[RECOMMENDATION]\nr\n"}

    original_call = report_api.model_client.call_model
    report_api.model_client.call_model = fake_model
    try:
        sections, notice = report_api.narrate(gathered)
    finally:
        report_api.model_client.call_model = original_call
    check("the narrative comes back parsed", sections["executive_summary"], "Four tests ran.")
    check("nothing is flagged as degraded on success", notice, "")
    check("the prompt forbids re-judging the attribution",
          "must NOT re-judge" in prompt_seen["text"], True)
    check("the prompt states that an agent fault is not a defect",
          "NOT a defect" in prompt_seen["text"], True)
    check("the model is shown the attribution it must respect",
          '"attribution": "agent"' in prompt_seen["text"], True)

    print("an unreachable model costs the prose, never the facts")

    def broken_model(prompt, **kwargs):
        raise RuntimeError("OpenRouter unavailable")

    report_api.model_client.call_model = broken_model
    try:
        sections, notice = report_api.narrate(gathered)
    finally:
        report_api.model_client.call_model = original_call
    check("no prose is invented when the model is down", sections, {})
    check("the reader is told the narrative is missing", "could not be generated" in notice, True)

    print("the PDF renders, including text the core fonts cannot encode")
    safe = report_api._pdf_safe("Delete — “quoted” … মুছে")
    check("an em dash becomes a hyphen", "-" in safe, True)
    check("smart quotes become plain ones", '"quoted"' in safe, True)
    check("the whole string is encodable by the PDF core fonts",
          bool(safe.encode("latin-1")), True)

    pdf_bytes = report_api.build_pdf(gathered, sections, notice)
    check("a PDF is produced even with no narrative", pdf_bytes[:4], b"%PDF")
    check("the PDF is not a stub", len(pdf_bytes) > 2000, True)

    full = report_api.build_pdf(
        gathered,
        {"executive_summary": "Four tests ran.", "scope_and_method": "s",
         "why_these_tests": "w", "test_narrative": "t", "findings_discussion": "f",
         "limitations": "l", "recommendation": "r"}, "")
    check("the narrated PDF also renders", full[:4], b"%PDF")
    check("the narrated report is longer than the degraded one", len(full) > len(pdf_bytes), True)

    print("the download filename is safe to put in a header")
    name = report_api._safe_filename("../../etc/pass wd")
    check("path separators cannot escape the filename", "/" not in name and "\\" not in name, True)
    check("it is still a pdf", name.endswith(".pdf"), True)

    print(f"\n{_passed}/{_passed + _failed} checks passed")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
