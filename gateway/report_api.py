"""The operator-facing run report: one PDF a human can hand to someone else.

Kept out of gateway/main.py to match its "thin router" design, and built here
rather than in rag_api/ because it only *reads* what the graph already knows —
there is no new analysis in this file. Every number comes from an existing
endpoint (/execution/logs, /findings, /graph/stats) and every attribution comes
from settings.classify_run.

WHAT THE MODEL IS AND IS NOT ALLOWED TO DO
The LLM writes prose only. It never decides whether a run found a defect: that
verdict is computed from the recorded error_type against the taxonomy in
settings.py, whose own comment records what happens when that classification is
allowed to drift (an OpenRouter outage was once filed as a defect in the site
under test, and autonomy read 100% when it was 67%). A report is the document
someone acts on, so a hallucinated "this is a bug" is the most expensive
mistake this file could make. The counts below are therefore passed to the
model already labelled, and the prompt forbids re-judging them.

When the model is unavailable the PDF is still produced, with the narrative
sections replaced by the deterministic facts and a visible notice — a report
that silently omits its own degradation is worse than one that admits it.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import requests
from fastapi import HTTPException

import settings
from observability import degradations
from planner import model_client
from planner.textutil import strip_reasoning

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Runs are grouped into a "batch" by the gap between them: one `targets run`
# writes its rounds back to back, so a long quiet period is the boundary
# between sessions. There is no batch id on ExecutionLog to group by, and
# inventing one would mean changing the write path for a read-only feature.
# The window that was actually used is printed in the report, so the reader can
# see what "this batch" meant rather than trusting the heuristic blindly.
_BATCH_GAP_SECONDS = 30 * 60

# Latin-1 is all the fpdf2 core fonts can encode. The sites under test are not
# all English (the DataGhurhi profile blocks Bengali control text), so findings
# and screen labels genuinely arrive outside it.
_TYPOGRAPHY = {
    "—": "-", "–": "-", "‘": "'", "’": "'",
    "“": '"', "”": '"', "…": "...", " ": " ",
    "•": "-", "→": "->", "×": "x",
}


def _pdf_safe(text: str) -> str:
    """Reduce arbitrary text to something the PDF core fonts can render.

    Non-Latin scripts are transliterated away to a marker rather than dropped
    silently: a Bengali button label rendered as an empty string would read as
    "no label", which is a different claim from "a label this document cannot
    display".
    """
    out = []
    for ch in str(text or ""):
        if ch in _TYPOGRAPHY:
            out.append(_TYPOGRAPHY[ch])
            continue
        try:
            ch.encode("latin-1")
            out.append(ch)
        except UnicodeEncodeError:
            decomposed = unicodedata.normalize("NFKD", ch)
            stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
            try:
                stripped.encode("latin-1")
                out.append(stripped)
            except UnicodeEncodeError:
                out.append("?")
    return "".join(out)


def _fetch(path: str, required: bool, **params) -> dict:
    """GET one RAG API endpoint. Optional sources degrade to {} instead of failing."""
    try:
        r = requests.get(f"{settings.RAG_URL}{path}", params=params, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        if required:
            raise HTTPException(
                status_code=503,
                detail=f"Could not read {path} from the RAG API ({settings.RAG_URL}): {e}",
            )
        degradations.record("report_source_unavailable", detail=f"{path}: {e}")
        return {}


def _parse_ts(value: str):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def latest_batch(logs: list[dict]) -> list[dict]:
    """The most recent contiguous group of runs, oldest-first.

    ``logs`` arrives newest-first (the endpoint's ORDER BY). Walking backwards
    in time and stopping at the first gap wider than _BATCH_GAP_SECONDS yields
    the session that just finished. Runs with no parseable timestamp end the
    walk rather than being dropped into the batch on a guess.
    """
    batch: list[dict] = []
    previous = None
    for entry in logs:
        stamp = _parse_ts(entry.get("created_at"))
        if stamp is None:
            break
        if previous is not None and (previous - stamp).total_seconds() > _BATCH_GAP_SECONDS:
            break
        batch.append(entry)
        previous = stamp
    return list(reversed(batch))


def _screenshot_for(entry: dict) -> str:
    """The web player's capture for this run, or '' when there isn't one.

    Mirrors web_player/browser.py's naming: the file is '<test id>-<verdict>.png'
    with everything outside [A-Za-z0-9-_.] replaced.
    """
    tc_id, verdict = entry.get("test_case_id") or "", entry.get("verdict") or ""
    if not tc_id:
        return ""
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in f"{tc_id}-{verdict}")
    path = os.path.join(settings.WEB_SCREENSHOT_DIR, f"{safe}.png")
    return path if os.path.isfile(path) else ""


def gather(project: str) -> dict:
    """Everything the report states as fact, with each run already attributed."""
    logs = _fetch("/execution/logs", True, project=project, limit=200).get("logs", [])
    if not logs:
        # A misspelled project looks exactly like an empty one, so name the
        # projects that DO have runs rather than leaving the caller to guess
        # which of the two happened.
        known = [p for p in _fetch("/projects", False).get("projects", [])
                 if p.get("run_count")]
        suggestion = ""
        if known:
            names = ", ".join(f"'{p['name']}' ({p['run_count']} runs)" for p in known[:8])
            suggestion = f" Projects with runs: {names}."
        raise HTTPException(
            status_code=404,
            detail=f"No execution logs for project '{project}' yet — run a batch "
                   f"first, or check the project name.{suggestion}",
        )

    batch = latest_batch(logs)
    findings = _fetch("/findings", False, project=project, limit=60).get("findings", [])
    stats = _fetch("/graph/stats", False, project=project)
    appmodel = _fetch("/appmodel/graph", False, project=project)

    runs = []
    counts = {"pass": 0, "app": 0, "agent": 0, "environment": 0, "unclassified": 0}
    for entry in batch:
        attribution = settings.classify_run(entry.get("verdict"), entry.get("error_type"))
        counts[attribution] += 1
        runs.append({
            "test_case_id": entry.get("test_case_id") or "",
            "title": entry.get("title") or "",
            "verdict": entry.get("verdict") or "",
            "error_type": entry.get("error_type") or "",
            "attribution": attribution,
            "steps": entry.get("device_steps") or 0,
            "seconds": round((entry.get("duration_ms") or 0) / 1000, 1),
            "where": [w for w in (entry.get("path_labels") or []) if w],
            "summary": entry.get("trajectory_summary") or "",
            "created_at": entry.get("created_at") or "",
            "screenshot": _screenshot_for(entry),
        })

    # The denominator that means something: runs that produced evidence about
    # the app at all. Agent and environment outcomes say nothing about it.
    informative = counts["pass"] + counts["app"]
    started = runs[0]["created_at"] if runs else ""
    ended = runs[-1]["created_at"] if runs else ""

    return {
        "project": project,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "window": {"from": started, "to": ended},
        "runs": runs,
        "counts": counts,
        "total": len(runs),
        "informative": informative,
        "total_steps": sum(r["steps"] for r in runs),
        "total_seconds": round(sum(r["seconds"] for r in runs), 1),
        # Only the kinds that are claims about the app. AGENT_DIFFICULTY and
        # CONTROL_DISCOVERED are deliberately excluded from the defect section:
        # findings.py defines the first as explicitly not-a-defect.
        "defect_findings": [
            f for f in findings
            if (f.get("kind") or "") in ("SPEC_VIOLATION", "SUSPECTED_DEFECT")
        ],
        "agent_difficulty": [
            f for f in findings if (f.get("kind") or "") == "AGENT_DIFFICULTY"
        ],
        "coverage": {
            "tests": stats.get("test_case_count", 0),
            "requirements": stats.get("requirement_count", 0),
            "covered_requirements": stats.get("covered_requirement_count", 0),
            "ui_states": appmodel.get("state_count", 0),
        },
    }


_PROMPT = """You are writing the narrative sections of a formal software test \
report for a stakeholder who did not watch the run.

Write in a neutral, official register: third person, past tense, no marketing \
language, no exclamation marks, no emoji, no bullet symbols.

THE FACTS ARE FIXED. Every run below has already been attributed by the test \
system itself:
  attribution="app"          the application misbehaved - this is a defect
  attribution="agent"        the automated tester could not complete the run -
                             a limitation of the tool, NOT a defect
  attribution="environment"  blocked by test data, permissions, a guardrail or
                             a step budget - NOT a defect
  attribution="pass"         the application behaved as expected

You must NOT re-judge these. Never call an "agent" or "environment" outcome a \
defect, and never downgrade an "app" outcome. Do not invent tests, screens, \
counts or causes that are absent from the data.

DATA (JSON):
{facts}

Write the seven sections below. Introduce each with its marker alone on its own \
line, exactly as written, and put the prose underneath. Plain prose only - no \
JSON, no markdown, no bullet lists, no headings of your own.

[EXECUTIVE_SUMMARY]
3-5 sentences: what was tested, how much was run, what was found, and the single \
most important takeaway.

[SCOPE_AND_METHOD]
2-4 sentences describing what the tests set out to verify and how they were executed.

[WHY_THESE_TESTS]
3-5 sentences explaining why this set of tests was worth running against this \
application, grounded in the areas and objectives shown.

[TEST_NARRATIVE]
One short paragraph, in plain language, describing what the planned tests \
actually check. Refer to them by title.

[FINDINGS_DISCUSSION]
2-5 sentences on the defects found and where they occurred. If there were none, \
say so plainly and do not speculate.

[LIMITATIONS]
2-4 sentences on runs that were lost to tool or environment limits, stating \
clearly that these are not defects in the application.

[RECOMMENDATION]
2-4 sentences of concrete next steps."""

_SECTION_ORDER = [
    ("executive_summary", "Executive Summary"),
    ("scope_and_method", "Scope and Method"),
    ("why_these_tests", "Why These Tests Were Necessary"),
    ("test_narrative", "What the Tests Examined"),
    ("findings_discussion", "Discussion of Findings"),
    ("limitations", "Limitations of This Run"),
    ("recommendation", "Recommendation"),
]


def _facts_for_model(facts: dict) -> dict:
    """The slice of the facts the model is shown. Prompts stay bounded: full
    trajectory summaries for 200 runs would dominate the context and buy nothing
    the titles and attributions do not already say."""
    return {
        "project": facts["project"],
        "counts": facts["counts"],
        "total_runs": facts["total"],
        "informative_runs": facts["informative"],
        "total_steps": facts["total_steps"],
        "total_seconds": facts["total_seconds"],
        "coverage": facts["coverage"],
        "runs": [
            {k: r[k] for k in ("title", "attribution", "error_type", "steps", "where")}
            for r in facts["runs"]
        ],
        "defects": [
            {"claim": f.get("claim", ""), "screen": f.get("screen", ""),
             "kind": f.get("kind", "")}
            for f in facts["defect_findings"]
        ],
    }


def parse_sections(raw: str) -> dict:
    """Split a marker-delimited reply into the report's sections.

    Markers rather than JSON because the payload is multi-paragraph prose: a
    single unescaped quote or apostrophe invalidates a whole JSON object, and
    that was not hypothetical — it cost a real report its entire narrative. A
    line-anchored marker cannot be broken by anything the prose contains.
    Sections the model omits come back empty rather than missing.
    """
    text = strip_reasoning(raw or "")
    keys = [key for key, _ in _SECTION_ORDER]
    pattern = re.compile(r"^[ \t]*\[?(" + "|".join(k.upper() for k in keys) + r")\]?[ \t]*:?[ \t]*$",
                         re.MULTILINE | re.IGNORECASE)
    sections = {key: "" for key in keys}
    matches = list(pattern.finditer(text))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end():end].strip()
        # Models sometimes restate the instruction line under its own marker;
        # keep the longest body when a marker appears more than once.
        key = match.group(1).lower()
        if len(body) > len(sections.get(key, "")):
            sections[key] = body
    return sections


def narrate(facts: dict) -> tuple[dict, str]:
    """Prose sections from the configured model backend.

    Returns (sections, notice). ``notice`` is non-empty when the model could not
    be reached, and is printed in the PDF so the reader knows the narrative is
    missing rather than concluding the run had nothing to say.
    """
    prompt = _PROMPT.format(facts=json.dumps(_facts_for_model(facts), indent=2))
    try:
        reply = model_client.call_model(
            prompt, max_new_tokens=2000, enable_thinking=False,
            app_label="QA Run Report",
        )
        sections = parse_sections(reply.get("answer") or "")
        if not any(sections.values()):
            raise ValueError("model returned no usable sections")
        return sections, ""
    except Exception as e:
        degradations.record("report_narrative_unavailable", detail=str(e))
        return {}, ("The narrative sections of this report could not be generated: the "
                    f"model backend was unavailable ({e}). Every figure and finding "
                    "below is recorded directly by the test system and is unaffected.")


# ── PDF ───────────────────────────────────────────────────────────────────────
# Colours and helper shapes follow scripts/make_user_guide.py so the two
# documents read as one family.
INK = (28, 32, 36)
ACCENT = (79, 70, 229)
MUTED = (110, 118, 129)
RULE = (223, 227, 232)
BAD = (185, 28, 28)
OK = (21, 128, 61)
WARN = (180, 83, 9)

_ATTRIBUTION_LABEL = {
    "pass": ("Passed", OK, "the application behaved as expected"),
    "app": ("Defect", BAD, "a fault in the application under test"),
    "agent": ("Tool limit", WARN, "our automated tester could not complete the run"),
    "environment": ("Blocked", MUTED, "test data, permissions, guardrail or step budget"),
    "unclassified": ("Unclassified", MUTED, "no attribution was recorded"),
}


def build_pdf(facts: dict, sections: dict, notice: str) -> bytes:
    from fpdf import FPDF

    project = _pdf_safe(facts["project"])

    class Report(FPDF):
        def multi_cell(self, w, h=None, text="", *args, **kwargs):
            kwargs.setdefault("new_x", "LMARGIN")
            kwargs.setdefault("new_y", "NEXT")
            return super().multi_cell(w, h, text, *args, **kwargs)

        def header(self):
            if self.page_no() == 1:
                return
            self.set_y(10)
            self.set_font("Helvetica", "", 8)
            self.set_text_color(*MUTED)
            self.cell(0, 6, f"Exploratory Test Report - {project}",
                      new_x="LMARGIN", new_y="NEXT")
            self.set_draw_color(*RULE)
            self.line(self.l_margin, 16, self.w - self.r_margin, 16)
            self.set_y(20)

        def footer(self):
            self.set_y(-14)
            self.set_font("Helvetica", "", 8)
            self.set_text_color(*MUTED)
            self.cell(0, 8, f"Page {self.page_no()}", align="C")

    pdf = Report(format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(18, 20, 18)
    epw = pdf.w - pdf.l_margin - pdf.r_margin

    def h1(text):
        if pdf.get_y() > pdf.h - 50:
            pdf.add_page()
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 15)
        pdf.set_text_color(*ACCENT)
        pdf.multi_cell(0, 8, _pdf_safe(text))
        pdf.set_draw_color(*ACCENT)
        pdf.line(pdf.l_margin, pdf.get_y() + 1, pdf.l_margin + 26, pdf.get_y() + 1)
        pdf.ln(4)

    def para(text, size=10.5):
        pdf.set_font("Helvetica", "", size)
        pdf.set_text_color(*INK)
        pdf.multi_cell(0, 5.4, _pdf_safe(text))
        pdf.ln(1.5)

    # ── Cover ────────────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.ln(26)
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(*ACCENT)
    pdf.multi_cell(0, 11, "Exploratory Test Report")
    pdf.ln(1)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(*INK)
    pdf.multi_cell(0, 8, project)
    pdf.ln(6)
    pdf.set_draw_color(*RULE)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(6)

    window = facts["window"]
    pdf.set_font("Helvetica", "", 10.5)
    pdf.set_text_color(*INK)
    for label, value in (
        ("Report generated", facts["generated_at"]),
        ("Run window", f"{window['from']}  to  {window['to']}" if window["from"] else "n/a"),
        ("Test cases executed", str(facts["total"])),
        ("Defects found", str(facts["counts"]["app"])),
        ("Prepared by", "Exploratory Testing Planner Agent (automated)"),
    ):
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(45, 6, _pdf_safe(label))
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, _pdf_safe(value))
    pdf.ln(8)

    if notice:
        pdf.set_font("Helvetica", "B", 9.5)
        pdf.set_text_color(*WARN)
        pdf.multi_cell(0, 5, "NOTICE")
        pdf.set_font("Helvetica", "", 9.5)
        pdf.set_text_color(*INK)
        pdf.multi_cell(0, 5, _pdf_safe(notice))
        pdf.ln(2)

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*MUTED)
    pdf.multi_cell(0, 5,
        _pdf_safe("Every outcome in this report is attributed by the test system itself. "
                  "A failed run is counted as a defect only when the application was at "
                  "fault; runs lost to the limits of the automated tester or to the test "
                  "environment are reported separately and are not defects."))

    # ── Narrative + results ──────────────────────────────────────────────────
    pdf.add_page()
    for key, title in _SECTION_ORDER:
        body = (sections or {}).get(key, "")
        if not body:
            continue
        h1(title)
        para(body)

    h1("Results by Attribution")
    counts, total = facts["counts"], facts["total"]
    pdf.set_font("Helvetica", "B", 9.5)
    pdf.set_text_color(*MUTED)
    pdf.cell(34, 6, "OUTCOME")
    pdf.cell(14, 6, "N", align="R")
    pdf.cell(18, 6, "SHARE", align="R")
    pdf.multi_cell(0, 6, "  MEANING")
    pdf.set_draw_color(*RULE)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(1.5)
    for key in ("pass", "app", "agent", "environment", "unclassified"):
        n = counts[key]
        if not n and key not in ("pass", "app"):
            continue
        label, colour, meaning = _ATTRIBUTION_LABEL[key]
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*colour)
        pdf.cell(34, 5.8, _pdf_safe(label))
        pdf.set_text_color(*INK)
        pdf.cell(14, 5.8, str(n), align="R")
        pdf.cell(18, 5.8, f"{(100 * n / total if total else 0):.0f}%", align="R")
        pdf.set_font("Helvetica", "", 9.5)
        pdf.set_text_color(*MUTED)
        pdf.multi_cell(0, 5.8, _pdf_safe("  " + meaning))
    pdf.ln(3)

    para(f"Total execution effort: {facts['total_steps']} steps across "
         f"{facts['total_seconds']:.0f} seconds. Of {total} runs, {facts['informative']} "
         f"produced evidence about the application; the remainder were lost to the "
         f"tester or the environment and say nothing about its quality.", size=10)

    coverage = facts["coverage"]
    para(f"Knowledge coverage: {coverage['tests']} test cases recorded, "
         f"{coverage['covered_requirements']} of {coverage['requirements']} requirements "
         f"covered, {coverage['ui_states']} user-interface states mapped.", size=10)

    # ── Test-by-test ─────────────────────────────────────────────────────────
    h1("Tests Executed")
    for index, run in enumerate(facts["runs"], start=1):
        if pdf.get_y() > pdf.h - 48:
            pdf.add_page()
        label, colour, _ = _ATTRIBUTION_LABEL[run["attribution"]]
        pdf.set_font("Helvetica", "B", 10.5)
        pdf.set_text_color(*INK)
        pdf.multi_cell(0, 5.6, _pdf_safe(f"{index}. {run['title'] or run['test_case_id']}"))
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*colour)
        pdf.cell(30, 5, _pdf_safe(label))
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*MUTED)
        detail = f"{run['steps']} steps, {run['seconds']:.0f}s"
        if run["error_type"]:
            detail += f", recorded as {run['error_type']}"
        pdf.multi_cell(0, 5, _pdf_safe(detail))
        if run["where"]:
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(*MUTED)
            pdf.multi_cell(0, 5, _pdf_safe("Path: " + " > ".join(run["where"][:8])))
        pdf.ln(2)

    # ── Defects ──────────────────────────────────────────────────────────────
    h1("Defects and Where They Occurred")
    defects = facts["defect_findings"]
    app_runs = [r for r in facts["runs"] if r["attribution"] == "app"]
    if not defects and not app_runs:
        para("No defects were attributed to the application in this run. Note that this "
             "is not a statement that the application is free of defects, only that none "
             "were observed within the scope and budget described above.")
    else:
        for run in app_runs:
            if pdf.get_y() > pdf.h - 60:
                pdf.add_page()
            pdf.set_font("Helvetica", "B", 10.5)
            pdf.set_text_color(*BAD)
            pdf.multi_cell(0, 5.6, _pdf_safe(run["title"] or run["test_case_id"]))
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(*INK)
            where = " > ".join(run["where"][:8]) if run["where"] else "not recorded"
            pdf.multi_cell(0, 5.2, _pdf_safe(f"Where: {where}"))
            if run["error_type"]:
                pdf.multi_cell(0, 5.2, _pdf_safe(f"Classified as: {run['error_type']} "
                                                 f"(a fault in the application)"))
            if run["summary"]:
                pdf.multi_cell(0, 5.2, _pdf_safe("Observed: " + run["summary"][:600]))
            if run["screenshot"]:
                try:
                    if pdf.get_y() > pdf.h - 90:
                        pdf.add_page()
                    pdf.ln(1)
                    pdf.image(run["screenshot"], w=min(epw, 150))
                    pdf.set_font("Helvetica", "", 8)
                    pdf.set_text_color(*MUTED)
                    pdf.multi_cell(0, 4.5, _pdf_safe(
                        f"Screen captured at the point of failure ({os.path.basename(run['screenshot'])})."))
                except Exception:
                    # An unreadable capture must not cost the reader the report.
                    degradations.record("report_screenshot_unreadable",
                                        detail=run["screenshot"], severity=degradations.MINOR)
            pdf.ln(3)

        for finding in defects:
            if pdf.get_y() > pdf.h - 40:
                pdf.add_page()
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(*INK)
            pdf.multi_cell(0, 5.2, _pdf_safe(finding.get("claim") or "(no claim recorded)"))
            pdf.set_font("Helvetica", "", 9.5)
            pdf.set_text_color(*MUTED)
            pdf.multi_cell(0, 5, _pdf_safe(
                f"{finding.get('kind', '')} - screen: {finding.get('screen') or 'not recorded'}"))
            pdf.ln(2)

    # ── Not defects ──────────────────────────────────────────────────────────
    lost = [r for r in facts["runs"] if r["attribution"] in ("agent", "environment")]
    if lost or facts["agent_difficulty"]:
        h1("Runs That Are Not Defects")
        para("The runs below did not complete, but the cause lies with the automated "
             "tester or the test environment rather than the application. They are "
             "excluded from the defect count above and must not be read as quality "
             "problems in the software under test.")
        for run in lost:
            if pdf.get_y() > pdf.h - 34:
                pdf.add_page()
            label, colour, meaning = _ATTRIBUTION_LABEL[run["attribution"]]
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(*INK)
            pdf.multi_cell(0, 5.2, _pdf_safe(run["title"] or run["test_case_id"]))
            pdf.set_font("Helvetica", "", 9.5)
            pdf.set_text_color(*MUTED)
            pdf.multi_cell(0, 5, _pdf_safe(
                f"{label}: {run['error_type'] or 'no type recorded'} - {meaning}"))
            pdf.ln(1.5)
        for finding in facts["agent_difficulty"][:10]:
            pdf.set_font("Helvetica", "", 9.5)
            pdf.set_text_color(*MUTED)
            pdf.multi_cell(0, 5, _pdf_safe(
                f"- {finding.get('claim', '')} (screen: {finding.get('screen') or 'not recorded'})"))

    return bytes(pdf.output())


def _safe_filename(project: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", project).strip("-.") or "report"
    return f"test-report-{stem[:60]}-{datetime.now(timezone.utc):%Y%m%d-%H%M}.pdf"


def build_report(project: str) -> tuple[bytes, str]:
    """The whole pipeline: read the graph, attribute, narrate, render."""
    facts = gather(project)
    sections, notice = narrate(facts)
    try:
        pdf_bytes = build_pdf(facts, sections, notice)
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="fpdf2 is not installed. Install it with: pip install fpdf2",
        )
    return pdf_bytes, _safe_filename(project)
