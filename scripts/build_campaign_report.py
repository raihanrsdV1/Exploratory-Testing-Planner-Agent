"""Render a self-contained HTML report for one campaign, ready to print to PDF.

    ./venv/bin/python scripts/build_campaign_report.py --since 2026-09-20T05:41:00

`--since` scopes the report to a campaign: ExecutionLogs at or after that instant, and
Findings *first seen* at or after it. Findings survive CLEAN_SLATE while executions do
not, so without the filter the finding counts silently include earlier sessions.
"""
from __future__ import annotations

import argparse
import html
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from neo4j import GraphDatabase

KIND_BLURB = {
    "SUSPECTED_DEFECT": "App behaved in a way that looks wrong. Needs a human to confirm.",
    "SPEC_VIOLATION": "App contradicts a written requirement in the SRS.",
    "SPEC_GAP": "The SRS does not say what should happen here.",
    "UNEXPECTED_BEHAVIOUR": "Surprising, but not clearly a defect.",
    "CONFIRMED_BEHAVIOUR": "App worked as expected — evidence the feature is sound.",
    "CONTROL_DISCOVERED": "A UI control the agent had not previously mapped.",
    "UNVERIFIED": "The run could not settle the question either way.",
    "AGENT_DIFFICULTY": "Our testing agent struggled. Says nothing about the app.",
}
KIND_ORDER = list(KIND_BLURB)
APP_KINDS = {"SUSPECTED_DEFECT", "SPEC_VIOLATION", "SPEC_GAP", "UNEXPECTED_BEHAVIOUR",
             "CONFIRMED_BEHAVIOUR", "CONTROL_DISCOVERED"}
ERROR_BLURB = {
    "STEP_LIMIT_EXCEEDED": ("Agent limitation", "The executor exhausted its interaction "
        "budget before it could assert. Not evidence of an app defect."),
    "ASSERTION_FAILURE": ("App evidence", "The agent reached its checkpoint and the app "
        "did not do what the test expected. These are the results that describe the app."),
    "NAVIGATION_FAILURE": ("Agent limitation", "The agent could not reach the screen the "
        "test targeted."),
    "PRECONDITION_NOT_MET": ("Environment", "The state the test assumed was not present on "
        "the device. Neither an app defect nor an agent limitation."),
    "CRASH": ("App evidence", "The application terminated unexpectedly."),
}
e = lambda s: html.escape(str(s if s is not None else ""))


def trim(s, n):
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[: n - 1].rsplit(" ", 1)[0] + "…"


def fetch(session, project, since):
    runs = [dict(r) for r in session.run(
        """MATCH (x:ExecutionLog {project:$p}) WHERE x.created_at >= $since
           RETURN x.test_case_id AS tid, x.title AS title, x.verdict AS verdict,
                  x.error_type AS error_type, x.error_message AS error_message,
                  x.trajectory_summary AS summary, x.device_steps AS steps,
                  x.duration_ms AS ms, x.path_labels AS path, x.created_at AS at,
                  x.states_visited AS states
           ORDER BY x.created_at""", p=project, since=since)]
    meta = {r["ext"]: r for r in (dict(x) for x in session.run(
        """MATCH (t:TestCase {project:$p}) WHERE t.external_id IS NOT NULL
           OPTIONAL MATCH (t)-[:COVERS]->(q:Requirement)
           RETURN t.external_id AS ext, t.area AS area, t.test_type AS type,
                  t.last_notes AS notes, collect(DISTINCT q.ref_id) AS reqs""", p=project))}
    for r in runs:
        m = meta.get(r["tid"], {})
        r["area"] = m.get("area") or "—"
        r["reqs"] = [q for q in (m.get("reqs") or []) if q]
        r["notes"] = m.get("notes") or ""
    findings = [dict(r) for r in session.run(
        """MATCH (f:Finding {project:$p}) WHERE f.first_seen >= $since
           RETURN f.kind AS kind, f.claim AS claim, f.evidence AS evidence,
                  f.screen_label AS screen, f.severity AS severity, f.status AS status,
                  f.times_seen AS seen, f.attempts AS attempts, f.resolution AS resolution,
                  f.confidence AS confidence, f.first_seen AS at
           ORDER BY f.first_seen""", p=project, since=since)]
    cov = session.run(
        """MATCH (:Project {name:$p})-[:HAS_REQUIREMENT]->(q:Requirement)
           OPTIONAL MATCH (q)<-[:COVERS]-(t:TestCase)
           RETURN count(DISTINCT q) AS total,
                  count(DISTINCT CASE WHEN t IS NOT NULL THEN q END) AS covered""",
        p=project).single()
    states = session.run("MATCH (u:UIState {project:$p}) RETURN count(u) AS n",
                         p=project).single()["n"]
    return runs, findings, dict(cov), states


def bar(pct, cls):
    return f'<div class="bar"><span class="{cls}" style="width:{pct:.1f}%"></span></div>'


def build(runs, findings, cov, states, project, since, app_desc=""):
    n = len(runs)
    npass = sum(1 for r in runs if r["verdict"] == "pass")
    nfail = n - npass
    errs = Counter(r["error_type"] or "(unclassified)" for r in runs if r["verdict"] == "failed")
    assertion = errs.get("ASSERTION_FAILURE", 0)
    agentlim = sum(v for k, v in errs.items() if ERROR_BLURB.get(k, ("", ""))[0] == "Agent limitation")
    areas = Counter(r["area"] for r in runs)
    kinds = Counter(f["kind"] for f in findings)
    app_findings = sum(v for k, v in kinds.items() if k in APP_KINDS)
    durs = sorted(r["ms"] for r in runs if r["ms"])
    median = durs[len(durs) // 2] / 1000 if durs else 0
    total_h = sum(durs) / 3_600_000 if durs else 0
    stat = Counter((f["status"] or "n/a") for f in findings)
    o = []
    w = o.append

    w(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Exploratory QA Agent — Campaign Report</title><style>
@page {{ size: A4; margin: 16mm 14mm; }}
* {{ box-sizing: border-box; }}
body {{ font: 10.5pt/1.5 "Helvetica Neue", Helvetica, Arial, sans-serif; color: #17303d;
  margin: 0; background: #fff; }}
h1 {{ font-size: 25pt; margin: 0 0 6px; letter-spacing: -.5px; }}
h2 {{ font-size: 15pt; margin: 26px 0 10px; padding-bottom: 5px;
  border-bottom: 2px solid #0f7ea8; color: #0b5f7f; break-after: avoid; }}
h3 {{ font-size: 11.5pt; margin: 18px 0 7px; color: #0b5f7f; break-after: avoid; }}
p {{ margin: 0 0 9px; }}
table {{ width: 100%; border-collapse: collapse; margin: 9px 0 14px; font-size: 9pt; }}
th {{ background: #eef6fa; text-align: left; padding: 6px 7px; border-bottom: 2px solid #cfe3ee;
  font-weight: 600; color: #0b5f7f; }}
td {{ padding: 5px 7px; border-bottom: 1px solid #e8eef2; vertical-align: top; }}
tr {{ break-inside: avoid; }}
.num {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
.cover {{ background: linear-gradient(140deg,#0b5f7f,#0f7ea8 55%,#0f9d76); color:#fff;
  padding: 30mm 16mm 20mm; margin: -16mm -14mm 22px; }}
.cover p {{ opacity: .93; }}
.sub {{ font-size: 13pt; font-weight: 300; }}
.kpis {{ display: flex; gap: 9px; margin: 14px 0 18px; }}
.kpi {{ flex: 1; border: 1px solid #d6e6ee; border-top: 3px solid #0f7ea8; border-radius: 4px;
  padding: 9px 11px; background: #fbfdfe; }}
.kpi b {{ display: block; font-size: 19pt; color: #0b5f7f; line-height: 1.15; }}
.kpi span {{ font-size: 8pt; color: #5b7383; text-transform: uppercase; letter-spacing: .4px; }}
.pass {{ color: #0f9d76; font-weight: 600; }} .fail {{ color: #c0392b; font-weight: 600; }}
.bar {{ background: #eaf1f5; border-radius: 3px; height: 9px; width: 100%; overflow: hidden; }}
.bar span {{ display: block; height: 100%; }}
.bpass {{ background: #0f9d76; }} .bfail {{ background: #d9704f; }} .bneutral {{ background: #0f7ea8; }}
.note {{ background: #f4fbf8; border-left: 3px solid #0f9d76; padding: 9px 12px; margin: 12px 0;
  font-size: 9.5pt; }}
.warn {{ background: #fff8f1; border-left: 3px solid #d9843f; padding: 9px 12px; margin: 12px 0;
  font-size: 9.5pt; }}
.tag {{ display: inline-block; font-size: 7.5pt; padding: 1px 6px; border-radius: 9px;
  background: #eef6fa; color: #0b5f7f; border: 1px solid #d6e6ee; white-space: nowrap; }}
.case {{ border: 1px solid #e0eaf0; border-left: 3px solid #0f9d76; border-radius: 4px;
  padding: 9px 12px; margin: 0 0 9px; break-inside: avoid; }}
.case.f {{ border-left-color: #d9704f; }}
.case h4 {{ margin: 0 0 5px; font-size: 10pt; }}
.case .meta {{ font-size: 8pt; color: #5b7383; margin: 0 0 6px; }}
.case .why {{ font-size: 9pt; margin: 5px 0 0; }}
.case .why b {{ color: #0b5f7f; }}
.mono {{ font-family: "SF Mono", Menlo, monospace; font-size: 8pt; }}
footer {{ margin-top: 26px; padding-top: 9px; border-top: 1px solid #e8eef2;
  font-size: 8pt; color: #7b8f9c; }}
.brk {{ break-before: page; }}
</style></head><body>""")

    w(f"""<div class="cover"><h1>Exploratory QA Test-Case Planner Agent</h1>
<p class="sub">Autonomous Android testing campaign — full results</p>
<p style="margin-top:20px"><b>Application under test:</b> {e(project)}{(' — ' + e(app_desc)) if app_desc else ''}<br>
<b>Campaign:</b> {n} autonomous test cases, generated and executed without human input<br>
<b>Executed:</b> {since[:10]} &nbsp;·&nbsp; <b>Report generated:</b>
{datetime.now().strftime('%d %b %Y')}</p></div>""")

    w(f"""<h2>1 · Executive summary</h2>
<p>An LLM-driven agent designed {n} exploratory test cases for an Android app it had
never seen before, ran each one on a real device, and wrote what it learned into a
knowledge graph that shaped the next test. No human wrote a test case, a selector, or a
script.</p>
<div class="kpis">
<div class="kpi"><b>{n}</b><span>Tests executed</span></div>
<div class="kpi"><b class="pass">{npass}</b><span>Passed ({npass*100//n}%)</span></div>
<div class="kpi"><b class="fail">{nfail}</b><span>Failed ({nfail*100//n}%)</span></div>
<div class="kpi"><b>{app_findings}</b><span>Findings about the app</span></div>
</div>
<div class="warn"><b>Read the pass rate carefully.</b> Of the {nfail} failures, only
<b>{assertion}</b> are statements about the application — cases where the agent reached its
checkpoint and the app did something unexpected. The other <b>{agentlim}</b> are our agent
running out of its own interaction budget or failing to navigate, which says nothing about
the app's quality. The honest headline is <b>{assertion} app-level failures across {n}
tests</b>, not a {nfail*100//n}% defect rate.</div>
<p>The campaign produced <b>{len(findings)} findings</b>: {app_findings} describe the
application, {kinds.get('AGENT_DIFFICULTY', 0)} record our own agent's difficulties, and
{kinds.get('UNVERIFIED', 0)} are questions a run raised but could not settle either way.
It covered <b>{cov['covered']}/{cov['total']} requirements</b>
({cov['covered']*100//cov['total']}%) across <b>{len(areas)} feature areas</b>, and mapped
<b>{states} distinct UI states</b>. Total device time was {total_h:.1f} hours, median
{median:.0f}s per test.</p>""")

    w(f"""<h2>2 · How the system works</h2>
<p>Three LLM agents share one Neo4j knowledge graph. Each test makes the graph a little
richer, so the next test is proposed from better facts.</p>
<table><tr><th style="width:17%">Agent</th><th style="width:36%">Job</th><th>What it reads and writes</th></tr>
<tr><td><b>Planner</b></td><td>Decides what to test next and writes the test case</td>
<td>Calls tools over the graph — untested requirements, screens already mapped, findings
so far, open questions, coverage gaps — then proposes a test. A validation gate rejects
proposals that duplicate earlier work or target screens that do not exist.</td></tr>
<tr><td><b>Executor</b></td><td>Drives the real device</td>
<td>Receives the test case in plain language and interacts with the live app, with a
budget of 50 steps. Emits a step-by-step trajectory.</td></tr>
<tr><td><b>Investigator</b></td><td>Turns the trajectory into knowledge</td>
<td>Reads the trajectory and writes <i>findings</i> — typed, deduplicated claims about the
app — back into the graph, and marks earlier open questions resolved when a run settles
them.</td></tr></table>
<p>Nothing is carried in conversation history. Every planner call is stateless and rebuilt
from the graph, so "learning" means the graph grew — not that the model remembers.</p>""")

    w(f"""<h2 class="brk">3 · Results</h2><h3>3.1 Outcome</h3>
<table><tr><th>Verdict</th><th class="num">Tests</th><th class="num">Share</th><th style="width:44%"></th></tr>
<tr><td><span class="pass">Passed</span></td><td class="num">{npass}</td>
<td class="num">{npass*100/n:.0f}%</td><td>{bar(npass*100/n,'bpass')}</td></tr>
<tr><td><span class="fail">Failed</span></td><td class="num">{nfail}</td>
<td class="num">{nfail*100/n:.0f}%</td><td>{bar(nfail*100/n,'bfail')}</td></tr></table>
<h3>3.2 Why tests failed — and what each reason means</h3>
<table><tr><th style="width:24%">Reason</th><th class="num">Count</th>
<th style="width:17%">Tells us about</th><th>Interpretation</th></tr>""")
    for k, v in errs.most_common():
        cls, blurb = ERROR_BLURB.get(k, ("Unclassified", "No classification recorded."))
        col = "#0b5f7f" if cls == "App evidence" else "#8a6d3b"
        w(f"""<tr><td class="mono">{e(k)}</td><td class="num">{v}</td>
<td style="color:{col}"><b>{cls}</b></td><td>{e(blurb)}</td></tr>""")
    w(f"""</table>
<div class="note"><b>This split is the most important number in the report.</b> A testing
agent that cannot finish its own test produces a failure that looks identical to a real
defect unless you separate them. {assertion} of {nfail} failures are app evidence; the rest
are our ceiling, and they are the clearest direction for future work.</div>""")

    w(f"""<h3>3.3 Coverage</h3>
<table><tr><th>Dimension</th><th class="num">Covered</th><th class="num">Total</th>
<th style="width:34%"></th></tr>
<tr><td>Requirements exercised by at least one test</td><td class="num">{cov['covered']}</td>
<td class="num">{cov['total']}</td>
<td>{bar(cov['covered']*100/cov['total'],'bneutral')}</td></tr>
<tr><td>Feature areas reached</td><td class="num">{len(areas)}</td><td class="num">{len(areas)}</td>
<td>{bar(100,'bneutral')}</td></tr>
<tr><td>Distinct UI states mapped</td><td class="num">{states}</td><td class="num">—</td><td></td></tr>
</table>
<h3>3.4 Spread across the app</h3>
<p>An exploratory agent's failure mode is tunnelling — hammering one feature it already
understands. Across {n} tests the agent touched <b>{len(areas)} areas</b>, the largest
taking {areas.most_common(1)[0][1]*100//n}% of the campaign.</p>
<table><tr><th style="width:38%">Feature area</th><th class="num">Tests</th>
<th class="num">Pass</th><th class="num">Fail</th><th style="width:26%"></th></tr>""")
    for a, c in areas.most_common():
        p = sum(1 for r in runs if r["area"] == a and r["verdict"] == "pass")
        w(f"""<tr><td>{e(a)}</td><td class="num">{c}</td><td class="num pass">{p}</td>
<td class="num fail">{c-p}</td><td>{bar(c*100/n,'bneutral')}</td></tr>""")
    w("</table>")

    w(f"""<h2 class="brk">4 · What the agent learned — findings</h2>
<p>A <i>finding</i> is a typed, deduplicated claim the investigator wrote into the graph.
Findings are deduplicated semantically, so a repeat observation increments a counter rather
than creating a new row. This campaign produced <b>{len(findings)}</b> of them.</p>
<table><tr><th style="width:23%">Kind</th><th class="num">Count</th><th>What it means</th>
<th style="width:17%"></th></tr>""")
    for k in KIND_ORDER:
        if not kinds.get(k):
            continue
        w(f"""<tr><td class="mono"><b>{e(k)}</b></td><td class="num">{kinds[k]}</td>
<td>{e(KIND_BLURB[k])}</td><td>{bar(kinds[k]*100/max(kinds.values()),'bneutral')}</td></tr>""")
    w(f"""</table>
<p><b>Lifecycle.</b> Findings that pose a question start <span class="mono">open</span> and
are closed when a later run settles them — after three inconclusive attempts they are marked
<span class="mono">inconclusive</span> rather than retried forever. Status now:
{', '.join(f'<b>{v}</b> {e(k)}' for k, v in stat.most_common())}.
<span class="mono">n/a</span> covers kinds that are not questions, such as
AGENT_DIFFICULTY.</p>""")

    for kind in ("SUSPECTED_DEFECT", "SPEC_VIOLATION", "SPEC_GAP", "UNEXPECTED_BEHAVIOUR"):
        sel = [f for f in findings if f["kind"] == kind]
        if not sel:
            continue
        w(f"""<h3>4.{KIND_ORDER.index(kind)+1} {e(kind)} — all {len(sel)}</h3>
<p>{e(KIND_BLURB[kind])}</p>
<table><tr><th style="width:44%">Claim</th><th style="width:15%">Screen</th>
<th class="num">Sev</th><th class="num">Seen</th><th>Evidence</th></tr>""")
        for f in sel:
            ev = f["evidence"]
            ev = " ".join(ev) if isinstance(ev, list) else (ev or "")
            w(f"""<tr><td>{e(trim(f['claim'],260))}</td><td>{e(trim(f['screen'],26) or '—')}</td>
<td class="num">{e(f['severity'] or '—')}</td><td class="num">{f['seen'] or 1}</td>
<td style="font-size:8pt;color:#5b7383">{e(trim(ev,180))}</td></tr>""")
        w("</table>")

    w(f"""<h2 class="brk">5 · Every test case</h2>
<p>All {n} tests in execution order. "Why" quotes the agent's own reasoning about what
happened on the device.</p>""")
    for i, r in enumerate(runs, 1):
        ok = r["verdict"] == "pass"
        cls, _ = ERROR_BLURB.get(r["error_type"] or "", ("", ""))
        reqs = (" ".join(f'<span class="tag">{e(q)}</span>' for q in r["reqs"][:6])) or ""
        w(f"""<div class="case {'' if ok else 'f'}">
<h4>{i}. {e(r['tid'] or '—')} — {e(r['title'] or '(untitled)')}</h4>
<p class="meta"><b class="{'pass' if ok else 'fail'}">{'PASS' if ok else 'FAIL'}</b>
 · area {e(r['area'])} · {r['steps'] or 0} device steps
 · {(r['ms'] or 0)/1000:.0f}s · {e(r['at'][11:16] if r['at'] else '')}
 {(' · <span class="mono">'+e(r['error_type'])+'</span>' if r['error_type'] else '')}
 {(' · '+cls if cls else '')} {reqs}</p>""")
        if not ok and r["error_message"]:
            w(f'<p class="why"><b>Why it failed:</b> {e(trim(r["error_message"],420))}</p>')
        if r["summary"]:
            w(f'<p class="why"><b>Agent\'s account:</b> {e(trim(r["summary"],460))}</p>')
        w("</div>")

    w(f"""<h2 class="brk">6 · Limitations</h2>
<p>Stated plainly, because they bound every number above.</p>
<table><tr><th style="width:28%">Limitation</th><th>Consequence</th></tr>
<tr><td><b>No ground truth</b></td><td>The app has no known, seeded defect list, so we
cannot compute recall. We know what the agent found; we cannot say what it missed. This is
the single biggest gap.</td></tr>
<tr><td><b>No ablation</b></td><td>Every component — graph memory, tools, the finding
lifecycle — is on. We have not measured what each contributes, so we cannot attribute the
results to any one design choice.</td></tr>
<tr><td><b>{agentlim} of {nfail} failures are ours</b></td><td>The 50-step executor budget
and navigation reliability, not the app, cap the usable yield of a campaign.</td></tr>
<tr><td><b>Findings are LLM-authored</b></td><td>A SUSPECTED_DEFECT is a model's judgement
from a device trajectory, not a verified bug. The "suspected" is load-bearing and each one
still needs a human.</td></tr>
<tr><td><b>Single app, single run</b></td><td>One application, one campaign, no
repetition. Nothing here establishes variance between runs.</td></tr>
<tr><td><b>Provider variance</b></td><td>Inference is routed through OpenRouter, where
different providers served the same model with order-of-magnitude differences in latency
and token usage, adding noise to any cost or timing figure.</td></tr>
</table>
<footer>Generated from the Neo4j campaign graph by
<span class="mono">scripts/build_campaign_report.py</span> · project
<span class="mono">{e(project)}</span> · campaign since
<span class="mono">{e(since)}</span> · {n} executions, {len(findings)} findings.</footer>
</body></html>""")
    return "".join(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="shobarkhamar")
    ap.add_argument("--since", required=True, help="ISO instant the campaign began")
    ap.add_argument("--out", default="reports/campaign_report.html")
    ap.add_argument("--app-desc", default="",
                    help="one-line description of the app under test, for the cover")
    a = ap.parse_args()
    load_dotenv(".env")
    driver = GraphDatabase.driver(os.getenv("NEO4J_URI"),
                                  auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")))
    with driver.session() as s:
        runs, findings, cov, states = fetch(s, a.project, a.since)
    driver.close()
    if not runs:
        sys.exit(f"No executions at or after {a.since} for project {a.project}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(build(runs, findings, cov, states, a.project, a.since, a.app_desc))
    print(f"{a.out}  ({len(runs)} executions, {len(findings)} findings)")


if __name__ == "__main__":
    main()
