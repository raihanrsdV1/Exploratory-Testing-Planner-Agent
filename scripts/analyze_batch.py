"""
Batch analysis for a completed exploratory-testing run.

Produces the two things a results section needs:

  1. QUANTITATIVE — outcomes split by attribution, so "failed" is never confused
     with "found a bug". Only ASSERTION_FAILURE and CRASH are candidate defects;
     agent- and environment-attributable outcomes are reported separately and
     excluded from the defect denominator.

  2. QUALITATIVE — a review sheet (CSV) with one row per generated test and the
     rubric columns left blank, for a human to score. Automated proxies are
     pre-filled where they can be computed honestly (grounding, novelty), so the
     reviewer only has to judge what actually needs judgement.

Usage:
    ./venv/bin/python scripts/analyze_batch.py [--project contacts-app] [--csv out.csv]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from settings import PROJECT, RAG_URL, GATEWAY_URL as GATEWAY  # noqa: E402

# Attribution → is this evidence about the APP, or about our agent/environment?
APP_FAULT = {"ASSERTION_FAILURE", "CRASH"}
AGENT_FAULT = {"TIMEOUT", "ELEMENT_NOT_FOUND", "NAVIGATION_FAILURE"}
ENV_FAULT = {"PRECONDITION_NOT_MET", "PERMISSION_DENIED", "STEP_LIMIT_EXCEEDED"}

RUBRIC = [
    "grounding",        # cites a real requirement / uses real UI control names
    "novelty",          # not a rephrase of an earlier test
    "defect_oriented",  # probes a failure mode vs. confirms the happy path
    "executable",       # steps map onto controls that exist
    "specificity",      # concrete inputs, not "enter some data"
]


def fetch(path: str, **params):
    r = requests.get(f"{RAG_URL}{path}", params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--csv", default="logs/review_sheet.csv")
    args = ap.parse_args()

    logs = fetch("/execution/logs", project=args.project, limit=200).get("logs", [])
    tests = fetch("/tests/recent", project=args.project, limit=200).get("tests", [])
    stats = fetch("/graph/stats", project=args.project)
    appmodel = fetch("/appmodel/graph", project=args.project)

    if not logs:
        print("No execution logs yet for", args.project)
        return 1

    by_test = {}
    for e in logs:  # newest first; keep the latest run per test
        by_test.setdefault(e.get("test_case_id"), e)
    runs = list(by_test.values())

    kinds = Counter()
    for e in runs:
        v, et = (e.get("verdict") or ""), (e.get("error_type") or "")
        if v in ("pass", "passed"):
            kinds["pass"] += 1
        elif et in APP_FAULT or (v == "failed" and not et):
            kinds["candidate_defect"] += 1
        elif et in AGENT_FAULT:
            kinds["agent_failure"] += 1
        elif et in ENV_FAULT:
            kinds["environment"] += 1
        else:
            kinds["unclassified"] += 1

    total = sum(kinds.values())
    # Runs that produced evidence about the app at all (the honest denominator).
    informative = kinds["pass"] + kinds["candidate_defect"]
    autonomy = (total - kinds["agent_failure"]) / total if total else 0.0
    steps = [e.get("device_steps") or 0 for e in runs]
    secs = [(e.get("duration_ms") or 0) / 1000 for e in runs]

    print(f"\n=== BATCH ANALYSIS — {args.project} ===")
    print(f"runs analysed: {total}\n")
    print(f"{'outcome':<20}{'n':>5}{'share':>9}   attribution")
    rows = [
        ("pass", "app behaved as expected"),
        ("candidate_defect", "APP fault — needs human confirmation"),
        ("agent_failure", "OUR agent could not complete"),
        ("environment", "test data / permissions / budget"),
        ("unclassified", "-"),
    ]
    for key, note in rows:
        n = kinds[key]
        if n or key in ("pass", "candidate_defect"):
            print(f"{key:<20}{n:>5}{(100*n/total if total else 0):>8.1f}%   {note}")

    print(f"\nautonomy rate        : {100*autonomy:5.1f}%   (runs not lost to agent failure)")
    print(f"informative runs     : {informative:5d}     (the denominator for precision)")
    if informative:
        print(f"candidate defect rate: {100*kinds['candidate_defect']/informative:5.1f}%   of informative runs")
    print(f"steps  mean/median   : {sum(steps)/len(steps):5.1f} / {sorted(steps)[len(steps)//2]:>5}")
    print(f"seconds mean         : {sum(secs)/len(secs):5.1f}")
    print(f"\ncoverage: {stats.get('test_case_count',0)} tests · "
          f"{stats.get('covered_requirement_count',0)}/{stats.get('requirement_count',0)} requirements · "
          f"{appmodel.get('state_count',0)} UI states mapped")

    # ── review sheet ─────────────────────────────────────────────────────────
    titles = {t.get("id"): t for t in tests}
    seen: list[str] = []
    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    with open(args.csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["test_id", "title", "area", "verdict", "attribution",
                    "steps", "seconds", "auto_novelty", *RUBRIC, "reviewer_notes"])
        for e in sorted(runs, key=lambda x: x.get("created_at") or ""):
            tid = e.get("test_case_id") or ""
            t = titles.get(tid, {})
            title = t.get("title") or e.get("title") or ""
            et = e.get("error_type") or ""
            attribution = ("pass" if e.get("verdict") in ("pass", "passed")
                           else "app" if et in APP_FAULT or not et
                           else "agent" if et in AGENT_FAULT
                           else "environment")
            # Cheap novelty proxy: token overlap against every earlier title.
            toks = set(title.lower().split())
            dup = max((len(toks & set(p.lower().split())) / max(1, len(toks | set(p.lower().split())))
                       for p in seen), default=0.0)
            seen.append(title)
            w.writerow([tid, title, t.get("area", ""), e.get("verdict", ""), attribution,
                        e.get("device_steps", 0), round((e.get("duration_ms") or 0) / 1000),
                        f"{1-dup:.2f}", *["" for _ in RUBRIC], ""])

    # ── learning-layer report ────────────────────────────────────────────────
    # These signals steer generation but are otherwise invisible, so a suite that
    # ran with an empty learning layer looks identical to one that used it.
    def _get(path, **params):
        try:
            return fetch(path, project=args.project, **params)
        except Exception:
            return {}

    print("\n=== LEARNING LAYER AFTER THIS SUITE ===")
    defects = _get("/defects/summary")
    risk = _get("/risk/scores").get("risk_scores", []) or []
    anomalies = _get("/anomalies", limit=20).get("anomalies", []) or []
    strategies = _get("/strategy/memory").get("strategies", []) or []
    patterns = _get("/execution/error-patterns").get("error_patterns", []) or []
    nav = _get("/navtree/stats")

    print(f"defect history       : {defects.get('total_defects', 0)} defects"
          + ("  (none ingested — this signal contributed nothing)" if not defects.get("total_defects") else ""))
    if defects.get("prone_areas"):
        print("  defect-prone areas :", ", ".join(str(a.get('area', a)) for a in defects["prone_areas"][:6]))

    print(f"regression risk      : {len(risk)} scored areas")
    for r in sorted(risk, key=lambda x: -(x.get("regression_risk_score") or x.get("score") or 0))[:6]:
        sc = r.get("regression_risk_score") or r.get("score") or 0
        print(f"    {sc:.2f}  {r.get('area') or r.get('feature') or '?'}")

    print(f"anomalies detected   : {len(anomalies)}")
    for a in anomalies[:6]:
        print(f"    [{a.get('severity','?')}] {str(a.get('description') or a.get('anomaly_type'))[:100]}")

    print(f"error patterns mined : {len(patterns)}")
    for pt in patterns[:5]:
        print(f"    x{pt.get('frequency','?')}  {str(pt.get('description') or pt.get('pattern_signature'))[:90]}")

    print(f"strategy memory      : {len(strategies)} strategies")
    for st in strategies[:5]:
        print(f"    {st.get('strategy_type','?'):<22} score={st.get('decayed_score', st.get('effectiveness_score', 0)):.2f}"
              f"  effective {st.get('times_effective',0)}/{st.get('times_applied',0)}")

    print(f"navigation memory    : {nav.get('nav_nodes',0)} nodes, {nav.get('avoid_nodes',0)} marked avoid, depth {nav.get('max_depth',0)}")

    # ── degradation report ───────────────────────────────────────────────────
    try:
        import urllib.request as _u, json as _j
        dash = _j.load(_u.urlopen(f"{GATEWAY}/dashboard/data?project={args.project}", timeout=60))
        deg = dash.get("degradations") or {}
    except Exception:
        deg = {}
    print("\n=== SILENT FALLBACKS DURING THIS SUITE ===")
    if not deg.get("total"):
        print("  none — no capability was silently lost")
    else:
        print(f"  {deg['total']} fallback(s), worst severity: {deg.get('worst_severity')}")
        if deg.get("trustworthy") is False:
            print("  *** RESULTS SHOULD NOT BE TRUSTED ***")
        for e in (deg.get("events") or [])[:8]:
            print(f"    [{e.get('severity')}] {e.get('kind')}: {e.get('detail','')[:100]}")

    print(f"\nreview sheet -> {args.csv}")
    print(f"score each test 1-5 on: {', '.join(RUBRIC)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
