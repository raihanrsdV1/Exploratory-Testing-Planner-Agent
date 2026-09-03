#!/usr/bin/env python3
"""Web player CLI — the batch loop.

Same six-step workflow as the Android executor, so a run of either is directly
comparable:

  1. ask the gateway for the next test case
  2. translate it into a browser goal
  3. execute it against the real site
  4. read a verdict out of the agent's report AND the browser's own signals
  5. log the verdict, which drives what the planner generates next
  6. repeat

Run:  py -m web_player.runner
      py -m web_player.runner --rounds 5
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
import traceback

# Importable when invoked as a module from the repo root or as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for _stream in (sys.stdout, sys.stderr):  # emoji on a cp1252 console
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import requests  # noqa: E402

import settings as cfg  # noqa: E402
from web_player import failures, gateway, goal as goal_mod  # noqa: E402
from web_player.agent import WebAgent  # noqa: E402
from web_player.browser import BrowserSession  # noqa: E402
from web_player.llm import ChatClient  # noqa: E402
from web_player.oracles import Collector  # noqa: E402


def _header(text: str) -> None:
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


# ──────────────────────────────────────────────────────────────────────────────
# Preflight
# ──────────────────────────────────────────────────────────────────────────────

def preflight() -> None:
    """Fail loudly, before the first test case, on anything that would waste a run."""
    _header("PREFLIGHT CHECK (web)")

    print("[1/5] Config...")
    if not cfg.WEB_BASE_URL:
        sys.exit("  ❌ WEB_BASE_URL is not set. Point it at the site under test.")
    print(f"  ✅ Site: {cfg.WEB_SITE_NAME} @ {cfg.WEB_BASE_URL}")
    print(f"     browser={cfg.WEB_BROWSER} headless={cfg.WEB_HEADLESS} "
          f"viewport={cfg.WEB_VIEWPORT} same_origin_only={cfg.WEB_SAME_ORIGIN_ONLY}")

    print("[2/5] Gateway...")
    resp = requests.get(f"{gateway.GATEWAY_URL}/health", timeout=30)
    resp.raise_for_status()
    print(f"  ✅ Gateway: {resp.json()}")

    print("[3/5] RAG API...")
    resp = requests.get(f"{gateway.RAG_URL}/health", timeout=30)
    resp.raise_for_status()
    print(f"  ✅ RAG API: {resp.json()}")

    print("[4/5] Executor model...")
    try:
        ChatClient(cfg)
    except Exception as exc:
        sys.exit(f"  ❌ {exc}")
    print(f"  ✅ {cfg.WEB_LLM_PROVIDER} / {cfg.WEB_LLM_MODEL}")

    print("[5/5] Playwright...")
    try:
        import playwright  # noqa: F401
    except ImportError:
        sys.exit("  ❌ Playwright is not installed.  pip install playwright"
                 "  &&  playwright install chromium")
    print("  ✅ Playwright importable (browser binary is checked on launch)")
    print("\n🚀 Preflight passed. Starting web executor loop.\n")


# ──────────────────────────────────────────────────────────────────────────────
# One test case
# ──────────────────────────────────────────────────────────────────────────────

async def execute_test_case(session: BrowserSession, collector: Collector,
                            client: ChatClient, tc: dict) -> dict:
    """Run one test case end to end. Returns {verdict, notes, duration_seconds}."""
    tc_id = tc.get("test_case_id", "?")
    title = tc.get("title", "?")
    goal = goal_mod.build_goal(tc)

    _header(f"EXECUTING IN BROWSER: {tc_id}")
    print(f"Title: {title}")
    print(f"Goal:\n{goal}\n" + "-" * 72)

    collector.reset()
    started = time.time()

    try:
        await session.reset_to_base()
    except Exception as exc:
        duration = time.time() - started
        notes = f"Could not open {cfg.WEB_BASE_URL}: {type(exc).__name__}: {exc}"
        print(f"\n❌ {notes}")
        gateway.log_execution(tc, "failed", duration * 1000, 0, [],
                              error_type="NAVIGATION_FAILURE", error_message=notes)
        return {"verdict": "failed", "notes": notes, "duration_seconds": duration}

    agent = WebAgent(session.page, cfg, client)

    try:
        result = await agent.run(goal, cfg.WEB_MAX_STEPS, cfg.WEB_TIMEOUT)
    except Exception as exc:
        duration = time.time() - started
        notes = (f"Web execution CRASHED after {duration:.1f}s. "
                 f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-500:]}")
        print(f"\n❌ CRASH: {exc}")
        await session.screenshot(f"{tc_id}-crash")
        gateway.log_execution(tc, "failed", duration * 1000, 0, [],
                              error_type="CRASH", error_message=str(exc))
        return {"verdict": "failed", "notes": notes, "duration_seconds": duration}

    success, reason, steps = result.success, result.reason, result.steps
    error_type = failures.classify(reason, success)
    recovery_action = ""

    # ── WP7 self-healing: one adaptive re-attempt for recoverable categories ──
    if not success and cfg.SELF_HEAL:
        strategy = failures.recovery_strategy(error_type)
        if strategy["retry"]:
            print(f"\n🔧 Self-heal: {error_type} → {strategy['action']} (retrying once)")
            retry_goal = goal_mod.build_retry_goal(tc, error_type, reason, strategy)
            budget = cfg.WEB_TIMEOUT * 2 if error_type == "TIMEOUT" else cfg.WEB_TIMEOUT
            try:
                retry = await agent.run(retry_goal, cfg.WEB_MAX_STEPS, budget)
                steps += retry.steps
                result.urls.extend(u for u in retry.urls if u not in result.urls)
                if retry.success:
                    success, reason = True, f"Recovered via self-heal: {retry.reason}"
                    recovery_action = f"{error_type}: {strategy['action']} -> RECOVERED"
                else:
                    reason = retry.reason
                    recovery_action = f"{error_type}: {strategy['action']} -> still failed"
            except Exception as exc:
                recovery_action = f"{error_type}: recovery attempt errored ({exc})"
        else:
            recovery_action = f"{error_type}: {strategy['action']} (no retry — logged for investigation)"

    # ── Browser oracles: signals the agent may not have noticed ──────────────
    findings = collector.findings
    override = findings.verdict_override(cfg)
    if success and override:
        error_type, override_reason = override
        success = False
        reason = f"{override_reason} (the agent reported success; the browser disagreed)"
        print(f"\n🔎 Oracle override: {error_type} — {override_reason}")

    duration = time.time() - started
    verdict = "pass" if success else "failed"
    logged_error_type = "" if success else (error_type or failures.classify(reason))

    shot = await session.screenshot(f"{tc_id}-{verdict}")
    notes = (
        f"Web execution completed in {duration:.1f}s. Steps taken: {steps}. "
        f"Success={success}. Reason: {reason} | Browser signals: {findings.summary()}"
        + (f" | Self-heal: {recovery_action}" if recovery_action else "")
        + (f" | Screenshot: {shot}" if shot else "")
    )

    print(f"\n{'✅' if success else '❌'} Web result: success={success}")
    print(f"   Steps taken: {steps}")
    print(f"   Reason: {reason[:300]}")
    print(f"   Browser signals: {findings.counts()}")
    if result.urls:
        print(f"   Route: {' -> '.join(result.urls[:6])}")

    gateway.log_execution(tc, verdict, duration * 1000, steps, result.urls,
                          error_type=logged_error_type,
                          error_message=("" if success else reason[:500]),
                          recovery_action=recovery_action)
    return {"verdict": verdict, "notes": notes, "duration_seconds": duration}


# ──────────────────────────────────────────────────────────────────────────────
# Batch loop
# ──────────────────────────────────────────────────────────────────────────────

def _show_testcase(tc: dict) -> None:
    print(f"  ID:       {tc.get('test_case_id', '?')}")
    print(f"  Title:    {tc.get('title', '?')}")
    print(f"  Page:     {tc.get('screen', '?')}")
    print(f"  Area:     {tc.get('area', '?')}")
    for i, step in enumerate(tc.get("steps", []), 1):
        print(f"    {i}. {step}")
    print(f"  Expected: {tc.get('expected_result', '?')}")


async def main(rounds: int) -> None:
    preflight()

    if cfg.CLEAN_SLATE:
        _header("CLEAN SLATE — resetting execution history")
        try:
            resp = requests.post(f"{gateway.RAG_URL}/project/reset", json={
                "project": cfg.PROJECT, "delete_tests": True,
                "delete_srs": False, "delete_figma": False, "delete_appmodel": False,
            }, timeout=120)
            resp.raise_for_status()
            print(f"  reset: {resp.json().get('deleted')}")
        except Exception as exc:
            print(f"  ⚠️  reset failed: {exc}")

    _header("PLANNER → GENERATING FIRST TEST CASE")
    tc = (gateway.next_testcase() or {}).get("next_testcase", {})
    if not tc or not tc.get("steps"):
        print("❌ Planner returned an empty test case. Aborting.")
        return
    print("Generated test case:")
    _show_testcase(tc)

    client = ChatClient(cfg)
    results: list[dict] = []

    async with BrowserSession(cfg) as session:
        collector = Collector(session.page, cfg)
        collector.attach()

        for i in range(1, rounds + 1):
            _header(f"ROUND {i}/{rounds}")
            outcome = await execute_test_case(session, collector, client, tc)
            results.append({
                "round": i,
                "test_case_id": tc.get("test_case_id"),
                "title": tc.get("title"),
                "area": tc.get("area", "?"),
                **outcome,
            })

            try:
                logged = gateway.log_verdict(tc, outcome["verdict"], outcome["notes"])
                print(f"  Logged: {logged.get('test_case_id')} | {outcome['verdict']}")
            except Exception as exc:
                print(f"  ⚠️  Failed to log verdict: {exc}")

            if i == rounds:
                break

            _header("PLANNER → GENERATING NEXT TEST CASE")
            tc = (gateway.next_testcase() or {}).get("next_testcase", {})
            if not tc or not tc.get("steps"):
                print("  ❌ Planner returned an empty test case. Ending loop.")
                break
            print("Next test case:")
            _show_testcase(tc)

    _summarize(results)


def _summarize(results: list[dict]) -> None:
    _header("EXECUTION SUMMARY (web)")
    total = len(results)
    passed = sum(1 for r in results if r["verdict"] == "pass")
    print(f"  Total Rounds:   {total}")
    print(f"  Passed:         {passed} ✅")
    print(f"  Failed:         {total - passed} ❌")
    print(f"  Total Duration: {sum(r['duration_seconds'] for r in results):.1f}s")
    print(f"  Pass Rate:      {(passed / total * 100) if total else 0:.0f}%")
    for r in results:
        status = "✅ PASS" if r["verdict"] == "pass" else "❌ FAILED"
        print(f"\n  {'─' * 66}")
        print(f"  Round {r['round']}: {r['test_case_id']} | {status} | {r['duration_seconds']:.1f}s")
        print(f"    Title: {r['title']}")
        print(f"    Area:  {r['area']}")
        print(f"    Notes: {r['notes'][:400]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Playwright web test executor")
    parser.add_argument("--rounds", type=int, default=cfg.WEB_ROUNDS,
                        help=f"test cases to run (default {cfg.WEB_ROUNDS}, from WEB_ROUNDS)")
    args = parser.parse_args()
    asyncio.run(main(rounds=args.rounds))
