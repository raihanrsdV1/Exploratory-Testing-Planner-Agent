#!/usr/bin/env python3
"""
crawl_runner.py — autonomous Live App Model crawler (WP1).

Maps a running Android app *before* goal-directed testing, for the zero-doc case
(no SRS, no Figma). It drives Droidrun with an app-agnostic exploration goal and
feeds every observed UI state into the Live App Model via ``/liveui/observe`` —
reusing the exact recording path the executor uses, so the crawl and normal test
runs build one shared, de-duplicated app map.

Needs a connected device/emulator + a configured executor LLM (same as
executor_runner). App-agnostic: only PROJECT + TARGET_APP_PACKAGE matter.

Run:  py ./clients/crawl_runner.py           # uses .env PROJECT / TARGET_APP_PACKAGE
Env:  CRAWL_ROUNDS (default 3), CRAWL_MAX_STEPS (default 25), CRAWL_GOAL (override)
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import requests

# Ensure the repo root is importable when run as `py ./clients/crawl_runner.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Reuse the executor's device/observe machinery (also loads .env + UTF-8 guard).
import clients.executor_runner as ex
from settings import CRAWL_ROUNDS, CRAWL_MAX_STEPS  # noqa: E402


CRAWL_GOAL = os.getenv("CRAWL_GOAL", "").strip() or (
    "You are systematically exploring this Android app to map its screens. "
    "Open the app, then visit as many DISTINCT screens, menus, tabs, and dialogs as you can: "
    "tap navigation items, open menus and settings, follow links, and scroll to reveal more. "
    "Prefer reaching screens you have not visited yet. Do NOT delete data, send messages, make "
    "purchases, or change account settings. After exploring, stop."
)


def _appmodel_stats() -> dict:
    try:
        r = requests.get(f"{ex.RAG_URL}/appmodel/graph", params={"project": ex.PROJECT}, timeout=15)
        r.raise_for_status()
        d = r.json()
        return {"states": d.get("state_count", 0), "edges": len(d.get("edges", []))}
    except Exception:
        return {"states": 0, "edges": 0}


async def _crawl_once(round_no: int) -> int:
    """Run one exploration episode; fold observed states into the Live App Model."""
    from mobilerun import MobileAgent, AndroidDriver, load_llm, MobileConfig, AgentConfig
    from mobilerun.config_manager.config_manager import LoggingConfig

    ex._attach_mobilerun_file_log()
    driver = AndroidDriver()

    api_key = ex.OPENROUTER_API_KEY if ex.EXECUTOR_LLM_PROVIDER.lower() == "openrouter" else ex.GEMINI_API_KEY
    provider = "OpenRouter" if ex.EXECUTOR_LLM_PROVIDER.lower() == "openrouter" else ex.EXECUTOR_LLM_PROVIDER
    llm = load_llm(provider, model=ex.EXECUTOR_LLM_MODEL, api_key=api_key)

    config = MobileConfig(
        agent=AgentConfig(max_steps=CRAWL_MAX_STEPS),
        logging=LoggingConfig(save_trajectory="all", trajectory_path="logs/trajectories", trajectory_gifs=False),
    )
    agent = MobileAgent(goal=CRAWL_GOAL, llms=llm, driver=driver, timeout=ex.EXECUTOR_TIMEOUT, config=config)

    observations = []
    handler = agent.run()
    try:
        from mobilerun.agent.common.events import RecordUIStateEvent
        async for event in handler.stream_events():
            if isinstance(event, RecordUIStateEvent):
                shot = await ex._safe_screenshot(driver)
                observations.append((getattr(event, "ui_state", None), shot))
    except Exception as e:
        print(f"  event streaming issue: {e}")
    try:
        await handler
    except Exception as e:
        print(f"  crawl episode ended: {e}")

    path = ex._record_observations(observations, f"crawl-{round_no}", driver_shot=await ex._safe_screenshot(driver))
    return len(path)


async def main():
    ex._print_header("AUTONOMOUS APP-MODEL CRAWL")
    print(f"Project: {ex.PROJECT}  |  App: {ex.TARGET_APP_PACKAGE}")
    print(f"Rounds: {CRAWL_ROUNDS}  |  Max steps/round: {CRAWL_MAX_STEPS}")

    before = _appmodel_stats()
    print(f"App model before: {before['states']} states, {before['edges']} transitions")

    for r in range(1, CRAWL_ROUNDS + 1):
        ex._print_header(f"CRAWL ROUND {r}/{CRAWL_ROUNDS}")
        try:
            recorded = await _crawl_once(r)
            snap = _appmodel_stats()
            print(f"  round {r}: recorded {recorded} states this episode → "
                  f"map now {snap['states']} states, {snap['edges']} transitions")
        except Exception as e:
            print(f"  round {r} failed: {e}")
            if "device" in str(e).lower() or "adb" in str(e).lower():
                print("  → No device found. Start an emulator / connect a device (adb devices) and retry.")
                break

    after = _appmodel_stats()
    ex._print_header("CRAWL COMPLETE")
    print(f"App model grew: {before['states']} → {after['states']} states, "
          f"{before['edges']} → {after['edges']} transitions")
    print(f"Inspect: {ex.RAG_URL}/appmodel/graph?project={ex.PROJECT}")


if __name__ == "__main__":
    asyncio.run(main())
