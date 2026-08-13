#!/usr/bin/env python3
"""
Dump the planner's generation prompt without spending an LLM call.

Rebuilds the prompt with the same builders the agent uses, writes it to
``logs/planner_prompt_preview.txt``, and prints a per-block size table so you can
see what the model is actually told and which blocks grow with session length.
See docs/PLANNER_PROMPT_ANATOMY.md for the interpretation.

Usage:  PROJECT=contacts-app python scripts/dump_prompt.py
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from planner import context_builders, coverage, prompts, rag_client  # noqa: E402
from planner.sources import registry as sources_registry  # noqa: E402
from planner.sources.base import RetrievalRequest  # noqa: E402

PROJECT = os.getenv("PROJECT", "contacts-app")
OBJECTIVE = "generate the next high-value exploratory test case"
OUT = Path(__file__).resolve().parent.parent / "logs" / "planner_prompt_preview.txt"


def main() -> None:
    brief = rag_client.get_brief_context(PROJECT)
    recent = brief.get("recent_tests", [])
    screens = brief.get("screen_index", [])
    overview = rag_client.get_figma_overview(PROJECT)
    cmap = coverage.compute_coverage_map(recent, screens)
    available = {s.name for s in sources_registry.available_sources(brief)}
    picked = context_builders.pick_relevant_screens(screens, [], recent)

    # Stand-in for one retrieval round: the SRS block the planner would have pulled.
    srs_context = rag_client.get_srs_and_history(
        PROJECT, "validation rules and error conditions", top_k=2
    ).get("context", "")

    figma_context = context_builders.build_figma_context(PROJECT, picked)
    live = sources_registry.get("live_ui")
    if live and live.is_available(brief):
        block = live.retrieve(PROJECT, RetrievalRequest(source="live_ui"), 8)
        if block:
            figma_context = (figma_context + "\n\n" + block.text).strip()

    defect, nav, failed_nav, strategy, risk, anomaly = context_builders.build_learned_context(
        PROJECT, available, OBJECTIVE, picked, [], []
    )

    prompt = prompts.build_testcase_prompt(
        app_name=os.getenv("APP_NAME", "the app under test"),
        objective=OBJECTIVE,
        srs_context=srs_context,
        figma_overview_context=context_builders.build_figma_overview_context(overview),
        figma_context=figma_context,
        figma_flow_context="",
        done_titles=[str(t.get("title", "")) for t in recent if t.get("title")],
        failed_titles=[
            str(t.get("title", "")) for t in recent
            if str(t.get("verdict", "")).lower() == "failed" and t.get("title")
        ],
        coverage_map=cmap,
        recent_tests=recent,
        defect_context=defect,
        nav_context=nav,
        failed_nav=failed_nav,
        strategy_context=strategy,
        risk_context=risk,
        anomaly_context=anomaly,
        failure_context=context_builders.build_failure_context(PROJECT, recent),
        requirements_context=context_builders.build_requirements_context(PROJECT),
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(prompt, encoding="utf-8")

    executed = [t for t in recent if str(t.get("verdict", "")).lower() != "planned"]
    failures = [t for t in recent if str(t.get("verdict", "")).lower() == "failed"]
    print(f"project: {PROJECT}   tests: {len(executed)} executed, {len(failures)} failed, "
          f"{len(recent)} logged")
    print(f"TOTAL:   {len(prompt):,} chars  ~{len(prompt) // 4:,} tokens")
    print(f"saved -> {OUT.relative_to(Path.cwd()) if OUT.is_relative_to(Path.cwd()) else OUT}\n")
    print(f"{'BLOCK':52} {'CHARS':>7}")
    print("-" * 62)
    for part in re.split(r"\n(?=## )", prompt):
        print(f"{part.splitlines()[0][:50]:52} {len(part):>7,}")


if __name__ == "__main__":
    main()
