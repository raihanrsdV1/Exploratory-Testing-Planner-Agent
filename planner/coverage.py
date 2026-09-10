"""
Live exploration coverage map + prompt directives.

All areas are derived at runtime from test history and the ingested UI graph —
there are no hardcoded feature areas, so this works for any application.
"""

from __future__ import annotations

import re

from . import config


# Failures that say nothing about the app: our agent, the environment, or missing
# test data. They must not create "hot spots", because a hot spot tells the planner
# "keep digging here" — and digging into an area we simply cannot reach produced a
# run of five near-duplicate tests that each burned the full step budget.
NON_INFORMATIVE_ERRORS = {
    "PRECONDITION_NOT_MET", "STEP_LIMIT_EXCEEDED", "NAVIGATION_LIVELOCK",
    "NAVIGATION_FAILURE", "ELEMENT_NOT_FOUND", "TIMEOUT", "PERMISSION_DENIED",
}


def compute_coverage_map(recent_tests: list[dict], figma_screens: list[dict]) -> dict:
    """Derive a coverage map from test history + known screens (purely data-driven)."""
    # Exclude not-yet-executed ("planned") tests so coverage reflects real runs only.
    executed = [t for t in recent_tests if str(t.get("verdict", "")).lower() != "planned"]
    area_stats: dict[str, dict] = {}
    for t in executed:
        area = re.sub(r"\s+", "_", str(t.get("area", "general")).lower().strip()) or "general"
        stats = area_stats.setdefault(
            area, {"total": 0, "passed": 0, "failed": 0, "unreachable": 0})
        stats["total"] += 1
        err = str(t.get("error_type", "") or "").upper()
        if str(t.get("verdict", "")).lower() == "pass":
            stats["passed"] += 1
        elif err in NON_INFORMATIVE_ERRORS:
            # Ran, but produced no evidence about the app.
            stats["unreachable"] += 1
        else:
            stats["failed"] += 1

    screen_purposes: set[str] = set()
    for s in figma_screens:
        p = str(s.get("purpose", "")).lower().strip()
        if p and p != "other":
            screen_purposes.add(p)

    tested_areas = set(area_stats.keys()) - {"general"}
    uncovered_purposes = sorted(screen_purposes - tested_areas)
    # Hot spot = the APP broke here repeatedly. Unreachable failures are excluded.
    hot_spots = sorted(a for a, s in area_stats.items() if s["failed"] >= 2)
    # Dead end = we keep trying and keep failing to even observe the app. Steer away.
    dead_ends = sorted(
        a for a, s in area_stats.items()
        if s["unreachable"] >= 2 and s["unreachable"] >= s["passed"] + s["failed"]
    )
    exhausted = sorted(
        a for a, s in area_stats.items()
        if s["total"] >= 4 and s["failed"] == 0 and s["unreachable"] == 0
    )
    cov_pct = round(100 * len(tested_areas) / len(screen_purposes)) if screen_purposes else 0

    return {
        "area_stats": area_stats,
        "uncovered_purposes": uncovered_purposes,
        "hot_spots": hot_spots,
        "dead_ends": dead_ends,
        "exhausted_areas": exhausted,
        "total_tests": len(executed),
        "total_areas_tested": len(tested_areas),
        "total_areas_available": len(screen_purposes),
        "coverage_pct": cov_pct,
    }


def build_coverage_block(coverage_map: dict) -> str:
    stats = coverage_map.get("area_stats", {})
    total = coverage_map.get("total_tests", 0)
    if not stats:
        return "No tests executed yet — start of exploration session. Begin with broad coverage."
    lines = [f"Total tests executed: {total}", "Per-area breakdown:"]
    for area, s in sorted(stats.items(), key=lambda x: (-x[1]["failed"], -x[1]["total"])):
        flag = " ⚠ REPEATED FAILURES" if s["failed"] >= 2 else (" ✓ stable" if s["failed"] == 0 and s["total"] >= 3 else "")
        lines.append(f"  {area}: {s['total']} tests ({s['passed']} pass / {s['failed']} fail){flag}")
    unc = coverage_map.get("uncovered_purposes", [])
    if unc:
        lines.append(f"Completely untested areas: {', '.join(unc)}")
    available = coverage_map.get("total_areas_available", 0)
    if available:
        lines.append(
            f"Overall coverage: {coverage_map.get('total_areas_tested', 0)}/"
            f"{available} known areas ({coverage_map.get('coverage_pct', 0)}%)"
        )
    else:
        # total_areas_available comes from Figma screen purposes. A project with
        # no Figma export (fully supported — Live App Model works without one)
        # always has 0 here, which rendered as the nonsensical "8/0 (0%)" —
        # claiming zero coverage on a project with real, substantial history.
        lines.append(
            f"Overall coverage: {coverage_map.get('total_areas_tested', 0)} area(s) tested so far "
            f"(no design-file area index available for this project to compute a % against)"
        )
    return "\n".join(lines)


def build_exploration_directive(coverage_map: dict, recent_tests: list[dict], mode: str | None = None) -> str:
    """Prioritised, data-driven next-action directive for the planner.

    ``mode`` sets the explore/exploit balance for the run (see config.EXPLORATION_MODE):
    ``exploit`` puts defect-prone depth first, ``explore`` puts untested breadth first,
    ``balanced`` investigates failures then expands.
    """
    mode = (mode or config.EXPLORATION_MODE or "balanced").strip().lower()
    recent_tests = [t for t in recent_tests if str(t.get("verdict", "")).lower() != "planned"]
    lines: list[str] = []
    hot_spots = coverage_map.get("hot_spots", [])
    dead_ends = coverage_map.get("dead_ends", [])
    uncovered = coverage_map.get("uncovered_purposes", [])
    exhausted = coverage_map.get("exhausted_areas", [])

    investigate = (
        f"[INVESTIGATE] Areas with repeated failures need deeper edge-case coverage: "
        f"{', '.join(hot_spots[:3])}"
    ) if hot_spots else ""
    expand = (
        f"[EXPAND] Areas with ZERO test coverage yet: {', '.join(uncovered[:5])}"
    ) if uncovered else ""

    if mode == "exploit":
        lines.append("[MODE: EXPLOIT] Go DEEP on areas that have already broken. Prefer another "
                     "angle on a known-fragile area over opening a new one.")
        ordered = [investigate, expand]
    elif mode == "explore":
        lines.append("[MODE: EXPLORE] Go BROAD. Cover untested areas first; only revisit a "
                     "known-fragile area once no untested area remains.")
        ordered = [expand, investigate]
    else:
        ordered = [investigate, expand]

    for i, block in enumerate([b for b in ordered if b], start=1):
        lines.append(f"[PRIORITY {i}] " + block)

    last_areas = [str(t.get("area", "")).lower().strip() for t in recent_tests[:4] if t.get("area")]
    if len(last_areas) >= 3 and len(set(last_areas)) == 1:
        last_verdicts = [str(t.get("verdict", "")).lower() for t in recent_tests[:4]]
        if all(v == "pass" for v in last_verdicts):
            lines.append(
                f"[PIVOT] Last {len(last_areas)} consecutive tests all PASSED in "
                f"'{last_areas[0]}'. This area is likely stable — move to a different area now."
            )
        else:
            lines.append(
                f"[VARY ANGLE] Still in '{last_areas[0]}' with mixed results. "
                "Try a different test type: boundary values, invalid input, or state transitions."
            )

    if dead_ends:
        lines.append(
            f"[AVOID — UNREACHABLE] Repeatedly could not observe the app in: {', '.join(dead_ends[:3])}. "
            "The preconditions cannot be met on this device (missing data, storage source, or permission). "
            "Do NOT generate further variants here — choose a different area, or a variant whose "
            "preconditions can be created from the current state."
        )
    if exhausted:
        lines.append(
            f"[DEPRIORITIZE] Well-covered stable areas (4+ tests, 0 failures — avoid repeating): "
            f"{', '.join(exhausted[:3])}"
        )

    if not lines:
        lines.append("[OPEN EXPLORATION] No specific signal. Aim for breadth first, then depth.")
    return "\n".join(lines)
