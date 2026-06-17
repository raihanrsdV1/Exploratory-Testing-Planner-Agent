"""
Live exploration coverage map + prompt directives.

All areas are derived at runtime from test history and the ingested UI graph —
there are no hardcoded feature areas, so this works for any application.
"""

from __future__ import annotations

import re


def compute_coverage_map(recent_tests: list[dict], figma_screens: list[dict]) -> dict:
    """Derive a coverage map from test history + known screens (purely data-driven)."""
    area_stats: dict[str, dict] = {}
    for t in recent_tests:
        area = re.sub(r"\s+", "_", str(t.get("area", "general")).lower().strip()) or "general"
        stats = area_stats.setdefault(area, {"total": 0, "passed": 0, "failed": 0})
        stats["total"] += 1
        if str(t.get("verdict", "")).lower() == "pass":
            stats["passed"] += 1
        else:
            stats["failed"] += 1

    screen_purposes: set[str] = set()
    for s in figma_screens:
        p = str(s.get("purpose", "")).lower().strip()
        if p and p != "other":
            screen_purposes.add(p)

    tested_areas = set(area_stats.keys()) - {"general"}
    uncovered_purposes = sorted(screen_purposes - tested_areas)
    hot_spots = sorted(a for a, s in area_stats.items() if s["failed"] >= 2)
    exhausted = sorted(a for a, s in area_stats.items() if s["total"] >= 4 and s["failed"] == 0)
    cov_pct = round(100 * len(tested_areas) / len(screen_purposes)) if screen_purposes else 0

    return {
        "area_stats": area_stats,
        "uncovered_purposes": uncovered_purposes,
        "hot_spots": hot_spots,
        "exhausted_areas": exhausted,
        "total_tests": len(recent_tests),
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
    lines.append(
        f"Overall coverage: {coverage_map.get('total_areas_tested', 0)}/"
        f"{coverage_map.get('total_areas_available', 0)} known areas ({coverage_map.get('coverage_pct', 0)}%)"
    )
    return "\n".join(lines)


def build_exploration_directive(coverage_map: dict, recent_tests: list[dict]) -> str:
    """Prioritised, data-driven next-action directive for the planner."""
    lines: list[str] = []
    hot_spots = coverage_map.get("hot_spots", [])
    uncovered = coverage_map.get("uncovered_purposes", [])
    exhausted = coverage_map.get("exhausted_areas", [])

    if hot_spots:
        lines.append(
            f"[PRIORITY 1 — INVESTIGATE] Areas with repeated failures need deeper edge-case coverage: "
            f"{', '.join(hot_spots[:3])}"
        )
    if uncovered:
        lines.append(f"[PRIORITY 2 — EXPAND] Areas with ZERO test coverage yet: {', '.join(uncovered[:5])}")

    last_areas = [str(t.get("area", "")).lower().strip() for t in recent_tests[:4] if t.get("area")]
    if len(last_areas) >= 3 and len(set(last_areas)) == 1:
        last_verdicts = [str(t.get("verdict", "")).lower() for t in recent_tests[:4]]
        if all(v == "pass" for v in last_verdicts):
            lines.append(
                f"[PRIORITY 3 — PIVOT] Last {len(last_areas)} consecutive tests all PASSED in "
                f"'{last_areas[0]}'. This area is likely stable — move to a different area now."
            )
        else:
            lines.append(
                f"[PRIORITY 3 — VARY ANGLE] Still in '{last_areas[0]}' with mixed results. "
                "Try a different test type: boundary values, invalid input, or state transitions."
            )

    if exhausted:
        lines.append(
            f"[DEPRIORITIZE] Well-covered stable areas (4+ tests, 0 failures — avoid repeating): "
            f"{', '.join(exhausted[:3])}"
        )

    if not lines:
        lines.append("[OPEN EXPLORATION] No specific signal. Aim for breadth first, then depth.")
    return "\n".join(lines)
