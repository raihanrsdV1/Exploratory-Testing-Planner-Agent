"""
Degradation tracking — make graceful fallbacks visible instead of silent.

The system is full of `except: ... continue` so that one bad screen cannot kill a
two-hour batch. That resilience is worth keeping, but it hid real failures:
requirement coverage read 0% for days because the POST that creates COVERS edges
failed inside a bare `except: pass`, and SRS extraction quietly substituted a
regex fallback that produced unusable requirements while the run reported "ok".

The rule this module enforces is: **degrade, but never quietly**. Every fallback
records what was lost and why. The counts surface on /health, on the dashboard,
and in the end-of-suite report, so a run that "succeeded" while silently
operating on worse data is visible as such.

Severity:
    critical — results from this run should not be trusted
    major    — a capability was lost (a source, a learning signal)
    minor    — transient, self-corrected (a retry that then succeeded)
"""

from __future__ import annotations

import threading
from collections import Counter
from datetime import datetime, timezone

_LOCK = threading.Lock()
_EVENTS: list[dict] = []
_COUNTS: Counter = Counter()

CRITICAL, MAJOR, MINOR = "critical", "major", "minor"


def record(kind: str, severity: str = MAJOR, detail: str = "", **context) -> None:
    """Record that the system fell back to a lesser behaviour."""
    with _LOCK:
        _COUNTS[kind] += 1
        _EVENTS.append({
            "kind": kind,
            "severity": severity,
            "detail": str(detail)[:400],
            "at": datetime.now(timezone.utc).isoformat(),
            **{k: str(v)[:200] for k, v in context.items()},
        })
        # Keep memory bounded; counts stay exact even when detail is trimmed.
        if len(_EVENTS) > 500:
            del _EVENTS[:250]


def snapshot(limit: int = 50) -> dict:
    """Current degradation state, newest first."""
    with _LOCK:
        events = list(reversed(_EVENTS[-limit:]))
        counts = dict(_COUNTS)
    worst = CRITICAL if any(e["severity"] == CRITICAL for e in events) else (
        MAJOR if any(e["severity"] == MAJOR for e in events) else (MINOR if events else "none"))
    return {
        "total": sum(counts.values()),
        "worst_severity": worst,
        "counts": counts,
        "events": events,
        "trustworthy": not any(e["severity"] == CRITICAL for e in events),
    }


def reset() -> None:
    """Clear state at the start of a run so a report covers only that run."""
    with _LOCK:
        _EVENTS.clear()
        _COUNTS.clear()
