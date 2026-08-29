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

import json
import os
import threading
from collections import Counter
from datetime import datetime, timezone

_LOCK = threading.Lock()
_EVENTS: list[dict] = []
_COUNTS: Counter = Counter()

CRITICAL, MAJOR, MINOR = "critical", "major", "minor"

# Cross-process sink. These globals are per-process, and the executor and the
# API run as SEPARATE processes — so every degradation that matters most (the
# device portal missing, observations dropped, fixtures unseeded) was recorded
# in the executor and could never be seen by the dashboard or the batch report,
# which read the API's copy and cheerfully printed "0 fallbacks, trustworthy".
# Appending to a shared file makes a run's degradations visible to whoever asks.
_SINK = os.environ.get("DEGRADATION_SINK") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "degradations.jsonl"
)


def _append_to_sink(event: dict) -> None:
    try:
        os.makedirs(os.path.dirname(_SINK), exist_ok=True)
        with open(_SINK, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event) + "\n")
    except Exception:
        pass          # the sink is a convenience; never break the caller


def _read_sink() -> list[dict]:
    try:
        with open(_SINK, encoding="utf-8") as fh:
            return [json.loads(l) for l in fh if l.strip()]
    except Exception:
        return []


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
        event = _EVENTS[-1]
        # Keep memory bounded; counts stay exact even when detail is trimmed.
        if len(_EVENTS) > 500:
            del _EVENTS[:250]
    _append_to_sink(event)


def snapshot(limit: int = 50) -> dict:
    """Current degradation state for the whole RUN, newest first.

    Merges this process's events with the shared sink, so an API process
    reporting on a run can see what the executor process lost. Without this the
    dashboard reported "0 fallbacks, trustworthy" while the executor had
    recorded a CRITICAL one seconds earlier.
    """
    with _LOCK:
        mine = list(_EVENTS)
    seen = {(e.get("at"), e.get("kind")) for e in mine}
    merged = mine + [e for e in _read_sink()
                     if (e.get("at"), e.get("kind")) not in seen]
    merged.sort(key=lambda e: e.get("at") or "")
    counts: Counter = Counter(e.get("kind", "?") for e in merged)
    events = list(reversed(merged[-limit:]))
    worst = CRITICAL if any(e.get("severity") == CRITICAL for e in merged) else (
        MAJOR if any(e.get("severity") == MAJOR for e in merged) else
        (MINOR if merged else "none"))
    return {
        "total": len(merged),
        "worst_severity": worst,
        "counts": dict(counts),
        "events": events,
        "trustworthy": not any(e.get("severity") == CRITICAL for e in merged),
    }


def reset() -> None:
    """Clear state at the start of a run so a report covers only that run.

    Clears the shared sink too, otherwise a run inherits every degradation ever
    recorded and "trustworthy" would be false forever after the first bad run.
    """
    with _LOCK:
        _EVENTS.clear()
        _COUNTS.clear()
    try:
        if os.path.exists(_SINK):
            os.remove(_SINK)
    except Exception:
        pass
