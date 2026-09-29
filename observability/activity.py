"""
Live agent activity — what each of the three agents is doing *right now*.

The dashboard already shows what has happened (trends, findings, traces). This
module answers the different question a live demo needs: which agent is working
at this instant, and on what. It is deliberately in-memory and lossy — it is a
status light, not a record. Everything durable is already in Neo4j and app.jsonl.

Planner and investigator run inside the gateway process, so they report here
directly. The executor runs in a separate client process and is *derived* rather
than reported: mobilerun writes one `ui_states/NNNN.json` per device step while
the run is in flight, so counting that directory gives a live step count without
the executor having to call back.

Usage:
    from observability import activity
    activity.emit("planner", "tool", "list_open_questions")
    activity.idle("planner")
"""

from __future__ import annotations

import threading
import time
from collections import deque

COMPONENTS = ("planner", "executor", "investigator")

# How long a component stays "active" without a fresh event before the view
# treats it as finished. A planner LLM call can take ~30s, so this must exceed
# the slowest single step we expect to see.
STALE_AFTER_S = 90.0

_EVENTS_KEPT = 120

_lock = threading.Lock()
_state: dict[str, dict] = {
    c: {"state": "idle", "detail": "", "since": 0.0, "updated": 0.0, "extra": {}}
    for c in COMPONENTS
}
_events: deque[dict] = deque(maxlen=_EVENTS_KEPT)
_seq = 0
# Which campaign round and test case the current work belongs to. Set by the
# executor at the top of each round, so every event in between can be labelled.
_run: dict = {"round": None, "rounds": None, "test_id": ""}


def emit(component: str, state: str, detail: str = "", **extra) -> None:
    """Record that `component` is doing `state` (`detail` names the specific thing).

    `state` is a coarse verb the view colours by — "thinking", "tool", "llm",
    "step", "evaluating". `detail` is what to print: a node name, a tool name, a
    model id.
    """
    if component not in COMPONENTS:
        return
    global _seq
    now = time.time()
    with _lock:
        cur = _state[component]
        # `since` marks when this component last went from idle to busy, so the
        # view can show how long the current turn has been running.
        if cur["state"] == "idle":
            cur["since"] = now
        # A model call is always made *by* a stage or a tool. Carry that name
        # along so the view can show "generate_testcase · qwen3.7-flash · 5040ms"
        # on one line instead of two unrelated-looking rows.
        if state == "llm" and cur["state"] in ("node", "tool", "thinking", "evaluating"):
            extra = {**extra, "stage": cur["detail"]}
            cur.update(state=state, detail=detail, updated=now, extra=extra)
        else:
            cur.update(state=state, detail=detail, updated=now, extra=extra)
        _seq += 1
        _events.append({
            "seq": _seq, "ts": now, "component": component,
            "state": state, "detail": detail,
            "round": _run["round"], "test_id": _run["test_id"], **extra,
        })


def idle(component: str) -> None:
    """Mark a component as no longer working."""
    if component not in COMPONENTS:
        return
    with _lock:
        _state[component].update(state="idle", detail="", updated=time.time(), extra={})


def set_run(round_no: int | None = None, rounds: int | None = None,
            test_id: str | None = None) -> None:
    """Label subsequent events with the campaign round and test case they belong to."""
    with _lock:
        if round_no is not None:
            _run["round"] = round_no
        if rounds is not None:
            _run["rounds"] = rounds
        if test_id is not None:
            _run["test_id"] = test_id


def run_context() -> dict:
    with _lock:
        return dict(_run)


def busiest(default: str = "planner") -> str:
    """Which agent is currently working — used to attribute a shared resource.

    The planner and the investigator call the same model client, so an LLM call
    on its own does not say who made it. Whichever agent is already marked busy
    is the one that did; if neither is, fall back to `default`.
    """
    now = time.time()
    with _lock:
        live = [
            (s["updated"], c) for c, s in _state.items()
            if s["state"] != "idle" and (now - s["updated"]) <= STALE_AFTER_S
        ]
    return max(live)[1] if live else default


def snapshot(since_seq: int = 0) -> dict:
    """Current state of all three components, plus events newer than `since_seq`.

    A component whose last update is older than STALE_AFTER_S is reported idle
    even if nothing called `idle()` — a crashed or killed run must not leave the
    demo showing a permanently lit agent.
    """
    now = time.time()
    with _lock:
        comps = {}
        for name, s in _state.items():
            stale = s["state"] != "idle" and (now - s["updated"]) > STALE_AFTER_S
            comps[name] = {
                "state": "idle" if stale else s["state"],
                "detail": "" if stale else s["detail"],
                "busy": not stale and s["state"] != "idle",
                "for_s": round(now - s["since"], 1) if s["since"] and not stale else 0.0,
                "age_s": round(now - s["updated"], 1) if s["updated"] else None,
                **({} if stale else s["extra"]),
            }
        events = [e for e in _events if e["seq"] > since_seq]
        seq = _seq
    return {"components": comps, "events": events, "seq": seq, "now": now,
            "run": run_context()}


def reset() -> None:
    """Clear all state. For tests."""
    global _seq
    with _lock:
        for c in COMPONENTS:
            _state[c] = {"state": "idle", "detail": "", "since": 0.0, "updated": 0.0, "extra": {}}
        _events.clear()
        _seq = 0
