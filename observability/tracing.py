"""
Trace context propagation and LangGraph node timing helpers.

Provides:
  - A contextvars.ContextVar that carries {request_id, project} across the
    synchronous LangGraph call chain (no async needed — LangGraph is sync).
  - set_trace() / get_trace() — bind / read the current trace payload.
  - timed_node(name, fn) — decorator that wraps a LangGraph node function
    with structured enter/exit timing logs.

Usage in langgraph_agent.py:
    from observability.tracing import timed_node

    def bootstrap_context(state):
        ...

    bootstrap_context = timed_node("bootstrap_context", bootstrap_context)
"""

from __future__ import annotations

import time
from contextvars import ContextVar
from functools import wraps
from typing import Any, Callable

import structlog

log = structlog.get_logger("agent")

# ── Trace context ─────────────────────────────────────────────────────────────

_trace_ctx: ContextVar[dict] = ContextVar("trace_ctx", default={})


def set_trace(request_id: str = "", project: str = "") -> None:
    """Bind a request_id (and optional project) into the current context."""
    current = _trace_ctx.get({})
    _trace_ctx.set({**current, "request_id": request_id, "project": project})


def get_trace() -> dict:
    """Return the current trace payload (request_id, project)."""
    return _trace_ctx.get({})


# ── LangGraph node timer ──────────────────────────────────────────────────────

def timed_node(name: str) -> Callable:
    """
    Wrap a LangGraph node function so that every execution logs:
      - `node_enter` with node name and current round number (if in state)
      - `node_exit`  with node name and wall-clock duration_ms

    The wrapper is transparent — it accepts and returns the same AgentState dict.
    """

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(state: dict, **kwargs: Any) -> dict:
            trace = get_trace()
            round_no = state.get("round_no", 0)

            # Bind node name into the structlog context-var store for this call
            structlog.contextvars.bind_contextvars(node=name)

            log.info(
                "node_enter",
                node=name,
                round=round_no,
                **{k: v for k, v in trace.items() if v},
            )

            start = time.perf_counter()
            try:
                result = fn(state, **kwargs)
            except Exception as exc:
                duration_ms = round((time.perf_counter() - start) * 1000, 1)
                log.error(
                    "node_error",
                    node=name,
                    duration_ms=duration_ms,
                    error=str(exc),
                    **{k: v for k, v in trace.items() if v},
                )
                raise

            duration_ms = round((time.perf_counter() - start) * 1000, 1)
            log.info(
                "node_exit",
                node=name,
                duration_ms=duration_ms,
                **{k: v for k, v in trace.items() if v},
            )

            # Un-bind node from context-vars after exit
            structlog.contextvars.unbind_contextvars("node")
            return result

        return wrapper
    return decorator
