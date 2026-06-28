"""
Observability package for the Exploratory Testing Planner Agent.

Provides:
  - Structured logging (structlog) with pretty console + JSONL file sinks
  - Optional Logtail / Better Stack cloud sink
  - FastAPI request-logging middleware (request_id, latency)
  - Lightweight in-process metrics counters
  - Context-var based trace propagation across the LangGraph call chain

Quick start:
    from observability import get_logger, setup_logging

    setup_logging()                        # call once at service startup
    log = get_logger(__name__)
    log.info("my_event", key="value")
"""

from .logger import get_logger, setup_logging
from .metrics import get_metrics, inc
from .tracing import get_trace, set_trace, timed_node

__all__ = [
    "get_logger",
    "setup_logging",
    "get_metrics",
    "inc",
    "get_trace",
    "set_trace",
    "timed_node",
]
