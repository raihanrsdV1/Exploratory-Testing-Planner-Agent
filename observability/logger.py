"""
Structured logging setup for the Exploratory Testing Planner Agent.

Features:
  - Pretty colored console output (human-readable during development)
  - JSONL file sink at logs/app.jsonl (machine-readable, grep-friendly)
  - Optional Logtail / Better Stack cloud sink (set LOGTAIL_SOURCE_TOKEN)
  - All loggers auto-bind request_id from the current trace context

Usage:
    from observability.logger import setup_logging, get_logger

    setup_logging()              # call once at service startup
    log = get_logger(__name__)
    log.info("srs_ingested", project="contacts-app", chunks=17)
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import structlog

# ── Public API ────────────────────────────────────────────────────────────────

def setup_logging(
    log_level: str | None = None,
    log_file: str | None = None,
) -> None:
    """
    Configure structlog and stdlib logging.  Call once at service startup.

    Reads LOG_LEVEL and LOG_FILE from environment if not supplied explicitly.
    """
    level_name = (log_level or os.getenv("LOG_LEVEL", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)

    log_path = Path(log_file or os.getenv("LOG_FILE", "logs/app.jsonl"))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Shared processors (applied to every log record) ───────────────────────
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,          # inject request_id etc.
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
    ]

    # ── Console renderer (pretty, colored) ────────────────────────────────────
    console_renderer = structlog.dev.ConsoleRenderer(
        colors=sys.stderr.isatty() or os.getenv("FORCE_COLOR", "0") == "1",
        exception_formatter=structlog.dev.plain_traceback,
    )

    # ── JSONL file renderer ───────────────────────────────────────────────────
    jsonl_handler = logging.FileHandler(log_path, encoding="utf-8")
    jsonl_handler.setLevel(level)

    # ── Root stdlib logger ────────────────────────────────────────────────────
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove any pre-existing handlers to avoid duplicates on hot-reload
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(jsonl_handler)

    # ── Optional Logtail / Better Stack cloud sink ────────────────────────────
    logtail_token = os.getenv("LOGTAIL_SOURCE_TOKEN", "")
    if logtail_token:
        try:
            from logtail import LogtailHandler
            cloud_handler = LogtailHandler(source_token=logtail_token)
            cloud_handler.setLevel(level)
            root_logger.addHandler(cloud_handler)
        except ImportError:
            pass  # logtail not installed — skip silently

    # Silence noisy third-party loggers
    for noisy in ("uvicorn.access", "httpx", "httpcore", "neo4j"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # ── structlog configuration ───────────────────────────────────────────────
    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure the stdlib formatter for the console handler
    console_formatter = structlog.stdlib.ProcessorFormatter(
        processor=console_renderer,
        foreign_pre_chain=shared_processors,
    )
    console_handler.setFormatter(console_formatter)

    # Configure the stdlib formatter for the JSONL file handler
    jsonl_formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
        foreign_pre_chain=shared_processors,
    )
    jsonl_handler.setFormatter(jsonl_formatter)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a named structlog logger that auto-binds the current trace context."""
    return structlog.get_logger(name)
