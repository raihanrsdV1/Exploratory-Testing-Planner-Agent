"""
FastAPI request-logging middleware.

For every inbound HTTP request this middleware:
  1. Generates a unique request_id (UUID4 short form)
  2. Binds it (and the request path) into the structlog context-var store
     so every log line emitted during that request automatically carries it
  3. Logs `request_started` before the request is processed
  4. Logs `request_done` after the response, including status code and wall-clock latency

Usage:
    from observability.middleware import RequestLoggingMiddleware
    app.add_middleware(RequestLoggingMiddleware)
"""

from __future__ import annotations

import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from observability import inc
from observability.tracing import set_trace

log = structlog.get_logger("http")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs every HTTP request with a unique request_id and latency."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = uuid.uuid4().hex[:12]
        path = request.url.path
        method = request.method

        # Bind to structlog context-vars so child logs automatically inherit request_id
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            path=path,
        )

        # Also bind into our own tracing context-var
        set_trace(request_id=request_id)

        start = time.perf_counter()
        inc("http_requests_total")

        log.info(
            "request_started",
            method=method,
            path=path,
            request_id=request_id,
        )

        try:
            response: Response = await call_next(request)
        except Exception as exc:
            latency_ms = round((time.perf_counter() - start) * 1000, 1)
            inc("http_errors_total")
            log.error(
                "request_error",
                method=method,
                path=path,
                error=str(exc),
                latency_ms=latency_ms,
                request_id=request_id,
            )
            raise

        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        status = response.status_code

        if status >= 500:
            inc("http_errors_total")
            log_fn = log.error
        elif status >= 400:
            log_fn = log.warning
        else:
            log_fn = log.info

        log_fn(
            "request_done",
            method=method,
            path=path,
            status=status,
            latency_ms=latency_ms,
            request_id=request_id,
        )

        # Propagate request_id in response headers for clients / curl debugging
        response.headers["X-Request-ID"] = request_id
        return response
