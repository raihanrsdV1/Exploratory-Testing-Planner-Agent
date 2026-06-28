"""
Lightweight in-process metrics counters.

Usage:
    from observability import inc, get_metrics
    
    inc("http_requests_total")
    metrics = get_metrics()
"""

import threading
from collections import Counter

_counters = Counter()
_lock = threading.Lock()

def inc(name: str, count: int = 1) -> None:
    """Increment a metric counter safely across threads."""
    with _lock:
        _counters[name] += count

def get_metrics() -> dict[str, int]:
    """Return a snapshot of all metrics."""
    with _lock:
        return dict(_counters)
