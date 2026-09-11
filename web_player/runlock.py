"""One batch at a time.

Two batches were once started against the same target within minutes of each
other. They shared the Neo4j project (each wiping the other's history through
`clean_slate`), they shared the browser-facing site, and they shared
`logs/web_player.log` — so the transcript interleaved two loops and read as a
single run doing impossible things: rounds in the order 1,2,2,3,3,4,5,4, eight
executions for a five-round batch, and one test reporting "the run is headless"
while a window was plainly open. An hour went into diagnosing a bug that did not
exist, because the evidence belonged to two processes at once.

Stamping each trace line with its writer made that *visible*. This makes it
*impossible*, which is the better fix: the second batch is refused at startup,
before it can corrupt anything, and is told exactly which run holds the lock.

Deliberately simple:

* A lock is a small JSON file. No daemon, no third-party dependency.
* It records the PID, so a lock left behind by a killed run is detected as stale
  and taken over rather than blocking every future batch — the usual failure of
  naive lock files, and the reason people learn to delete them by hand.
* Releasing never raises. Failing to clean up a lock must not turn a finished
  run into a crashed one.
"""

from __future__ import annotations

import json
import os
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCK_PATH = os.environ.get("WEB_RUN_LOCK") or os.path.join(_ROOT, "logs", ".web_player.lock")


class RunInProgress(RuntimeError):
    """Another batch holds the lock. Carries enough detail to act on."""


def _pid_alive(pid: int) -> bool:
    """Is this process still running? False on any doubt, so a stale lock clears."""
    if pid <= 0:
        return False
    try:
        if os.name == "nt":
            import ctypes

            # QUERY_LIMITED_INFORMATION succeeds for processes we may not signal.
            handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
            if not handle:
                return False
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        os.kill(pid, 0)  # signal 0 tests existence without touching the process
        return True
    except Exception:
        return False


def _read() -> dict | None:
    try:
        with open(LOCK_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


class RunLock:
    """Context manager holding the batch lock for this process."""

    def __init__(self, profile: str = "", rounds: int = 0):
        self.profile = profile
        self.rounds = rounds
        self.held = False

    def __enter__(self) -> "RunLock":
        existing = _read()
        if existing:
            pid = int(existing.get("pid") or 0)
            if _pid_alive(pid):
                started = existing.get("started_at", "?")
                raise RunInProgress(
                    f"Another batch is already running: pid {pid}, profile "
                    f"'{existing.get('profile', '?')}', started {started}. Two batches "
                    f"share the knowledge graph, the site and the trace log, and each "
                    f"would corrupt the other's results. Wait for it, or stop it first.\n"
                    f"  lock file: {LOCK_PATH}"
                )
            # The holder is gone — a killed or crashed run. Take it over silently;
            # a lock that outlives its process must not block every later batch.
        os.makedirs(os.path.dirname(LOCK_PATH), exist_ok=True)
        with open(LOCK_PATH, "w", encoding="utf-8", newline="\n") as fh:
            json.dump({
                "pid": os.getpid(),
                "profile": self.profile,
                "rounds": self.rounds,
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, fh, indent=2)
        self.held = True
        return self

    def __exit__(self, *_exc) -> None:
        self.release()

    def release(self) -> None:
        """Drop the lock if we own it. Never raises."""
        if not self.held:
            return
        try:
            current = _read()
            if current and int(current.get("pid") or 0) == os.getpid():
                os.remove(LOCK_PATH)
        except Exception:
            pass
        finally:
            self.held = False
