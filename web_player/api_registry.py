"""What API routes this site actually has — learned, not guessed.

The planner once wrote *"verify that `GET /api/projects` returns only projects
owned by the current session"* into a test objective. That endpoint does not
exist: the real one is `/api/project`, singular. The executor obediently chased
the invented route six times and the test died in a livelock.

No amount of prompting fixes an endpoint list nobody has verified, so this keeps
one grounded in evidence:

* **Seeded** from a discovery walk of the running application (every `/api/*`
  request the app itself issued).
* **Grown** during ordinary runs — the oracle already sees every response, so a
  route the agent stumbles onto is remembered with the status it returned.
* **Shown to the agent** so it works from the real list instead of inventing a
  plausible-looking one, and can tell "this route is unknown" from "this route
  is known to 404".

Paths are normalised (`/api/project/270/surveys` -> `/api/project/:id/surveys`)
so one row covers every id.
"""

from __future__ import annotations

import json
import os
import re
import threading

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORE_DIR = os.path.join(_ROOT, "data", "api")

# Anything that looks like an id becomes ":id" — numeric, uuid, or a long slug.
_ID_PATTERNS = (
    (re.compile(r"/\d+(?=/|$)"), "/:id"),
    (re.compile(r"/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(?=/|$)", re.I), "/:id"),
    # Short hyphenated slugs like "e07-ba4" identify one survey; without this,
    # every survey the agent opened would add its own row and the list would grow
    # without ever getting more useful.
    #
    # The segment MUST contain a digit. A rule of "letters-and-a-hyphen" ate real
    # route names: /api/template-share/shared-with-me collapsed to
    # /api/:slug/shared-with-me, and /api/collaborator/all-projects became
    # /api/collaborator/:slug - destroying the very names the list exists to hold.
    (re.compile(r"/(?=[0-9a-z-]*\d)[0-9a-z]{1,10}-[0-9a-z]{1,10}(?=/|$)", re.I), "/:slug"),
)

_lock = threading.Lock()


def normalise(url_or_path: str) -> str:
    """Reduce a concrete URL to a comparable route."""
    path = url_or_path
    if "://" in path:
        path = "/" + path.split("://", 1)[1].split("/", 1)[-1] if "/" in path.split("://", 1)[1] else "/"
    path = path.split("?", 1)[0].split("#", 1)[0].rstrip("/") or "/"
    for pattern, repl in _ID_PATTERNS:
        path = pattern.sub(repl, path)
    return path


class ApiRegistry:
    """Known routes for one project, persisted between runs."""

    def __init__(self, project: str):
        self.project = project or "default"
        self.path = os.path.join(STORE_DIR, f"{self.project}.json")
        self.routes: dict[str, dict] = {}
        self._dirty = False
        self._load()

    # ── persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as fh:
                data = json.load(fh)
            self.routes = data.get("endpoints", {}) or {}
        except (OSError, json.JSONDecodeError):
            self.routes = {}

    def save(self) -> bool:
        """Persist if anything new was learned. Never raises."""
        if not self._dirty:
            return False
        try:
            os.makedirs(STORE_DIR, exist_ok=True)
            with open(self.path, "w", encoding="utf-8", newline="\n") as fh:
                json.dump({"project": self.project, "endpoints": self.routes},
                          fh, indent=2, ensure_ascii=False, sort_keys=True)
                fh.write("\n")
            self._dirty = False
            return True
        except OSError:
            return False

    # ── learning ─────────────────────────────────────────────────────────────

    def record(self, method: str, url: str, status: int) -> None:
        """Remember a route the application actually called. Never raises."""
        try:
            if "/api/" not in url:
                return
            route = normalise(url)
            entry = self.routes.setdefault(route, {"methods": {}, "seen": 0})
            entry["seen"] = entry.get("seen", 0) + 1
            methods = entry.setdefault("methods", {})
            key = (method or "GET").upper()
            # Keep the best status seen: a route that has ever answered 200 is
            # real, even if a later call 404s on a missing id.
            prior = methods.get(key)
            if prior is None or (status and status < prior):
                methods[key] = status
            self._dirty = True
        except Exception:
            pass

    # ── reading ──────────────────────────────────────────────────────────────

    def working(self) -> list[str]:
        """Routes that have answered successfully at least once."""
        out = []
        for route, entry in self.routes.items():
            for method, status in (entry.get("methods") or {}).items():
                if status and 200 <= status < 400:
                    out.append(f"{method} {route}")
        return sorted(set(out))

    def broken(self) -> list[str]:
        """Routes only ever seen failing — worth knowing, not worth retrying."""
        out = []
        for route, entry in self.routes.items():
            statuses = [s for s in (entry.get("methods") or {}).values() if s]
            if statuses and all(s >= 400 for s in statuses):
                out.append(route)
        return sorted(set(out))

    def prompt_block(self, limit: int = 40) -> str:
        """The list the agent sees. Empty when nothing is known yet."""
        working = self.working()
        if not working:
            return ""
        lines = [
            "KNOWN API ROUTES on this site (observed, not guessed). Use ONLY these "
            "if a test asks you to check an endpoint; do NOT invent a plausible "
            "path - a wrong one wastes the whole test:",
        ]
        lines += [f"  {route}" for route in working[:limit]]
        if len(working) > limit:
            lines.append(f"  ... and {len(working) - limit} more")
        broken = self.broken()
        if broken:
            lines.append("Known NOT to work: " + ", ".join(broken[:8]))
        return "\n".join(lines)

    def is_known(self, url: str) -> bool:
        return normalise(url) in self.routes


def for_project(project: str) -> ApiRegistry:
    with _lock:
        return ApiRegistry(project)
