"""Passive browser signals — the oracles the app gives us for free.

This is the one thing web testing has that the Android path does not. On a
device, "did it work?" is almost always the agent's own judgement of a
screenshot. In a browser the page reports its own failures: an uncaught
exception, a request that came back 5xx, an error logged to the console. None of
that depends on an LLM believing anything.

They are collected always and used two ways:

* Always folded into the notes, so a test that "passed" while the page threw is
  visible as such to a human and to the planner's next round.
* Optionally allowed to fail a test on their own (``WEB_FAIL_ON_PAGE_ERROR`` /
  ``WEB_FAIL_ON_HTTP_5XX``), which is OFF by default — a site with a noisy
  console would otherwise fail every test it has, and a run where everything
  fails carries no information.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Findings:
    """Everything the browser reported during one test case."""

    console_errors: list[str] = field(default_factory=list)
    page_errors: list[str] = field(default_factory=list)
    http_failures: list[str] = field(default_factory=list)   # 5xx — the app broke
    http_client_errors: list[str] = field(default_factory=list)  # 4xx — often expected

    def is_empty(self) -> bool:
        return not (self.console_errors or self.page_errors
                    or self.http_failures or self.http_client_errors)

    def counts(self) -> dict[str, int]:
        return {
            "console_errors": len(self.console_errors),
            "page_errors": len(self.page_errors),
            "http_5xx": len(self.http_failures),
            "http_4xx": len(self.http_client_errors),
        }

    def summary(self, limit: int = 3) -> str:
        """One line per signal class, for the verdict notes."""
        if self.is_empty():
            return "Browser reported no console errors, page exceptions or failed requests."
        parts = []
        for label, items in (
            ("page exception", self.page_errors),
            ("HTTP 5xx", self.http_failures),
            ("console error", self.console_errors),
            ("HTTP 4xx", self.http_client_errors),
        ):
            if items:
                shown = "; ".join(items[:limit])
                more = f" (+{len(items) - limit} more)" if len(items) > limit else ""
                parts.append(f"{len(items)} {label}(s): {shown}{more}")
        return " | ".join(parts)

    def verdict_override(self, cfg) -> tuple[str, str] | None:
        """A signal serious enough to fail a test on its own, or None.

        Returns ``(error_type, message)``. Only the two unambiguous classes are
        eligible: an uncaught page exception and a 5xx from the app's own
        backend. A console error is reported but never fails a test by itself —
        too many sites log errors during normal operation.
        """
        if cfg.WEB_FAIL_ON_PAGE_ERROR and self.page_errors:
            return "PAGE_ERROR", f"Uncaught page exception: {self.page_errors[0]}"
        if cfg.WEB_FAIL_ON_HTTP_5XX and self.http_failures:
            return "HTTP_ERROR", f"Server error during the test: {self.http_failures[0]}"
        return None


class Collector:
    """Attaches to a Playwright page and accumulates findings for one test case."""

    def __init__(self, page, cfg):
        self.page = page
        self.cfg = cfg
        self.findings = Findings()
        self._attached = False

    def attach(self) -> None:
        if self._attached:
            return
        if self.cfg.WEB_COLLECT_CONSOLE:
            self.page.on("console", self._on_console)
            self.page.on("pageerror", self._on_page_error)
        if self.cfg.WEB_COLLECT_NETWORK:
            self.page.on("response", self._on_response)
            self.page.on("requestfailed", self._on_request_failed)
        self._attached = True

    def reset(self) -> None:
        """Start a fresh set of findings — called between test cases."""
        self.findings = Findings()

    # ── handlers (never raise: a listener that throws kills the page) ─────────

    def _on_console(self, msg) -> None:
        try:
            if msg.type != "error":
                return
            text = (msg.text or "").strip()
            if self._ignored(text):
                return
            self._add(self.findings.console_errors, text)
        except Exception:
            pass

    def _on_page_error(self, err) -> None:
        try:
            text = str(err).strip().splitlines()[0] if str(err).strip() else "unknown error"
            if self._ignored(text):
                return
            self._add(self.findings.page_errors, text)
        except Exception:
            pass

    def _on_response(self, response) -> None:
        try:
            status = response.status
            if status < 400:
                return
            entry = f"{status} {response.request.method} {_short_url(response.url)}"
            bucket = (self.findings.http_failures if status >= 500
                      else self.findings.http_client_errors)
            self._add(bucket, entry)
        except Exception:
            pass

    def _on_request_failed(self, request) -> None:
        try:
            # Blocked/aborted requests are mostly ad-blockers and analytics; only
            # a same-origin failure says anything about the app under test.
            if not request.url.lower().startswith(self.cfg.WEB_BASE_URL.lower()):
                return
            failure = getattr(request, "failure", None) or "request failed"
            self._add(self.findings.http_failures,
                      f"FAILED {request.method} {_short_url(request.url)} ({failure})")
        except Exception:
            pass

    # ── helpers ──────────────────────────────────────────────────────────────

    def _ignored(self, text: str) -> bool:
        low = text.lower()
        return any(pat in low for pat in self.cfg.WEB_CONSOLE_IGNORE)

    @staticmethod
    def _add(bucket: list[str], entry: str, cap: int = 25) -> None:
        """De-duplicate and cap: one broken poll can emit the same error 400 times."""
        entry = entry[:300]
        if entry not in bucket and len(bucket) < cap:
            bucket.append(entry)


def _short_url(url: str, limit: int = 120) -> str:
    return url if len(url) <= limit else url[:limit] + "…"
