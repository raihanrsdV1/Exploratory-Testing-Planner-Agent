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

import asyncio
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
from dataclasses import dataclass, field


@dataclass
class Findings:
    """Everything the browser reported during one test case."""

    console_errors: list[str] = field(default_factory=list)
    page_errors: list[str] = field(default_factory=list)
    http_failures: list[str] = field(default_factory=list)   # 5xx — the app broke
    http_client_errors: list[str] = field(default_factory=list)  # 4xx — often expected
    request_failures: list[str] = field(default_factory=list)  # no response at all — never a 5xx
    dialogs: list[str] = field(default_factory=list)  # native alert/confirm/prompt — the site's own messages

    def is_empty(self) -> bool:
        return not (self.console_errors or self.page_errors
                    or self.http_failures or self.http_client_errors or self.request_failures
                    or self.dialogs)

    def counts(self) -> dict[str, int]:
        return {
            "console_errors": len(self.console_errors),
            "page_errors": len(self.page_errors),
            "http_5xx": len(self.http_failures),
            "http_4xx": len(self.http_client_errors),
            "request_failures": len(self.request_failures),
            "dialogs": len(self.dialogs),
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
            ("failed request", self.request_failures),
            ("browser dialog", self.dialogs),
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

    def __init__(self, page, cfg, registry=None):
        self.page = page
        self.cfg = cfg
        self.findings = Findings()
        self._attached = False
        # Every API call the app makes passes through here already. Feeding it to
        # the registry is how the known-route list grows without a separate crawl.
        self.registry = registry
        # Native dialogs. Playwright DISMISSES every dialog when no handler is
        # registered, so alert() text never reached the agent and confirm() was
        # silently answered "Cancel" - the action simply did not happen, and the
        # agent read that as a control that does nothing. This application calls
        # alert() 309 times and confirm() 16 times, so that was not a rare path.
        self.dialogs: list[str] = []

    def attach(self) -> None:
        if self._attached:
            return
        if self.cfg.WEB_COLLECT_CONSOLE:
            self.page.on("console", self._on_console)
            self.page.on("pageerror", self._on_page_error)
        if self.cfg.WEB_COLLECT_NETWORK:
            self.page.on("response", self._on_response)
            self.page.on("requestfailed", self._on_request_failed)
        self.page.on("dialog", self._on_dialog)
        self._attached = True

    def reset(self) -> None:
        """Start a fresh set of findings — called between test cases."""
        self.findings = Findings()
        self.dialogs = []

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
            if self.registry is not None:
                self.registry.record(response.request.method, response.url, status)
            if status < 400:
                return
            if urlsplit(response.url).netloc != urlsplit(self.cfg.WEB_BASE_URL).netloc:
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
            if urlsplit(request.url).netloc != urlsplit(self.cfg.WEB_BASE_URL).netloc:
                return
            failure = getattr(request, "failure", None) or "request failed"
            # Navigating away mid-stream cancels requests routinely: that is noise,
            # not evidence. What is left got no response, so it is still never a 5xx.
            if "ERR_ABORTED" in failure or "NS_BINDING_ABORTED" in failure:
                return
            self._add(self.findings.request_failures,
                      f"FAILED {request.method} {_short_url(request.url)} ({failure})")
        except Exception:
            pass

    def _on_dialog(self, dialog) -> None:
        """Answer a native dialog, and remember what it said.

        Policy, and why:
          * alert   - accept. There is no other option, and the message is often
                      the only confirmation the application gives.
          * confirm - accept by default. Dismissing was the old behaviour and it
                      made confirm-gated actions silently not happen. The label
                      guardrails already refuse the destructive controls BEFORE
                      the click, so they are the safety layer, not this.
          * prompt  - dismiss. We cannot know what to type.
          * beforeunload - accept, so navigation is not blocked.
        """
        try:
            kind = dialog.type
            message = " ".join((dialog.message or "").split())[:300]
            accept = kind != "prompt"
            if kind == "confirm":
                accept = bool(getattr(self.cfg, "WEB_ACCEPT_CONFIRM", True))
            self._add(self.dialogs, f"{kind}: {message} [{'accepted' if accept else 'dismissed'}]")
            if accept:
                asyncio.ensure_future(dialog.accept())
            else:
                asyncio.ensure_future(dialog.dismiss())
        except Exception:
            try:
                asyncio.ensure_future(dialog.dismiss())
            except Exception:
                pass

    def take_dialogs(self) -> list[str]:
        """Hand over the dialogs seen since the last call, and forget them."""
        seen, self.dialogs = list(self.dialogs), []
        return seen

    # ── helpers ──────────────────────────────────────────────────────────────

    def _ignored(self, text: str) -> bool:
        low = text.lower()
        return any(pat in low for pat in self.cfg.WEB_CONSOLE_IGNORE)

    @staticmethod
    def _add(bucket: list[str], entry: str, cap: int = 25) -> None:
        """De-duplicate and cap: one broken poll can emit the same error 400 times."""
        # Collapse whitespace FIRST. A JS error arrives with its whole stack
        # attached, and storing the newlines spilled a column of "at Ike (...)"
        # frames and blank lines straight into the verdict notes.
        entry = " ".join(entry.split())[:300]
        if entry not in bucket and len(bucket) < cap:
            bucket.append(entry)


def _short_url(url: str, limit: int = 120) -> str:
    parsed = urlsplit(url)
    query = [(k, "[REDACTED]" if any(s in k.lower() for s in ("token", "key", "password", "secret", "auth")) else v)
             for k, v in parse_qsl(parsed.query, keep_blank_values=True)]
    url = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(query), ""))
    return url if len(url) <= limit else url[:limit] + "…"
