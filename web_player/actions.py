"""The action vocabulary, and how each action is performed against a page.

Kept deliberately small. Every action the agent can take is listed in
``ACTION_SPEC`` (which *is* the prompt text — one definition, so the prompt can
never describe an action the dispatcher does not implement), and each one either
succeeds with an observation-worthy note or returns a failure the agent can read
and react to.

Guardrails live here rather than in the prompt, because a prompt is a request and
a dispatcher is a rule: an agent that decides to click "Delete account" anyway
must still be stopped.
"""

from __future__ import annotations

from urllib.parse import urlparse

from . import snapshot

# The action reference injected into the system prompt verbatim.
ACTION_SPEC = """\
click        {"action":"click","ref":"e12"}                      activate a control
fill         {"action":"fill","ref":"e3","text":"a@b.com"}       replace a field's value
press        {"action":"press","key":"Enter","ref":"e3"}         send a key ("ref" optional)
select       {"action":"select","ref":"e7","value":"Large"}      choose a dropdown option
goto         {"action":"goto","url":"/settings"}                 navigate (same origin only)
back         {"action":"back"}                                   browser back
scroll       {"action":"scroll","direction":"down"}              down | up | top | bottom
wait         {"action":"wait","seconds":2}                       let the page settle (max 5)
finish       {"action":"finish","success":true,"reason":"..."}   end the test with a verdict\
"""

_SCROLL_JS = {
    "down": "window.scrollBy(0, window.innerHeight * 0.8)",
    "up": "window.scrollBy(0, -window.innerHeight * 0.8)",
    "top": "window.scrollTo(0, 0)",
    "bottom": "window.scrollTo(0, document.body.scrollHeight)",
}


class ActionError(Exception):
    """An action could not be performed. ``category`` feeds failure attribution."""

    def __init__(self, message: str, category: str = "ASSERTION_FAILURE"):
        super().__init__(message)
        self.category = category


class Dispatcher:
    """Performs actions against one page, under one set of guardrails."""

    def __init__(self, page, cfg):
        self.page = page
        self.cfg = cfg
        self._origin = _origin_of(cfg.WEB_BASE_URL)

    # ── guardrails ───────────────────────────────────────────────────────────

    def _check_label(self, element: dict | None) -> None:
        """Refuse to activate a control whose label is on the blocked list."""
        name = (element or {}).get("name", "").lower()
        if not name:
            return
        for blocked in self.cfg.WEB_BLOCKED_TEXTS:
            if blocked and blocked in name:
                raise ActionError(
                    f"Refused to activate '{element.get('name')}' — it matches the "
                    f"blocked control '{blocked}'. This would end the session or "
                    f"destroy the account.",
                    category="BLOCKED_BY_GUARDRAIL",
                )

    def _check_url(self, url: str) -> str:
        """Resolve a possibly-relative URL and enforce the origin/pattern rules."""
        absolute = url if urlparse(url).scheme else _join(self.cfg.WEB_BASE_URL, url)
        low = absolute.lower()
        for pattern in self.cfg.WEB_BLOCKED_URL_PATTERNS:
            if pattern and pattern in low:
                raise ActionError(
                    f"Refused to navigate to {absolute} — it matches the blocked "
                    f"URL pattern '{pattern}'.",
                    category="BLOCKED_BY_GUARDRAIL",
                )
        if self.cfg.WEB_SAME_ORIGIN_ONLY and _origin_of(absolute) != self._origin:
            raise ActionError(
                f"Refused to navigate off-origin to {absolute} "
                f"(the site under test is {self._origin}).",
                category="BLOCKED_BY_GUARDRAIL",
            )
        return absolute

    # ── dispatch ─────────────────────────────────────────────────────────────

    async def perform(self, action: dict, snap: dict) -> str:
        """Run one action. Returns a short note; raises ActionError on refusal."""
        kind = (action.get("action") or "").lower()
        handler = getattr(self, f"_do_{kind}", None)
        if handler is None:
            raise ActionError(f"Unknown action '{kind}'. Use one of the listed actions.")
        return await handler(action, snap)

    async def _do_click(self, action: dict, snap: dict) -> str:
        el = self._require(action, snap)
        self._check_label(el)
        await self._locator(el["ref"]).click(timeout=self.cfg.WEB_ACTION_TIMEOUT_MS)
        return f"clicked [{el['ref']}] {el.get('role')} \"{el.get('name')}\""

    async def _do_fill(self, action: dict, snap: dict) -> str:
        el = self._require(action, snap)
        text = str(action.get("text", ""))
        locator = self._locator(el["ref"])
        # fill() clears first, which is what we want; a controlled input that
        # rejects the value is caught by the agent reading the field back.
        await locator.fill(text, timeout=self.cfg.WEB_ACTION_TIMEOUT_MS)
        shown = "*" * len(text) if el.get("role") == "password" else text
        return f"filled [{el['ref']}] \"{el.get('name')}\" with '{shown}'"

    async def _do_press(self, action: dict, snap: dict) -> str:
        key = str(action.get("key") or "Enter")
        ref = action.get("ref")
        if ref:
            el = self._require(action, snap)
            await self._locator(el["ref"]).press(key, timeout=self.cfg.WEB_ACTION_TIMEOUT_MS)
            return f"pressed {key} on [{el['ref']}]"
        await self.page.keyboard.press(key)
        return f"pressed {key}"

    async def _do_select(self, action: dict, snap: dict) -> str:
        el = self._require(action, snap)
        value = str(action.get("value", ""))
        locator = self._locator(el["ref"])
        try:
            await locator.select_option(label=value, timeout=self.cfg.WEB_ACTION_TIMEOUT_MS)
        except Exception:
            # Fall back to matching by value when the visible label is not it.
            await locator.select_option(value=value, timeout=self.cfg.WEB_ACTION_TIMEOUT_MS)
        return f"selected '{value}' in [{el['ref']}] \"{el.get('name')}\""

    async def _do_goto(self, action: dict, _snap: dict) -> str:
        url = self._check_url(str(action.get("url") or ""))
        await self.page.goto(url, timeout=self.cfg.WEB_NAV_TIMEOUT_MS, wait_until="domcontentloaded")
        return f"navigated to {url}"

    async def _do_back(self, _action: dict, _snap: dict) -> str:
        await self.page.go_back(timeout=self.cfg.WEB_NAV_TIMEOUT_MS)
        return "went back"

    async def _do_scroll(self, action: dict, _snap: dict) -> str:
        direction = str(action.get("direction") or "down").lower()
        js = _SCROLL_JS.get(direction)
        if js is None:
            raise ActionError(f"Unknown scroll direction '{direction}'. Use down/up/top/bottom.")
        await self.page.evaluate(js)
        return f"scrolled {direction}"

    async def _do_wait(self, action: dict, _snap: dict) -> str:
        seconds = min(max(float(action.get("seconds") or 1), 0.1), 5.0)
        await self.page.wait_for_timeout(seconds * 1000)
        return f"waited {seconds:g}s"

    # ── helpers ──────────────────────────────────────────────────────────────

    def _require(self, action: dict, snap: dict) -> dict:
        ref = action.get("ref")
        if not ref:
            raise ActionError(f"Action '{action.get('action')}' needs a 'ref'.")
        el = snapshot.find(snap, ref)
        if el is None:
            raise ActionError(
                f"No element '{ref}' in the current observation. Pick a ref that is "
                f"actually listed.",
                category="ELEMENT_NOT_FOUND",
            )
        if el.get("disabled"):
            raise ActionError(
                f"[{ref}] \"{el.get('name')}\" is disabled — it cannot be used yet.",
                category="ASSERTION_FAILURE",
            )
        return el

    def _locator(self, ref: str):
        return self.page.locator(f'[data-etp-ref="{ref}"]')


def _origin_of(url: str) -> str:
    parsed = urlparse(url or "")
    return f"{parsed.scheme}://{parsed.netloc}".lower()


def _join(base: str, path: str) -> str:
    return base.rstrip("/") + "/" + path.lstrip("/")
