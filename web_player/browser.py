"""Playwright lifecycle: one browser, one context, one page, for a whole batch.

The context is created once and reused across test cases. That mirrors the
Android default (``DEVICE_RESET_SCOPE=suite``): state accumulates between tests
within a run, which is what makes later tests able to find data earlier ones
created, while every *batch* still starts from an identical place.

``storage_state`` is the web equivalent of "the device is already signed in".
Producing it once by hand and pointing ``WEB_STORAGE_STATE`` at it removes the
single largest waste of agent steps — re-logging-in before every test — and is
also the only reliable way past a login the agent cannot complete on its own.
"""

from __future__ import annotations

import os
import time

from .actions import _goto_settled

class BrowserSession:
    """Async context manager owning the Playwright objects."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._pw = None
        self.headless = bool(getattr(cfg, "WEB_HEADLESS", True))
        self.browser = None
        self.context = None
        self.page = None
        self.shot_dir = getattr(cfg, "WEB_SCREENSHOT_DIR", "")

    async def __aenter__(self) -> "BrowserSession":
        from playwright.async_api import async_playwright

        self._pw = await async_playwright().start()
        launcher = getattr(self._pw, self.cfg.WEB_BROWSER, None)
        if launcher is None:
            raise RuntimeError(
                f"WEB_BROWSER='{self.cfg.WEB_BROWSER}' is not a Playwright browser "
                f"(use chromium, firefox or webkit)."
            )
        # slow_mo makes a headed run followable; it costs nothing headless.
        # Remember what we really launched. The CAPTCHA hand-off depends on whether
        # a window exists, and deciding that from config at the moment of use left
        # one test reporting "the run is headless" while the next one correctly
        # asked for help - in the same process, with the same settings.
        self.headless = bool(self.cfg.WEB_HEADLESS)
        self.browser = await launcher.launch(
            headless=self.cfg.WEB_HEADLESS,
            slow_mo=max(0, int(getattr(self.cfg, "WEB_SLOW_MO_MS", 0))),
        )

        context_args = {"viewport": self.cfg.web_viewport()}
        state_path = self.cfg.WEB_STORAGE_STATE
        if state_path:
            if not os.path.exists(state_path):
                # Loud, not silent: a run that quietly starts logged-out produces a
                # page of sign-in tests and no evidence about the app.
                raise FileNotFoundError(
                    f"WEB_STORAGE_STATE points at '{state_path}', which does not exist. "
                    f"Create it once with: playwright codegen --save-storage={state_path} <url>"
                )
            context_args["storage_state"] = state_path

        self.context = await self.browser.new_context(**context_args)
        self.context.set_default_timeout(self.cfg.WEB_ACTION_TIMEOUT_MS)
        self.context.set_default_navigation_timeout(self.cfg.WEB_NAV_TIMEOUT_MS)
        self.page = await self.context.new_page()
        # One folder per batch, because test ids restart at TC-001 every campaign.
        self.shot_dir = os.path.join(self.cfg.WEB_SCREENSHOT_DIR, time.strftime("%Y%m%d-%H%M%S"))
        return self

    async def __aexit__(self, *_exc) -> None:
        for closer in (self.context, self.browser):
            try:
                if closer:
                    await closer.close()
            except Exception:
                pass
        try:
            if self._pw:
                await self._pw.stop()
        except Exception:
            pass

    async def reset_to_base(self) -> None:
        """Return to the site's entry point before a test case.

        Waits for the network to settle so the first observation is not of an
        unrendered page - but falls back, because an app that polls (or embeds a
        CAPTCHA widget) may never reach idle at all. Without the fallback this
        threw at the navigation timeout and cost a whole test case before it had
        taken a single step.
        """
        await _goto_settled(self.page, self.cfg.WEB_BASE_URL, self.cfg)

    async def screenshot(self, name: str) -> str:
        """Best-effort screenshot; returns the path written, or '' on failure."""
        try:
            folder = self.shot_dir or self.cfg.WEB_SCREENSHOT_DIR
            os.makedirs(folder, exist_ok=True)
            safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
            path = os.path.join(folder, f"{safe}.png")
            await self.page.screenshot(path=path, full_page=False)
            return path
        except Exception:
            return ""
