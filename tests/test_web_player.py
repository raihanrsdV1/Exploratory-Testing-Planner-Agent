#!/usr/bin/env python3
"""Web player: the pure logic, without a browser.

Everything here is a decision the player makes before or after Playwright is
involved — failure attribution, guardrail enforcement, observation rendering,
action parsing. Each check corresponds to a way the player could quietly lie
about a run: counting its own budget exhaustion as a discovered defect, clicking
a control that ends the session, or accepting a model reply it did not parse.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import settings as st  # noqa: E402
from web_player import failures, llm, snapshot  # noqa: E402
from web_player.actions import ActionError, Dispatcher  # noqa: E402
from web_player.oracles import Findings  # noqa: E402

_passed = _failed = 0


def check(label, got, want):
    global _passed, _failed
    ok = got == want
    _passed, _failed = _passed + ok, _failed + (not ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + ("" if ok else f"  (got {got!r}, want {want!r})"))


class _FakeCfg:
    """Only the fields the dispatcher's guardrails read."""
    WEB_BASE_URL = "https://shop.example.com/app"
    WEB_BLOCKED_TEXTS = ("delete account", "log out")
    WEB_BLOCKED_URL_PATTERNS = ("/logout",)
    WEB_SAME_ORIGIN_ONLY = True
    WEB_ACTION_TIMEOUT_MS = 5000
    WEB_NAV_TIMEOUT_MS = 5000


def main():
    print("the agent's own failure strings never count as app defects")
    # Every string below is emitted verbatim by WebAgent.run or Dispatcher. If any
    # falls through to ASSERTION_FAILURE it is counted as a discovered bug, which
    # is the single most damaging misclassification in the system.
    emitted = {
        "Step limit reached: used all 30 steps without reaching a verdict.": "STEP_LIMIT_EXCEEDED",
        "Livelock: repeated the same action against an unchanged page 5 times. "
        "The page is not responding to it.": "NAVIGATION_LIVELOCK",
        "Timed out after 420s at step 12/30.": "TIMEOUT",
        "Refused to activate 'Log out' — it matches the blocked control 'log out'.": "BLOCKED_BY_GUARDRAIL",
        "Refused to navigate off-origin to https://accounts.google.com": "BLOCKED_BY_GUARDRAIL",
        "No element 'e9' in the current observation.": "ELEMENT_NOT_FOUND",
    }
    for reason, want in emitted.items():
        check(f"{want:22} <- {reason[:46]}", failures.classify(reason), want)

    print("app misbehaviour is still attributed to the app")
    check("uncaught exception -> PAGE_ERROR",
          failures.classify("Uncaught TypeError: cart.total is not a function"), "PAGE_ERROR")
    check("server error -> HTTP_ERROR",
          failures.classify("Server error during the test: 500 POST /api/orders"), "HTTP_ERROR")
    check("unmet expectation -> ASSERTION_FAILURE",
          failures.classify("The confirmation banner never appeared"), "ASSERTION_FAILURE")
    check("a passing run has no category", failures.classify("all good", success=True), "")

    print("every category the player emits has a recovery strategy")
    emitted_categories = set(emitted.values()) | {
        "PAGE_ERROR", "HTTP_ERROR", "ASSERTION_FAILURE", "STALE_ELEMENT",
        "NAVIGATION_FAILURE", "PRECONDITION_NOT_MET",
    }
    for cat in sorted(emitted_categories):
        strat = failures.recovery_strategy(cat)
        check(f"{cat} has an action", bool(strat.get("action")), True)
    check("no unrecoverable category asks for a retry",
          [c for c in ("STEP_LIMIT_EXCEEDED", "NAVIGATION_LIVELOCK", "BLOCKED_BY_GUARDRAIL",
                       "PAGE_ERROR", "HTTP_ERROR")
           if failures.recovery_strategy(c)["retry"]], [])

    print("every category is attributable — none is silently unclassified")
    known = st.APP_FAULT | st.AGENT_FAULT | st.ENV_FAULT
    for cat in sorted(emitted_categories):
        check(f"{cat} is in the shared taxonomy", cat in known, True)

    print("guardrails refuse destructive controls and foreign origins")
    d = Dispatcher(page=None, cfg=_FakeCfg)
    snap = {"elements": [
        {"ref": "e1", "role": "button", "name": "Log out"},
        {"ref": "e2", "role": "button", "name": "Save changes"},
        {"ref": "e3", "role": "button", "name": "Delete Account", "disabled": True},
    ]}

    def refuses(action):
        try:
            asyncio.run(d.perform(action, snap))
            return ""
        except ActionError as exc:
            return exc.category
        except Exception:
            # A non-guardrail failure means it got past the check to a None page.
            return "PASSED_GUARD"

    check("clicking 'Log out' is refused",
          refuses({"action": "click", "ref": "e1"}), "BLOCKED_BY_GUARDRAIL")
    check("case does not matter ('Delete Account')",
          refuses({"action": "click", "ref": "e3"}) in ("BLOCKED_BY_GUARDRAIL", "ASSERTION_FAILURE"), True)
    check("an ordinary control is not refused",
          refuses({"action": "click", "ref": "e2"}), "PASSED_GUARD")
    check("off-origin navigation is refused",
          refuses({"action": "goto", "url": "https://accounts.google.com/signin"}),
          "BLOCKED_BY_GUARDRAIL")
    check("a blocked URL pattern is refused",
          refuses({"action": "goto", "url": "/logout"}), "BLOCKED_BY_GUARDRAIL")
    check("a same-origin relative path is allowed",
          refuses({"action": "goto", "url": "/settings"}), "PASSED_GUARD")
    check("a ref not in the observation is ELEMENT_NOT_FOUND",
          refuses({"action": "click", "ref": "e99"}), "ELEMENT_NOT_FOUND")
    check("an unknown action is rejected",
          refuses({"action": "teleport"}), "ASSERTION_FAILURE")

    print("guardrails match whole words, so they do not over-block")
    # A bare substring test would refuse "Credits" for containing "edit" and
    # "Remove" for containing "move" — spurious BLOCKED_BY_GUARDRAIL against
    # controls that were never dangerous.
    from web_player.actions import _matches_word  # noqa: E402
    for needle, name, want in (
        ("edit", "edit", True),
        ("edit", "edit source", True),
        ("edit", "credits", False),
        ("edit", "edition", False),
        ("move", "move page", True),
        ("move", "remove", False),
        ("talk", "talk", True),
        ("talk", "talking", False),
        ("delete account", "delete account", True),
        ("log out", "log out of wikipedia", True),
    ):
        check(f"{needle!r} blocks {name!r}: {want}", _matches_word(needle, name), want)

    print("the observation renders every fact the agent needs to act")
    snap2 = {
        "url": "https://shop.example.com/checkout", "title": "Checkout",
        "headings": ["Payment"], "messages": ["Card number is invalid"], "dialog_open": True,
        "texts": ["Items: 0", "Total: $42.00"],
        "elements": [
            {"ref": "e1", "role": "textbox", "name": "Card number", "value": "4111", "required": True},
            {"ref": "e2", "role": "password", "name": "CVC", "value": "***"},
            {"ref": "e3", "role": "button", "name": "Pay", "disabled": True},
            {"ref": "e4", "role": "checkbox", "name": "Save card", "checked": False},
        ],
    }
    text = snapshot.render(snap2)
    # "Items: 0" matters as much as any control: an assertion about a value on
    # the page is unobservable without it, and the agent thrashes looking for it.
    for needle in ("[e1]", "Card number", "required", "DISABLED", "unchecked",
                   "Card number is invalid", "MODAL DIALOG", "Items: 0", "Total: $42.00"):
        check(f"rendered observation mentions {needle!r}", needle in text, True)
    check("a password value is never echoed in clear", "***" in text, True)
    check("find() resolves a ref", (snapshot.find(snap2, "e3") or {}).get("name"), "Pay")
    check("find() returns None for an unknown ref", snapshot.find(snap2, "e77"), None)
    check("an empty page still renders", "none found" in snapshot.render(
        {"url": "u", "elements": []}), True)

    print("our own display limits are never mistaken for the app's behaviour")
    # A real run typed 300 characters into Wikipedia's search box, read back the
    # 100-character rendering, and concluded the field had truncated its input.
    # The truncation was ours. The observation must say so.
    long_snap = {"url": "u", "elements": [
        {"ref": "e1", "role": "textbox", "name": "Search", "value": "a" * 100,
         "value_length": 300},
        {"ref": "e2", "role": "textbox", "name": "Short", "value": "abc"},
    ]}
    long_text = snapshot.render(long_snap)
    check("a shortened value declares its true length", "300 chars total" in long_text, True)
    check("and says the shortening was the observer's",
          "NOT truncated" in long_text, True)
    check("an untruncated value claims no length", "chars total" in long_text.split("Short")[1], False)

    print("an executor-model outage is never a defect in the site")
    # OpenRouter returned a Cloudflare challenge mid-run. It was recorded as
    # CRASH — an APP fault — so an outage on our side counted as a bug found in
    # the site under test.
    for reason in ("OpenRouter 403: <!DOCTYPE html> Just a moment...",
                   "Gemini 400: API key not valid. Please pass a valid API key.",
                   "Model backend (OpenRouter) unavailable: 403 Client Error"):
        check(f"LLM_UNAVAILABLE <- {reason[:34]}", failures.classify(reason), "LLM_UNAVAILABLE")
    check("LLM_UNAVAILABLE is an environment fault, not an app fault",
          ("LLM_UNAVAILABLE" in st.ENV_FAULT, "LLM_UNAVAILABLE" in st.APP_FAULT), (True, False))
    check("an outage is never retried", failures.recovery_strategy("LLM_UNAVAILABLE")["retry"], False)
    # The status codes inside a provider error must not read as the site's own.
    check("a provider 403 does not become an HTTP_ERROR about the app",
          failures.classify("OpenRouter 503: upstream overloaded"), "LLM_UNAVAILABLE")

    print("model replies are parsed out of whatever wrapping they arrive in")
    check("bare JSON", llm.parse_action('{"action":"click","ref":"e1"}')["ref"], "e1")
    check("fenced JSON", llm.parse_action('```json\n{"action":"click","ref":"e2"}\n```')["ref"], "e2")
    check("JSON buried in prose",
          llm.parse_action('Sure! I will click it.\n{"action":"click","ref":"e3"}\nDone.')["ref"], "e3")
    check("finish action survives parsing",
          llm.parse_action('{"action":"finish","success":false,"reason":"no banner"}')["success"], False)
    check("unparseable reply becomes an _error action, not an exception",
          llm.parse_action("I cannot do that.")["action"], "_error")
    check("empty reply becomes an _error action", llm.parse_action("")["action"], "_error")

    print("browser findings summarise honestly")
    empty = Findings()
    check("a clean run says so", "no console errors" in empty.summary(), True)
    check("a clean run overrides nothing", empty.verdict_override(st), None)
    noisy = Findings(page_errors=["TypeError: x"], http_failures=["500 POST /api/o"])
    check("findings are counted", noisy.counts()["page_errors"], 1)
    check("findings appear in the summary", "TypeError: x" in noisy.summary(), True)
    check("oracles do not fail a test while the flags are off",
          noisy.verdict_override(st), None)

    class _FailOn:
        WEB_FAIL_ON_PAGE_ERROR = True
        WEB_FAIL_ON_HTTP_5XX = True
    check("a page exception can fail a test when enabled",
          (noisy.verdict_override(_FailOn) or ("", ""))[0], "PAGE_ERROR")
    check("console noise alone never fails a test",
          Findings(console_errors=["boom"]).verdict_override(_FailOn), None)

    print("web config guards")
    check("WEB_BASE_URL has no baked-in default", st.WEB_BASE_URL, os.environ.get("WEB_BASE_URL", ""))
    check("a malformed viewport falls back instead of crashing",
          isinstance(st.web_viewport().get("width"), int), True)
    for name in ("site_identity_block", "web_input_block", "web_safety_block"):
        check(f"{name} produces guidance", len(getattr(st, name)()) > 40, True)
    check("web_login_block hides the password when no user is configured",
          st.web_login_block() if not st.WEB_LOGIN_USER else "", "")

    print(f"\n{_passed}/{_passed + _failed} checks passed")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
