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

# Stub agents in these tests emit real trace lines. Send them to a temp file
# so fixture runs never land in logs/web_player.log, which is the operator
# transcript and the dashboard's live feed.
import tempfile  # noqa: E402
os.environ.setdefault("WEB_TRACE_FILE",
                      os.path.join(tempfile.gettempdir(), "web_player_tests.log"))
import settings as st  # noqa: E402
from web_player import agent as agent_mod  # noqa: E402
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


class _FakeChatClient:
    """Replies with a DIFFERENT action each turn (varying ``text``, which the
    exact-repeat signature keys on), so that guard never fires — only the
    content-only wandering guard should catch this."""

    def __init__(self):
        self.turn = 0

    def chat(self, messages):
        self.turn += 1
        direction = ("down", "up", "top", "bottom")[(self.turn - 1) % 4]
        return f'{{"thought":"t","action":"scroll","direction":"{direction}"}}'


async def _check_wandering_guard():
    """Different actions every turn, but the page content never changes: the
    stall (content-only) guard must end the run well before WEB_MAX_STEPS,
    which the exact-repeat guard (keyed on page+action together) would miss."""
    fixed_snap = {"url": "https://shop.example.com/catalog", "elements": [], "messages": [], "texts": []}

    async def fake_observe(page, max_elements):
        return dict(fixed_snap)

    async def fake_perform(action, snap):
        return "scrolled"

    orig_observe = snapshot.observe
    agent_mod.snapshot.observe = fake_observe
    try:
        a = agent_mod.WebAgent(page=None, cfg=st, client=_FakeChatClient())
        a.dispatcher.perform = fake_perform
        result = await a.run("goal: reach the bottom of an infinite page", max_steps=30, timeout_s=999)
    finally:
        agent_mod.snapshot.observe = orig_observe

    check("stops before the step budget is exhausted", result.steps < 30, True)
    check("stops at the configured stall limit", result.steps, st.WEB_STALL_STEPS + 1)
    check("ends unsuccessfully", result.success, False)
    check("reason classifies as NAVIGATION_LIVELOCK", failures.classify(result.reason), "NAVIGATION_LIVELOCK")


def main():
    print("the agent's own failure strings never count as app defects")
    # Every string below is emitted verbatim by WebAgent.run or Dispatcher. If any
    # falls through to ASSERTION_FAILURE it is counted as a discovered bug, which
    # is the single most damaging misclassification in the system.
    emitted = {
        "Step limit reached: used all 30 steps without reaching a verdict.": "STEP_LIMIT_EXCEEDED",
        "Livelock: repeated the same action against an unchanged page 5 times. "
        "The page is not responding to it.": "NAVIGATION_LIVELOCK",
        "Livelock: tried 6 different actions in a row and the page never changed. "
        "The page is not responding to this line of exploration.": "NAVIGATION_LIVELOCK",
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

    print("an oscillating agent is stopped, not just a repeating one")
    # A real run cycled "Edit Survey" -> "Cancel" thirteen times and burned its
    # whole 30-step budget: the old guard only looked at the PREVIOUS turn, so an
    # A-B-A-B loop was invisible to it.
    import asyncio as _asyncio

    from web_player import agent as agent_mod

    class _StubClient:
        """Alternates between two actions forever, exactly like the real stall."""
        def __init__(self):
            self.calls = 0

        def chat(self, _messages):
            self.calls += 1
            return ('{"action":"click","ref":"e1"}' if self.calls % 2
                    else '{"action":"click","ref":"e2"}')

    pages = [
        {"url": "u", "elements": [{"ref": "e1", "role": "button", "name": "Edit"}],
         "messages": [], "texts": []},
        {"url": "u", "elements": [{"ref": "e2", "role": "button", "name": "Cancel"}],
         "messages": [], "texts": []},
    ]
    state = {"i": 0}

    async def _fake_observe(_page, _max):
        snap = pages[state["i"] % 2]
        state["i"] += 1
        return snap

    class _FakeDispatcher:
        async def perform(self, action, _snap):
            return f"clicked {action.get('ref')}"

    real_observe = agent_mod.snapshot.observe
    agent_mod.snapshot.observe = _fake_observe
    try:
        a = agent_mod.WebAgent(page=None, cfg=st, client=_StubClient())
        a.dispatcher = _FakeDispatcher()
        res = _asyncio.run(a.run("goal", max_steps=30, timeout_s=30))
    finally:
        agent_mod.snapshot.observe = real_observe

    check("an A-B-A-B oscillation is caught", "livelock" in res.reason.lower(), True)
    check("and is stopped well before the step budget", res.steps < 12, True)
    check("it is classified as an agent fault, not a defect",
          failures.classify(res.reason), "NAVIGATION_LIVELOCK")
    check("NAVIGATION_LIVELOCK is an agent fault", "NAVIGATION_LIVELOCK" in st.AGENT_FAULT, True)

    print("an oracle entry never spills a stack trace into the notes")
    from web_player.oracles import Collector as _Coll
    bucket = []
    nl = chr(10)
    noisy = ("Translation error: AxiosError 403" + nl +
             "    at Ike (x.js:62)" + nl + nl + "    at Ike (x.js:62)")
    _Coll._add(bucket, noisy)
    check("newlines are collapsed", nl in bucket[0], False)
    check("the message survives", bucket[0].startswith("Translation error"), True)

    print("a provider/transport failure is ours, never the site's")
    # A ConnectionResetError to openrouter.ai was recorded as CRASH — an APP
    # fault — filing a false defect against the site under test.
    import requests as _rq
    from web_player.llm import LLMError, _as_llm_error, _is_transient
    dropped = _rq.exceptions.ConnectionError(
        ("Connection aborted.", ConnectionResetError(10054, "forcibly closed", None, 10054, None)))
    check("a dropped connection is retried", _is_transient(dropped), True)
    check("a read timeout is retried", _is_transient(_rq.exceptions.Timeout("read timed out")), True)
    check("a bad API key is NOT retried", _is_transient(LLMError("OpenRouter 401: invalid api key")), False)
    check("transport failures normalise to LLMError",
          isinstance(_as_llm_error(dropped), LLMError), True)
    check("an LLMError passes through unchanged",
          _as_llm_error(LLMError("boom")).args[0], "boom")
    check("LLM_UNAVAILABLE is an environment fault, never an app fault",
          ("LLM_UNAVAILABLE" in st.ENV_FAULT, "LLM_UNAVAILABLE" in st.APP_FAULT), (True, False))

    print("replies that drift off contract are recovered, not thrown away")
    # Every shape below was emitted by a real executor model and cost a turn.
    for label, raw, want in (
        ("array of actions", '[{"action":"fill","element_id":"e1","text":"x"},{"action":"click"}]', ("fill", "e1")),
        ("underscore keys", '{"_action":"click","_element":"[e3]"}', ("click", "e3")),
        ("bracketed ref", '{"action":"click","ref":"[e7]"}', ("click", "e7")),
        ("value instead of text", '{"action":"fill","ref":"e2","value":"hello"}', ("fill", "e2")),
    ):
        got = llm.parse_action(raw)
        check(f"{label} parses", (got.get("action"), got.get("ref")), want)
    check("scroll_down becomes scroll+direction",
          (llm.parse_action('{"action":"scroll_down"}').get("action"),
           llm.parse_action('{"action":"scroll_down"}').get("direction")), ("scroll", "down"))
    check("a genuinely unparseable reply keeps the text for diagnosis",
          "I will click the button" in llm.parse_action("I will click the button")["reason"], True)
    check("an empty reply says so", "(empty reply)" in llm.parse_action("")["reason"], True)

    print("a password is never rendered as something the model can retype")
    # A model read '********' out of its own observation and typed it as a new
    # password on a live account. Only the site's policy stopped the change.
    pw = snapshot._render_element(
        {"ref": "e5", "role": "password", "name": "Current password", "filled_chars": 12})
    check("no asterisk run appears", "***" in pw, False)
    check("the length is described, not the value", "12 hidden characters" in pw, True)
    check("an empty password field says empty",
          "(empty)" in snapshot._render_element({"ref": "e6", "role": "password", "name": "New"}), True)

    print("an action that achieves nothing is reported as such")
    # A real run clicked a dead "Preview" button five times. Its history line read
    # 'clicked [e8] button "Preview"', which reads like success, so the agent kept
    # going back to it until a guard fired 21 steps later.
    import asyncio as _aio
    from web_player import agent as _ag

    class _DeadPageClient:
        """Always clicks the same inert control, like the model did."""
        def __init__(self): self.calls = 0
        def chat(self, messages):
            self.calls += 1
            self.last = messages[-1]["content"]
            return '{"action":"click","ref":"e1"}'

    frozen = {"url": "u", "messages": [], "texts": [],
              "elements": [{"ref": "e1", "role": "button", "name": "Preview"}]}

    async def _frozen_observe(_page, _max):
        return frozen

    class _NoopDispatcher:
        async def perform(self, action, _snap):
            return f'clicked [{action["ref"]}] button "Preview"'

    real = _ag.snapshot.observe
    _ag.snapshot.observe = _frozen_observe
    try:
        cl = _DeadPageClient()
        ag = _ag.WebAgent(page=None, cfg=st, client=cl)
        ag.dispatcher = _NoopDispatcher()
        res = _aio.run(ag.run("goal", max_steps=10, timeout_s=30))
        prompt = cl.last
    finally:
        _ag.snapshot.observe = real

    check("the unchanged observation is named on the action's own line",
          "NO VISIBLE CHANGE" in prompt, True)
    check("unchanged DOM alone does not permanently blacklist a control",
          "CONTROLS THAT DO NOTHING" in prompt, False)
    check("and the run still terminates rather than spinning", res.steps < 10, True)
    check("attributed to the agent, not the app",
          failures.classify(res.reason), "NAVIGATION_LIVELOCK")

    print("deliberately revisiting a page is not mistaken for a stall")
    # A draft-retention test MUST open a URL, navigate away, and open it again.
    # An earlier guard counted every repeat and warned at the second, telling the
    # agent mid-test that it was going in circles - and it abandoned a correct run.
    import asyncio as _a2
    from web_player import agent as _ag2

    pages = [
        {"url": "/survey", "messages": [], "texts": ["survey page"],
         "elements": [{"ref": "e1", "role": "textbox", "name": "Short answer"}]},
        {"url": "/home", "messages": [], "texts": ["home page"],
         "elements": [{"ref": "e1", "role": "button", "name": "Home"}]},
    ]
    flip = {"i": 0}

    async def _alternating(_page, _max):
        snap = pages[flip["i"] % 2]
        flip["i"] += 1
        return snap

    class _RoundTripClient:
        """goto survey / goto home, repeatedly - the legitimate pattern."""
        def __init__(self): self.n = 0; self.last = ""
        def chat(self, messages):
            self.n += 1
            self.last = messages[-1]["content"]
            url = "/survey" if self.n % 2 else "/home"
            return '{"action":"goto","url":"%s"}' % url

    class _RealNav:
        async def perform(self, action, _snap):
            return "navigated to %s" % action["url"]

    real2 = _ag2.snapshot.observe
    _ag2.snapshot.observe = _alternating
    try:
        c2 = _RoundTripClient()
        ag2 = _ag2.WebAgent(page=None, cfg=st, client=c2)
        ag2.dispatcher = _RealNav()
        res2 = _a2.run(ag2.run("goal", max_steps=5, timeout_s=30))
        prompt2 = c2.last
    finally:
        _ag2.snapshot.observe = real2

    check("five round-trip navigations are allowed to run", res2.steps, 5)
    check("the agent is never told it is going in circles",
          "going in circles" in prompt2 or "inert" in prompt2, False)
    check("nothing is marked as doing nothing", "THIS DID NOTHING" in prompt2, False)
    check("it ends on the step budget, not a livelock",
          failures.classify(res2.reason), "STEP_LIMIT_EXCEEDED")

    print("a repeated sequence is offered as evidence before it is called a stall")
    # The dominant real-world stall: "type a value, leave without saving, come
    # back and see if it survived" answers itself on the second pass. The agent
    # instead read its own repetition as "that did not take" and restarted six
    # times. It must be told the repetition IS the finding, before any guard
    # punishes it as a loop.
    import asyncio as _a3
    from web_player import agent as _ag3

    two = [
        {"url": "/form", "messages": [], "texts": ["empty form"],
         "elements": [{"ref": "e1", "role": "textbox", "name": "Project Name"}]},
        {"url": "/list", "messages": [], "texts": ["project list"],
         "elements": [{"ref": "e1", "role": "button", "name": "New Project"}]},
    ]
    turn = {"i": 0}

    async def _cycle(_page, _max):
        snap = two[turn["i"] % 2]
        turn["i"] += 1
        return snap

    class _RestartClient:
        def __init__(self): self.n = 0; self.seen = []
        def chat(self, messages):
            self.n += 1
            self.seen.append(messages[-1]["content"])
            return '{"action":"click","ref":"e1"}'

    class _Nav:
        async def perform(self, action, _snap):
            return "clicked [%s]" % action["ref"]

    real3 = _ag3.snapshot.observe
    _ag3.snapshot.observe = _cycle
    try:
        c3 = _RestartClient()
        ag3 = _ag3.WebAgent(page=None, cfg=st, client=c3)
        ag3.dispatcher = _Nav()
        _a3.run(ag3.run("goal", max_steps=12, timeout_s=30))
        prompts = chr(10).join(c3.seen)
    finally:
        _ag3.snapshot.observe = real3

    check("the agent is told the repetition may be the answer",
          "that is your evidence" in prompts, True)
    check("and told repeating again cannot help",
          "cannot tell you anything new" in prompts, True)
    check("the nudge arrives before the abort",
          prompts.index("that is your evidence") > 0, True)
    check("it is a nudge, not an accusation of being stuck",
          "going in circles" in prompts, False)

    print("status codes are matched as numbers, not substrings")
    # Our own budget-overrun message contains "4000", which contains "400", so
    # every overrun was classified as a permanent HTTP 400 and never retried --
    # the escalating retry was implemented, tested, and silently never ran.
    from web_player.llm import _is_transient as _tr
    overrun = ("OpenRouter returned an empty answer (finish_reason='length', 18720 "
               "chars of reasoning). The model spent its whole 4000-token budget on "
               "its scratchpad - raise WEB_LLM_MAX_TOKENS.")
    check("a 4000-token overrun is retried, not read as HTTP 400",
          _tr(llm.LLMError(overrun)), True)
    check("an 8000-token overrun is retried too",
          _tr(llm.LLMError(overrun.replace("4000", "8000"))), True)
    check("a genuine HTTP 400 is still permanent",
          _tr(llm.LLMError("OpenRouter 400: bad request")), False)
    check("a genuine HTTP 404 is still permanent",
          _tr(llm.LLMError("OpenRouter 404: no such model")), False)
    check("HTTP 429 is still retried", _tr(llm.LLMError("OpenRouter 429: slow down")), True)

    print("a CAPTCHA is seen, named, and blamed on the environment")
    # reCAPTCHA renders in a cross-origin iframe, so the element walk never saw
    # it. The agent submitted a form that silently refused, had no idea why, and
    # wandered off looking for a different survey.
    cap = snapshot.render({"url": "u", "captcha": True,
                           "elements": [{"ref": "e1", "role": "button", "name": "Submit"}]})
    check("the observation announces it", "CAPTCHA" in cap, True)
    check("and forbids attempting it", "must not try" in cap, True)
    check("and says what is still testable", "still testable" in cap, True)
    check("a clean page says nothing about captchas",
          "CAPTCHA" in snapshot.render({"url": "u", "elements": []}), False)

    for reason in ("Blocked by CAPTCHA: cannot submit",
                   "the page shows an 'I am not a robot' challenge",
                   "reCAPTCHA prevents submission"):
        check(f"classified from {reason[:32]!r}", failures.classify(reason), "BLOCKED_BY_CAPTCHA")
    check("it is an environment fault, not the app's",
          ("BLOCKED_BY_CAPTCHA" in st.ENV_FAULT, "BLOCKED_BY_CAPTCHA" in st.APP_FAULT), (True, False))
    check("never retried - a retry cannot solve it",
          failures.recovery_strategy("BLOCKED_BY_CAPTCHA")["retry"], False)
    check("the fix is named in the recovery hint",
          "test keys" in failures.recovery_strategy("BLOCKED_BY_CAPTCHA")["action"], True)
    print("the browser mode decides who deals with a CAPTCHA")
    import asyncio as _a4
    from web_player import agent as _ag4

    class _Cfg(dict):
        def __getattr__(self, k):
            try: return self[k]
            except KeyError: return getattr(st, k)

    capt = {"url": "u", "captcha": True, "messages": [], "texts": [],
            "elements": [{"ref": "e1", "role": "button", "name": "Submit"}]}

    async def _capt_observe(_p, _m):
        return capt

    class _Never:
        def chat(self, _m): return '{"action":"click","ref":"e1"}'

    real4 = _ag4.snapshot.observe
    _ag4.snapshot.observe = _capt_observe
    try:
        # Headless: nobody is watching, so fail at once rather than burn the budget.
        ag4 = _ag4.WebAgent(page=None, cfg=_Cfg(WEB_HEADLESS=True), client=_Never())
        res4 = _a4.run(ag4.run("goal", max_steps=20, timeout_s=20))
    finally:
        _ag4.snapshot.observe = real4

    check("headless fails immediately", res4.steps, 1)
    check("headless names the CAPTCHA", "CAPTCHA" in res4.reason, True)
    check("headless says why nobody can help", "headless" in res4.reason, True)
    check("headless suggests the real fix", "test keys" in res4.reason, True)
    check("it classifies as a CAPTCHA block",
          failures.classify(res4.reason), "BLOCKED_BY_CAPTCHA")
    check("which is an environment fault, never the app's",
          "BLOCKED_BY_CAPTCHA" in st.ENV_FAULT, True)
    check("headed waiting has a sane built-in default",
          _ag4._CAPTCHA_WAIT_DEFAULT >= 60, True)

    print("a browser that has gone away is not our navigation failing")
    # One batch left the browser open 28 minutes at a CAPTCHA; it closed, and the
    # four remaining tests each failed in 0s as NAVIGATION_FAILURE - an AGENT
    # fault, blaming our navigation for a browser that no longer existed.
    gone = ("Could not open https://x: TargetClosedError: Page.goto: Target page, "
            "context or browser has been closed")
    check("classified as the browser being gone", failures.classify(gone), "BROWSER_CLOSED")
    check("an environment fault", "BROWSER_CLOSED" in st.ENV_FAULT, True)
    check("never an agent fault", "BROWSER_CLOSED" in st.AGENT_FAULT, False)
    check("never an app fault", "BROWSER_CLOSED" in st.APP_FAULT, False)
    check("not retried - there is nothing to retry into",
          failures.recovery_strategy("BROWSER_CLOSED")["retry"], False)

    print("a wall-clock timeout is not retried with a bigger clock")
    # Self-heal used to retry a TIMEOUT with double the budget: one case ran
    # 1728s - nearly half an hour - and still failed.
    check("TIMEOUT is not retried", failures.recovery_strategy("TIMEOUT")["retry"], False)
    check("and says what to do instead",
          "WEB_TIMEOUT" in failures.recovery_strategy("TIMEOUT")["action"], True)

    print("a blocked click reports what is blocking it")
    # Playwright names the element sitting on top; we were discarding it and
    # telling the agent only "covered, off-screen, or disabled" - three guesses
    # and no way to choose. Three such clicks cost 10s each in one run.
    from web_player.actions import _why_not_actionable as _why
    pw = ("Locator.click: Timeout 10000ms exceeded." + chr(10) +
          "  -   element is visible, enabled and stable" + chr(10) +
          '  -   <div class="modal-backdrop"></div> intercepts pointer events')
    msg = _why(Exception(pw))
    check("the blocking element is named", "modal-backdrop" in msg, True)
    check("and why it blocks", "intercepts pointer events" in msg, True)
    check("a bare timeout still gets an honest fallback",
          "no reason" in _why(Exception("Timeout 10000ms exceeded.")), True)
    check("the explanation is bounded", len(_why(Exception(pw * 20))) < 400, True)

    print("the CAPTCHA hand-off follows the real browser, not the config")
    # One run reported "the run is headless" on one test and correctly asked for
    # help on the next - same process, same settings. Deciding from config at the
    # moment of use was unreliable; the launched browser is the authority.
    import asyncio as _a5
    from web_player import agent as _ag5

    capt5 = {"url": "u", "captcha": True, "messages": [], "texts": [], "elements": []}

    async def _obs5(_p, _m):
        return capt5

    real5 = _ag5.snapshot.observe
    _ag5.snapshot.observe = _obs5
    try:
        # cfg claims headed, but the browser really is headless: trust the browser.
        class _CfgHeaded:
            def __getattr__(self, k):
                return False if k == "WEB_HEADLESS" else getattr(st, k)
        ag5 = _ag5.WebAgent(page=None, cfg=_CfgHeaded(), client=None)
        ag5.headless = True
        res5 = _a5.run(ag5._handle_captcha(capt5))
    finally:
        _ag5.snapshot.observe = real5

    check("a truly headless browser fails fast even if cfg says headed",
          "headless" in res5[1], True)
    check("the agent defaults to asking config when unset",
          _ag5.WebAgent(page=None, cfg=st, client=None).headless, None)

    print("one test case writes exactly one execution record")
    # A 5-round batch produced 8 execution rows, one round logging twice. Every
    # metric built on those rows - steps, durations, verdict counts - was wrong.
    import re as _re
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "web_player", "runner.py"), encoding="utf-8").read()
    nl2 = chr(10)
    body = src.split("async def execute_test_case")[1].split(nl2 + "def ")[0]
    check("every exit path goes through the guarded logger",
          "gateway.log_execution(tc," in body, False)
    check("the guard exists", "def log_once(" in body, True)
    check("and it refuses a second write", "already recorded" in body, True)
    check("all four exit paths use it", body.count("log_once(tc,"), 4)

    print("navigation falls back when a page never goes idle")
    # reset_to_base waited for networkidle with no fallback; an app that polls
    # never reaches it, so a whole test case died at the navigation timeout
    # before taking a single step.
    bsrc = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "web_player", "browser.py"), encoding="utf-8").read()
    reset = bsrc.split("async def reset_to_base")[1].split("async def")[0]
    check("reset_to_base uses the shared bounded navigation helper", "_goto_settled" in reset, True)

    print("two batches cannot run at once")
    # Two concurrent batches shared the knowledge graph, the site and the trace
    # log. Their transcripts interleaved into rounds 1,2,2,3,3,4,5,4 and eight
    # executions for a five-round run, and an hour went into diagnosing a bug
    # that did not exist because the evidence belonged to two processes.
    import os as _os
    import tempfile as _tf
    from web_player import runlock as _rl

    original = _rl.LOCK_PATH
    _rl.LOCK_PATH = _os.path.join(_tf.mkdtemp(), "batch.lock")
    # Neutralise the live-process scan: this block tests the lock FILE. Leaving it
    # on made the suite fail whenever a real batch happened to be running, which
    # is a reading of the machine, not a regression in the code.
    _orig_scan = _rl.find_other_batches
    _rl.find_other_batches = lambda: []
    try:
        first = _rl.RunLock(profile="p", rounds=5).__enter__()
        check("the first batch takes the lock", _os.path.exists(_rl.LOCK_PATH), True)

        try:
            _rl.RunLock(profile="p", rounds=5).__enter__()
            check("a second batch is refused", "not refused", "refused")
        except _rl.RunInProgress as exc:
            check("a second batch is refused", True, True)
            check("it names the holding process", str(_os.getpid()) in str(exc), True)
            check("and says why it matters", "corrupt" in str(exc), True)

        first.release()
        check("releasing removes the lock", _os.path.exists(_rl.LOCK_PATH), False)

        second = _rl.RunLock(profile="p", rounds=5).__enter__()
        check("a later batch may then start", second.held, True)
        second.release()

        # A lock left behind by a killed run must not block every future batch --
        # the classic failure that teaches people to delete lock files by hand.
        import json as _json
        with open(_rl.LOCK_PATH, "w", encoding="utf-8") as fh:
            _json.dump({"pid": 999999, "profile": "dead", "started_at": "?"}, fh)
        taken = _rl.RunLock(profile="p", rounds=1).__enter__()
        check("a stale lock from a dead run is taken over", taken.held, True)
        taken.release()

        check("a dead pid reads as dead", _rl._pid_alive(999999), False)
        check("our own pid reads as alive", _rl._pid_alive(_os.getpid()), True)
    finally:
        _rl.LOCK_PATH = original
        _rl.find_other_batches = _orig_scan

    print("the batch detector never mistakes a run for its own rival")
    # The venv launcher runs as a parent/child pair with identical command lines.
    # If the detector counted its own parent or child, every legitimate run would
    # refuse itself at startup and nothing could ever start.
    import sys as _sys
    import types as _types
    from web_player import runlock as _rl2

    me = _os.getpid()

    class _P:
        def __init__(self, pid, cmd, name="python.exe"):
            self.info = {"pid": pid, "name": name, "cmdline": cmd.split()}

    class _Self:
        def ppid(self): return 4242          # our launcher parent
        def children(self, recursive=True):  # a child we spawned
            return [_types.SimpleNamespace(pid=4343)]

    fake = _types.ModuleType("psutil")
    fake.Process = lambda pid=None: _Self()
    fake.process_iter = lambda attrs=None: [
        _P(me,    "python -m targets.run profile"),      # ourselves
        _P(4242,  "python -m targets.run profile"),      # our venv parent
        _P(4343,  "python -m targets.run profile"),      # our child
        _P(9001,  "python -m targets.run other"),        # a REAL rival
        _P(9002,  "python -m http.server"),              # unrelated python
        _P(9003,  "node server.js", name="node.exe"),    # not python
    ]
    saved = _sys.modules.get("psutil")
    _sys.modules["psutil"] = fake
    try:
        found = _rl2.find_other_batches()
    finally:
        if saved is not None: _sys.modules["psutil"] = saved
        else: _sys.modules.pop("psutil", None)

    pids = [pid for pid, _ in found]
    check("the rival batch is found", 9001 in pids, True)
    check("we never count ourselves", me in pids, False)
    check("nor our venv launcher parent", 4242 in pids, False)
    check("nor a child we spawned", 4343 in pids, False)
    check("unrelated python is ignored", 9002 in pids, False)
    check("non-python is ignored", 9003 in pids, False)
    check("exactly one rival", len(found), 1)

    # Without psutil the detector must degrade to silence, not to an exception.
    _sys.modules["psutil"] = None
    try:
        degraded = _rl2.find_other_batches()
    except Exception as exc:
        degraded = "raised: %s" % exc
    finally:
        if saved is not None: _sys.modules["psutil"] = saved
        else: _sys.modules.pop("psutil", None)
    check("no psutil degrades to the lock file alone", degraded, [])

    print("a solved challenge stops blocking, a present one does not")
    # A solved reCAPTCHA does NOT disappear - it shows a green tick and the widget
    # stays. Testing presence alone kept reporting "CAPTCHA on the page" after a
    # person had solved it: the run waited out its full 300s and failed the test
    # as unsolved, with the way actually clear the whole time.
    js = io.open("web_player/snapshot.py", encoding="utf-8").read() if False else None
    import io as _io
    src = _io.open("web_player/snapshot.py", encoding="utf-8").read()
    check("the response token is what is inspected",
          "g-recaptcha-response" in src, True)
    check("hCaptcha's token too", "h-captcha-response" in src, True)
    check("Turnstile's token too", "cf-turnstile-response" in src, True)
    check("blocking requires present AND unsolved", "&& !solved" in src, True)
    # The rendered observation must only warn while it genuinely blocks.
    check("an unsolved challenge is announced",
          "CAPTCHA" in snapshot.render({"url": "u", "captcha": True, "elements": []}), True)
    check("a solved one is not",
          "CAPTCHA" in snapshot.render({"url": "u", "captcha": False, "elements": []}), False)

    print("the CAPTCHA pause is audible and keeps reminding")
    # A run sat silently at a CAPTCHA for five minutes and failed with someone at
    # the machine throughout: the terminal bell alone is ignored by Windows
    # Terminal and is switched off on many systems.
    check("an alert helper exists", callable(getattr(agent_mod, "_alert", None)), True)
    check("it never raises, whatever happens", agent_mod._alert(0), None)
    check("reminders are frequent enough to catch someone returning",
          0 < agent_mod._CAPTCHA_REMIND_SECONDS <= 30, True)
    check("and far shorter than the wait itself",
          agent_mod._CAPTCHA_REMIND_SECONDS < agent_mod._CAPTCHA_WAIT_DEFAULT, True)
    src_agent = _io.open("web_player/agent.py", encoding="utf-8").read() if False else None
    import io as _io2
    src2 = _io2.open("web_player/agent.py", encoding="utf-8").read()
    check("it does not rely on the terminal bell alone", "winsound" in src2, True)
    check("the wait loop re-alerts", "next_reminder = waited" in src2, True)
    check("and chirps once when released", "free to walk away again" in src2, True)

    print("a second, different loop is nudged too")
    # The nudge was capped at one per test case. A real run spent it on a search
    # box at step 3, then looped goto-then-back eleven times from step 30 with no
    # prompt at all, and only the hard guard stopped it at step 41.
    import asyncio as _a5
    from web_player import agent as _ag5

    # Two distinct A/B loops, one after the other.
    seq = [
        {"url": "/one", "messages": [], "texts": ["one"],
         "elements": [{"ref": "e1", "role": "button", "name": "One"}]},
        {"url": "/two", "messages": [], "texts": ["two"],
         "elements": [{"ref": "e1", "role": "button", "name": "Two"}]},
        {"url": "/three", "messages": [], "texts": ["three"],
         "elements": [{"ref": "e1", "role": "button", "name": "Three"}]},
        {"url": "/four", "messages": [], "texts": ["four"],
         "elements": [{"ref": "e1", "role": "button", "name": "Four"}]},
    ]
    st5 = {"i": 0}

    async def _two_loops(_p, _m):
        i = st5["i"]
        st5["i"] += 1
        # first loop alternates 0/1 four times, then a second loop alternates 2/3
        return seq[(i % 2) if i < 8 else 2 + (i % 2)]

    class _Looper:
        def __init__(self): self.seen = []
        def chat(self, messages):
            self.seen.append(messages[-1]["content"])
            return '{"action":"click","ref":"e1"}'

    class _Nop:
        async def perform(self, action, _snap):
            return "clicked [%s]" % action["ref"]

    real5 = _ag5.snapshot.observe
    _ag5.snapshot.observe = _two_loops
    try:
        c5 = _Looper()
        ag5 = _ag5.WebAgent(page=None, cfg=st, client=c5)
        ag5.dispatcher = _Nop()
        _a5.run(ag5.run("goal", max_steps=16, timeout_s=30))
        nudges = sum(p.count("that is your evidence") for p in c5.seen[-1:])
        allp = chr(10).join(c5.seen)
    finally:
        _ag5.snapshot.observe = real5

    check("the nudge is keyed per loop, not per run",
          "nudged: set[str]" in open(os.path.join(
              os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
              "web_player", "agent.py"), encoding="utf-8").read(), True)
    check("a nudge was issued at all", "that is your evidence" in allp, True)

    print("the agent works from a verified API list, not an invented one")
    # A planner objective cited GET /api/projects. It does not exist - the real
    # route is /api/project - and the executor chased it six times until the test
    # died in a livelock.
    import tempfile as _tf
    from web_player import api_registry as _ar

    real_dir = _ar.STORE_DIR
    _ar.STORE_DIR = _tf.mkdtemp()
    try:
        reg = _ar.ApiRegistry("unit-test")
        check("nothing known means nothing claimed", reg.prompt_block(), "")

        reg.record("GET", "https://x.tld/api/project", 200)
        reg.record("GET", "https://x.tld/api/project/270/surveys", 200)
        reg.record("GET", "https://x.tld/api/projects", 404)

        block = reg.prompt_block()
        check("the real route is offered", "GET /api/project" in block, True)
        check("ids are collapsed so one row covers all",
              "/api/project/:id/surveys" in block, True)
        check("a 404-only route is not offered as usable",
              "GET /api/projects" in block, False)
        check("but is named as known-bad", "/api/projects" in reg.broken(), True)
        check("the agent is told not to invent", "do NOT invent" in block, True)

        # It must GROW: a route met mid-run is remembered for the next one.
        before = len(reg.routes)
        reg.record("POST", "https://x.tld/api/question-bank/create", 201)
        check("a newly met route is learned", len(reg.routes), before + 1)
        check("and survives a save/load round trip",
              (reg.save(), len(_ar.ApiRegistry("unit-test").routes))[1], before + 1)
    finally:
        _ar.STORE_DIR = real_dir

    check("slugs are collapsed too",
          _ar.normalise("https://x/api/fetch-survey-user/e07-ba4"),
          "/api/fetch-survey-user/:slug")
    # A slug rule of "letters-and-a-hyphen" ate real route names:
    # /api/template-share/shared-with-me collapsed to /api/:slug/shared-with-me.
    for raw, want in (
        ("/api/template-share/shared-with-me", "/api/template-share/shared-with-me"),
        ("/api/collaborator/all-projects", "/api/collaborator/all-projects"),
        ("/api/survey-collaborator/all-invitations", "/api/survey-collaborator/all-invitations"),
        ("/api/get-user-packages", "/api/get-user-packages"),
    ):
        check(f"hyphenated route name survives: {raw[:34]}", _ar.normalise(raw), want)
    check("a bad URL never raises", _ar.ApiRegistry("unit-test2").record("GET", None, 200), None)

    print("the shipped registry matches what the site really serves")
    shipped = _ar.for_project("dataghurhi-auth")
    check("seeded from the discovery walk", len(shipped.routes) >= 14, True)
    check("the real project route is known",
          any(r.endswith("/api/project") for r in shipped.routes), True)
    check("the invented plural is NOT offered as usable",
          "GET /api/projects" in shipped.prompt_block(), False)

    print("native dialogs are answered and reported, not silently swallowed")
    # Playwright DISMISSES every dialog when no handler is registered. This app
    # calls alert() 309 times and confirm() 16 times, so alert text never reached
    # the agent and every confirm was answered "Cancel" - the action did not
    # happen, and the agent read it as a control that does nothing.
    from web_player.oracles import Collector as _C

    class _Cfg2:
        WEB_COLLECT_CONSOLE = True
        WEB_COLLECT_NETWORK = True
        WEB_CONSOLE_IGNORE = ()
        WEB_BASE_URL = "https://x.tld"
        WEB_ACCEPT_CONFIRM = True

    class _Dialog:
        def __init__(self, kind, message):
            self.type, self.message = kind, message
            self.answered = None
        async def accept(self): self.answered = "accept"
        async def dismiss(self): self.answered = "dismiss"

    col = _C(page=None, cfg=_Cfg2)
    for kind, msg in (("alert", "Saved successfully!"),
                      ("confirm", "Delete this item?"),
                      ("prompt", "Name?")):
        col._on_dialog(_Dialog(kind, msg))
    seen = col.take_dialogs()
    check("the alert text is captured",
          any("Saved successfully!" in d for d in seen), True)
    check("a confirm is accepted by default, not cancelled",
          any("confirm" in d and "accepted" in d for d in seen), True)
    check("a prompt is dismissed (we cannot know what to type)",
          any("prompt" in d and "dismissed" in d for d in seen), True)
    check("taking them clears them", col.take_dialogs(), [])

    class _CfgNo(_Cfg2):
        WEB_ACCEPT_CONFIRM = False
    col2 = _C(page=None, cfg=_CfgNo)
    col2._on_dialog(_Dialog("confirm", "Delete?"))
    check("confirm handling is configurable",
          any("dismissed" in d for d in col2.take_dialogs()), True)
    check("a broken dialog object never raises", col._on_dialog(None) or True, True)

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

    print("wandering guard ends a stalled run before the step budget")
    asyncio.run(_check_wandering_guard())

    print(f"\n{_passed}/{_passed + _failed} checks passed")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
