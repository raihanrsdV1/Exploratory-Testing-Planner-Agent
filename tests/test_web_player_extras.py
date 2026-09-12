#!/usr/bin/env python3
"""Web player: the fixes for what a run could not see or tell the truth about.

Each check here corresponds to a way a campaign silently produced a wrong
report. A native alert was answered by Playwright before the agent could read
it, and a validation message the site DID show was filed as a silent refusal —
a defect that did not exist. Controls behind a modal looked clickable and burned
the step budget. A run's identity came from the gateway's own environment, so an
Android session's role leaked into a web run's prompts. Screenshots overwrote
each other because test ids restart at TC-001 every campaign.
"""
import asyncio
import inspect
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import tempfile as _tf  # noqa: E402
os.environ.setdefault("WEB_TRACE_FILE",
                      os.path.join(_tf.gettempdir(), "web_player_tests.log"))
import settings as st  # noqa: E402
from web_player import agent as agent_mod  # noqa: E402
from web_player import snapshot  # noqa: E402
from web_player.actions import ActionError, Dispatcher, fixture_path  # noqa: E402
from web_player.browser import dialog_answer  # noqa: E402
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


def _check_dialog_rules():
    print("a native dialog is answered by rule, never ignored")
    blocked = ("delete account", "log out")
    # An alert has only an OK button; refusing it would hang the page.
    check("alert is accepted", dialog_answer("alert", "Please fill in all required fields", blocked), "accept")
    check("benign confirm is accepted", dialog_answer("confirm", "Save your changes?", blocked), "accept")
    check("a destructive confirm is cancelled",
          dialog_answer("confirm", "Delete this project permanently?", blocked), "dismiss")
    check("a guardrailed confirm is cancelled",
          dialog_answer("confirm", "Log out of this device?", blocked), "dismiss")
    # There is no way to know what a prompt wants typed, and a wrong value is a write.
    check("prompt is dismissed", dialog_answer("prompt", "New name?", blocked), "dismiss")
    check("a page-leave warning is accepted", dialog_answer("beforeunload", "", blocked), "accept")
    check("the rule holds with no guardrails configured",
          dialog_answer("confirm", "Remove this item?", ()), "dismiss")

    note = agent_mod._dialog_note({"type": "alert", "message": "Please fill in all required fields",
                                   "answer": "accept"})
    for needle in ("POPUP", "alert", "Please fill in all required fields", "OK"):
        check(f"the popup note states {needle!r}", needle in note, True)


async def _check_dialog_reaches_the_model():
    """The regression that matters: the site DID answer, and the agent must see it."""
    seen = []

    class Client:
        def __init__(self):
            self.turn = 0

        def chat(self, messages):
            seen.append(str(messages[-1].get("content", "")))
            self.turn += 1
            if self.turn == 1:
                return '{"thought":"t","action":"click","ref":"e1"}'
            return '{"thought":"t","action":"finish","success":false,"reason":"the site refused the name"}'

    page_snap = {"url": "https://shop.example.com/projects", "title": "Projects",
                 "elements": [{"ref": "e1", "role": "button", "name": "Create"}],
                 "messages": [], "texts": []}
    log = []

    async def fake_observe(page, max_elements):
        return dict(page_snap)

    async def fake_perform(action, snap):
        # What the browser does on a click that trips validation.
        log.append({"type": "alert", "message": "Please fill in all required fields",
                    "answer": "accept"})
        return "clicked [e1]"

    orig = agent_mod.snapshot.observe
    agent_mod.snapshot.observe = fake_observe
    try:
        a = agent_mod.WebAgent(page=None, cfg=st, client=Client())
        a.dialog_log = log
        a.dispatcher.perform = fake_perform
        result = await a.run("goal: create a project with a blank name", max_steps=5, timeout_s=999)
    finally:
        agent_mod.snapshot.observe = orig

    check("the popup text reaches the model's next observation",
          any("Please fill in all required fields" in m for m in seen), True)
    check("a popup is not mistaken for a page that ignored the click",
          result.success, False)


def _check_honest_bail_out():
    """Every livelock is a run that reported nothing. The agent must be told,
    in the prompt AND at the moment it starts repeating itself, that saying
    "I could not find it" is a real answer."""
    print("the agent is offered an honest exit instead of clicking on")
    prompt = agent_mod._SYSTEM_PROMPT
    check("the prompt names the phrase to use", "Precondition not met:" in prompt, True)
    check("it covers a feature that cannot be found, not just missing data",
          "cannot find" in prompt, True)
    check("it says that is not a defect", "not a defect" in prompt, True)
    # The two interventions that fire before a livelock abort must name the exit.
    src = inspect.getsource(agent_mod.WebAgent.run)
    check("the wandering warning offers the exit",
          src.count('Precondition not met:') >= 2, True)


async def _check_repetition_is_refused():
    """The livelock fix that does not depend on the model taking advice.

    A campaign lost 16 of 20 runs to repeated no-op actions. The agent was told
    to do something different and did not, so the same click must simply stop
    being carried out.
    """
    print("an action that has changed nothing is refused, not performed again")
    performed = []

    class Client:
        def chat(self, messages):
            return '{"thought":"the button must work eventually","action":"click","ref":"e1"}'

    page_snap = {"url": "https://shop.example.com/analysis", "title": "Analysis",
                 "elements": [{"ref": "e1", "role": "button", "name": "Analyze"}],
                 "messages": [], "texts": []}

    async def fake_observe(page, max_elements):
        return dict(page_snap)

    async def fake_perform(action, snap):
        performed.append(action)
        return "clicked [e1]"

    orig = agent_mod.snapshot.observe
    agent_mod.snapshot.observe = fake_observe
    try:
        a = agent_mod.WebAgent(page=None, cfg=st, client=Client())
        a.dispatcher.perform = fake_perform
        result = await a.run("goal: run the analysis", max_steps=15, timeout_s=999)
    finally:
        agent_mod.snapshot.observe = orig

    check("the useless action is carried out at most twice",
          len(performed) <= 2, True)
    check("the run still ends rather than spinning", result.success, False)
    check("the refusal names the honest exit",
          any("Precondition not met:" in h for h in result.history), True)
    check("the refusal says it was not performed",
          any("was not performed again" in h for h in result.history), True)


def _check_covered_controls():
    print("controls behind a modal are marked and refused, not clicked at")
    snap = {
        "url": "https://shop.example.com/projects", "title": "Projects",
        "dialog_open": True, "dialog_title": "Create New Project", "covered_count": 1,
        "headings": [], "messages": [], "texts": [],
        "elements": [
            {"ref": "e1", "role": "textbox", "name": "Project name"},
            {"ref": "e2", "role": "button", "name": "Export", "covered": True},
        ],
    }
    text = snapshot.render(snap)
    for needle in ("MODAL DIALOG", "Create New Project", "COVERED"):
        check(f"the observation states {needle!r}", needle in text, True)

    d = Dispatcher(page=None, cfg=_FakeCfg())
    try:
        d._require({"action": "click", "ref": "e2"}, snap)
        check("a covered control is refused", "not refused", "refused")
    except ActionError as exc:
        check("a covered control is refused at once", "behind an open dialog" in str(exc), True)
        check("refusing it is not reported as a missing element type",
              exc.category, "ELEMENT_NOT_FOUND")
    check("a control inside the dialog is still usable",
          d._require({"action": "fill", "ref": "e1"}, snap)["name"], "Project name")


def _check_form_validation_visible():
    print("a form the browser itself refused is not a dead click")
    snap = {
        "url": "https://shop.example.com/projects", "title": "Projects",
        "headings": [], "messages": [], "texts": [], "elements": [],
        "validation_messages": ["[e31] Research Field: Please fill out this field."],
    }
    text = snapshot.render(snap)
    for needle in ("REFUSED TO SUBMIT", "Research Field", "Please fill out this field",
                   "never sent"):
        check(f"the observation states {needle!r}", needle in text, True)
    check("a form with nothing wrong says nothing about validation",
          "REFUSED TO SUBMIT" in snapshot.render({"url": "u", "elements": []}), False)


def _check_inert_control_marked_inline():
    """A control proven inert must say so on its own line.

    The agent was already given a prose list of dead control names and clicked
    the same dead control eleven times anyway: it chooses a ref from the element
    list, so that is where the warning has to be.
    """
    print("a control already proven inert is marked where the choice is made")
    marked = _render_one({"ref": "e54", "role": "button", "name": "Close",
                          "inert": True})
    check("the element line says it was already tried", "ALREADY TRIED" in marked, True)
    check("it tells the agent to pick something else",
          "choose something else" in marked, True)
    fresh = _render_one({"ref": "e54", "role": "button", "name": "Close"})
    check("an untried control carries no such mark", "ALREADY TRIED" in fresh, False)


def _check_whitespace_value_visible():
    """A whitespace-only entry must not read back as an empty field.

    A live run typed three spaces into 'Project Name', saw no value in the next
    observation, and concluded the field "does not accept any input - it discards
    values silently". The field was fine; the observer had trimmed the value away.
    """
    print("whitespace typed into a field is reported as content, not as nothing")
    filled = _render_one({"ref": "e1", "role": "textbox", "name": "Project Name",
                          "whitespace_chars": 3})
    check("the observation says the field holds whitespace",
          "whitespace only" in filled, True)
    check("it states the typing was accepted", "WAS accepted" in filled, True)
    check("it gives the count", "3 space characters" in filled, True)

    empty = _render_one({"ref": "e1", "role": "textbox", "name": "Project Name"})
    check("a genuinely empty field still says nothing about whitespace",
          "whitespace" in empty, False)
    # A real value must be unaffected.
    real = _render_one({"ref": "e1", "role": "textbox", "name": "Project Name",
                        "value": "Survey A"})
    check("an ordinary value is still shown", 'value="Survey A"' in real, True)
    check("and is not described as whitespace", "whitespace" in real, False)


def _render_one(el: dict) -> str:
    return snapshot.render({"url": "u", "title": "t", "headings": [], "messages": [],
                            "texts": [], "elements": [el]})


def _check_fixture_files():
    print("only the profile's own sample files can ever be uploaded")
    allowed = (os.path.join(ROOT, "data", "fixtures", "web", "survey_sample.csv"),)
    check("a listed file resolves to an absolute path",
          fixture_path("survey_sample.csv", allowed), os.path.abspath(allowed[0]))
    for attempt in ("/etc/passwd", "../../../../etc/passwd", "auth.json", ""):
        try:
            fixture_path(attempt, allowed)
            check(f"{attempt!r} is refused", "accepted", "refused")
        except ActionError as exc:
            check(f"{attempt!r} is refused", "not an available sample file" in str(exc), True)
    try:
        fixture_path("survey_sample.csv", ())
        check("with nothing configured, upload is refused", "accepted", "refused")
    except ActionError as exc:
        check("with nothing configured, upload is refused",
              "none are configured" in str(exc), True)


def _check_dialog_findings():
    print("the oracle records dialogs as evidence")
    f = Findings(dialogs=['alert: "Please fill in all required fields"'])
    check("a dialog alone makes a run non-empty", f.is_empty(), False)
    check("dialogs are counted", f.counts()["dialogs"], 1)
    check("the summary names them", "browser dialog" in f.summary(), True)
    check("a clean run is still empty", Findings().is_empty(), True)


def _check_session_isolation():
    print("a web run's identity comes from the request, not the gateway's environment")
    role, state = st.APP_LOGIN_ROLE, st.APP_ACCOUNT_STATE
    st.APP_LOGIN_ROLE, st.APP_ACCOUNT_STATE = "android admin", "3 saved contacts."
    try:
        web = st.app_session_block(account_state="5 projects", login_role="signed-in researcher")
        check("the caller's role is used", "signed-in researcher" in web, True)
        check("the gateway's own role cannot leak in", "android admin" in web, False)
        check("the gateway's own account state cannot leak in", "3 saved contacts" in web, False)
        check("the live account state is used", "5 projects" in web, True)

        own = st.app_session_block()
        check("with no caller identity the process's own role still applies",
              "android admin" in own, True)
        check("and its configured account state still applies", "3 saved contacts" in own, True)

        check("a caller that is signed in as nobody gets no role sentence",
              st.app_session_block(login_role=""), "")
    finally:
        st.APP_LOGIN_ROLE, st.APP_ACCOUNT_STATE = role, state


def _check_no_invented_credentials():
    print("with no account, the agent is told not to invent one")
    user, storage = st.WEB_LOGIN_USER, st.WEB_STORAGE_STATE
    st.WEB_LOGIN_USER, st.WEB_STORAGE_STATE = "", ""
    try:
        block = st.web_no_credentials_block()
        check("it names a safe address to use instead", "example.invalid" in block, True)
        check("it forbids made-up passwords", "password" in block.lower(), True)
        st.WEB_LOGIN_USER = "tester@example.com"
        check("with an account configured it says nothing", st.web_no_credentials_block(), "")
    finally:
        st.WEB_LOGIN_USER, st.WEB_STORAGE_STATE = user, storage


def _check_screenshot_lookup():
    print("a report finds the capture from its own campaign")
    from gateway import report_api

    root = tempfile.mkdtemp(prefix="shots-")
    old_batch = os.path.join(root, "20260101-090000")
    new_batch = os.path.join(root, "20260912-120000")
    os.makedirs(old_batch)
    os.makedirs(new_batch)
    old = os.path.join(old_batch, "TC-001-PASS.png")
    new = os.path.join(new_batch, "TC-001-PASS.png")
    for path in (old, new):
        with open(path, "wb") as fh:
            fh.write(b"x")
    # Same name, different campaigns: only the timestamp tells them apart.
    os.utime(old, (1767258000, 1767258000))     # 2026-01-01T09:00Z
    os.utime(new, (1789560000, 1789560000))     # 2026-09-12T12:00Z

    original = st.WEB_SCREENSHOT_DIR
    st.WEB_SCREENSHOT_DIR = root
    try:
        entry = {"test_case_id": "TC-001", "verdict": "PASS", "created_at": "2026-09-12T12:00:05"}
        check("the capture from this run's batch is chosen",
              report_api._screenshot_for(entry), new)
        older = {"test_case_id": "TC-001", "verdict": "PASS", "created_at": "2026-01-01T09:00:02"}
        check("an older run still resolves to its own capture",
              report_api._screenshot_for(older), old)
        check("a run with no capture reports none",
              report_api._screenshot_for({"test_case_id": "TC-404", "verdict": "PASS"}), "")
        # Captures written before the per-batch folders existed.
        flat = os.path.join(root, "TC-009-FAIL.png")
        with open(flat, "wb") as fh:
            fh.write(b"x")
        check("a capture saved directly in the root is still found",
              report_api._screenshot_for({"test_case_id": "TC-009", "verdict": "FAIL"}), flat)
    finally:
        st.WEB_SCREENSHOT_DIR = original


def main():
    _check_dialog_rules()
    asyncio.run(_check_dialog_reaches_the_model())
    _check_honest_bail_out()
    asyncio.run(_check_repetition_is_refused())
    _check_covered_controls()
    _check_form_validation_visible()
    _check_whitespace_value_visible()
    _check_inert_control_marked_inline()
    _check_fixture_files()
    _check_dialog_findings()
    _check_session_isolation()
    _check_no_invented_credentials()
    _check_screenshot_lookup()

    print(f"\n{_passed}/{_passed + _failed} checks passed")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
