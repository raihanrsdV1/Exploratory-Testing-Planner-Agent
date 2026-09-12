"""The observe -> decide -> act loop.

One test case, one agent run. The agent sees a compact observation of the page,
picks a single action, and we perform it; that repeats until it calls ``finish``
or runs out of budget.

Two things this loop does that a naive version does not:

* **Only the current observation is sent in full.** Earlier turns are compressed
  to their one-line action notes. A page with 60 controls is ~1.5k tokens; ten of
  them in the history is most of a context window spent on pages the agent has
  already left.

* **It detects its own livelock, two ways.** The most common agent failure is
  clicking the same control forever because nothing visibly changed — after a
  few identical no-op actions the agent is told so explicitly, and shortly after
  that the run ends as NAVIGATION_LIVELOCK. The second, subtler failure is
  *wandering*: trying a different action each time (e.g. scrolling further and
  further) while the page itself never changes at all. A same-action repeat
  never fires there, so a separate content-only signature (``WEB_STALL_STEPS``,
  default 6) catches it too. Without either guard the run burns its whole
  budget and reports an ASSERTION_FAILURE, which would be counted as a
  discovered bug.
"""

from __future__ import annotations

import hashlib
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field

from . import actions as actions_mod
from . import llm as llm_mod
from . import snapshot
from . import trace
from . import review
from .exploration import create_recorder

# Occurrences of one (page, action) pair before warning, then before giving up.
_LIVELOCK_WARN = 2
_LIVELOCK_ABORT = 4
# Raw cycle ceiling: high enough that deliberate revisiting is never
# mistaken for a stall, low enough to stop a loop long before the budget.
_CYCLE_ABORT = 6
# When a sequence comes round a second time, the repetition is usually the
# test's own answer rather than a stall. Nudge before any guard punishes it.
_CYCLE_CONCLUDE = 2
# How long a headed run waits for a person, when no explicit limit is set,
# and how often it re-checks whether the challenge has cleared.
_CAPTCHA_WAIT_DEFAULT = 180
_CAPTCHA_POLL_MS = 3000
# How often to beep again while still waiting for a person.
_CAPTCHA_REMIND_SECONDS = 20


def _alert(times: int = 3) -> None:
    """Make an audible noise, because nobody is watching the terminal.

    The terminal bell (``\\a``) alone is not enough: Windows Terminal ignores it
    by default, and on many systems the bell sound is switched off entirely. A
    run sat at a CAPTCHA in silence for five minutes and failed, with someone at
    the machine the whole time.

    So on Windows this plays an actual two-tone chime through ``winsound``, which
    does not depend on any terminal setting, and falls back to the bell elsewhere.
    Never raises: no alert is worth interrupting a run for.
    """
    try:
        if os.name == "nt":
            import winsound

            for _ in range(max(1, times)):
                winsound.Beep(880, 180)   # A5
                winsound.Beep(1320, 180)  # E6 - a rising pair carries across a room
            return
    except Exception:
        pass
    try:
        sys.stdout.write("\a" * max(1, times))
        sys.stdout.flush()
    except Exception:
        pass

_SYSTEM_PROMPT = """\
You are an exploratory QA engineer driving a real web browser to execute one test case.

You will be shown the current page as a list of interactive elements. Each has a
ref like [e12]. You act by replying with EXACTLY ONE JSON object and nothing else
— no prose, no markdown fence, no explanation outside the JSON.

Available actions:
{actions}

Every reply must have this shape:
  {{"thought": "<one short sentence on why>", "action": "<name>", ...action fields}}

Rules:
- One action per reply. You will see the result and the new page before the next one.
- Only use a ref that appears in the CURRENT observation. Refs change every turn.
- Read fields back after typing into them; a page can silently discard a value.
- If an action produced no visible change, do something DIFFERENT — never repeat it.
- Before judging the outcome, look at the page where the result would be visible.
- Running the test once gives you the answer. If you have entered a value, left,
  and come back, what you now see IS the result - do not run the sequence again
  to "make sure the procedure worked". An outcome you dislike is still an outcome.
- Call finish as soon as you can judge the expected result, with success true or
  false and a specific reason naming what you actually observed.
- A pass requires EVERY assertion and prerequisite in the test. If a prerequisite
  is missing, finish with success=false and 'Precondition not met'. Never redefine
  the objective or report a partial check as a pass. If an expected result is
  impossible to observe with your tools, report it as blocked, not successful.
"""


@dataclass
class AgentResult:
    """Outcome of one agent run. Mirrors mobilerun's ResultEvent contract."""

    success: bool
    reason: str
    steps: int
    history: list[str] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)   # ordered, de-duplicated route trace


class WebAgent:
    def __init__(self, page, cfg, client: llm_mod.ChatClient):
        self.page = page
        self.cfg = cfg
        self.client = client
        self.reviewer = None
        if isinstance(client, llm_mod.ChatClient) and getattr(cfg, "WEB_VERIFY_VERDICTS", True):
            reviewer_model = getattr(cfg, "EVALUATOR_MODEL", "") if client.provider == "openrouter" else client.model
            self.reviewer = llm_mod.ChatClient(cfg, model=reviewer_model or client.model, response_schema=review.SCHEMA)
        self.dispatcher = actions_mod.Dispatcher(page, cfg)
        # Progress so far, readable after an exception unwinds the loop. Without
        # these the crash path logged 0 steps and an empty route for a test that
        # had really taken ten.
        self.last_step = 0
        self.last_urls: list[str] = []
        # Known API routes, so the agent works from the real list instead of
        # inventing one. A planner-written objective once cited GET /api/projects,
        # which does not exist; the executor chased it until the test died.
        self.api_registry = None
        # Native dialog text, handed over by the oracle each turn. Without it the
        # only feedback an alert() gives is invisible, and the agent reads a
        # successful action as one that changed nothing.
        self.collector = None
        # Set by the runner from the live BrowserSession. None means "ask config".
        self.headless: bool | None = None

    async def run(self, goal: str, max_steps: int, timeout_s: float) -> AgentResult:
        self.last_step = 0
        self.last_urls = []
        self.exploration = create_recorder(self.cfg)
        try:
            return await asyncio.wait_for(self._run(goal, max_steps, timeout_s), timeout=timeout_s)
        except asyncio.TimeoutError:
            return AgentResult(False, f"Timed out after {timeout_s:.0f}s at step {self.last_step}/{max_steps}.",
                               self.last_step, urls=self.last_urls)
        finally:
            if self.exploration is not None:
                self.exploration.close()

    async def _run(self, goal: str, max_steps: int, timeout_s: float) -> AgentResult:
        started = time.time()
        history: list[str] = []
        urls: list[str] = []
        # How many times each (page state, intended action) pair has come up.
        # Counting occurrences rather than consecutive repeats is deliberate: the
        # commonest real stall is an A-B-A-B oscillation — open a dialog, cancel
        # it, open it again — which a consecutive-only check never sees. One run
        # cycled Edit Survey / Cancel thirteen times and used its whole budget.
        seen: dict[str, int] = {}
        stall_signature, stall_count = None, 0
        stall_limit = getattr(self.cfg, "WEB_STALL_STEPS", 6)
        # An action that achieves nothing is recorded as "clicked X", which reads
        # like success, so the agent tries it again. A real run clicked a dead
        # "Preview" button five times for exactly that reason. These carry the page
        # state from before the last action, so the next observation can say on the
        # action's own history line whether it accomplished anything.
        pending = None
        dead_controls: list[str] = []
        cycles: dict[str, int] = {}
        # One nudge per DISTINCT loop, not one per test case. Capping it at one
        # per run was too blunt: an agent that circles briefly early on spends the
        # only nudge there, and a genuine stall later gets none. A real run burned
        # its first nudge on a search box at step 3, then looped
        # goto-/api/projects-then-back eleven times from step 30 with no prompt at
        # all. Bloat stays bounded because each loop is nudged once and history
        # lines are capped.
        nudged: set[str] = set()
        captcha_handled = False
        previous_observation = ""
        parse_errors = 0
        review_corrections = 0

        for step in range(1, max_steps + 1):
            if time.time() - started > timeout_s:
                return AgentResult(
                    False,
                    f"Timed out after {timeout_s:.0f}s at step {step - 1}/{max_steps}.",
                    step - 1, history, urls,
                )

            self.last_step = step
            snap = await snapshot.observe(self.page, self.cfg.WEB_SNAPSHOT_MAX_ELEMENTS)
            if snap.get("captcha") and not captcha_handled:
                captcha_handled = True
                snap, blocked = await self._handle_captcha(snap)
                if blocked:
                    if self.exploration is not None:
                        self.exploration.observe(snap)
                    trace.outcome(blocked, ok=False)
                    return AgentResult(False, blocked, step, history, urls)
            _track_url(urls, snap.get("url", ""))
            self.last_urls = urls
            observation = snapshot.render(snap)
            if previous_observation:
                history.append("Observed after the previous action: " + _evidence(snap))
            else:
                history.append("Initial baseline: " + _evidence(snap))
            previous_observation = observation
            if self.collector is not None:
                observation += "\nBROWSER DIAGNOSTICS (observed during this test): " + self.collector.findings.summary()
                dialogs = self.collector.take_dialogs()
                if dialogs:
                    snap["browser_dialogs"] = dialogs
                    history.append("Browser dialogs: " + "; ".join(dialogs))
                    nl = chr(10)
                    observation += (nl + nl + "BROWSER DIALOGS since your last "
                                    "action (already answered for you):" + nl
                                    + "  " + (nl + "  ").join(dialogs))
            if self.exploration is not None:
                self.exploration.observe(snap)
            content_signature = _content_signature(snap)

            if pending is not None:
                idx, before, label, act_sig = pending
                if content_signature == before:
                    history[idx] += ("   <-- NO VISIBLE CHANGE: the page was identical "
                                     "afterwards. This may be the expected validation result. "
                                     "Judge the objective before choosing another action.")
                    # An unchanged DOM is also the expected result of validation,
                    # focus, scrolling, copying a link, or opening another tab.
                    # Never permanently blacklist a control from this alone.
                    # Only an action PROVEN inert counts toward livelock. Counting
                    # every repeat punished legitimate revisiting: a draft-retention
                    # test must open a URL, navigate away and open it again, and the
                    # guard told the agent mid-test that it was going in circles.
                    times = seen[act_sig] = seen.get(act_sig, 0) + 1
                    if times >= _LIVELOCK_ABORT:
                        trace.outcome("LIVELOCK - an action that changes nothing has "
                                      "been repeated", ok=False)
                        return AgentResult(
                            False,
                            f"Livelock: an action that changes nothing has been repeated "
                            f"{times} times. The loop is not making progress.",
                            step, history, urls,
                        )
                    if times == _LIVELOCK_WARN:
                        history.append(
                            "WARNING: you have now repeated an action that does nothing. "
                            "That control or route is inert - take a genuinely different "
                            "one, or finish and report that it does not work."
                        )
                pending = None

            messages = self._messages(goal, history, observation, step, max_steps, dead_controls)
            remaining = max(0.1, timeout_s - (time.time() - started))
            kwargs = {"timeout_s": remaining} if isinstance(self.client, llm_mod.ChatClient) else {}
            reply = await asyncio.to_thread(self.client.chat, messages, **kwargs)
            action = llm_mod.parse_action(reply)
            trace.step(step, max_steps, snap, action)
            trace.observation(snap)

            if action.get("action") == "finish":
                if self.reviewer is not None:
                    remaining = max(0.1, timeout_s - (time.time() - started))
                    checked = review.parse(await asyncio.to_thread(
                        self.reviewer.chat, review.messages(goal, history, observation, action), timeout_s=remaining))
                    if checked is None:
                        return AgentResult(False, "Verdict unverified: reviewer returned invalid evidence assessment.", step, history, urls)
                    trace.outcome("Evidence review: " + checked["decision"] + " - " + checked["reason"])
                    if checked["decision"] == "blocked":
                        return AgentResult(False, "Precondition not met: " + checked["reason"], step, history, urls)
                    if checked["decision"] == "continue":
                        review_corrections += 1
                        if review_corrections > 2:
                            return AgentResult(False, "Verdict unverified: " + checked["reason"], step, history, urls)
                        history.append("EVIDENCE REVIEW: your verdict was premature. " + checked["reason"])
                        continue
                success = action.get("success") is True
                reason = str(action.get("reason") or "no reason given").strip()
                history.append(f"finish(success={success}): {reason}")
                trace.outcome(f"FINISH success={success}: {reason}", ok=success)
                return AgentResult(success, reason, step, history, urls)

            if action.get("action") == "_error":
                parse_errors += 1
                if parse_errors >= 3:
                    return AgentResult(False, "Model backend returned three invalid JSON actions; no reliable action can be executed.",
                                       step, history, urls)
                history.append(
                    f"step {step}: your last reply was not a JSON action and was "
                    f"discarded ({action.get('reason', '')}). Reply with ONE JSON "
                    f"object and nothing else."
                )
                trace.outcome(f"model reply was not a JSON action — reprompting "
                              f"| {action.get('reason', '')}"[:300], ok=False)
                continue
            parse_errors = 0

            # Identifies (page state, action). Counted two ways, deliberately:
            #
            #  * above, once the next observation PROVES the action inert - precise,
            #    warns the agent, and never fires on useful work;
            #  * here, as a raw cycle count - a pure safety net with a high ceiling
            #    and NO advice injected into the history. An earlier version warned
            #    at the second repeat and aborted at the fourth, which killed correct
            #    tests: a draft-retention test must open a URL, leave, and open it
            #    again, and it was being told mid-test that it was going in circles.
            #    Legitimate revisiting needs two or three; the pathological run that
            #    motivated this cycled thirteen times.
            signature = _signature(snap, action)
            cycles[signature] = cycles.get(signature, 0) + 1
            if cycles[signature] == _CYCLE_CONCLUDE and signature not in nudged:
                nudged.add(signature)
                # The decisive fix for the commonest stall. A test like "type a
                # value, leave without saving, come back and see whether it is
                # still there" ANSWERS ITSELF on the second pass: the field is
                # empty again, and that is the finding. The agent instead read its
                # own repetition as "the procedure did not take" and started over,
                # six times. Repetition here is evidence, not failure - so say so
                # before any guard treats it as a stall.
                history.append(
                    "NOTE: you have now done this twice and the page is back to a "
                    "state you already saw. If that repeated result IS what your "
                    "test was checking (a field empty again, a value that did or "
                    "did not survive), that is your evidence - call finish and "
                    "report it. Doing it again cannot tell you anything new."
                )
                history.append(
                    "If you are searching for a feature instead: this route has already been "
                    "checked. Do not reopen the same dialog or repeat this route. Choose a "
                    "different visible control, or finish with 'Precondition not met' and "
                    "name the feature you could not reach."
                )
                trace.outcome("nudge: this repetition may already be the answer", ok=True)
            if cycles[signature] >= _CYCLE_ABORT:
                trace.outcome(f"LIVELOCK - the same action on the same page for the "
                              f"{cycles[signature]}th time", ok=False)
                return AgentResult(
                    False,
                    f"Livelock: the same action on the same page state has now been "
                    f"taken {cycles[signature]} times without reaching a verdict.",
                    step, history, urls,
                )

            # Wandering guard: the page itself (not the action) hasn't changed for
            # several steps in a row, even though each action tried was different —
            # e.g. scrolling repeatedly with no new content. The exact-repeat guard
            # above misses this because it keys on (page, action) together.
            stall_count = stall_count + 1 if content_signature == stall_signature else 0
            stall_signature = content_signature
            if stall_count >= stall_limit:
                trace.outcome(
                    f"WANDERING — {stall_count} different actions in a row, unchanged page", ok=False)
                return AgentResult(
                    False,
                    f"Livelock: tried {stall_count} different actions in a row and the "
                    "page never changed. The page is not responding to this line of "
                    "exploration.",
                    step, history, urls,
                )
            if stall_count == max(1, stall_limit - 2):
                history.append(
                    "WARNING: the last few actions were all different, but the page "
                    "has not changed at all. Whatever you are trying is not working — "
                    "try a completely different control, or finish and report why."
                )
                trace.outcome("warning: several different actions, unchanged page", ok=False)

            if self.exploration is not None:
                self.exploration.attempted(action, snap)
            try:
                note = await self.dispatcher.perform(action, snap)
                history.append(f"step {step}: {note}")
                trace.outcome(note)
                element = snapshot.find(snap, action.get("ref", "")) or {}
                pending = (len(history) - 1, content_signature,
                           element.get("name", ""), signature)
            except actions_mod.ActionError as exc:
                if self.exploration is not None:
                    self.exploration.attempted(action, snap, exc.category)
                history.append(f"step {step}: FAILED — {exc}")
                trace.outcome(f"REFUSED/FAILED — {exc}", ok=False)
                if exc.category == "BLOCKED_BY_GUARDRAIL":
                    # Not negotiable, and not worth spending the rest of the
                    # budget discovering that it is still not negotiable.
                    return AgentResult(False, str(exc), step, history, urls)
            except Exception as exc:
                if self.exploration is not None:
                    self.exploration.attempted(action, snap, type(exc).__name__)
                history.append(f"step {step}: FAILED — {type(exc).__name__}: {exc}")
                trace.outcome(f"FAILED — {type(exc).__name__}: {str(exc)[:160]}", ok=False)

        trace.outcome(f"STEP LIMIT — used all {max_steps} steps without a verdict", ok=False)
        return AgentResult(
            False,
            f"Step limit reached: used all {max_steps} steps without reaching a verdict.",
            max_steps, history, urls,
        )

    async def _handle_captcha(self, snap: dict) -> tuple[dict, str]:
        """A CAPTCHA is the one obstacle the agent must hand back to a person.

        The browser mode decides what that means, because it decides whether
        anyone is there to help:

        * **Headed** - a window is open, so ask. The run waits, watching for the
          challenge to disappear, and carries straight on once it is solved. This
          is the only way to reach anything behind the challenge until the
          application itself is configured with reCAPTCHA test keys.
        * **Headless** - nobody is watching a window that does not exist, so
          waiting would burn the per-test timeout and end in the same place.
          Fail the test immediately, naming the CAPTCHA.

        Returns ``(snapshot, blocked_reason)``; a non-empty reason ends the test.
        """
        headless = self.headless
        if headless is None:
            headless = bool(getattr(self.cfg, "WEB_HEADLESS", True))
        trace.emit(f"      CAPTCHA detected (browser headless={headless})")
        if headless:
            return snap, (
                "Blocked by CAPTCHA: a human-verification challenge stands between "
                "the agent and this behaviour, and the run is headless so nobody can "
                "solve it. Re-run with WEB_HEADLESS=false to solve it by hand, or "
                "configure reCAPTCHA test keys in the test environment."
            )

        limit = getattr(self.cfg, "WEB_CAPTCHA_PAUSE_SECONDS", 0) or _CAPTCHA_WAIT_DEFAULT
        trace.emit("")
        trace.emit("      " + "=" * 62)
        trace.emit("      CAPTCHA - YOUR INPUT IS NEEDED")
        trace.emit("      Solve the challenge in the browser window that is open.")
        trace.emit(f"      The run resumes by itself the moment it clears "
                   f"(waiting up to {limit}s).")
        trace.emit("      " + "=" * 62)
        _alert(3)

        waited = 0
        next_reminder = _CAPTCHA_REMIND_SECONDS
        while waited < limit:
            try:
                await self.page.wait_for_timeout(_CAPTCHA_POLL_MS)
            except Exception:
                break
            waited += _CAPTCHA_POLL_MS / 1000
            fresh = await snapshot.observe(self.page, self.cfg.WEB_SNAPSHOT_MAX_ELEMENTS)
            if not fresh.get("captcha"):
                trace.emit(f"      solved after {waited:.0f}s - continuing")
                _alert(1)  # one short chirp: you are free to walk away again
                return fresh, ""
            if waited >= next_reminder:
                # Keep nagging. A single beep at the start is easy to miss if
                # you were out of the room, and five minutes of silence looks
                # exactly like a run that is merely slow.
                remaining = int(limit - waited)
                trace.emit(f"      still waiting for the CAPTCHA - {remaining}s left")
                _alert(2)
                next_reminder = waited + _CAPTCHA_REMIND_SECONDS

        return snap, (
            f"Blocked by CAPTCHA: the challenge was still unsolved after {limit}s. "
            f"Configure reCAPTCHA test keys in the test environment, or raise "
            f"WEB_CAPTCHA_PAUSE_SECONDS."
        )

    def _messages(self, goal: str, history: list[str], observation: str,
                  step: int, max_steps: int, dead: list[str] | None = None) -> list[dict]:
        recent = [h if len(h) <= 600 else h[:597] + "..." for h in history[-24:]]
        log = "\n".join(recent) if recent else "(nothing yet — this is your first action)"
        if history and history[0].startswith("Initial baseline:") and len(history) > 24:
            log = history[0][:1200] + "\n" + log
        if dead:
            # Refs change every turn, so these are remembered by label. Without it
            # the agent forgets a control was inert as soon as the note scrolls out
            # of the last-12 window, and goes straight back to it.
            nl = chr(10)
            listed = ", ".join('"%s"' % d for d in dead[-8:])
            log += (nl + nl + "CONTROLS THAT DO NOTHING on this site - "
                    "never activate these again, find another route: " + listed)
        if self.api_registry is not None:
            block = self.api_registry.prompt_block()
            if block:
                log += chr(10) + chr(10) + block
        return [
            {"role": "system",
             "content": _SYSTEM_PROMPT.format(actions=actions_mod.ACTION_SPEC)},
            {"role": "user", "content": (
                f"## Test case goal\n{goal}\n\n"
                f"## What you have done so far\n{log}\n\n"
                f"## Current page\n{observation}\n\n"
                f"You are on step {step} of at most {max_steps}. Reply with one JSON action."
            )},
        ]


def _track_url(urls: list[str], url: str) -> None:
    """Append the URL only when it actually changed — a route trace, not a log."""
    if url and (not urls or urls[-1] != url):
        urls.append(url)


def _signature(snap: dict, action: dict) -> str:
    """Hash of (page contents, intended action), for repeat detection only.

    This is NOT a persisted state identity — the Live App Model is deliberately
    out of scope for the web player. It lives and dies inside one agent run.
    """
    kind = action.get("action")
    fields = {"action": kind, **{k: action[k] for k in actions_mod.ACTION_FIELDS.get(kind, ()) if k in action}}
    ref = fields.pop("ref", None)
    if ref:
        el = snapshot.find(snap, ref) or {}
        fields["target"] = [el.get("role", ""), el.get("name", ""), el.get("href", "")]
        # Equal labels may name different rows. Preserve their relative occurrence,
        # while ignoring ref numbers that shift when unrelated elements appear.
        peers = [e for e in snap.get("elements", []) if
                 (e.get("role"), e.get("name"), e.get("href")) ==
                 (el.get("role"), el.get("name"), el.get("href"))]
        fields["target"].append(next((i for i, e in enumerate(peers) if e.get("ref") == ref), 0))
    return _content_signature(snap) + "||" + json.dumps(fields, sort_keys=True, ensure_ascii=False)


def _content_signature(snap: dict) -> str:
    """Hash of the page's content alone, independent of what action is next.

    Used for the wandering guard: several different actions in a row that never
    change this signature mean the agent is trying different things against a
    page that isn't responding to any of them.
    """
    page = snap.get("url", "") + "|" + "|".join(
        f"{e.get('role')}{e.get('name')}{e.get('value', '')}{e.get('disabled', False)}" for e in snap.get("elements") or []
    ) + "|" + "|".join(snap.get("messages") or []) + "|" + "|".join(snap.get("texts") or [])
    page += "|" + "|".join(snap.get("headings") or []) + "|" + str(snap.get("dialog_open", False))
    page += "|" + "|".join(snap.get("browser_dialogs") or [])
    return hashlib.sha1(page.encode("utf-8", "replace")).hexdigest()


def _evidence(snap):
    fields = [f"{e.get('name', '')}={e.get('value', '')!r}" for e in snap.get("elements", [])
              if "value" in e and e.get("role") != "password"]
    return ("URL=" + snap.get("url", "") + "; " + "; ".join(
        (snap.get("messages") or []) + (snap.get("headings") or []) + fields[:8] + (snap.get("texts") or [])[:4]))[:1200]
