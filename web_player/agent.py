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
import time
from dataclasses import dataclass, field

from . import actions as actions_mod
from . import llm as llm_mod
from . import snapshot
from . import trace

# Occurrences of one (page, action) pair before warning, then before giving up.
_LIVELOCK_WARN = 2
_LIVELOCK_ABORT = 4
# Raw cycle ceiling: high enough that deliberate revisiting is never
# mistaken for a stall, low enough to stop a loop long before the budget.
_CYCLE_ABORT = 6
# When a sequence comes round a second time, the repetition is usually the
# test's own answer rather than a stall. Nudge before any guard punishes it.
_CYCLE_CONCLUDE = 2

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
        self.dispatcher = actions_mod.Dispatcher(page, cfg)
        # Progress so far, readable after an exception unwinds the loop. Without
        # these the crash path logged 0 steps and an empty route for a test that
        # had really taken ten.
        self.last_step = 0
        self.last_urls: list[str] = []

    async def run(self, goal: str, max_steps: int, timeout_s: float) -> AgentResult:
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
        nudged = False
        captcha_handled = False

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
                snap = await self._offer_human_solve(snap)
            _track_url(urls, snap.get("url", ""))
            self.last_urls = urls
            observation = snapshot.render(snap)
            content_signature = _content_signature(snap)

            if pending is not None:
                idx, before, label, act_sig = pending
                if content_signature == before:
                    history[idx] += ("   <-- THIS DID NOTHING: the page was identical "
                                     "afterwards. Do not repeat it; take another route.")
                    if label and label not in dead_controls:
                        dead_controls.append(label)
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

            reply = self.client.chat(
                self._messages(goal, history, observation, step, max_steps, dead_controls))
            action = llm_mod.parse_action(reply)
            trace.step(step, max_steps, snap, action)
            trace.observation(snap)

            if action.get("action") == "finish":
                success = bool(action.get("success"))
                reason = str(action.get("reason") or "no reason given").strip()
                history.append(f"finish(success={success}): {reason}")
                trace.outcome(f"FINISH success={success}: {reason}", ok=success)
                return AgentResult(success, reason, step, history, urls)

            if action.get("action") == "_error":
                history.append(
                    f"step {step}: your last reply was not a JSON action and was "
                    f"discarded ({action.get('reason', '')}). Reply with ONE JSON "
                    f"object and nothing else."
                )
                trace.outcome(f"model reply was not a JSON action — reprompting "
                              f"| {action.get('reason', '')}"[:300], ok=False)
                continue

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
            if cycles[signature] == _CYCLE_CONCLUDE and not nudged:
                nudged = True
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

            try:
                note = await self.dispatcher.perform(action, snap)
                history.append(f"step {step}: {note}")
                trace.outcome(note)
                element = snapshot.find(snap, action.get("ref", "")) or {}
                pending = (len(history) - 1, content_signature,
                           element.get("name", ""), signature)
            except actions_mod.ActionError as exc:
                history.append(f"step {step}: FAILED — {exc}")
                trace.outcome(f"REFUSED/FAILED — {exc}", ok=False)
                if exc.category == "BLOCKED_BY_GUARDRAIL":
                    # Not negotiable, and not worth spending the rest of the
                    # budget discovering that it is still not negotiable.
                    return AgentResult(False, str(exc), step, history, urls)
            except Exception as exc:
                history.append(f"step {step}: FAILED — {type(exc).__name__}: {exc}")
                trace.outcome(f"FAILED — {type(exc).__name__}: {str(exc)[:160]}", ok=False)

        trace.outcome(f"STEP LIMIT — used all {max_steps} steps without a verdict", ok=False)
        return AgentResult(
            False,
            f"Step limit reached: used all {max_steps} steps without reaching a verdict.",
            max_steps, history, urls,
        )

    async def _offer_human_solve(self, snap: dict) -> dict:
        """Pause for a person to solve a CAPTCHA, when someone is actually watching.

        Off by default. An unattended batch must never block on a human: it would
        hang until the per-test timeout and teach us nothing. With a headed browser
        and a person present, though, solving it by hand is far better than
        abandoning the test.
        """
        seconds = getattr(self.cfg, "WEB_CAPTCHA_PAUSE_SECONDS", 0)
        if seconds <= 0 or getattr(self.cfg, "WEB_HEADLESS", True):
            trace.emit("")
            trace.emit("      CAPTCHA on the page - the agent cannot solve it. "
                       "Set WEB_CAPTCHA_PAUSE_SECONDS (with WEB_HEADLESS=false) "
                       "to solve it by hand, or use reCAPTCHA test keys.")
            return snap
        trace.emit("")
        trace.emit(f"      CAPTCHA on the page - PAUSING {seconds}s for you to solve "
                   f"it in the browser window.")
        try:
            await self.page.wait_for_timeout(seconds * 1000)
        except Exception:
            return snap
        fresh = await snapshot.observe(self.page, self.cfg.WEB_SNAPSHOT_MAX_ELEMENTS)
        trace.emit("      resumed - CAPTCHA still present: %s" % bool(fresh.get("captcha")))
        return fresh

    def _messages(self, goal: str, history: list[str], observation: str,
                  step: int, max_steps: int, dead: list[str] | None = None) -> list[dict]:
        recent = [h if len(h) <= 320 else h[:317] + "..." for h in history[-12:]]
        log = "\n".join(recent) if recent else "(nothing yet — this is your first action)"
        if dead:
            # Refs change every turn, so these are remembered by label. Without it
            # the agent forgets a control was inert as soon as the note scrolls out
            # of the last-12 window, and goes straight back to it.
            nl = chr(10)
            listed = ", ".join('"%s"' % d for d in dead[-8:])
            log += (nl + nl + "CONTROLS THAT DO NOTHING on this site - "
                    "never activate these again, find another route: " + listed)
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
    return f"{_content_signature(snap)}||{action.get('action')}:{action.get('ref', '')}:{action.get('text', '')}"


def _content_signature(snap: dict) -> str:
    """Hash of the page's content alone, independent of what action is next.

    Used for the wandering guard: several different actions in a row that never
    change this signature mean the agent is trying different things against a
    page that isn't responding to any of them.
    """
    page = snap.get("url", "") + "|" + "|".join(
        f"{e.get('ref')}{e.get('name')}{e.get('value', '')}" for e in snap.get("elements") or []
    ) + "|" + "|".join(snap.get("messages") or []) + "|" + "|".join(snap.get("texts") or [])
    return hashlib.sha1(page.encode("utf-8", "replace")).hexdigest()
