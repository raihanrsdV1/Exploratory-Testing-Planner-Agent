"""The observe -> decide -> act loop.

One test case, one agent run. The agent sees a compact observation of the page,
picks a single action, and we perform it; that repeats until it calls ``finish``
or runs out of budget.

Two things this loop does that a naive version does not:

* **Only the current observation is sent in full.** Earlier turns are compressed
  to their one-line action notes. A page with 60 controls is ~1.5k tokens; ten of
  them in the history is most of a context window spent on pages the agent has
  already left.

* **It detects its own livelock.** The single most common agent failure is
  clicking the same control forever because nothing visibly changed. After a few
  identical no-op actions the agent is told so explicitly, and shortly after that
  the run is ended as NAVIGATION_LIVELOCK — an agent fault, not a defect. Without
  this the run burns its whole budget and then reports an ASSERTION_FAILURE,
  which would be counted as a discovered bug.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

from . import actions as actions_mod
from . import llm as llm_mod
from . import snapshot

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

    async def run(self, goal: str, max_steps: int, timeout_s: float) -> AgentResult:
        started = time.time()
        history: list[str] = []
        urls: list[str] = []
        repeat_signature, repeat_count = None, 0

        for step in range(1, max_steps + 1):
            if time.time() - started > timeout_s:
                return AgentResult(
                    False,
                    f"Timed out after {timeout_s:.0f}s at step {step - 1}/{max_steps}.",
                    step - 1, history, urls,
                )

            snap = await snapshot.observe(self.page, self.cfg.WEB_SNAPSHOT_MAX_ELEMENTS)
            _track_url(urls, snap.get("url", ""))
            observation = snapshot.render(snap)

            reply = self.client.chat(self._messages(goal, history, observation, step, max_steps))
            action = llm_mod.parse_action(reply)

            if action.get("action") == "finish":
                success = bool(action.get("success"))
                reason = str(action.get("reason") or "no reason given").strip()
                history.append(f"finish(success={success}): {reason}")
                return AgentResult(success, reason, step, history, urls)

            if action.get("action") == "_error":
                history.append(
                    f"step {step}: your last reply was not a JSON action and was "
                    f"discarded ({action.get('reason', '')}). Reply with ONE JSON "
                    f"object and nothing else."
                )
                continue

            # Livelock guard: identical action against an identical page.
            signature = _signature(snap, action)
            repeat_count = repeat_count + 1 if signature == repeat_signature else 0
            repeat_signature = signature
            if repeat_count >= 4:
                return AgentResult(
                    False,
                    "Livelock: repeated the same action against an unchanged page "
                    f"{repeat_count + 1} times. The page is not responding to it.",
                    step, history, urls,
                )
            if repeat_count == 2:
                history.append(
                    "WARNING: you have now repeated the same action on an unchanged "
                    "page. It is not working. Do something different, or finish and "
                    "report what the unchanged page means."
                )

            try:
                note = await self.dispatcher.perform(action, snap)
                history.append(f"step {step}: {note}")
            except actions_mod.ActionError as exc:
                history.append(f"step {step}: FAILED — {exc}")
                if exc.category == "BLOCKED_BY_GUARDRAIL":
                    # Not negotiable, and not worth spending the rest of the
                    # budget discovering that it is still not negotiable.
                    return AgentResult(False, str(exc), step, history, urls)
            except Exception as exc:
                history.append(f"step {step}: FAILED — {type(exc).__name__}: {exc}")

        return AgentResult(
            False,
            f"Step limit reached: used all {max_steps} steps without reaching a verdict.",
            max_steps, history, urls,
        )

    def _messages(self, goal: str, history: list[str], observation: str,
                  step: int, max_steps: int) -> list[dict]:
        recent = history[-12:]
        log = "\n".join(recent) if recent else "(nothing yet — this is your first action)"
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
    page = snap.get("url", "") + "|" + "|".join(
        f"{e.get('ref')}{e.get('name')}{e.get('value', '')}" for e in snap.get("elements") or []
    ) + "|" + "|".join(snap.get("messages") or []) + "|" + "|".join(snap.get("texts") or [])
    act = f"{action.get('action')}:{action.get('ref', '')}:{action.get('text', '')}"
    return hashlib.sha1(f"{page}||{act}".encode("utf-8", "replace")).hexdigest()
