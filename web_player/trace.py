"""Live run trace — what the agent is doing, as it does it.

The Android side tees mobilerun's agent chatter to ``logs/mobilerun.log`` so a
run is observable while it happens. The web agent had no equivalent: it ran for
minutes inside one ``agent.run()`` call and printed nothing until it was over, so
a slow run and a hung run looked identical.

Everything here goes to two places at once:

  * **stdout**, so you can watch a run in the terminal
  * **``logs/web_player.log``**, so you can ``tail -f`` it from elsewhere, keep it
    after the terminal is gone, and grep a finished run

Deliberately not the ``logging`` module: this is a transcript meant for a human
to read in order, not levelled diagnostics, and mixing it into the root logger
would interleave it with library noise.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(_ROOT, "logs", "web_player.log")

_fh = None


def _handle():
    """Open the log file lazily, and never let logging break a run."""
    global _fh
    if _fh is None:
        try:
            os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
            _fh = open(LOG_PATH, "a", encoding="utf-8", errors="replace")
        except Exception:
            _fh = False  # tried and failed; don't retry on every line
    return _fh or None


def emit(text: str = "") -> None:
    """Print a line and append it to the run log."""
    print(text)
    fh = _handle()
    if not fh:
        return
    try:
        stamp = datetime.now().strftime("%H:%M:%S")
        for line in str(text).splitlines() or [""]:
            fh.write(f"{stamp} {line}\n")
        fh.flush()
    except Exception:
        pass


def header(text: str) -> None:
    emit("\n" + "=" * 72)
    emit(text)
    emit("=" * 72)


def run_start(tc_id: str, title: str, url: str) -> None:
    emit(f"▶ {tc_id} — {title}")
    emit(f"  start: {url}")


def step(n: int, max_steps: int, snap: dict, action: dict) -> None:
    """One agent turn: where it is, what it decided, and why.

    The model's own ``thought`` is the single most useful line in the whole
    trace — it is the difference between watching an exploration and watching a
    cursor move.
    """
    url = _short(snap.get("url", ""), 70)
    thought = str(action.get("thought") or "").strip()
    emit(f"\n  [{n}/{max_steps}] {url}")
    if thought:
        emit(f"      think: {thought}")
    emit(f"      act:   {_compact(action)}")


def observation(snap: dict) -> None:
    """One line on what the page looks like, without dumping the whole snapshot."""
    bits = [f"{len(snap.get('elements') or [])} controls"]
    if snap.get("dialog_open"):
        bits.append("modal open")
    if snap.get("messages"):
        bits.append(f"messages: {' | '.join(snap['messages'][:2])}")
    emit(f"      page:  {', '.join(bits)}")


def outcome(note: str, ok: bool = True) -> None:
    emit(f"      {'->' if ok else '!!'}     {note}")


def _compact(action: dict) -> str:
    """The action without the thought — the thought is already on its own line."""
    payload = {k: v for k, v in action.items() if k != "thought"}
    try:
        return json.dumps(payload, ensure_ascii=False)[:200]
    except (TypeError, ValueError):
        return str(payload)[:200]


def _short(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[: limit - 1] + "…"
