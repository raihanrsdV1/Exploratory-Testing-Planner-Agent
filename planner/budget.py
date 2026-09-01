"""
Global prompt budget with per-block priority.

Replaces the scattered per-block magic-number caps with one explicit token
budget. Blocks are filled highest-priority-first; when the budget runs out the
*lowest*-value content is truncated or dropped, instead of every block being
clipped to an arbitrary local limit regardless of how much room is actually
left.

Why a budget at all, given a 1M-token window: cost scales linearly with prompt
size on every call, and long-context models measurably degrade at retrieving
facts buried mid-prompt. So the aim is not "use the whole window" but "never
drop something important while something unimportant is still in".

Priority 0 blocks are never dropped — if the budget cannot hold them it is
simply exceeded, and that is reported in the accounting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Rough chars-per-token for English prose + JSON. Good enough for budgeting;
# we never need exact token counts, only proportions.
CHARS_PER_TOKEN = 4


@dataclass
class Block:
    """One prompt section competing for budget."""

    name: str
    text: str
    priority: int = 2       # 0 = never drop, higher = dropped/truncated first
    min_chars: int = 240    # below this a truncated block is worthless — drop instead
    truncatable: bool = True


@dataclass
class BudgetReport:
    """What the budgeter actually did — surfaced in debug_trace."""

    budget_chars: int
    used_chars: int
    kept: list[str] = field(default_factory=list)
    truncated: list[tuple[str, int, int]] = field(default_factory=list)  # name, from, to
    dropped: list[str] = field(default_factory=list)

    @property
    def used_tokens(self) -> int:
        return self.used_chars // CHARS_PER_TOKEN

    def as_dict(self) -> dict:
        return {
            "budget_chars": self.budget_chars,
            "budget_tokens": self.budget_chars // CHARS_PER_TOKEN,
            "used_chars": self.used_chars,
            "used_tokens": self.used_tokens,
            "kept": self.kept,
            "truncated": [{"block": n, "from": a, "to": b} for n, a, b in self.truncated],
            "dropped": self.dropped,
        }


def _truncate(text: str, limit: int) -> str:
    """Clip on a line boundary where possible so a block never ends mid-fact."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    nl = cut.rfind("\n")
    if nl > limit * 0.6:  # only honour the line break if we keep most of the budget
        cut = cut[:nl]
    return cut.rstrip() + "\n… (truncated to fit prompt budget)"


def fit(blocks: list[Block], budget_tokens: int) -> tuple[dict[str, str], BudgetReport]:
    """Fit blocks into a token budget, dropping lowest-priority content first.

    Returns ``({name: text}, report)``. Blocks that were dropped are absent from
    the mapping, so callers can treat "missing" and "empty" identically.
    """
    budget_chars = max(1, budget_tokens) * CHARS_PER_TOKEN
    report = BudgetReport(budget_chars=budget_chars, used_chars=0)
    out: dict[str, str] = {}

    # Priority 0 first (never dropped), then ascending priority. Within the same
    # priority, smaller blocks first so one huge block cannot starve several small
    # high-value ones.
    ordered = sorted(
        [b for b in blocks if (b.text or "").strip()],
        key=lambda b: (b.priority, len(b.text)),
    )

    remaining = budget_chars
    for b in ordered:
        text = b.text
        size = len(text)

        if b.priority == 0:
            # Mandatory: always included, even if it overshoots the budget.
            out[b.name] = text
            remaining -= size
            report.used_chars += size
            report.kept.append(b.name)
            continue

        if size <= remaining:
            out[b.name] = text
            remaining -= size
            report.used_chars += size
            report.kept.append(b.name)
        elif b.truncatable and remaining >= b.min_chars:
            clipped = _truncate(text, remaining)
            out[b.name] = clipped
            report.used_chars += len(clipped)
            remaining -= len(clipped)
            report.truncated.append((b.name, size, len(clipped)))
        else:
            report.dropped.append(b.name)

    return out, report
