"""
Validation for a proposed test case — the planner's terminal step.

The old pipeline generated a test and then patched it after the fact: a
duplicate triggered one blind regeneration, and two prompt rules ("never invent
a requirement id", "screen_hint should be a real name") were enforced by nothing
at all. Measured consequence: the planner's named screen matched a real observed
screen about 16% of the time, and the executor's goal text had to defensively
call it "a LEAD, not a fact" while the agent burned its step budget hunting for
screens that do not exist.

Here the same checks run BEFORE the test leaves the planner, and a failure is
returned to the model as a correctable message rather than a wasted device run.

Pure except for the graph reads it needs; no LLM calls.
"""

from __future__ import annotations

import settings as _settings

from . import rag_client, textutil

# Fields a dispatchable test case must carry. `steps` is deliberately NOT here:
# the executor works out HOW from the live screen (see build_droidrun_goal), and
# a planner-written tap script is usually wrong.
_REQUIRED = ("title", "objective", "expected_result")

# Accepted when the planner genuinely cannot ground a screen. Explicitly better
# than a confident guess: the executor is told to explore rather than to hunt
# for a name that does not exist.
_UNKNOWN_SCREEN = {"unknown", "", "n/a", "none", "any"}

_DUPLICATE_THRESHOLD = 0.90


def _observed_screens(project: str) -> list[str]:
    try:
        graph = rag_client.rag_get("/appmodel/graph", {"project": project})
    except Exception:
        return []
    nodes = [n for n in (graph.get("nodes") or []) if not n.get("stale")] or (graph.get("nodes") or [])
    return [str(n.get("label") or "") for n in nodes if n.get("label")]


def _citable_ids(project: str) -> set[str]:
    try:
        return set(rag_client.rag_get("/requirements/ids", {"project": project}).get("ref_ids") or [])
    except Exception:
        return set()


def validate(project: str, proposal: dict, done_titles: list[str] | None = None) -> tuple[bool, list[str]]:
    """Check a proposed test case. Returns (ok, errors).

    Each error is phrased as an instruction the model can act on in its next
    turn — naming the valid alternatives wherever possible, because "that screen
    does not exist" without the real list just invites a second guess.
    """
    errors: list[str] = []
    if not isinstance(proposal, dict):
        return False, ["The proposal must be a JSON object."]

    for field in _REQUIRED:
        if not str(proposal.get(field) or "").strip():
            errors.append(f"'{field}' is required and must be non-empty.")

    # ── screen_hint must name a screen that actually exists ──────────────────
    hint = str(proposal.get("screen_hint") or "").strip()
    observed = _observed_screens(project)
    if observed and hint.lower() not in _UNKNOWN_SCREEN:
        if not any(hint.lower() == s.lower() or hint.lower() in s.lower() or s.lower() in hint.lower()
                   for s in observed):
            errors.append(
                f"screen_hint '{hint}' is not a screen this app has been observed to have. "
                f"Call get_screen to check, then use one of: {observed[:12]} — "
                f"or set screen_hint to 'unknown' and let the executor explore."
            )

    # ── requirement ids must exist, or COVERS edges silently fail ────────────
    cited = proposal.get("requirement_ids")
    cited = [str(c).strip() for c in cited if str(c).strip()] if isinstance(cited, list) else []
    if cited:
        citable = _citable_ids(project)
        unknown = [c for c in cited if c not in citable]
        if citable and unknown:
            errors.append(
                f"These requirement ids do not exist: {unknown}. Call search_requirements or "
                f"list_untested_requirements to get real ids, or use an empty list."
            )

    # ── out-of-scope areas override everything, including coverage pressure ──
    if _settings.OUT_OF_SCOPE:
        hay = " ".join(str(proposal.get(f) or "") for f in
                       ("title", "objective", "expected_result", "area", "screen_hint")).lower()
        hay += " " + " ".join(str(p) for p in (proposal.get("preconditions") or [])).lower()
        for area in _settings.OUT_OF_SCOPE:
            if area and area.lower() in hay:
                errors.append(
                    f"This test requires '{area}', which is OUT OF SCOPE for this agent "
                    f"(it needs something outside the agent's control). Treat it as already "
                    f"satisfied and test what it unlocks instead."
                )
                break

    # ── preconditions must be creatable through the UI in this test's own steps
    for pre in (proposal.get("preconditions") or []):
        low = str(pre).lower()
        if any(k in low for k in _settings.UNACHIEVABLE_PRECONDITIONS):
            errors.append(
                f"Precondition '{str(pre)[:80]}' cannot be established by the tester — it asserts "
                f"a state that can only be observed, not created. Rewrite it so the test's own "
                f"first steps create what it needs."
            )
            break

    # ── semantic duplicate of an already-executed test ───────────────────────
    title = str(proposal.get("title") or "").strip()
    if title:
        if textutil.is_similar_to_existing(title, list(done_titles or []), threshold=0.60):
            errors.append(
                f"'{title}' is too similar to a test that has already been executed. "
                f"Call list_findings to see what is already established and choose a "
                f"different behaviour, rule or screen."
            )
        else:
            try:
                dupe = rag_client.semantic_dedup_check(project, title, threshold=_DUPLICATE_THRESHOLD)
                if dupe.get("is_duplicate"):
                    errors.append(
                        f"'{title}' is {round(dupe.get('similarity', 0) * 100)}% similar to the "
                        f"existing test '{dupe.get('most_similar_title', '')}'. Choose a "
                        f"materially different behaviour."
                    )
            except Exception:
                pass  # dedup is best-effort; never block a proposal on an outage

    return (not errors), errors


def normalize(proposal: dict) -> dict:
    """Coerce a validated proposal into the exact dict the executor consumes."""
    out = {
        "title": str(proposal.get("title") or "").strip(),
        "screen_hint": str(proposal.get("screen_hint") or "").strip(),
        "objective": str(proposal.get("objective") or "").strip(),
        "expected_result": str(proposal.get("expected_result") or "").strip(),
        "area": str(proposal.get("area") or "general").strip() or "general",
        "priority": str(proposal.get("priority") or "medium").strip(),
        "test_type": str(proposal.get("test_type") or "").strip(),
        "rationale": str(proposal.get("rationale") or "").strip(),
        "preconditions": [str(p) for p in (proposal.get("preconditions") or [])][:6],
        "requirement_ids": [str(r).strip() for r in (proposal.get("requirement_ids") or [])
                            if str(r).strip()][:6],
    }
    if out["screen_hint"].lower() in _UNKNOWN_SCREEN:
        out["screen_hint"] = ""
    return out
