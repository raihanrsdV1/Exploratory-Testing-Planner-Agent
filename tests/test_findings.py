#!/usr/bin/env python3
"""Findings storage: a restatement must reinforce, never duplicate.

The investigator used to emit one long prose report per run, and every later
prompt re-read the whole thing — 86k of one 98k-character evaluation was
previous reports. Atomic findings fix that only if two things hold: the same
observation phrased differently collapses onto ONE node (otherwise the graph
grows exactly like the prose did), and the kinds route apart (otherwise
'our agent got stuck' reaches the planner as evidence the app is broken).
These checks pin both.

Needs Neo4j. Skips cleanly (exit 0) when it is not reachable, so the suite still
runs on a machine with no database.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Stub agents in these tests emit real trace lines. Send them to a temp file
# so fixture runs never land in logs/web_player.log, which is the operator
# transcript and the dashboard's live feed.
import tempfile  # noqa: E402
os.environ.setdefault("WEB_TRACE_FILE",
                      os.path.join(tempfile.gettempdir(), "web_player_tests.log"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from rag_api import embeddings, findings as F  # noqa: E402

_passed = _failed = 0
PROJECT = "__findings_selftest__"
NOW = "2026-01-01T00:00:00+00:00"


def check(label, got, want):
    global _passed, _failed
    ok = got == want
    _passed, _failed = _passed + ok, _failed + (not ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + ("" if ok else f"  (got {got!r}, want {want!r})"))


def _embed(texts):
    try:
        vecs = embeddings.embed_texts(texts)
        return vecs if vecs else [None] * len(texts)
    except Exception:
        return [None] * len(texts)


def _wipe(session):
    session.run("MATCH (p:Project {name:$p})-[:HAS_FINDING]->(f:Finding) DETACH DELETE f", p=PROJECT)
    session.run("MATCH (p:Project {name:$p}) DETACH DELETE p", p=PROJECT)


def main():
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI") or "neo4j://127.0.0.1:7687",
            auth=(os.getenv("NEO4J_USER") or "neo4j", os.getenv("NEO4J_PASSWORD") or "hihi"))
        driver.verify_connectivity()
    except Exception as e:
        print(f"  [SKIP] Neo4j unreachable ({str(e)[:80]}) — 0/0 checks passed")
        return 0

    print("taxonomy is closed — an invented kind can never reach a prompt block")
    check("known kind survives", F.normalize_kind("spec_violation"), "SPEC_VIOLATION")
    check("hyphen/space forms normalise", F.normalize_kind("agent difficulty"), "AGENT_DIFFICULTY")
    check("unknown kind coerced, not dropped", F.normalize_kind("totally_made_up"), "UNEXPECTED_BEHAVIOUR")
    check("empty kind coerced", F.normalize_kind(""), "UNEXPECTED_BEHAVIOUR")
    check("every routed kind is in the taxonomy",
          (F.ORACLE_KINDS | F.UI_KINDS | F.AGENT_KINDS) <= set(F.KINDS), True)
    check("agent difficulty is NOT in the bug oracle",
          bool(F.AGENT_KINDS & F.ORACLE_KINDS), False)
    check("control discoveries are NOT in the bug oracle",
          bool(F.UI_KINDS & F.ORACLE_KINDS), False)

    with driver.session() as s:
        _wipe(s)

        print("\na run's findings land as separate, addressable nodes")
        batch = [
            {"claim": "Saving Farm Info with an empty farm name shows a success message and no validation error",
             "kind": "SPEC_VIOLATION", "screen": "Farm Details Update",
             "evidence": "steps 12-14: cleared name, tapped save, success toast shown",
             "severity": "high", "requirement_ids": ["FR-FARM-04"]},
            {"claim": "The Chats screen shows 'No chats yet' when the account has no conversations",
             "kind": "CONFIRMED_BEHAVIOUR", "screen": "Chats", "evidence": "steps 1-6 empty state visible"},
            {"claim": "Bottom navigation items are ImageView controls rather than labelled buttons",
             "kind": "CONTROL_DISCOVERED", "screen": "Chats", "evidence": "step 1 click at (540,2269)"},
            {"claim": "The agent tapped the empty-state message three times with no visible effect",
             "kind": "AGENT_DIFFICULTY", "screen": "Chats", "evidence": "steps 3,4,5 identical taps"},
            {"claim": "", "kind": "SPEC_VIOLATION"},
        ]
        r1 = F.record(s, PROJECT, batch, log_id="", test_case_id="TC-X", embed_texts=_embed, now=NOW)
        check("four findings created, the claimless one dropped", (r1["created"], r1["reinforced"]), (4, 0))

        print("\nthe evaluator names what it is confirming, and that wins")
        # Cosine cannot do this job: measured on these very claims, a true
        # restatement scores 0.846 while an ADJACENT VARIANT ("over-long name"
        # vs "empty name") scores 0.853. The model saw the steps; it decides.
        spec = [f for f in F.query(s, PROJECT, kinds=["SPEC_VIOLATION"], limit=5)][0]
        restated = [
            {"claim": "Submitting the farm profile with a blank name field produces a success toast instead of an error",
             "kind": "SPEC_VIOLATION", "screen": "Farm Details Update",
             "evidence": "steps 20-22: same behaviour observed again", "confirms": spec["ref"]},
            {"claim": "The products list renders nothing at all when the seller has no products",
             "kind": "SUSPECTED_DEFECT", "screen": "My Products", "evidence": "step 9 blank body"},
        ]
        r2 = F.record(s, PROJECT, restated, log_id="", test_case_id="TC-Y", embed_texts=_embed, now=NOW)
        check("cited restatement reinforces; the new one is created",
              (r2["created"], r2["reinforced"]), (1, 1))
        check("the merge is attributed to the model, not the embedding",
              [f["matched_by"] for f in r2["findings"] if f["status"] == "reinforced"], ["model"])

        rows = F.query(s, PROJECT, limit=50)
        check("distinct findings after both runs", len(rows), 5)

        print("\nrepetition becomes a signal instead of duplicated text")
        top = rows[0]
        check("twice-observed finding ranks first", top["times_seen"], 2)
        check("both runs' evidence retained on the one node", len(top["evidence"]), 2)

        print("\na false ref is ignored rather than silently merging somewhere")
        r_bad = F.record(s, PROJECT, [{"claim": "A completely unrelated observation about the login screen",
                                       "kind": "SPEC_GAP", "screen": "Login", "confirms": "F-deadbeef"}],
                         log_id="", test_case_id="TC-B", embed_texts=_embed, now=NOW)
        check("unresolvable confirms falls through to create", r_bad["created"], 1)

        print("\nthe dangerous merge — a defect absorbed by its own passing case — cannot happen")
        # These two score 0.849 on cosine: high enough to merge under any
        # threshold that also catches real restatements. Only the group gate
        # stops the defect being erased from the oracle.
        r_pol = F.record(s, PROJECT, [
            {"claim": "Saving Farm Info with a valid farm name persists the value across an app restart",
             "kind": "CONFIRMED_BEHAVIOUR", "screen": "Farm Details Update", "evidence": "steps 1-8"}],
            log_id="", test_case_id="TC-P", embed_texts=_embed, now=NOW)
        check("confirmation does not absorb the defect", r_pol["created"], 1)
        check("the defect is still in the oracle",
              len(F.query(s, PROJECT, kinds=["SPEC_VIOLATION"], limit=10)), 1)

        print("\nkinds route apart, so one prompt block never inherits another's noise")
        check("bug oracle sees only oracle kinds",
              sorted({f["kind"] for f in F.query(s, PROJECT, kinds=list(F.ORACLE_KINDS), limit=50)}),
              ["CONFIRMED_BEHAVIOUR", "SPEC_GAP", "SPEC_VIOLATION", "SUSPECTED_DEFECT"])
        check("agent difficulty is retrievable on its own",
              [f["kind"] for f in F.query(s, PROJECT, kinds=list(F.AGENT_KINDS), limit=50)],
              ["AGENT_DIFFICULTY"])
        check("screen filter narrows to one screen's findings",
              len(F.query(s, PROJECT, screens=["Chats"], limit=50)), 3)
        check("unknown screen returns nothing rather than everything",
              len(F.query(s, PROJECT, screens=["No Such Screen"], limit=50)), 0)

        print("\nover-long model output is clamped, not rejected")
        long_batch = [{"claim": "x" * 900, "kind": "UNVERIFIED", "screen": "Chats", "evidence": "y" * 900}]
        F.record(s, PROJECT, long_batch, log_id="", test_case_id="TC-Z", embed_texts=_embed, now=NOW)
        stored = [f for f in F.query(s, PROJECT, kinds=["UNVERIFIED"], limit=5)]
        check("claim clamped to the field cap", len(stored[0]["claim"]), F.CLAIM_MAX)
        check("evidence clamped to the field cap", len(stored[0]["evidence"][0]), F.EVIDENCE_MAX)

        print("\na runaway batch cannot flood the graph")
        many = [{"claim": f"distinct observation number {i} about a control", "kind": "CONTROL_DISCOVERED",
                 "screen": "Chats", "evidence": f"step {i}"} for i in range(40)]
        r3 = F.record(s, PROJECT, many, log_id="", test_case_id="TC-W", embed_texts=_embed, now=NOW)
        check("batch truncated to the per-run guard",
              r3["created"] + r3["reinforced"], F._MAX_FINDINGS_PER_RUN)

        _wipe(s)
    driver.close()

    print(f"\n{_passed}/{_passed + _failed} checks passed")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
