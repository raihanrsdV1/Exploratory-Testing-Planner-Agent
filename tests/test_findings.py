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

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from rag_api import embeddings, findings as F, learning as L  # noqa: E402

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

        print("\nlifecycle: questions open, answers do not")
        # Kind decides. A finding the app might still answer differently is a
        # question; one that records what was seen to work is already an answer.
        check("an unverified item opens a question", F.initial_status("UNVERIFIED"), F.OPEN)
        check("a suspected defect opens a question", F.initial_status("SUSPECTED_DEFECT"), F.OPEN)
        check("a confirmed behaviour is already an answer",
              F.initial_status("CONFIRMED_BEHAVIOUR"), F.RESOLVED)
        check("our own agent's difficulty is NOT an app question",
              F.initial_status("AGENT_DIFFICULTY"), "")

        F.record(s, PROJECT, [
            {"claim": "The run never established whether the disease list loads at all",
             "kind": "UNVERIFIED", "screen": "Chats", "evidence": "steps 1-50: budget spent navigating"}],
            log_id="", test_case_id="TC-Q", embed_texts=_embed, now=NOW)
        # Match on the claim, not the kind: an earlier check in this file also
        # creates an UNVERIFIED finding, which is legitimately a question too.
        q = [x for x in F.open_questions(s, PROJECT, limit=20)
             if x["claim"].startswith("The run never established")]
        check("the question reaches the open queue", len(q), 1)
        ref = q[0]["ref"]
        check("it starts with a full attempt budget", q[0]["attempts_left"], F.MAX_FINDING_ATTEMPTS)
        check("resolved findings are not in the queue",
              any(x["kind"] == "CONFIRMED_BEHAVIOUR" for x in F.open_questions(s, PROJECT, limit=20)), False)
        check("agent difficulties are never queued as questions",
              any(x["kind"] == "AGENT_DIFFICULTY" for x in F.open_questions(s, PROJECT, limit=20)), False)

        print("\ncuriosity is bounded — a question cannot be chased forever")
        for i in range(F.MAX_FINDING_ATTEMPTS - 1):
            out = F.record_attempt(s, PROJECT, ref, NOW)
            check(f"attempt {i+1} keeps it open", out["status"], F.OPEN)
        final = F.record_attempt(s, PROJECT, ref, NOW)
        check("the last attempt closes it as inconclusive", final["status"], F.INCONCLUSIVE)
        check("an exhausted question leaves the queue",
              any(x["ref"] == ref for x in F.open_questions(s, PROJECT, limit=20)), False)
        check("an unknown ref is reported, not silently counted",
              F.record_attempt(s, PROJECT, "F-deadbeef", NOW)["status"], "unknown_ref")

        print("\nan answer closes a question")
        q2 = F.record(s, PROJECT, [
            {"claim": "The favourites control does not exist anywhere in the product detail flow",
             "kind": "UNVERIFIED", "screen": "Medicine", "evidence": "steps 4-9"}],
            log_id="", test_case_id="TC-R", embed_texts=_embed, now=NOW)
        ref2 = q2["findings"][0]["ref"]
        F.record(s, PROJECT, [
            {"claim": "Favourites is not implemented on product detail — no such control after full search",
             "kind": "CONFIRMED_BEHAVIOUR", "screen": "Medicine",
             "evidence": "steps 1-12 exhaustive", "resolves": ref2}],
            log_id="", test_case_id="TC-S", embed_texts=_embed, now=NOW)
        closed = [x for x in F.query(s, PROJECT, limit=60) if x["ref"] == ref2]
        check("the question is now resolved", closed[0]["status"], F.RESOLVED)
        check("the answer is stored on it", bool(closed[0]["resolution"]), True)
        check("a resolved question leaves the queue",
              any(x["ref"] == ref2 for x in F.open_questions(s, PROJECT, limit=20)), False)

        print("\na runaway batch cannot flood the graph")
        many = [{"claim": f"distinct observation number {i} about a control", "kind": "CONTROL_DISCOVERED",
                 "screen": "Chats", "evidence": f"step {i}"} for i in range(40)]
        r3 = F.record(s, PROJECT, many, log_id="", test_case_id="TC-W", embed_texts=_embed, now=NOW)
        check("batch truncated to the per-run guard",
              r3["created"] + r3["reinforced"], F._MAX_FINDINGS_PER_RUN)

        print("\na question can be looked up by the ref a test names")
        # The chain planner -> executor -> investigator carries only a ref, so
        # the ref has to resolve back to the question on its own.
        r = F.record(s, PROJECT, [
            {"claim": "Whether the order receipt can be exported as a PDF was never determined",
             "kind": "UNVERIFIED", "screen": "Orders", "evidence": "steps 4-9: never reached"}],
            log_id="", test_case_id="TC-REF", embed_texts=_embed, now=NOW)
        ref = r["findings"][0]["ref"]
        got = F.by_ref(s, PROJECT, ref)
        check("a short ref resolves to its finding", got["ref"], ref)
        check("it carries the remaining attempt budget",
              got["attempts_left"], F.MAX_FINDING_ATTEMPTS)
        check("it carries the evidence that explains why it is open",
              bool(got["evidence"]), True)
        check("the full id resolves too", F.by_ref(s, PROJECT, got["id"])["ref"], ref)
        check("an unknown ref returns nothing rather than a wrong finding",
              F.by_ref(s, PROJECT, "F-deadbeef"), None)
        check("an empty ref returns nothing", F.by_ref(s, PROJECT, ""), None)

        print("\nthe rollup covers everything a truncated list cannot")
        st = F.stats(s, PROJECT)
        check("stats reports a per-screen breakdown", "by_screen" in st, True)
        check("the rollup accounts for every finding",
              sum(r["total"] for r in st["by_screen"]), st["total"])
        screens = {r["screen"]: r for r in st["by_screen"]}
        check("it counts open work per screen", screens["Chats"]["open"] >= 1, True)
        check("it separates candidate defects from other kinds",
              screens["My Products"]["defects"], 1)
        check("agent difficulties are counted but NOT as defects",
              (screens["Chats"]["agent_trouble"], screens["Chats"]["defects"]), (1, 0))
        check("screens are ordered by where the open work is",
              [r["open"] for r in st["by_screen"]] ==
              sorted([r["open"] for r in st["by_screen"]], reverse=True), True)

        print("\nthe open queue balances started questions against fresh ones")
        # Neither sort alone works. attempts ASC starves started questions (each
        # run mints new ones, so a half-investigated question is never shown
        # again and nothing is ever concluded). attempts DESC starves fresh
        # discoveries. Slots are reserved for both.
        _wipe(s)
        # Claims must be genuinely different, or the embedding safety net merges
        # them into one finding — correctly — and there is no queue to balance.
        FRESH = [
            "Whether the checkout total includes delivery charges was never determined",
            "The vaccination reminder schedule could not be reached from any menu",
            "Whether product photos survive an app restart remains untested",
            "The feed order cancellation flow was never exercised",
            "Whether a sold animal disappears from the marketplace is unknown",
            "The medicine expiry date field was never validated",
        ]
        F.record(s, PROJECT, [
            {"claim": c, "kind": "UNVERIFIED", "screen": f"Screen{i}", "evidence": f"step {i}"}
            for i, c in enumerate(FRESH)],
            log_id="", test_case_id="TC-F", embed_texts=_embed, now=NOW)

        STARTED = [
            ("Saving a farm with a blank name shows success instead of a validation error", 2),
            ("The chat search box ignores leading and trailing whitespace entirely", 1),
            ("Tapping a sold listing opens a blank detail page rather than a notice", 2),
        ]
        started = []
        for claim, n_attempts in STARTED:
            r = F.record(s, PROJECT, [
                {"claim": claim, "kind": "SUSPECTED_DEFECT",
                 "screen": "Marketplace", "evidence": "steps 3-7"}],
                log_id="", test_case_id="TC-S2", embed_texts=_embed, now=NOW)
            ref = r["findings"][0]["ref"]
            for _ in range(n_attempts):          # 1 or 2, never the cap
                F.record_attempt(s, PROJECT, ref, NOW)
            started.append(ref)
        check("the fixtures did not collapse into one finding",
              len(F.open_questions(s, PROJECT, limit=50)) >= 8, True)

        picked = F.open_questions(s, PROJECT, limit=5)
        n_started = sum(1 for q in picked if q["attempts"] > 0)
        n_fresh = len(picked) - n_started
        check("started questions are not starved by fresh ones", n_started >= 1, True)
        check("fresh questions are not starved by started ones", n_fresh >= 1, True)
        check("the slot budget is respected", len(picked), 5)
        started_attempts = [q["attempts"] for q in picked if q["attempts"] > 0]
        check("closest-to-conclusion comes first among started",
              started_attempts == sorted(started_attempts, reverse=True), True)

        print("\nwhen one pool is empty the other takes the spare slots")
        _wipe(s)
        F.record(s, PROJECT, [
            {"claim": c, "kind": "UNVERIFIED", "screen": f"Only{i}", "evidence": "step 1"}
            for i, c in enumerate([
                "Whether the profile photo upload accepts a large file is unknown",
                "The district dropdown was never opened during any run",
                "Whether an order receipt can be shared was never checked",
                "The livestock weight field accepted no input this run",
            ])],
            log_id="", test_case_id="TC-O", embed_texts=_embed, now=NOW)
        only_fresh = F.open_questions(s, PROJECT, limit=5)
        check("all slots go to fresh when nothing is started", len(only_fresh), 4)
        check("and none of them are marked started",
              all(q["attempts"] == 0 for q in only_fresh), True)
        _wipe(s)

        print("\ncampaign snapshots survive the reset that destroys everything else")
        # The balance test above wiped the project, so seed one finding: the
        # snapshot is meant to capture what existed at wipe time.
        F.record(s, PROJECT, [
            {"claim": "A finding that exists at the moment the campaign is wiped",
             "kind": "CONFIRMED_BEHAVIOUR", "screen": "Chats", "evidence": "step 1"}],
            log_id="", test_case_id="TC-SNAP", embed_texts=_embed, now=NOW)
        # CLEAN_SLATE deletes tests and execution logs at the start of every
        # campaign, which is correct for a clean measurement but destroyed the
        # evidence a campaign-over-campaign comparison needs. The snapshot is
        # taken in the one moment the outgoing campaign still exists.
        snap = L.snapshot_campaign(s, PROJECT, NOW, reason="selftest")
        check("an empty project is not recorded as a campaign",
              snap.get("snapshotted"), False)

        s.run("""MERGE (p:Project {name:$p})
                 MERGE (t:TestCase {id:$p + '::tc::selftest'})
                 SET t.last_verdict='pass', t.external_id='TC-SELFTEST'
                 MERGE (p)-[:HAS_TEST]->(t)""", p=PROJECT)
        snap = L.snapshot_campaign(s, PROJECT, NOW, reason="selftest")
        check("a campaign with tests IS recorded", snap.get("snapshotted"), True)
        check("it counts the tests", snap.get("tests"), 1)
        check("it counts the findings that existed at wipe time",
              snap.get("findings_total") > 0, True)
        rows = L.list_campaigns(s, PROJECT, limit=5)
        check("the summary is listable", len(rows) >= 1, True)
        check("its properties are flat, not nested under a key",
              "ended_at" in (rows[0] if rows else {}), True)
        s.run("MATCH (c:CampaignSummary) WHERE c.project=$p DETACH DELETE c", p=PROJECT)
        s.run("MATCH (t:TestCase) WHERE t.id STARTS WITH $p DETACH DELETE t", p=PROJECT)

        _wipe(s)
    driver.close()

    print(f"\n{_passed}/{_passed + _failed} checks passed")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
