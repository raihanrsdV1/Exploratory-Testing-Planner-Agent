"""
End-to-end verification for ETA-REQ-301 / 302 / 303 against a running RAG API.

LLM-free: seeds a throwaway project with synthetic UI states, executions and
defects, then asserts every new endpoint returns correct graph-derived data.

Usage:  python scripts/verify_enhancements.py         (RAG_URL defaults to :9010)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ingestion import extractor

RAG = os.getenv("RAG_URL", "http://127.0.0.1:9010").rstrip("/")
PROJECT = os.getenv("VERIFY_PROJECT", "verify-enh")

_passed = 0
_failed = 0


def check(name: str, cond: bool, detail: str = ""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}  {detail}")


def post(path: str, payload: dict) -> dict:
    r = requests.post(f"{RAG}{path}", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


def get(path: str, params: dict) -> dict:
    r = requests.get(f"{RAG}{path}", params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def _state(pkg: str, activity: str, controls: list[str]) -> dict:
    """A minimal normalized UI observation the app_state abstractor accepts."""
    return {
        "phone_state": {"package": pkg, "activity": activity},
        "nodes": [{"resource_id": f"{pkg}:id/{c}", "class": "android.widget.Button",
                   "content_description": c, "clickable": True} for c in controls],
    }


def main() -> int:
    print(f"Verifying enhancements against {RAG} (project={PROJECT})\n")

    # Clean slate.
    post("/project/reset", {"project": PROJECT, "delete_tests": True, "delete_srs": True, "delete_figma": True})

    # ── REQ-301: Defect history ───────────────────────────────────────────────
    print("REQ-301 Defect History")
    defects = [
        {"id": "BUG-1", "title": "Crash saving contact with empty name", "severity": "critical",
         "status": "open", "area": "create_contact", "root_cause_category": "validation"},
        {"id": "BUG-2", "title": "App crashes when saving a contact without a name", "severity": "high",
         "status": "open", "area": "create_contact", "root_cause_category": "validation"},
        {"id": "BUG-3", "title": "Search returns wrong results for partial query", "severity": "medium",
         "status": "closed", "area": "search", "root_cause_category": "logic"},
    ]
    ing = post("/ingest/defects", {"project": PROJECT, "defects": defects})
    check("defects ingested", ing.get("defects_written") == 3, str(ing))
    check("defect areas scored (density)", ing.get("areas_scored", 0) >= 2, str(ing))

    summ = get("/defects/summary", {"project": PROJECT})
    check("defect summary totals", summ.get("total_defects") == 3 and summ.get("unresolved_defects") == 2, str(summ.get("total_defects")))
    check("defect summary has prone areas", len(summ.get("prone_areas", [])) >= 1)
    check("defect summary text non-empty", bool(summ.get("summary_text")))

    prone = get("/defects/prone-areas", {"project": PROJECT}).get("prone_areas", [])
    check("create_contact is top prone area", prone and prone[0]["area"].replace(" ", "_") == "create_contact", str(prone[:1]))

    dctx = get("/defects/context", {"project": PROJECT, "query": "save contact", "area": "create_contact"}).get("context", "")
    check("defect context block references a known defect", "BUG-1" in dctx or "BUG-2" in dctx, dctx[:80])

    # ── Seed a live app model (states + transitions) ──────────────────────────
    print("\nSeeding live app model (UI states + transitions)")
    pkg = "com.example.app"
    s_list = post("/liveui/observe", {"project": PROJECT, "normalized": _state(pkg, ".ListActivity", ["search", "fab_add"])})
    s_create = post("/liveui/observe", {"project": PROJECT, "normalized": _state(pkg, ".CreateActivity", ["first_name", "phone", "save"]),
                                        "from_state_id": s_list["state_id"], "action": "tap fab_add", "element": "fab_add"})
    s_detail = post("/liveui/observe", {"project": PROJECT, "normalized": _state(pkg, ".DetailActivity", ["edit", "delete", "back"]),
                                        "from_state_id": s_create["state_id"], "action": "tap save", "element": "save"})
    list_id, create_id, detail_id = s_list["state_id"], s_create["state_id"], s_detail["state_id"]
    check("three distinct states observed", len({list_id, create_id, detail_id}) == 3)

    # ── REQ-302: NavTree (auto-recorded from execution logs) ──────────────────
    print("\nREQ-302 Navigation Tree")
    # The planner always logs the test case before it is executed — mirror that so
    # the nav tree can attach RESOLVES_TEST to a real TestCase node.
    post("/tests/log", {"project": PROJECT, "test_case_id": "TC-1", "title": "Create a valid contact",
                        "verdict": "pass", "area": "create_contact", "test_type": "state_transition"})
    # A passing test that walks list -> create -> detail (3 steps).
    post("/execution/log", {"project": PROJECT, "test_case_id": "TC-1", "title": "Create a valid contact",
                            "verdict": "pass", "path": [list_id, create_id, detail_id],
                            "path_labels": [s_list["label"], s_create["label"], s_detail["label"]]})
    rp = get("/navtree/retrieve-path", {"project": PROJECT, "test_id": "TC-1"})
    check("navtree recorded a resolving path for TC-1", rp.get("found") and rp.get("length") == 3, str(rp.get("length")))

    # A shorter successful path for the same test (list -> detail, 2 steps) should replace it.
    post("/navtree/record-path", {"project": PROJECT, "test_case_id": "TC-1", "title": "Create a valid contact",
                                  "verdict": "pass", "path": [list_id, detail_id],
                                  "path_labels": [s_list["label"], s_detail["label"]], "actions": ["(entry)", "shortcut"]})
    rp2 = get("/navtree/retrieve-path", {"project": PROJECT, "test_id": "TC-1"})
    check("shortest path replaced the longer one (3 -> 2)", rp2.get("length") == 2, str(rp2.get("length")))

    # Retrieve by target screen (cross-test reuse).
    rp3 = get("/navtree/retrieve-path", {"project": PROJECT, "screen": s_detail["label"]})
    check("path retrievable by target screen label", rp3.get("found") is True, str(rp3))

    # Failed navigation: repeatedly fail reaching 'create' so its node is flagged avoid.
    for _ in range(3):
        post("/execution/log", {"project": PROJECT, "test_case_id": "TC-2", "title": "Broken create flow",
                                "verdict": "failed", "error_type": "element_not_found",
                                "path": [list_id, create_id],
                                "path_labels": [s_list["label"], s_create["label"]]})
    fps = get("/navtree/failed-paths", {"project": PROJECT}).get("failed_paths", [])
    check("failed navigation path flagged as avoid", any(s_create["label"] in (f.get("screen") or "") for f in fps), str(fps))

    # ── REQ-303: Experiential learning ────────────────────────────────────────
    print("\nREQ-303 Experiential Learning")
    eps = get("/execution/error-patterns", {"project": PROJECT}).get("error_patterns", [])
    check("error pattern mined from failed executions", any(e["error_type"] == "element_not_found" for e in eps), str(eps[:1]))
    check("error pattern carries a mitigation", eps and bool(eps[0].get("suggested_mitigation")))

    heat = get("/coverage/heatmap", {"project": PROJECT})
    check("coverage heatmap reports observed screens", heat.get("observed_screens") == 3, str(heat.get("observed_screens")))
    check("coverage heatmap persisted last_updated", bool(heat.get("last_updated")))

    strat = get("/strategy/memory", {"project": PROJECT}).get("strategies", [])
    check("strategy memory recorded from executions", len(strat) >= 1, str(strat[:2]))

    ss = post("/session/start", {"project": PROJECT, "focus_area": "create_contact", "strategy": "boundary_probe"})
    check("session started", ss.get("status") == "active" and ss.get("session_id"), str(ss))
    sc = get("/session/context", {"project": PROJECT})
    check("session context active", sc.get("active") is True and sc.get("focus_area") == "create_contact", str(sc))
    se = post("/session/end", {"project": PROJECT, "session_id": ss.get("session_id", "")})
    check("session ended", se.get("ended") is True, str(se))

    # ── REQ-301.6: defect-area traceability on a logged test ──────────────────
    print("\nREQ-301.6 Defect traceability")
    post("/tests/log", {"project": PROJECT, "test_case_id": "TC-3",
                        "title": "Probe empty-name validation on create", "verdict": "failed",
                        "area": "create_contact", "test_type": "boundary"})
    # brief exposes counts used by the planner's source-availability gating.
    brief = post("/context/brief", {"project": PROJECT, "recent_limit": 10})
    check("brief exposes defect_count", brief.get("defect_count") == 3, str(brief.get("defect_count")))
    check("brief exposes navtree_node_count", brief.get("navtree_node_count", 0) >= 3, str(brief.get("navtree_node_count")))
    check("brief carries defect_summary text", bool(brief.get("defect_summary")))

    # ── WP1: Live App Model knowledge-decay (recency weighting) ───────────────
    print("\nWP1 Live App Model decay")
    am = get("/appmodel/graph", {"project": PROJECT})
    nodes = am.get("nodes", [])
    check("appmodel nodes carry a recency_weight", nodes and "recency_weight" in nodes[0], str(nodes[:1]))
    check("freshly-observed states are not stale", am.get("stale_states") == 0, str(am.get("stale_states")))

    # ── WP2: sample defect file + defect-weighted retrieval ───────────────────
    print("\nWP2 sample defects + defect-weighted /retrieve")
    ingf = post("/ingest/defects", {"project": PROJECT, "source_path": "./data/inputs/defects_sample.json"})
    check("sample defect file ingested (6 defects)", ingf.get("defects_written") == 6, str(ingf.get("defects_written")))

    retr = post("/retrieve", {"project": PROJECT, "query": "save a contact", "top_k": 5, "include_history": False})
    check("retrieval context surfaces defect-prone-area bias", "Defect-prone areas" in retr.get("context", ""), retr.get("context", "")[:80])

    # ── WP5: Business-Logic Intelligence — multi-pass extraction + critique ───
    print("\nWP5 multi-pass extraction + self-critique (stub model, no tokens)")

    def stub_model(prompt, max_tokens, thinking):
        if "EXTRACTION UNDER REVIEW" in prompt:  # self-critique pass
            return {"answer": json.dumps({
                "confidence": {"FR-1": 0.9, "FR-2": 0.3},
                "missing": [{"id": "FR-9", "type": "functional",
                             "text": "Sessions must expire after inactivity", "feature": "security", "priority": "high"}],
                "open_questions": ["Duplicate-match threshold TBD after Q3 research"],
            })}
        return {"answer": json.dumps({  # per-section extraction (same ids each chunk → dedup)
            "document_title": "Test SRS",
            "requirements": [
                {"id": "FR-1", "type": "functional", "text": "Name must not be empty", "feature": "create", "priority": "high"},
                {"id": "FR-2", "type": "functional", "text": "Phone must be valid", "feature": "create", "priority": "medium"},
            ],
            "entities": [{"name": "Contact", "description": "a contact"}],
            "validation_rules": [{"field": "name", "rule": "name required", "requirement_id": "FR-1"}],
        })}

    merged = extractor._merge_extractions([
        extractor._normalize_extraction(json.loads(stub_model("x", 0, False)["answer"])),
        extractor._normalize_extraction(json.loads(stub_model("x", 0, False)["answer"])),
    ])
    check("synthesis pass dedups requirements across sections", len(merged["requirements"]) == 2, str(len(merged["requirements"])))

    ex, src = extractor.extract("3.1 A\nname text\n\n3.2 B\nphone text", model_call=stub_model, multipass=True)
    check("extraction source is multipass", src == "multipass", src)
    check("self-critique added a missing requirement (FR-9)", any(r["id"] == "FR-9" for r in ex["requirements"]))
    fr2 = next((r for r in ex["requirements"] if r["id"] == "FR-2"), {})
    check("low critique confidence flags needs_review", fr2.get("needs_review") is True, str(fr2.get("confidence")))
    check("self-critique surfaced an open question", any("TBD" in q or "threshold" in q.lower() for q in ex.get("open_questions", [])))

    fb, fbsrc = extractor.extract("FR-1: The name field shall not be empty.\nTODO: define max length later.", model_call=None)
    check("rule-based fallback attaches confidence", fb["requirements"] and "confidence" in fb["requirements"][0], str(fb["requirements"][:1]))
    check("fallback surfaces TODO as open question", any("todo" in q.lower() for q in fb.get("open_questions", [])), str(fb.get("open_questions")))

    # ── WP5: provenance + confidence in the graph, and SRS drift detection ────
    print("\nWP5 provenance/confidence + SRS versioning & drift")

    def ingest_srs(label, text):
        extraction, _ = extractor.extract(text, model_call=None)  # rule-based, no tokens
        return post("/ingest/srs", {"project": PROJECT, "source_path": f"inline://{label}",
                                    "srs_text": text, "extraction": extraction})

    srs_v1 = ("FR-1: The name field shall not be empty when saving a contact.\n"
              "FR-2: The phone number must conform to international format.\n")
    r1 = ingest_srs("v1", srs_v1)
    check("SRS v1 recorded as version 1", r1.get("srs_version") == 1, str(r1.get("srs_version")))
    check("v1 is not flagged as a re-ingest", r1.get("areas_flagged_for_retest", 0) == 0)

    blr = get("/business-logic/rules", {"project": PROJECT})
    check("business-logic rules carry confidence", blr.get("rules") and "confidence" in blr["rules"][0], str(blr.get("count")))
    check("rules carry needs_review flag", blr.get("rules") and "needs_review" in blr["rules"][0])
    check("at least one rule has SRS-chunk provenance", any(r.get("provenance_chunk") for r in blr.get("rules", [])), "no provenance (embeddings off?)")

    srs_v2 = ("FR-1: The name field shall not be empty when saving a contact.\n"
              "FR-2: The phone number must conform to national format only.\n"
              "FR-3: The email address shall be validated before saving.\n")
    r2 = ingest_srs("v2", srs_v2)
    check("re-ingest creates SRS version 2", r2.get("srs_version") == 2, str(r2.get("srs_version")))
    check("drift detected rule changes on re-ingest", (r2.get("rules_added", 0) + r2.get("rules_removed", 0)) >= 2, str(r2))
    check("re-ingest flagged areas for re-test", r2.get("areas_flagged_for_retest", 0) >= 1, str(r2.get("areas_flagged_for_retest")))

    drift = get("/srs/drift", {"project": PROJECT})
    check("drift endpoint reports latest version 2", (drift.get("latest") or {}).get("version") == 2, str(drift.get("total_versions")))
    check("drift endpoint lists areas needing re-test", len(drift.get("areas_needing_retest", [])) >= 1, str(drift.get("areas_needing_retest")))

    # ── WP5: decay-weighted strategy suggestions ─────────────────────────────
    print("\nWP5 decay-weighted strategy memory")
    strat2 = get("/strategy/memory", {"project": PROJECT}).get("strategies", [])
    check("strategies carry a decay-weighted score", strat2 and "decayed_score" in strat2[0], str(strat2[:1]))

    # ── WP6: multi-dimensional partitioning + filtering + transfer ────────────
    print("\nWP6 multi-dimensional KG")
    # Reset just the SRS slice so all chunks carry one platform tag (clean filter demo).
    post("/project/reset", {"project": PROJECT, "delete_tests": False, "delete_srs": True, "delete_figma": False})
    srs_dim = ("FR-1: The contact list shall display saved contacts alphabetically.\n"
               "FR-2: The user shall refresh the contact list by pulling down.\n")
    extraction, _ = extractor.extract(srs_dim, model_call=None)
    post("/ingest/srs", {"project": PROJECT, "source_path": "inline://android", "srs_text": srs_dim,
                         "extraction": extraction, "platform": "android"})

    dl = get("/dimensions/list", {"project": PROJECT})
    check("project registers platform=android dimension", "android" in dl.get("platform", []), str(dl))

    r_in = post("/retrieve", {"project": PROJECT, "query": "refresh contact list", "top_k": 5,
                              "include_history": False, "platform": "android"})
    check("matching-dimension retrieval returns content", len(r_in.get("chunks", [])) > 0, str(len(r_in.get("chunks", []))))
    r_miss = post("/retrieve", {"project": PROJECT, "query": "refresh contact list", "top_k": 5,
                                "include_history": False, "platform": "tizen"})
    check("out-of-dimension retrieval excludes tagged content (no leakage)", len(r_miss.get("chunks", [])) == 0, str(len(r_miss.get("chunks", []))))

    post("/tests/log", {"project": PROJECT, "test_case_id": "TD-C", "title": "Verify contact save flow",
                        "verdict": "pass", "area": "create_contact", "application": "contacts", "platform": "android"})
    post("/tests/log", {"project": PROJECT, "test_case_id": "TD-S", "title": "Verify settings toggle",
                        "verdict": "pass", "area": "settings", "application": "settings", "platform": "android"})
    ts = get("/dimensions/transfer-suggestions",
             {"project": PROJECT, "application": "contacts", "platform": "windows"}).get("suggestions", [])
    titles = [s["title"] for s in ts]
    check("transfer suggests same-app test for the untested platform", "Verify contact save flow" in titles, str(titles))
    check("transfer carries a confidence score", ts and 0 < ts[0].get("transfer_confidence", 0) <= 1, str(ts[:1]))
    check("no cross-application leakage in transfer", "Verify settings toggle" not in titles, str(titles))

    print(f"\n{'='*60}\nRESULT: {_passed} passed, {_failed} failed\n{'='*60}")
    return 1 if _failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except requests.HTTPError as e:
        print(f"\nHTTP error: {e}\nResponse: {getattr(e.response, 'text', '')[:500]}")
        sys.exit(2)
    except requests.ConnectionError:
        print(f"\nCannot reach RAG API at {RAG}. Start it first:\n"
              "  uvicorn rag_api.main:app --host 127.0.0.1 --port 9010")
        sys.exit(2)
