"""
Run findings — atomic, durable knowledge mined from execution trajectories.

The trajectory evaluator ("investigator") used to emit one long prose report per
run, which was then re-fed whole into both its own next prompt and the planner's
generation prompt. Prose has no addressable units, so nothing could be retrieved
selectively, nothing could be deduplicated, and the same discoveries were
restated every round until the prompts collapsed under their own weight (one
measured evaluation: 86k of 98k characters were previous reports).

A finding is the addressable unit that fixes that: one claim, one kind, the
screen it concerns, and the step evidence behind it. Findings accumulate in the
graph, are matched semantically on write so a repeat reinforces the existing
node instead of creating a new one, and are retrieved per screen/kind at the
point of use.

Pure graph logic over a Neo4j ``session`` handed in by ``main.py`` (same contract
as defects.py / risk.py — no driver or embedding import here). App-agnostic:
screens, requirements and areas are resolved from the ingested graph at write
time, never hardcoded.

Graph model written:
    (:Project)-[:HAS_FINDING]->(:Finding)
    (:Finding)-[:ABOUT_SCREEN]->(:UIState)        # best-effort by observed label
    (:Finding)-[:FOUND_BY]->(:ExecutionLog)       # every run that evidenced it
    (:Finding)-[:CONCERNS]->(:Requirement)        # when the claim cites one
"""

from __future__ import annotations

import hashlib
import math

# ── Taxonomy ──────────────────────────────────────────────────────────────────
# Kinds are defined by how each one is CONSUMED, not by what it describes — a
# category nothing routes on is a category the model will fill inconsistently.
KINDS: dict[str, str] = {
    # The app did what it should. Tells the planner "proven — don't re-verify".
    "CONFIRMED_BEHAVIOUR": "verified working as specified",
    # Contradicts a named requirement. Strongest defect signal; must cite ids.
    "SPEC_VIOLATION": "observed behaviour contradicts a cited requirement",
    # Violates a universal expectation (validation, feedback, no crash, no data
    # loss) with no requirement to cite. The "potential bug" bucket.
    "SUSPECTED_DEFECT": "looks wrong against universal expectations, no requirement covers it",
    # Surprising but not judgeable as right or wrong without a human.
    "UNEXPECTED_BEHAVIOUR": "surprising or undocumented, needs judgement",
    # The app does something real the SRS never describes (requirements drift).
    "SPEC_GAP": "real behaviour the requirements do not describe",
    # Could not be determined this run, and why. Stops the planner assuming
    # coverage it does not have.
    "UNVERIFIED": "could not be determined this run",
    # A real control/screen seen on the device. UI context, never bug evidence.
    "CONTROL_DISCOVERED": "a control or screen observed at runtime",
    # OUR agent flailed (repeat with no effect, found-then-lost path). Steers
    # test design. Must never be presented as evidence the app is broken.
    "AGENT_DIFFICULTY": "our own agent struggled here — not an app defect",
}

# Consumer groupings. Each maps to exactly one prompt block downstream, which is
# the whole point of having kinds at all.
DEFECT_KINDS = frozenset({"SPEC_VIOLATION", "SUSPECTED_DEFECT"})
ORACLE_KINDS = frozenset({"SPEC_VIOLATION", "SUSPECTED_DEFECT", "CONFIRMED_BEHAVIOUR",
                          "UNEXPECTED_BEHAVIOUR", "SPEC_GAP", "UNVERIFIED"})
UI_KINDS = frozenset({"CONTROL_DISCOVERED"})
AGENT_KINDS = frozenset({"AGENT_DIFFICULTY"})

# Named groups callers ask for by name. Consumers (the planner's prompt blocks,
# the dashboard) must never restate this mapping locally: a duplicated taxonomy
# is how this project once reported 100% autonomy when it was 67%.
GROUPS: dict[str, frozenset] = {
    "oracle": ORACLE_KINDS,   # the bug oracle: what the app does / fails to do
    "defect": DEFECT_KINDS,   # candidate defects only, for review and risk
    "ui": UI_KINDS,           # observed controls and screens
    "agent": AGENT_KINDS,     # our own agent's difficulties, never app evidence
}


def group_kinds(name: str) -> list[str]:
    """Kinds in a named group; empty list for an unknown name (= no filter)."""
    return sorted(GROUPS.get(str(name or "").strip().lower(), frozenset()))

_DEFAULT_KIND = "UNEXPECTED_BEHAVIOUR"

# Per-field caps. Bounding happens here, at the field level, rather than by
# telling the model to "be brief" — the previous prompt's explicit "no length
# limit" existed because an earlier word limit had compressed away the specific
# detail that makes a finding actionable. Structure bounds the volume; the
# fields stay long enough to carry evidence.
CLAIM_MAX = 240
EVIDENCE_MAX = 500
_MAX_EVIDENCE_KEPT = 5     # distinct evidence strings retained per finding
_MAX_FINDINGS_PER_RUN = 15  # runaway guard; dedup does the real bounding

# Merging is deliberately MODEL-LED, not embedding-led. Measured on this
# project's own claims, cosine cannot separate a restatement from an adjacent
# variant: "empty farm name -> false success" vs "over-long farm name -> silent
# truncation" scores 0.853 while a true restatement of the first scores 0.846.
# Those are two distinct defects on one screen, which is exactly what
# exploratory testing produces most of.
#
# So the evaluator — which holds the full step evidence and can trivially tell
# "empty" from "over-long" — names the finding it is confirming, and cosine is
# demoted to a safety net that fires only on near-verbatim repeats.
#
# The asymmetry that sets this threshold: a FALSE MERGE destroys knowledge (a
# second defect silently disappears into the first), while a FALSE SPLIT costs
# one duplicate node of ~240 chars. Cheap mistake, expensive mistake — so this
# is tuned to make false merges vanishingly unlikely.
_SIMILAR_THRESHOLD = 0.90

# A finding may only absorb another from the same group. Without this, a defect
# and the passing case that disproves it can merge on lexical overlap alone
# ("...with an empty farm name" vs "...with a valid farm name" = 0.849) and the
# defect is erased from the oracle.
_MERGE_GROUP = {
    "SPEC_VIOLATION": "defect",
    "SUSPECTED_DEFECT": "defect",
    "CONFIRMED_BEHAVIOUR": "confirmed",
    "UNEXPECTED_BEHAVIOUR": "observation",
    "SPEC_GAP": "observation",
    "UNVERIFIED": "observation",
    "CONTROL_DISCOVERED": "ui",
    "AGENT_DIFFICULTY": "agent",
}


def merge_group(kind: str) -> str:
    return _MERGE_GROUP.get(normalize_kind(kind), "observation")


def short_ref(finding_id: str) -> str:
    """Compact handle the evaluator can cite back ('F-1a2b3c4d')."""
    return "F-" + str(finding_id or "").rsplit("::", 1)[-1][:8]


def _cosine(a, b) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def normalize_kind(kind: str) -> str:
    """Coerce a model-supplied kind to the taxonomy (never raises)."""
    k = str(kind or "").strip().upper().replace(" ", "_").replace("-", "_")
    return k if k in KINDS else _DEFAULT_KIND


def _clean(value, limit: int) -> str:
    """Collapse whitespace and clamp, cutting at a word boundary.

    A mid-word cut ("...did not persist the single sele") reads as corruption
    once the claim is shown back to the planner, and these strings exist to be
    read. Falls back to a hard cut when there is no space to break on.
    """
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    # Reserve the ellipsis inside the budget so `limit` is a true bound — the
    # field caps are asserted downstream, and "limit + 1" is still a bug.
    cut = text[:limit - 1]
    space = cut.rfind(" ")
    return (cut[:space] if space > (limit - 1) * 0.6 else cut).rstrip(" ,;:-") + "…"


def _finding_id(project: str, kind: str, claim: str) -> str:
    digest = hashlib.sha1(f"{kind}|{claim.lower()}".encode("utf-8")).hexdigest()[:16]
    return f"{project}::finding::{digest}"


def _resolve_screens(session, project: str, labels: list[str]) -> dict[str, str]:
    """Map observed screen labels -> UIState ids (only those that exist)."""
    labels = [l for l in {str(x or "").strip() for x in labels} if l]
    if not labels:
        return {}
    rows = session.run(
        """
        MATCH (p:Project {name:$project})-[:HAS_STATE]->(s:UIState)
        WHERE s.label IN $labels OR s.id IN $labels
        RETURN s.label AS label, s.id AS id
        """,
        project=project, labels=labels,
    )
    out: dict[str, str] = {}
    for r in rows:
        out[r["label"]] = r["id"]
        out[r["id"]] = r["id"]
    return out


def record(session, project, findings, *, log_id, test_case_id, embed_texts, now,
           threshold: float = _SIMILAR_THRESHOLD) -> dict:
    """Persist a run's findings, reinforcing existing ones instead of duplicating.

    ``findings`` is the evaluator's parsed list of
    ``{claim, kind, screen, evidence, severity, confidence, requirement_ids}``.
    Returns counts plus the resolved ids, so the caller can report honestly what
    the run actually contributed rather than how much text it produced.
    """
    findings = [f for f in (findings or []) if isinstance(f, dict)][:_MAX_FINDINGS_PER_RUN]
    if not findings:
        return {"created": 0, "reinforced": 0, "findings": []}

    # Normalise first so dedup compares what will actually be stored.
    prepared = []
    for f in findings:
        claim = _clean(f.get("claim"), CLAIM_MAX)
        if not claim:
            continue
        prepared.append({
            "claim": claim,
            "kind": normalize_kind(f.get("kind")),
            "screen": _clean(f.get("screen"), 200),
            "evidence": _clean(f.get("evidence"), EVIDENCE_MAX),
            "severity": _clean(f.get("severity"), 16).lower() or "medium",
            "confidence": _clean(f.get("confidence"), 16).lower() or "medium",
            "requirement_ids": [_clean(r, 40) for r in (f.get("requirement_ids") or [])
                                if _clean(r, 40)][:5],
            # The evaluator's own judgement that this restates a finding it was
            # shown. Authoritative when it resolves — it saw the evidence.
            "confirms": _clean(f.get("confirms"), 80),
        })
    if not prepared:
        return {"created": 0, "reinforced": 0, "findings": []}

    screen_ids = _resolve_screens(session, project, [p["screen"] for p in prepared])
    vectors = embed_texts([p["claim"] for p in prepared])

    # Existing findings to match against, loaded once. Scoped to this project;
    # kind is compared in Python so a claim can still match when the model
    # relabels the same observation under a neighbouring kind.
    existing = [dict(r) for r in session.run(
        """
        MATCH (p:Project {name:$project})-[:HAS_FINDING]->(f:Finding)
        RETURN f.id AS id, f.claim AS claim, f.kind AS kind, f.embedding AS embedding
        """,
        project=project,
    )]
    by_ref = {short_ref(e["id"]): e["id"] for e in existing}
    by_ref.update({e["id"]: e["id"] for e in existing})

    created, reinforced, results = 0, 0, []
    for item, vec in zip(prepared, vectors):
        # 1. The evaluator named the finding this confirms — trust it.
        match_id = by_ref.get(item["confirms"], "")
        matched_by = "model" if match_id else ""

        # 2. Safety net: near-verbatim repeat the evaluator did not flag,
        #    restricted to its own merge group so a defect can never be absorbed
        #    by the passing case that would disprove it.
        if not match_id and vec:
            group = merge_group(item["kind"])
            best_sim = 0.0
            for e in existing:
                if merge_group(e["kind"]) != group or not e.get("embedding"):
                    continue
                sim = _cosine(vec, e["embedding"])
                if sim > best_sim:
                    best_sim, match_id = sim, e["id"]
            if best_sim < threshold:
                match_id = ""
            else:
                matched_by = "embedding"

        if match_id:
            session.run(
                """
                MATCH (f:Finding {id:$id})
                SET f.times_seen = coalesce(f.times_seen, 0) + 1,
                    f.last_seen = $now,
                    f.evidence = [e IN (coalesce(f.evidence, []) + [$evidence])
                                  WHERE e <> ''][0..$keep]
                """,
                id=match_id, now=now, evidence=item["evidence"], keep=_MAX_EVIDENCE_KEPT,
            )
            reinforced += 1
            results.append({"id": match_id, "ref": short_ref(match_id), "claim": item["claim"],
                            "kind": item["kind"], "status": "reinforced", "matched_by": matched_by})
        else:
            fid = _finding_id(project, item["kind"], item["claim"])
            session.run(
                """
                MERGE (p:Project {name:$project})
                MERGE (f:Finding {id:$id})
                ON CREATE SET f.first_seen = $now, f.times_seen = 0
                SET f.project = $project, f.claim = $claim, f.kind = $kind,
                    f.severity = $severity, f.confidence = $confidence,
                    f.screen_label = $screen, f.last_seen = $now,
                    f.times_seen = coalesce(f.times_seen, 0) + 1,
                    f.evidence = [e IN [$evidence] WHERE e <> '']
                """
                + ("SET f.embedding = $embedding\n" if vec else "")
                + "MERGE (p)-[:HAS_FINDING]->(f)",
                project=project, id=fid, now=now, claim=item["claim"], kind=item["kind"],
                severity=item["severity"], confidence=item["confidence"],
                screen=item["screen"], evidence=item["evidence"], embedding=vec,
            )
            existing.append({"id": fid, "claim": item["claim"], "kind": item["kind"],
                             "embedding": vec})
            match_id = fid
            created += 1
            results.append({"id": fid, "ref": short_ref(fid), "claim": item["claim"],
                            "kind": item["kind"], "status": "created", "matched_by": ""})

        # Provenance + attachment, applied on both paths: a reinforced finding
        # gains the new run as additional evidence, which is what makes
        # times_seen mean "independently observed N times".
        sid = screen_ids.get(item["screen"])
        if sid:
            session.run(
                "MATCH (f:Finding {id:$id}) MATCH (s:UIState {id:$sid}) "
                "MERGE (f)-[:ABOUT_SCREEN]->(s)", id=match_id, sid=sid)
        if log_id:
            session.run(
                "MATCH (f:Finding {id:$id}) MATCH (e:ExecutionLog {id:$log_id}) "
                "MERGE (f)-[:FOUND_BY]->(e)", id=match_id, log_id=log_id)
        for rid in item["requirement_ids"]:
            session.run(
                """
                MATCH (f:Finding {id:$id})
                MATCH (p:Project {name:$project})-[:HAS_REQUIREMENT]->(r:Requirement)
                WHERE r.ref_id = $rid
                MERGE (f)-[:CONCERNS]->(r)
                """, id=match_id, project=project, rid=rid)

    return {"created": created, "reinforced": reinforced, "findings": results}


def query(session, project, *, screens=None, kinds=None, limit: int = 20,
          exclude_log_id: str = "") -> list[dict]:
    """Findings for this project, optionally narrowed to screens and/or kinds.

    ``screens`` accepts UIState ids or observed labels. Ordering puts
    independently-confirmed findings first (times_seen), then recency — so a
    bounded slice keeps what is most established rather than most recent.
    """
    kinds = sorted({normalize_kind(k) for k in (kinds or [])}) or None
    screens = [s for s in {str(x or "").strip() for x in (screens or [])} if s] or None

    cypher = [
        "MATCH (p:Project {name:$project})-[:HAS_FINDING]->(f:Finding)",
    ]
    if screens:
        cypher.append("OPTIONAL MATCH (f)-[:ABOUT_SCREEN]->(s:UIState)")
        # WITH detaches the filter below from the OPTIONAL MATCH. Without it the
        # WHERE is read as part of the optional pattern, every finding survives
        # with s=null, and narrowing by screen quietly returns everything.
        cypher.append("WITH f, s")
    cypher.append("WHERE 1=1")
    if kinds:
        cypher.append("AND f.kind IN $kinds")
    if screens:
        cypher.append("AND (s.label IN $screens OR s.id IN $screens OR f.screen_label IN $screens)")
    if exclude_log_id:
        cypher.append("AND NOT EXISTS { MATCH (f)-[:FOUND_BY]->(:ExecutionLog {id:$exclude_log_id}) }")
    cypher.append(
        "RETURN DISTINCT f.id AS id, f.claim AS claim, f.kind AS kind, f.severity AS severity, "
        "f.confidence AS confidence, f.screen_label AS screen, f.times_seen AS times_seen, "
        "f.evidence AS evidence, f.last_seen AS last_seen "
        "ORDER BY f.times_seen DESC, f.last_seen DESC LIMIT $limit")

    rows = session.run("\n".join(cypher), project=project, kinds=kinds, screens=screens,
                       exclude_log_id=exclude_log_id, limit=max(1, min(limit, 200)))
    out = []
    for r in rows:
        d = dict(r)
        # Short handle the evaluator cites in `confirms` to reinforce this exact
        # finding instead of restating it as a new one.
        d["ref"] = short_ref(d["id"])
        out.append(d)
    return out


def stats(session, project) -> dict:
    """Counts per kind — for the dashboard and for run-to-run 'is it learning?'."""
    rows = session.run(
        """
        MATCH (p:Project {name:$project})-[:HAS_FINDING]->(f:Finding)
        RETURN f.kind AS kind, count(f) AS n, sum(f.times_seen) AS observations
        ORDER BY n DESC
        """, project=project)
    by_kind = [dict(r) for r in rows]
    return {"by_kind": by_kind, "total": sum(r["n"] for r in by_kind)}
