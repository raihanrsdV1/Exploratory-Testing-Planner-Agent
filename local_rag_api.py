from datetime import datetime, timezone
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
import json
import os
import re

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import PlainTextResponse
from neo4j import GraphDatabase
from pydantic import BaseModel, Field

from dotenv import load_dotenv

import embeddings
from ingestion import ui_normalizer

load_dotenv()

# `or default` (not just getenv default) so an empty value in .env doesn't crash startup.
NEO4J_URI = os.getenv("NEO4J_URI") or "neo4j://127.0.0.1:7687"
NEO4J_USER = os.getenv("NEO4J_USER") or "neo4j"
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD") or "hihi"
RAG_API_KEY = os.getenv("RAG_API_KEY", "")

CHUNK_VECTOR_INDEX = "chunk_embedding"
REQUIREMENT_VECTOR_INDEX = "requirement_embedding"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def _ensure_vector_indexes(session) -> None:
    """
    Create native Neo4j vector indexes for semantic retrieval (Neo4j 5.13+).
    Skipped silently when embeddings are disabled or the server is too old.
    """
    if not embeddings.is_enabled():
        return
    dim = embeddings.embedding_dim()
    if not dim:
        return
    for index_name, label in ((CHUNK_VECTOR_INDEX, "Chunk"), (REQUIREMENT_VECTOR_INDEX, "Requirement")):
        try:
            session.run(
                f"""
                CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                FOR (n:{label}) ON (n.embedding)
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: $dim,
                    `vector.similarity_function`: 'cosine'
                }}}}
                """,
                dim=dim,
            )
        except Exception as e:  # noqa: BLE001 - older Neo4j or unsupported edition
            print(f"[rag] vector index '{index_name}' not created: {e}")


@asynccontextmanager
async def lifespan(_: FastAPI):
    with driver.session() as session:
        session.run("CREATE CONSTRAINT project_name IF NOT EXISTS FOR (p:Project) REQUIRE p.name IS UNIQUE")
        session.run("CREATE CONSTRAINT srs_id IF NOT EXISTS FOR (s:SRS) REQUIRE s.id IS UNIQUE")
        session.run("CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
        session.run("CREATE CONSTRAINT testcase_id IF NOT EXISTS FOR (t:TestCase) REQUIRE t.id IS UNIQUE")
        session.run("CREATE CONSTRAINT testrun_id IF NOT EXISTS FOR (r:TestRun) REQUIRE r.id IS UNIQUE")
        session.run("CREATE CONSTRAINT figma_screen_id IF NOT EXISTS FOR (fs:FigmaScreen) REQUIRE fs.id IS UNIQUE")
        session.run("CREATE CONSTRAINT figma_element_id IF NOT EXISTS FOR (fe:UIElement) REQUIRE fe.id IS UNIQUE")
        session.run("CREATE CONSTRAINT feature_key IF NOT EXISTS FOR (f:FeatureArea) REQUIRE f.key IS UNIQUE")
        session.run("CREATE CONSTRAINT summary_id IF NOT EXISTS FOR (s:Summary) REQUIRE s.id IS UNIQUE")
        # Entity-graph node types (GraphRAG schema).
        session.run("CREATE CONSTRAINT requirement_id IF NOT EXISTS FOR (r:Requirement) REQUIRE r.id IS UNIQUE")
        session.run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
        session.run("CREATE CONSTRAINT vrule_id IF NOT EXISTS FOR (v:ValidationRule) REQUIRE v.id IS UNIQUE")
        _ensure_vector_indexes(session)
    yield


app = FastAPI(title="Local Neo4j RAG API", lifespan=lifespan)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _split_text(text: str, chunk_chars: int = 1200, overlap: int = 120) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []

    # Semantic-ish SRS chunking: group tagged requirement lines first (better
    # retrieval granularity). Convention-agnostic: FR/NFR/REQ/US/R + number,
    # or "1.2.3"-style numbered clauses.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    req_lines = [
        ln for ln in lines
        if re.match(r"^(FR|NFR|REQ|US|R)[-_ ]?\d+|^\d+(\.\d+)*[.)]\s", ln, flags=re.IGNORECASE)
    ]
    if req_lines:
        grouped: list[str] = []
        group_size = 8
        step = 6  # overlap of 2 requirements
        for i in range(0, len(req_lines), step):
            block = req_lines[i:i + group_size]
            if block:
                grouped.append("\n".join(block))
        return grouped

    # Fallback sentence-aware chunking for non-FR texts
    units = [u.strip() for u in re.split(r"\n\s*\n|(?<=[.!?])\s+", text) if u.strip()]
    chunks: list[str] = []
    current: list[str] = []
    cur_len = 0
    for u in units:
        if cur_len + len(u) + 1 <= chunk_chars or not current:
            current.append(u)
            cur_len += len(u) + 1
        else:
            chunks.append("\n".join(current).strip())
            # small overlap from previous chunk tail
            tail = current[-1:] if overlap > 0 else []
            current = tail + [u]
            cur_len = sum(len(x) + 1 for x in current)
    if current:
        chunks.append("\n".join(current).strip())
    return chunks


def _query_tokens(q: str) -> list[str]:
    toks = [t.strip().lower() for t in re.findall(r"[a-zA-Z0-9_]+", q or "")]
    out: list[str] = []
    stop = {"the", "and", "for", "with", "that", "this", "what", "when", "from", "into", "about"}
    for t in toks:
        if len(t) < 3 or t in stop:
            continue
        if t not in out:
            out.append(t)
    return out[:10]


def _strip_markdown_fence(raw: str) -> str:
    """Allow ingesting Figma JSON copied from markdown/codeblock sources."""
    text = (raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _build_srs_summary(text: str) -> str:
    """Fallback project-level SRS summary for planner stage-1 context."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    title = lines[0] if lines else "Product requirements"
    fr_lines = [ln for ln in lines if re.match(r"^FR-\d+", ln, flags=re.IGNORECASE)]
    nfr_lines = [ln for ln in lines if re.match(r"^NFR-\d+", ln, flags=re.IGNORECASE)]
    if not nfr_lines:
        nfr_lines = [
            ln for ln in lines
            if any(k in ln.lower() for k in ["non-functional", "performance", "security", "usability", "reliability", "availability"])
        ]

    summary = [
        f"Document: {title}",
        f"Total FR requirements: {len(fr_lines) if fr_lines else 'unknown'}",
        f"Total NFR requirements: {len(nfr_lines) if nfr_lines else 0}",
        "Functional requirements summary:",
    ]
    for ln in (fr_lines[:12] if fr_lines else lines[:12]):
        summary.append(f"- {ln[:220]}")

    summary.append("Non-functional requirements summary:")
    if nfr_lines:
        for ln in nfr_lines[:8]:
            summary.append(f"- {ln[:220]}")
    else:
        summary.append("- Not explicitly tagged in source text.")

    summary.append("Validation and constraints:")
    val = [
        ln for ln in (fr_lines + nfr_lines + lines)
        if any(k in ln.lower() for k in ["validate", "must", "shall", "required", "format", "constraint"])
    ]
    for ln in val[:8]:
        summary.append(f"- {ln[:220]}")
    return "\n".join(summary)


def _build_figma_summary(screens: list[dict]) -> str:
    if not screens:
        return "No Figma screens available"
    lines = [f"Total screens: {len(screens)}", "Screens:"]
    for s in screens[:25]:
        lines.append(
            f"- {s.get('screen_name','Unknown')} (purpose={s.get('purpose','other')}, "
            f"elements={len(s.get('elements', []))})"
        )
    return "\n".join(lines)


def _slug(text: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", (text or "").lower())).strip("_") or "general"


# ── Figma parser ──────────────────────────────────────────────────────────────
# Parsing now lives in `ingestion/ui_normalizer.py` (design-tool-agnostic, no
# hardcoded per-app purpose map). This module consumes the canonical UI IR it
# produces, or an IR supplied directly by the gateway (e.g. with LLM-classified
# purposes).


def _embed_texts(texts: list[str]) -> list[list[float] | None]:
    """Embed a list of texts, returning a same-length list (None entries on failure)."""
    if not texts or not embeddings.is_enabled():
        return [None] * len(texts)
    try:
        vecs = embeddings.embed_texts(texts)
        return vecs if vecs else [None] * len(texts)
    except Exception as e:  # noqa: BLE001 - never let embedding failure break ingestion
        print(f"[rag] embedding failed: {e}")
        return [None] * len(texts)


# ── Request models ────────────────────────────────────────────────────────────

class IngestFigmaRequest(BaseModel):
    project: str = Field(..., min_length=1)
    source_path: str = Field(..., min_length=1)
    figma_json: str | None = None
    # Optional pre-normalized canonical UI IR (e.g. produced by the gateway).
    ui_ir: dict | None = None
    # Optional {screen_name: purpose} hints (e.g. LLM-classified feature areas).
    purpose_hints: dict[str, str] | None = None


class RetrieveRequest(BaseModel):
    project: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    top_k: int = 5
    include_history: bool = True


class BriefContextRequest(BaseModel):
    project: str = Field(..., min_length=1)
    recent_limit: int = 12


class IngestSRSRequest(BaseModel):
    project: str = Field(..., min_length=1)
    source_path: str = Field(..., min_length=1)
    srs_text: str | None = None
    chunk_chars: int = 700
    srs_summary: str | None = None
    # Structured entity-graph payload from ingestion.extractor (requirements/entities/rules).
    extraction: dict | None = None


class LogTestRequest(BaseModel):
    project: str = Field(..., min_length=1)
    test_case_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    verdict: Literal["pass", "failed"]
    notes: str = ""
    area: str = "general"
    # Original requirement refs (e.g. ["FR-5","FR-7"]) this test covers -> COVERS edges.
    requirement_ids: list[str] = Field(default_factory=list)


class ResetProjectRequest(BaseModel):
    project: str = Field(..., min_length=1)
    delete_tests: bool = True
    delete_srs: bool = True
    delete_figma: bool = True


class GraphSubgraphRequest(BaseModel):
    project: str = Field(..., min_length=1)
    max_nodes: int = 300
    max_rels: int = 800


class SeedDemoTestsRequest(BaseModel):
    project: str = Field(..., min_length=1)
    area: str = "general"
    count: int = 6
    verdict_pattern: Literal["alternating", "all_pass", "all_failed"] = "alternating"


def _check_auth(authorization: str | None):
    if not RAG_API_KEY:
        return
    expected = f"Bearer {RAG_API_KEY}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _safe_node_id(n) -> str:
    if n is None:
        return "unknown::none"
    nid = n.get("id")
    if nid:
        return nid
    labels = list(n.labels) if getattr(n, "labels", None) else []
    label = labels[0] if labels else "Node"
    return f"{label}::{hash(str(dict(n)))}"


def _fetch_project_subgraph(project: str, max_nodes: int, max_rels: int) -> tuple[list[dict], list[dict]]:
    with driver.session() as session:
        node_rows = session.run(
            """
            MATCH (p:Project {name:$project})
            CALL {
              WITH p
              MATCH (p)-[:HAS_SUMMARY|HAS_SRS|HAS_FIGMA|HAS_TEST|HAS_FEATURE]->(n)
              RETURN n
              UNION
              WITH p
              MATCH (p)-[:HAS_SRS]->(:SRS)-[:HAS_CHUNK]->(n:Chunk)
              RETURN n
              UNION
              WITH p
              MATCH (p)-[:HAS_FIGMA]->(:FigmaScreen)-[:HAS_ELEMENT]->(n:UIElement)
              RETURN n
              UNION
              WITH p
              MATCH (p)-[:HAS_TEST]->(:TestCase)-[:HAS_RUN]->(n:TestRun)
              RETURN n
            }
            RETURN DISTINCT n
            LIMIT $max_nodes
            """,
            project=project,
            max_nodes=max_nodes,
        )

        nodes = []
        node_seen = set()
        for r in node_rows:
            n = r["n"]
            nid = _safe_node_id(n)
            if nid in node_seen:
                continue
            node_seen.add(nid)
            nodes.append({"id": nid, "labels": list(n.labels), "properties": dict(n)})

        rel_rows = session.run(
            """
            MATCH (p:Project {name:$project})
            OPTIONAL MATCH path = (p)-[*1..4]->(n)
            WHERE n IS NOT NULL
            UNWIND relationships(path) AS r
            RETURN DISTINCT r
            LIMIT $max_rels
            """,
            project=project,
            max_rels=max_rels,
        )

        relationships = []
        rel_seen = set()
        for rr in rel_rows:
            r = rr["r"]
            if r.id in rel_seen:
                continue
            rel_seen.add(r.id)
            start_node = r.start_node
            end_node = r.end_node
            sid = _safe_node_id(start_node)
            tid = _safe_node_id(end_node)
            relationships.append(
                {
                    "id": r.id,
                    "type": r.type,
                    "start": sid,
                    "end": tid,
                    "properties": dict(r),
                }
            )

    return nodes, relationships


def _graph_summary_payload(project: str, top: int) -> dict:
    top = max(3, min(top, 50))
    with driver.session() as session:
        rel_rows = session.run(
            """
            MATCH (p:Project {name:$project})
            CALL {
              WITH p
              MATCH (p)-[r:HAS_SUMMARY|HAS_SRS|HAS_FIGMA|HAS_TEST|HAS_FEATURE]->()
              RETURN r
              UNION
              WITH p
              MATCH (p)-[:HAS_SRS]->(:SRS)-[r:HAS_CHUNK|SUMMARIZED_AS]->()
              RETURN r
              UNION
              WITH p
              MATCH (p)-[:HAS_FIGMA]->(:FigmaScreen)-[r:HAS_ELEMENT|IN_FEATURE|SUMMARIZED_AS|RELATED_SCREEN]->()
              RETURN r
              UNION
              WITH p
              MATCH (p)-[:HAS_FIGMA]->(:FigmaScreen)-[:HAS_ELEMENT]->(:UIElement)-[r:IN_FEATURE|NEXT_UI|SAME_AS_UI]->()
              RETURN r
                            UNION
                            WITH p
                            MATCH (p)-[:HAS_FIGMA]->(:FigmaScreen)-[:HAS_ELEMENT]->(:UIElement)-[r:NAVIGATES_TO]->(:FigmaScreen)
                            RETURN r
              UNION
              WITH p
              MATCH (p)-[:HAS_TEST]->(:TestCase)-[r:HAS_RUN|COVERS_FEATURE]->()
              RETURN r
            }
            RETURN type(r) AS rel_type, count(DISTINCT r) AS count
            ORDER BY count DESC, rel_type ASC
            """,
            project=project,
        )
        relationship_types = [dict(r) for r in rel_rows]

        top_screens = [
            dict(r)
            for r in session.run(
                """
                MATCH (:Project {name:$project})-[:HAS_FIGMA]->(fs:FigmaScreen)
                RETURN fs.screen_name AS name,
                       fs.purpose AS purpose,
                       fs.element_count AS element_count,
                       fs.interactive_count AS interactive_count
                ORDER BY fs.element_count DESC, fs.screen_name ASC
                LIMIT $top
                """,
                project=project,
                top=top,
            )
        ]

        top_tests = [
            dict(r)
            for r in session.run(
                """
                MATCH (:Project {name:$project})-[:HAS_TEST]->(t:TestCase)
                OPTIONAL MATCH (t)-[:HAS_RUN]->(run:TestRun)
                RETURN t.id AS id,
                       t.title AS title,
                       t.last_verdict AS verdict,
                       t.last_run_at AS last_run_at,
                       count(run) AS run_count
                ORDER BY last_run_at DESC
                LIMIT $top
                """,
                project=project,
                top=top,
            )
        ]

        sample_links = [
            dict(r)
            for r in session.run(
                """
                MATCH (p:Project {name:$project})-[:HAS_FIGMA]->(fs:FigmaScreen)-[:HAS_ELEMENT]->(el:UIElement)
                OPTIONAL MATCH (el)-[:NEXT_UI]->(n:UIElement)
                RETURN fs.screen_name AS screen,
                       el.kind AS kind,
                       el.label AS element,
                       n.label AS next_element
                ORDER BY fs.screen_name ASC, el.order ASC
                LIMIT $top
                """,
                project=project,
                top=top,
            )
        ]

    return {
        "project": project,
        "relationship_types": relationship_types,
        "top_screens": top_screens,
        "recent_tests": top_tests,
        "sample_ui_flow": sample_links,
    }


def _render_terminal_graph(summary: dict) -> str:
    rels = summary.get("relationship_types", [])
    screens = summary.get("top_screens", [])
    tests = summary.get("recent_tests", [])
    ui_flow = summary.get("sample_ui_flow", [])

    lines = []
    lines.append(f"Project: {summary.get('project', '')}")
    lines.append("")
    lines.append("Relationship counts")
    lines.append("-------------------")
    if not rels:
        lines.append("(no relationships found)")
    else:
        for row in rels:
            lines.append(f"- {row.get('rel_type','?'):16} {row.get('count',0)}")

    lines.append("")
    lines.append("Top screens")
    lines.append("-----------")
    if not screens:
        lines.append("(no screens found)")
    else:
        for s in screens:
            lines.append(
                f"- {s.get('name','?')} | purpose={s.get('purpose','?')} | "
                f"elements={s.get('element_count',0)} | interactive={s.get('interactive_count',0)}"
            )

    lines.append("")
    lines.append("Recent tests")
    lines.append("------------")
    if not tests:
        lines.append("(no tests found)")
    else:
        for t in tests:
            lines.append(
                f"- [{t.get('verdict','unknown')}] {t.get('id','')} "
                f"({t.get('run_count',0)} runs) :: {t.get('title','')}"
            )

    lines.append("")
    lines.append("Sample UI flow links")
    lines.append("--------------------")
    if not ui_flow:
        lines.append("(no UI flow links found)")
    else:
        for r in ui_flow:
            nxt = r.get("next_element") or "(end)"
            lines.append(f"- {r.get('screen','?')}: {r.get('element','?')} -> {nxt}")

    return "\n".join(lines)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "neo4j_uri": NEO4J_URI}


@app.post("/project/reset")
def project_reset(req: ResetProjectRequest, authorization: str | None = Header(default=None)):
    """Delete project data slices to rebuild graph cleanly (tests/srs/figma)."""
    _check_auth(authorization)
    with driver.session() as session:
        if req.delete_tests:
            session.run(
                """
                MATCH (p:Project {name:$project})-[:HAS_TEST]->(t:TestCase)
                OPTIONAL MATCH (t)-[:HAS_RUN]->(r:TestRun)
                DETACH DELETE t, r
                """,
                project=req.project,
            )

        if req.delete_srs:
            session.run(
                """
                MATCH (p:Project {name:$project})-[:HAS_SRS]->(s:SRS)
                OPTIONAL MATCH (s)-[:HAS_CHUNK]->(c:Chunk)
                DETACH DELETE s, c
                """,
                project=req.project,
            )
            session.run(
                """
                MATCH (:Project {name:$project})-[:HAS_SUMMARY]->(sum:Summary {kind:'srs'})
                DETACH DELETE sum
                """,
                project=req.project,
            )
            # Entity-graph slice (requirements / rules / entities) is derived from the SRS.
            session.run(
                """
                MATCH (p:Project {name:$project})-[:HAS_REQUIREMENT]->(req:Requirement)
                OPTIONAL MATCH (req)-[:HAS_RULE]->(v:ValidationRule)
                DETACH DELETE req, v
                """,
                project=req.project,
            )
            session.run(
                "MATCH (e:Entity {project:$project}) DETACH DELETE e",
                project=req.project,
            )

        if req.delete_figma:
            session.run(
                """
                MATCH (p:Project {name:$project})-[:HAS_FIGMA]->(fs:FigmaScreen)
                OPTIONAL MATCH (fs)-[:HAS_ELEMENT]->(fe:UIElement)
                DETACH DELETE fs, fe
                """,
                project=req.project,
            )
            session.run(
                """
                MATCH (:Project {name:$project})-[:HAS_SUMMARY]->(sum:Summary {kind:'figma'})
                DETACH DELETE sum
                """,
                project=req.project,
            )

        # remove orphan feature nodes belonging to this project
        session.run(
            """
            MATCH (fa:FeatureArea {project:$project})
            WHERE NOT (fa)<-[:IN_FEATURE]-(:FigmaScreen)
              AND NOT (fa)<-[:COVERS_FEATURE]-(:TestCase)
            DETACH DELETE fa
            """,
            project=req.project,
        )

        session.run(
            """
            MATCH (p:Project {name:$project})
            SET p.updated_at = $now
            """,
            project=req.project,
            now=_utc_now(),
        )

    return {
        "status": "ok",
        "project": req.project,
        "deleted": {
            "tests": req.delete_tests,
            "srs": req.delete_srs,
            "figma": req.delete_figma,
        },
    }


@app.post("/ingest/srs")
def ingest_srs(req: IngestSRSRequest, authorization: str | None = Header(default=None)):
    _check_auth(authorization)

    text = req.srs_text
    if not text:
        src = Path(req.source_path)
        if not src.exists():
            raise HTTPException(status_code=404, detail=f"SRS file not found: {req.source_path}")
        text = src.read_text(encoding="utf-8", errors="ignore")

    overlap = max(40, min(120, int(req.chunk_chars * 0.12)))
    chunks = _split_text(text, chunk_chars=req.chunk_chars, overlap=overlap)
    srs_summary = (req.srs_summary or "").strip() or _build_srs_summary(text)
    now = _utc_now()
    srs_id = f"{req.project}::srs::{Path(req.source_path).name}"
    srs_summary_id = f"{req.project}::summary::srs"

    with driver.session() as session:
        session.run(
            """
            MERGE (p:Project {name:$project})
            ON CREATE SET p.created_at = $now
            SET p.updated_at = $now,
                p.srs_summary = $srs_summary
            MERGE (s:SRS {id:$srs_id})
            SET s.project = $project,
                s.source_path = $source_path,
                s.text = $text,
                s.updated_at = $now
            MERGE (p)-[:HAS_SRS]->(s)
            MERGE (sum:Summary {id:$summary_id})
            SET sum.project = $project,
                sum.kind = 'srs',
                sum.text = $srs_summary,
                sum.updated_at = $now
            MERGE (p)-[:HAS_SUMMARY]->(sum)
            MERGE (s)-[:SUMMARIZED_AS]->(sum)
            WITH s
            OPTIONAL MATCH (s)-[r:HAS_CHUNK]->(old:Chunk)
            DELETE r, old
            """,
            project=req.project, srs_id=srs_id,
            source_path=req.source_path, text=text, now=now, srs_summary=srs_summary,
            summary_id=srs_summary_id,
        )
        # Embed all chunks in one batch for semantic (vector) retrieval.
        chunk_vecs = _embed_texts(chunks)
        for idx, ch in enumerate(chunks):
            session.run(
                """
                MATCH (s:SRS {id:$srs_id})
                MERGE (c:Chunk {id:$chunk_id})
                SET c.project = $project, c.source = 'srs', c.order = $idx,
                    c.text = $text, c.updated_at = $now,
                    c.embedding = $embedding
                MERGE (s)-[:HAS_CHUNK]->(c)
                """,
                srs_id=srs_id, chunk_id=f"{srs_id}::chunk::{idx}",
                project=req.project, idx=idx, text=ch, now=now,
                embedding=chunk_vecs[idx],
            )

        # Build / refresh the requirement entity graph (GraphRAG layer).
        entity_stats = _write_entity_graph(session, req.project, req.extraction, now)

    return {
        "status": "ok",
        "project": req.project,
        "srs_id": srs_id,
        "chunks_written": len(chunks),
        "chunks_embedded": sum(1 for v in chunk_vecs if v is not None),
        "embedding_backend": embeddings.backend_name(),
        **entity_stats,
    }


def _write_entity_graph(session, project: str, extraction: dict | None, now: str) -> dict:
    """
    Persist the requirement entity graph extracted from the SRS:

        (Requirement)-[:INVOLVES]->(Entity)
        (Requirement)-[:HAS_RULE]->(ValidationRule)
        (Requirement)-[:IN_FEATURE]->(FeatureArea)
        (Project)-[:HAS_REQUIREMENT]->(Requirement)

    Requirements/entities/rules are project-scoped and fully replaced on re-ingest.
    """
    if not extraction or not isinstance(extraction, dict):
        return {"requirements_written": 0, "entities_written": 0, "validation_rules_written": 0}

    # Wipe previous entity-graph slice for this project (keep feature areas — shared with figma/tests).
    session.run(
        """
        MATCH (p:Project {name:$project})-[:HAS_REQUIREMENT]->(req:Requirement)
        OPTIONAL MATCH (req)-[:HAS_RULE]->(v:ValidationRule)
        DETACH DELETE req, v
        """,
        project=project,
    )
    session.run(
        "MATCH (e:Entity {project:$project}) DETACH DELETE e",
        project=project,
    )

    requirements = extraction.get("requirements", []) or []
    entities = extraction.get("entities", []) or []
    rules = extraction.get("validation_rules", []) or []

    # Embed requirement texts in one batch.
    req_texts = [str(r.get("text", "")) for r in requirements]
    req_vecs = _embed_texts(req_texts)

    for i, r in enumerate(requirements):
        ref_id = str(r.get("id", "") or f"R{i+1}")
        req_uid = f"{project}::req::{ref_id}"
        feature = ui_normalizer.slug(r.get("feature") or "general")
        session.run(
            """
            MATCH (p:Project {name:$project})
            MERGE (req:Requirement {id:$req_uid})
            SET req.project = $project, req.ref_id = $ref_id, req.type = $rtype,
                req.text = $text, req.actor = $actor, req.action = $action,
                req.priority = $priority, req.feature = $feature,
                req.objects = $objects, req.constraints = $constraints,
                req.acceptance = $acceptance, req.embedding = $embedding,
                req.updated_at = $now
            MERGE (p)-[:HAS_REQUIREMENT]->(req)
            MERGE (fa:FeatureArea {key:$feature_key})
            SET fa.project = $project, fa.label = $feature_label, fa.updated_at = $now
            MERGE (p)-[:HAS_FEATURE]->(fa)
            MERGE (req)-[:IN_FEATURE]->(fa)
            """,
            project=project, req_uid=req_uid, ref_id=ref_id,
            rtype=str(r.get("type", "functional")), text=str(r.get("text", "")),
            actor=str(r.get("actor", "")), action=str(r.get("action", "")),
            priority=str(r.get("priority", "medium")), feature=feature,
            objects=r.get("objects", []) or [], constraints=r.get("constraints", []) or [],
            acceptance=r.get("acceptance", []) or [], embedding=req_vecs[i],
            feature_key=f"{project}::{feature}", feature_label=feature.replace("_", " "),
            now=now,
        )

        # Link requirement to its domain objects as Entity nodes.
        for obj in (r.get("objects", []) or []):
            ent_name = str(obj).strip()
            if not ent_name:
                continue
            session.run(
                """
                MATCH (req:Requirement {id:$req_uid})
                MERGE (e:Entity {id:$ent_id})
                SET e.project = $project, e.name = $name, e.updated_at = $now
                MERGE (req)-[:INVOLVES]->(e)
                """,
                req_uid=req_uid, ent_id=f"{project}::entity::{ui_normalizer.slug(ent_name)}",
                project=project, name=ent_name, now=now,
            )

    # Standalone entities (with descriptions) from the extractor.
    for e in entities:
        name = str(e.get("name", "")).strip() if isinstance(e, dict) else str(e).strip()
        if not name:
            continue
        desc = str(e.get("description", "")).strip() if isinstance(e, dict) else ""
        session.run(
            """
            MATCH (p:Project {name:$project})
            MERGE (e:Entity {id:$ent_id})
            SET e.project = $project, e.name = $name, e.description = $desc, e.updated_at = $now
            """,
            project=project, ent_id=f"{project}::entity::{ui_normalizer.slug(name)}",
            name=name, desc=desc, now=now,
        )

    # Validation rules attached to their owning requirement.
    rules_written = 0
    for j, v in enumerate(rules):
        rule_text = str(v.get("rule", "")).strip()
        if not rule_text:
            continue
        owner_ref = str(v.get("requirement_id", "")).strip()
        owner_uid = f"{project}::req::{owner_ref}" if owner_ref else None
        session.run(
            """
            MATCH (p:Project {name:$project})
            MERGE (vr:ValidationRule {id:$vr_id})
            SET vr.project = $project, vr.field = $field, vr.rule = $rule, vr.updated_at = $now
            WITH vr
            OPTIONAL MATCH (owner:Requirement {id:$owner_uid})
            FOREACH (_ IN CASE WHEN owner IS NULL THEN [] ELSE [1] END |
                MERGE (owner)-[:HAS_RULE]->(vr)
            )
            """,
            project=project, vr_id=f"{project}::vrule::{j}",
            field=str(v.get("field", "")), rule=rule_text, owner_uid=owner_uid, now=now,
        )
        rules_written += 1

    return {
        "requirements_written": len(requirements),
        "entities_written": len({ui_normalizer.slug(str(e.get('name', '') if isinstance(e, dict) else e)) for e in entities} | {ui_normalizer.slug(str(o)) for r in requirements for o in (r.get('objects', []) or [])}),
        "validation_rules_written": rules_written,
        "requirements_embedded": sum(1 for v in req_vecs if v is not None),
    }


@app.post("/ingest/figma")
def ingest_figma(req: IngestFigmaRequest, authorization: str | None = Header(default=None)):
    _check_auth(authorization)

    raw = req.figma_json
    if not raw:
        src = Path(req.source_path)
        if not src.exists():
            raise HTTPException(status_code=404, detail=f"Figma JSON not found: {req.source_path}")
        raw = src.read_text(encoding="utf-8", errors="ignore")

    raw = _strip_markdown_fence(raw)

    try:
        figma_data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    # Accept a pre-normalized canonical UI IR (from the gateway, possibly with
    # LLM-classified purposes) or normalize the raw Figma export here.
    if req.ui_ir and isinstance(req.ui_ir, dict) and req.ui_ir.get("screens"):
        ui_ir = req.ui_ir
    else:
        ui_ir = ui_normalizer.normalize_figma(figma_data, req.purpose_hints)
    screens = ui_ir.get("screens", [])
    if not screens:
        raise HTTPException(status_code=400, detail="No screens found in Figma JSON")

    now = _utc_now()
    figma_name = ui_ir.get("source_name") or figma_data.get("name", Path(req.source_path).stem)
    figma_summary = _build_figma_summary(screens)
    total_elements = 0
    figma_summary_id = f"{req.project}::summary::figma"

    with driver.session() as session:
        # Wipe old Figma data for this project
        session.run(
            """
            MATCH (p:Project {name:$project})-[:HAS_FIGMA]->(fs:FigmaScreen)
            OPTIONAL MATCH (fs)-[:HAS_ELEMENT]->(fe:UIElement)
            DETACH DELETE fe, fs
            """,
            project=req.project,
        )

        session.run(
            """
            MERGE (p:Project {name:$project})
            ON CREATE SET p.created_at = $now
            SET p.updated_at = $now,
                p.figma_source = $figma_name,
                p.figma_summary = $figma_summary
            MERGE (sum:Summary {id:$summary_id})
            SET sum.project = $project,
                sum.kind = 'figma',
                sum.text = $figma_summary,
                sum.updated_at = $now
            MERGE (p)-[:HAS_SUMMARY]->(sum)
            """,
            project=req.project,
            now=now,
            figma_name=figma_name,
            figma_summary=figma_summary,
            summary_id=figma_summary_id,
        )

        for screen in screens:
            screen_id = f"{req.project}::figma::{screen['node_id']}"

            # Count interactive vs static
            interactive_count = sum(1 for e in screen["elements"] if e.get("interactive"))
            by_kind: dict[str, list[str]] = {}
            for el in screen["elements"]:
                by_kind.setdefault(el["kind"], []).append(el["label"])

            session.run(
                """
                MATCH (p:Project {name:$project})
                MERGE (fa:FeatureArea {key:$feature_key})
                SET fa.project = $project, fa.label = $feature_label, fa.updated_at = $now
                MERGE (fs:FigmaScreen {id:$screen_id})
                SET fs.project = $project,
                    fs.screen_name = $screen_name,
                    fs.purpose = $purpose,
                    fs.figma_node_id = $figma_node_id,
                    fs.element_count = $element_count,
                    fs.interactive_count = $interactive_count,
                    fs.updated_at = $now
                MERGE (p)-[:HAS_FIGMA]->(fs)
                MERGE (p)-[:HAS_FEATURE]->(fa)
                MERGE (fs)-[:IN_FEATURE]->(fa)
                MERGE (sum:Summary {id:$figma_summary_id})
                MERGE (fs)-[:SUMMARIZED_AS]->(sum)
                """,
                project=req.project,
                screen_id=screen_id,
                screen_name=screen["screen_name"],
                purpose=screen["purpose"],
                feature_key=f"{req.project}::{screen['purpose']}",
                feature_label=screen["purpose"].replace("_", " "),
                figma_node_id=screen["node_id"],
                element_count=len(screen["elements"]),
                interactive_count=interactive_count,
                now=now,
                figma_summary_id=figma_summary_id,
            )

            for idx, el in enumerate(screen["elements"]):
                el_id = f"{screen_id}::el::{idx}"
                session.run(
                    """
                    MATCH (fs:FigmaScreen {id:$screen_id})-[:IN_FEATURE]->(fa:FeatureArea)
                    MERGE (fe:UIElement {id:$el_id})
                    SET fe.project = $project,
                        fe.kind = $kind,
                        fe.label = $label,
                        fe.name = $name,
                        fe.order = $order,
                        fe.interactive = $interactive,
                        fe.figma_node_id = $figma_node_id,
                        fe.updated_at = $now
                    MERGE (fs)-[:HAS_ELEMENT]->(fe)
                    MERGE (fe)-[:IN_FEATURE]->(fa)
                    """,
                    screen_id=screen_id,
                    el_id=el_id,
                    project=req.project,
                    kind=el["kind"],
                    label=el["label"],
                    name=el["name"],
                    order=idx,
                    interactive=el.get("interactive", False),
                    figma_node_id=el.get("node_id", ""),
                    now=now,
                )
                total_elements += 1

            # Link elements in local sequence for navigable UI flow context
            session.run(
                """
                MATCH (fs:FigmaScreen {id:$screen_id})-[:HAS_ELEMENT]->(e:UIElement)
                WITH e ORDER BY e.order ASC
                WITH collect(e) AS els
                UNWIND range(0, size(els)-2) AS i
                WITH els[i] AS a, els[i+1] AS b
                MERGE (a)-[:NEXT_UI]->(b)
                """,
                screen_id=screen_id,
            )

    # Infer soft relationships between screens by shared purpose/element overlap.
    with driver.session() as session:
        # Screen-level similarity by shared element semantics
        session.run(
            """
            MATCH (p:Project {name:$project})-[:HAS_FIGMA]->(a:FigmaScreen)
            MATCH (p)-[:HAS_FIGMA]->(b:FigmaScreen)
            WHERE a.id < b.id
            OPTIONAL MATCH (a)-[:HAS_ELEMENT]->(ea:UIElement)
            OPTIONAL MATCH (b)-[:HAS_ELEMENT]->(eb:UIElement)
            WITH a,b,
                 sum(CASE WHEN ea.kind = eb.kind AND toLower(ea.label) = toLower(eb.label) THEN 1 ELSE 0 END) AS shared,
                 CASE WHEN a.purpose = b.purpose THEN 1 ELSE 0 END AS same_purpose
            WHERE same_purpose = 1 OR shared >= 2
            MERGE (a)-[r:RELATED_SCREEN]->(b)
            SET r.same_purpose = same_purpose, r.shared_elements = shared
            """,
            project=req.project,
        )

        # Element-level similarity across screens
        session.run(
            """
            MATCH (p:Project {name:$project})-[:HAS_FIGMA]->(:FigmaScreen)-[:HAS_ELEMENT]->(a:UIElement)
            MATCH (p)-[:HAS_FIGMA]->(:FigmaScreen)-[:HAS_ELEMENT]->(b:UIElement)
            WHERE a.id < b.id
              AND a.kind = b.kind
              AND toLower(a.label) = toLower(b.label)
            MERGE (a)-[r:SAME_AS_UI]->(b)
            SET r.confidence = 1.0
            """,
            project=req.project,
        )

        # Inferred button/navigation to screen transitions (NAVIGATES_TO)
        session.run(
            """
            MATCH (:Project {name:$project})-[:HAS_FIGMA]->(:FigmaScreen)-[:HAS_ELEMENT]->(e:UIElement)-[r:NAVIGATES_TO]->(:FigmaScreen)
            DELETE r
            """,
            project=req.project,
        )
        session.run(
            """
            MATCH (p:Project {name:$project})-[:HAS_FIGMA]->(src:FigmaScreen)-[:HAS_ELEMENT]->(btn:UIElement)
            MATCH (p)-[:HAS_FIGMA]->(dst:FigmaScreen)
            WHERE src.id <> dst.id
              AND btn.kind IN ['button','navigation']
              AND (
                    toLower(btn.label) CONTAINS toLower(dst.purpose)
                 OR toLower(btn.label) CONTAINS toLower(dst.screen_name)
                 OR any(tok IN split(toLower(dst.screen_name), ' ') WHERE size(tok) >= 4 AND toLower(btn.label) CONTAINS tok)
              )
            MERGE (btn)-[r:NAVIGATES_TO]->(dst)
            SET r.inferred = true,
                r.updated_at = $now
            """,
            project=req.project,
            now=now,
        )

        # Real prototype transitions from the UI IR (Figma reactions/transitionNodeID).
        # These are ground-truth links and override any label-inferred guess.
        real_transitions = 0
        for screen in screens:
            for tr in screen.get("transitions", []) or []:
                via = tr.get("via_node_id")
                to_node = tr.get("to_node_id")
                if not via or not to_node:
                    continue
                res = session.run(
                    """
                    MATCH (p:Project {name:$project})-[:HAS_FIGMA]->(dst:FigmaScreen {figma_node_id:$to_node})
                    MATCH (p)-[:HAS_FIGMA]->(:FigmaScreen)-[:HAS_ELEMENT]->(btn:UIElement {figma_node_id:$via})
                    MERGE (btn)-[r:NAVIGATES_TO]->(dst)
                    SET r.inferred = false, r.updated_at = $now
                    RETURN count(r) AS c
                    """,
                    project=req.project, to_node=to_node, via=via, now=now,
                ).single()
                real_transitions += (res["c"] if res else 0)

    return {
        "status": "ok",
        "project": req.project,
        "figma_name": figma_name,
        "screens_written": len(screens),
        "elements_written": total_elements,
        "real_transitions_written": real_transitions,
    }


@app.post("/context/brief")
def context_brief(req: BriefContextRequest, authorization: str | None = Header(default=None)):
    """Small stage-1 context: summaries + recent tests. No heavy chunks/elements."""
    _check_auth(authorization)
    with driver.session() as session:
        proj_row = session.run(
            """
            MATCH (p:Project {name:$project})
            RETURN p.srs_summary AS srs_summary,
                   p.figma_summary AS figma_summary,
                   p.figma_source AS figma_source
            """,
            project=req.project,
        ).single()

        tests = session.run(
            """
            MATCH (p:Project {name:$project})-[:HAS_TEST]->(t:TestCase)
            RETURN coalesce(t.external_id, t.id) AS id, t.title AS title, t.area AS area,
                   t.last_verdict AS verdict, t.last_run_at AS ts
            ORDER BY t.last_run_at DESC
            LIMIT $limit
            """,
            project=req.project,
            limit=req.recent_limit,
        )
        recent_tests = [dict(r) for r in tests]

        screens = session.run(
            """
            MATCH (p:Project {name:$project})-[:HAS_FIGMA]->(fs:FigmaScreen)
            RETURN fs.screen_name AS screen_name, fs.purpose AS purpose,
                   fs.interactive_count AS interactive_count
            ORDER BY fs.interactive_count DESC, fs.screen_name ASC
            LIMIT 20
            """,
            project=req.project,
        )
        screen_index = [dict(r) for r in screens]

    return {
        "project": req.project,
        "srs_summary": (proj_row["srs_summary"] if proj_row else "") or "",
        "figma_summary": (proj_row["figma_summary"] if proj_row else "") or "",
        "figma_source": (proj_row["figma_source"] if proj_row else "") or "",
        "recent_tests": recent_tests,
        "screen_index": screen_index,
    }


@app.get("/figma/screens")
def figma_screens(project: str, authorization: str | None = Header(default=None)):
    """List all Figma screens for a project with element counts."""
    _check_auth(authorization)
    with driver.session() as session:
        rows = session.run(
            """
            MATCH (p:Project {name:$project})-[:HAS_FIGMA]->(fs:FigmaScreen)
            RETURN fs.screen_name AS screen_name,
                   fs.purpose AS purpose,
                   fs.element_count AS element_count,
                   fs.interactive_count AS interactive_count
            ORDER BY fs.screen_name
            """,
            project=project,
        )
        screens = [dict(r) for r in rows]
    return {"project": project, "screens": screens}


@app.get("/figma/overview")
def figma_overview(project: str, top_labels: int = 4, authorization: str | None = Header(default=None)):
    """Compact all-screen UI overview for planner prompts."""
    _check_auth(authorization)
    top_labels = max(2, min(top_labels, 10))
    with driver.session() as session:
        rows = session.run(
            """
            MATCH (:Project {name:$project})-[:HAS_FIGMA]->(fs:FigmaScreen)
            OPTIONAL MATCH (fs)-[:HAS_ELEMENT]->(e:UIElement)
            WITH fs,
                 [x IN collect(CASE WHEN e.kind='button' THEN e.label END) WHERE x IS NOT NULL][0..$top_labels] AS buttons,
                 [x IN collect(CASE WHEN e.kind='input' THEN e.label END) WHERE x IS NOT NULL][0..$top_labels] AS inputs,
                 [x IN collect(CASE WHEN e.kind='navigation' THEN e.label END) WHERE x IS NOT NULL][0..$top_labels] AS nav
            RETURN fs.screen_name AS screen_name,
                   fs.purpose AS purpose,
                   fs.interactive_count AS interactive_count,
                   buttons,
                   inputs,
                   nav
            ORDER BY fs.interactive_count DESC, fs.screen_name ASC
            """,
            project=project,
            top_labels=top_labels,
        )
        screens = [dict(r) for r in rows]
    return {"project": project, "screens": screens}


@app.get("/figma/transitions")
def figma_transitions(
    project: str,
    screen_name: str | None = None,
    limit: int = 80,
    authorization: str | None = Header(default=None),
):
    """Return inferred UI navigation transitions (button/navigation element -> target screen)."""
    _check_auth(authorization)
    limit = max(10, min(limit, 400))
    with driver.session() as session:
        if screen_name:
            rows = session.run(
                """
                MATCH (:Project {name:$project})-[:HAS_FIGMA]->(src:FigmaScreen {screen_name:$screen_name})
                      -[:HAS_ELEMENT]->(e:UIElement)-[r:NAVIGATES_TO]->(dst:FigmaScreen)
                RETURN src.screen_name AS from_screen,
                       e.label AS via_element,
                       e.kind AS element_kind,
                       dst.screen_name AS to_screen,
                       dst.purpose AS to_purpose,
                       r.inferred AS inferred
                ORDER BY src.screen_name, via_element
                LIMIT $limit
                """,
                project=project,
                screen_name=screen_name,
                limit=limit,
            )
        else:
            rows = session.run(
                """
                MATCH (:Project {name:$project})-[:HAS_FIGMA]->(src:FigmaScreen)
                      -[:HAS_ELEMENT]->(e:UIElement)-[r:NAVIGATES_TO]->(dst:FigmaScreen)
                RETURN src.screen_name AS from_screen,
                       e.label AS via_element,
                       e.kind AS element_kind,
                       dst.screen_name AS to_screen,
                       dst.purpose AS to_purpose,
                       r.inferred AS inferred
                ORDER BY src.screen_name, via_element
                LIMIT $limit
                """,
                project=project,
                limit=limit,
            )
        items = [dict(r) for r in rows]
    return {"project": project, "screen_name": screen_name, "transitions": items}


@app.get("/figma/elements")
def figma_elements(
    project: str,
    screen_name: str,
    interactive_only: bool = True,
    authorization: str | None = Header(default=None),
):
    """
    Return UIElements for a specific screen.
    Returns grouped dict: {kind: [labels...]} — compact for prompt injection.
    """
    _check_auth(authorization)
    with driver.session() as session:
        query = """
            MATCH (p:Project {name:$project})-[:HAS_FIGMA]->(fs:FigmaScreen {screen_name:$screen_name})
                  -[:HAS_ELEMENT]->(fe:UIElement)
        """
        if interactive_only:
            query += " WHERE fe.interactive = true"
        query += " RETURN fe.kind AS kind, fe.label AS label ORDER BY fe.kind, fe.label"

        rows = session.run(query, project=project, screen_name=screen_name)
        by_kind: dict[str, list[str]] = {}
        for r in rows:
            by_kind.setdefault(r["kind"], []).append(r["label"])

    return {
        "project": project,
        "screen_name": screen_name,
        "elements": by_kind,
    }


@app.post("/tests/log")
def log_test(req: LogTestRequest, authorization: str | None = Header(default=None)):
    _check_auth(authorization)
    now = _utc_now()
    # Keep a stable internal testcase key by semantic title so repeated external IDs
    # (e.g., TC-001 in multiple rounds) do not overwrite different testcases.
    internal_test_id = f"{req.project}::tc::{_slug(req.title)}"
    run_id = f"{internal_test_id}::run::{now}"
    area_slug = _slug(req.area)

    with driver.session() as session:
        session.run(
            """
            MERGE (p:Project {name:$project})
            ON CREATE SET p.created_at = $now
            SET p.updated_at = $now
            MERGE (t:TestCase {id:$internal_test_id})
            SET t.project = $project, t.title = $title, t.area = $area,
                t.external_id = $external_test_case_id,
                t.last_verdict = $verdict, t.last_notes = $notes,
                t.last_run_at = $now, t.updated_at = $now
            MERGE (p)-[:HAS_TEST]->(t)
            MERGE (fa:FeatureArea {key:$feature_key})
            SET fa.project = $project, fa.label = $area, fa.updated_at = $now
            MERGE (p)-[:HAS_FEATURE]->(fa)
            MERGE (t)-[:COVERS_FEATURE]->(fa)
            MERGE (r:TestRun {id:$run_id})
            SET r.project = $project, r.verdict = $verdict,
                r.notes = $notes, r.created_at = $now
            MERGE (t)-[:HAS_RUN]->(r)
            """,
            project=req.project,
            internal_test_id=internal_test_id,
            external_test_case_id=req.test_case_id,
            title=req.title, area=req.area, verdict=req.verdict,
            notes=req.notes, now=now, run_id=run_id,
            feature_key=f"{req.project}::{area_slug}",
        )

        # Graph-native coverage: link this test to the requirements it exercises.
        covered = 0
        for ref in (req.requirement_ids or []):
            ref = str(ref).strip()
            if not ref:
                continue
            res = session.run(
                """
                MATCH (t:TestCase {id:$internal_test_id})
                MATCH (req:Requirement {id:$req_uid})
                MERGE (t)-[:COVERS]->(req)
                RETURN count(req) AS c
                """,
                internal_test_id=internal_test_id,
                req_uid=f"{req.project}::req::{ref}",
            ).single()
            covered += (res["c"] if res else 0)

    return {
        "status": "ok",
        "project": req.project,
        "test_case_id": req.test_case_id,
        "run_id": run_id,
        "requirements_linked": covered,
    }


@app.get("/coverage/requirements")
def coverage_requirements(project: str, authorization: str | None = Header(default=None)):
    """
    Graph-native requirement coverage: which requirements have tests linked via
    COVERS edges, computed from the entity graph rather than free-text area strings.
    """
    _check_auth(authorization)
    with driver.session() as session:
        totals = session.run(
            """
            MATCH (p:Project {name:$project})-[:HAS_REQUIREMENT]->(req:Requirement)
            OPTIONAL MATCH (req)<-[:COVERS]-(t:TestCase)
            WITH req, count(DISTINCT t) AS test_count,
                 sum(CASE WHEN t.last_verdict = 'failed' THEN 1 ELSE 0 END) AS fail_count
            RETURN count(req) AS total,
                   sum(CASE WHEN test_count > 0 THEN 1 ELSE 0 END) AS covered,
                   sum(CASE WHEN fail_count > 0 THEN 1 ELSE 0 END) AS failing
            """,
            project=project,
        ).single()

        per_feature = [
            dict(r)
            for r in session.run(
                """
                MATCH (p:Project {name:$project})-[:HAS_REQUIREMENT]->(req:Requirement)
                OPTIONAL MATCH (req)<-[:COVERS]-(t:TestCase)
                WITH req.feature AS feature, req,
                     CASE WHEN count(t) > 0 THEN 1 ELSE 0 END AS is_covered
                RETURN feature,
                       count(req) AS total,
                       sum(is_covered) AS covered
                ORDER BY total DESC, feature ASC
                """,
                project=project,
            )
        ]

        uncovered = [
            dict(r)
            for r in session.run(
                """
                MATCH (p:Project {name:$project})-[:HAS_REQUIREMENT]->(req:Requirement)
                WHERE NOT (req)<-[:COVERS]-(:TestCase)
                RETURN req.ref_id AS ref_id, req.feature AS feature, req.text AS text
                ORDER BY req.ref_id ASC
                LIMIT 100
                """,
                project=project,
            )
        ]

    total = (totals["total"] if totals else 0) or 0
    covered = (totals["covered"] if totals else 0) or 0
    failing = (totals["failing"] if totals else 0) or 0
    return {
        "project": project,
        "total_requirements": total,
        "covered_requirements": covered,
        "failing_requirements": failing,
        "coverage_pct": round(100 * covered / total) if total else 0,
        "per_feature": per_feature,
        "uncovered_requirements": uncovered,
    }


@app.get("/tests/recent")
def tests_recent(project: str, limit: int = 20, authorization: str | None = Header(default=None)):
    _check_auth(authorization)
    with driver.session() as session:
        rows = session.run(
            """
            MATCH (p:Project {name:$project})-[:HAS_TEST]->(t:TestCase)
            RETURN t.id AS id, t.title AS title, t.last_verdict AS verdict,
                   t.last_notes AS notes, t.last_run_at AS ts
            ORDER BY t.last_run_at DESC
            LIMIT $limit
            """,
            project=project, limit=limit,
        )
        tests = [dict(r) for r in rows]
    return {"project": project, "tests": tests}


@app.post("/demo/tests/seed")
def seed_demo_tests(req: SeedDemoTestsRequest, authorization: str | None = Header(default=None)):
    """Create deterministic demo tests so UI and retrieval endpoints are easy to validate."""
    _check_auth(authorization)
    count = max(1, min(req.count, 50))
    now = _utc_now()
    area_slug = _slug(req.area)

    created = []
    with driver.session() as session:
        session.run(
            """
            MERGE (p:Project {name:$project})
            ON CREATE SET p.created_at = $now
            SET p.updated_at = $now
            MERGE (fa:FeatureArea {key:$feature_key})
            SET fa.project = $project, fa.label = $area, fa.updated_at = $now
            MERGE (p)-[:HAS_FEATURE]->(fa)
            """,
            project=req.project,
            now=now,
            area=req.area,
            feature_key=f"{req.project}::{area_slug}",
        )

        for i in range(1, count + 1):
            tc_id = f"demo-{area_slug}-{i:03d}"
            title = f"Demo {req.area} testcase {i:03d}"
            if req.verdict_pattern == "all_pass":
                verdict = "pass"
            elif req.verdict_pattern == "all_failed":
                verdict = "failed"
            else:
                verdict = "failed" if i % 2 == 0 else "pass"

            notes = (
                "Auto-seeded demo testcase for endpoint validation."
                if verdict == "pass"
                else "Auto-seeded demo failure to test retry/regression flow."
            )
            run_id = f"{req.project}::{tc_id}::run::{now}::{i}"

            session.run(
                """
                MATCH (p:Project {name:$project})
                MERGE (t:TestCase {id:$test_case_id})
                SET t.project = $project, t.title = $title, t.area = $area,
                    t.last_verdict = $verdict, t.last_notes = $notes,
                    t.last_run_at = $now, t.updated_at = $now
                MERGE (p)-[:HAS_TEST]->(t)
                MATCH (fa:FeatureArea {key:$feature_key})
                MERGE (t)-[:COVERS_FEATURE]->(fa)
                MERGE (r:TestRun {id:$run_id})
                SET r.project = $project, r.verdict = $verdict,
                    r.notes = $notes, r.created_at = $now
                MERGE (t)-[:HAS_RUN]->(r)
                """,
                project=req.project,
                test_case_id=tc_id,
                title=title,
                area=req.area,
                verdict=verdict,
                notes=notes,
                now=now,
                run_id=run_id,
                feature_key=f"{req.project}::{area_slug}",
            )

            created.append(
                {
                    "test_case_id": tc_id,
                    "title": title,
                    "verdict": verdict,
                    "run_id": run_id,
                }
            )

    return {
        "status": "ok",
        "project": req.project,
        "created_count": len(created),
        "tests": created,
    }


def _keyword_chunks(session, project: str, tokens: list[str], limit: int) -> list[dict]:
    if not tokens:
        return []
    rows = session.run(
        """
        MATCH (p:Project {name:$project})-[:HAS_SRS]->(:SRS)-[:HAS_CHUNK]->(c:Chunk)
        WHERE c.text IS NOT NULL
        WITH c,
             reduce(sc = 0, t IN $tokens |
                sc + CASE WHEN toLower(c.text) CONTAINS t THEN 1 ELSE 0 END
             ) AS score
        WHERE score > 0
        RETURN c.id AS id, c.text AS text, score
        ORDER BY score DESC, c.order ASC
        LIMIT $limit
        """,
        project=project, tokens=tokens, limit=limit,
    )
    return [{"id": r["id"], "text": r["text"]} for r in rows if r.get("text")]


def _vector_chunks(session, project: str, qvec: list[float], limit: int) -> list[dict]:
    """Semantic chunk search via the native Neo4j vector index."""
    try:
        rows = session.run(
            """
            CALL db.index.vector.queryNodes($index, $k, $qvec) YIELD node, score
            WHERE node.project = $project AND node.text IS NOT NULL
            RETURN node.id AS id, node.text AS text, score
            ORDER BY score DESC
            LIMIT $limit
            """,
            index=CHUNK_VECTOR_INDEX, k=limit * 5, qvec=qvec, project=project, limit=limit,
        )
        return [{"id": r["id"], "text": r["text"]} for r in rows if r.get("text")]
    except Exception as e:  # noqa: BLE001 - index missing / old server: degrade to keyword
        print(f"[rag] vector chunk search unavailable: {e}")
        return []


def _vector_requirements(session, project: str, qvec: list[float], limit: int) -> list[dict]:
    """Semantic requirement search + their validation rules (graph hop)."""
    try:
        rows = session.run(
            """
            CALL db.index.vector.queryNodes($index, $k, $qvec) YIELD node, score
            WHERE node.project = $project
            OPTIONAL MATCH (node)-[:HAS_RULE]->(v:ValidationRule)
            WITH node, score, collect(v.rule) AS rules
            RETURN node.ref_id AS ref_id, node.text AS text, node.feature AS feature,
                   node.priority AS priority, rules
            ORDER BY score DESC
            LIMIT $limit
            """,
            index=REQUIREMENT_VECTOR_INDEX, k=limit * 5, qvec=qvec, project=project, limit=limit,
        )
        return [dict(r) for r in rows]
    except Exception as e:  # noqa: BLE001
        print(f"[rag] vector requirement search unavailable: {e}")
        return []


def _rrf_merge(keyword: list[dict], vector: list[dict], limit: int, k: int = 60) -> list[str]:
    """Reciprocal-rank fusion of keyword + vector chunk lists, returning unique texts."""
    scores: dict[str, float] = {}
    text_by_id: dict[str, str] = {}
    for ranked in (keyword, vector):
        for rank, item in enumerate(ranked):
            cid = item.get("id") or item.get("text", "")[:64]
            text_by_id.setdefault(cid, item.get("text", ""))
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    out: list[str] = []
    seen_norm: set[str] = set()
    for cid, _ in ordered:
        text = text_by_id.get(cid, "")
        norm = re.sub(r"\s+", " ", text.lower()).strip()
        if norm and norm not in seen_norm:
            seen_norm.add(norm)
            out.append(text)
        if len(out) >= limit:
            break
    return out


@app.post("/retrieve")
def retrieve(req: RetrieveRequest, authorization: str | None = Header(default=None)):
    """
    Hybrid retrieval over the knowledge graph:
      - semantic vector search over SRS chunks (Neo4j vector index)
      - keyword token-overlap search over SRS chunks
      - fused via reciprocal-rank fusion
      - graph hop: semantically matched Requirements + their ValidationRules

    Figma is NOT included here — query /figma/elements per screen separately.
    """
    _check_auth(authorization)
    tokens = _query_tokens(req.query)
    qvec = embeddings.embed_query(req.query) if embeddings.is_enabled() else None

    with driver.session() as session:
        keyword_hits = _keyword_chunks(session, req.project, tokens, req.top_k)
        vector_hits = _vector_chunks(session, req.project, qvec, req.top_k) if qvec else []
        req_hits = _vector_requirements(session, req.project, qvec, req.top_k) if qvec else []

        if vector_hits or keyword_hits:
            srs_chunks = _rrf_merge(keyword_hits, vector_hits, req.top_k)
            retrieval_mode = "hybrid" if (vector_hits and keyword_hits) else ("vector" if vector_hits else "keyword")
        else:
            fallback = session.run(
                """
                MATCH (p:Project {name:$project})-[:HAS_SRS]->(:SRS)-[:HAS_CHUNK]->(c:Chunk)
                RETURN c.text AS text ORDER BY c.order ASC LIMIT $top_k
                """,
                project=req.project, top_k=req.top_k,
            )
            srs_chunks = [r["text"] for r in fallback if r.get("text")]
            retrieval_mode = "fallback_ordered"

        recent_tests = []
        if req.include_history:
            test_rows = session.run(
                """
                MATCH (p:Project {name:$project})-[:HAS_TEST]->(t:TestCase)
                RETURN t.id AS id, t.title AS title, t.last_verdict AS verdict,
                       t.last_notes AS notes, t.last_run_at AS ts
                ORDER BY t.last_run_at DESC
                LIMIT $top_k
                """,
                project=req.project, top_k=req.top_k,
            )
            recent_tests = [dict(r) for r in test_rows]

    history_lines = [
        f"- [{t.get('verdict','unknown')}] {t.get('id','')}: {t.get('title','')}"
        + (f" | notes: {t.get('notes','')}" if t.get("notes") else "")
        for t in recent_tests
    ]

    requirement_lines = []
    for r in req_hits:
        line = f"- [{r.get('ref_id','?')}] {r.get('text','')}"
        if r.get("rules"):
            line += " (rules: " + "; ".join([x for x in r["rules"] if x][:3]) + ")"
        requirement_lines.append(line)

    context_parts = []
    if srs_chunks:
        context_parts.append("SRS requirements:\n" + "\n\n".join(srs_chunks))
    if requirement_lines:
        context_parts.append("Related requirements & validation rules:\n" + "\n".join(requirement_lines))
    if req.include_history and history_lines:
        context_parts.append("Recent test history:\n" + "\n".join(history_lines))

    return {
        "project": req.project,
        "context": "\n\n".join(context_parts).strip(),
        "chunks": srs_chunks,
        "requirements": req_hits,
        "recent_tests": recent_tests,
        "retrieval_mode": retrieval_mode,
    }


@app.get("/graph/stats")
def graph_stats(project: str, authorization: str | None = Header(default=None)):
    """Quick relationship diagnostics for knowledge-graph integrity checks."""
    _check_auth(authorization)
    # COUNT {} subqueries are evaluated independently — no cartesian-product blow-up
    # (the old stacked OPTIONAL MATCH form multiplied chunks × elements × requirements × ...).
    with driver.session() as session:
        row = session.run(
            """
            MATCH (p:Project {name:$project})
            RETURN
              COUNT { (p)-[:HAS_SRS]->(:SRS) } AS srs_count,
              COUNT { (p)-[:HAS_SRS]->(:SRS)-[:HAS_CHUNK]->(:Chunk) } AS chunk_count,
              COUNT { MATCH (p)-[:HAS_SRS]->(:SRS)-[:HAS_CHUNK]->(c:Chunk) WHERE c.embedding IS NOT NULL } AS embedded_chunk_count,
              COUNT { (p)-[:HAS_SUMMARY]->(:Summary) } AS summary_count,
              COUNT { (p)-[:HAS_FIGMA]->(:FigmaScreen) } AS figma_screen_count,
              COUNT { (p)-[:HAS_FIGMA]->(:FigmaScreen)-[:HAS_ELEMENT]->(:UIElement) } AS figma_element_count,
              COUNT { (p)-[:HAS_FEATURE]->(:FeatureArea) } AS feature_count,
              COUNT { (p)-[:HAS_TEST]->(:TestCase) } AS test_case_count,
              COUNT { (p)-[:HAS_TEST]->(:TestCase)-[:HAS_RUN]->(:TestRun) } AS test_run_count,
              COUNT { (p)-[:HAS_TEST]->(:TestCase)-[:COVERS_FEATURE]->(:FeatureArea) } AS covered_feature_count,
              COUNT { (p)-[:HAS_REQUIREMENT]->(:Requirement) } AS requirement_count,
              COUNT { (p)-[:HAS_REQUIREMENT]->(:Requirement)-[:HAS_RULE]->(:ValidationRule) } AS validation_rule_count,
              COUNT { MATCH (e:Entity) WHERE e.project = $project } AS entity_count,
              COUNT { (p)-[:HAS_REQUIREMENT]->(:Requirement)<-[:COVERS]-(:TestCase) } AS covered_requirement_count
            """,
            project=project,
        ).single()
    stats = dict(row or {})
    stats["figma_nodes_estimate"] = stats.get("figma_screen_count", 0) + stats.get("figma_element_count", 0)
    return {"project": project, "embedding_backend": embeddings.backend_name(), **stats}


@app.get("/graph/summary")
def graph_summary(project: str, top: int = 12, authorization: str | None = Header(default=None)):
    """Compact graph summary for quick inspection without large node/edge payloads."""
    _check_auth(authorization)
    return _graph_summary_payload(project=project, top=top)


@app.get("/graph/terminal", response_class=PlainTextResponse)
def graph_terminal(project: str, top: int = 12, authorization: str | None = Header(default=None)):
    """Terminal-friendly text view of graph relationships and sample links."""
    _check_auth(authorization)
    summary = _graph_summary_payload(project=project, top=top)
    return _render_terminal_graph(summary)


@app.post("/graph/subgraph")
def graph_subgraph(req: GraphSubgraphRequest, authorization: str | None = Header(default=None)):
    """Dedicated endpoint returning connected graph payload (nodes + relationships)."""
    _check_auth(authorization)
    max_nodes = max(10, min(req.max_nodes, 2000))
    max_rels = max(10, min(req.max_rels, 5000))
    nodes, relationships = _fetch_project_subgraph(req.project, max_nodes=max_nodes, max_rels=max_rels)

    return {
        "project": req.project,
        "node_count": len(nodes),
        "relationship_count": len(relationships),
        "nodes": nodes,
        "relationships": relationships,
    }


@app.get("/graph/visualize")
def graph_visualize(
    project: str,
    max_nodes: int = 300,
    max_rels: int = 800,
    include_properties: bool = False,
    authorization: str | None = Header(default=None),
):
    """
    Return graph in Cytoscape-friendly format for quick frontend/browser visualization.
    """
    _check_auth(authorization)
    max_nodes = max(10, min(max_nodes, 2000))
    max_rels = max(10, min(max_rels, 5000))
    nodes, relationships = _fetch_project_subgraph(project, max_nodes=max_nodes, max_rels=max_rels)

    cy_nodes = []
    for n in nodes:
        labels = n.get("labels") or ["Node"]
        data = {
            "id": n["id"],
            "label": labels[0],
            "labels": labels,
            "title": n.get("properties", {}).get("title") or n.get("properties", {}).get("screen_name") or n["id"],
        }
        if include_properties:
            data["properties"] = n.get("properties", {})
        cy_nodes.append({"data": data})

    cy_edges = []
    for r in relationships:
        data = {
            "id": str(r["id"]),
            "source": r["start"],
            "target": r["end"],
            "label": r["type"],
            "type": r["type"],
        }
        if include_properties:
            data["properties"] = r.get("properties", {})
        cy_edges.append({"data": data})

    return {
        "project": project,
        "format": "cytoscape",
        "elements": {
            "nodes": cy_nodes,
            "edges": cy_edges,
        },
        "counts": {
            "nodes": len(cy_nodes),
            "edges": len(cy_edges),
        },
    }


@app.get("/demo/endpoints")
def demo_endpoints():
    """Quick endpoint catalog with consistent sample payloads for demos and smoke tests."""
    return {
        "version": "v1",
        "groups": {
            "health": [
                {"method": "GET", "path": "/health"},
                {"method": "GET", "path": "/graph/stats?project=my-app"},
            ],
            "ingest": [
                {
                    "method": "POST",
                    "path": "/ingest/srs",
                    "body": {
                        "project": "my-app",
                        "source_path": "./data/inputs/SRS1.txt",
                    },
                },
                {
                    "method": "POST",
                    "path": "/ingest/figma",
                    "body": {
                        "project": "my-app",
                        "source_path": "./data/inputs/GENERATED_JSON.json",
                    },
                },
            ],
            "tests": [
                {
                    "method": "POST",
                    "path": "/demo/tests/seed",
                    "body": {
                        "project": "my-app",
                        "area": "data_entry",
                        "count": 6,
                        "verdict_pattern": "alternating",
                    },
                },
                {"method": "GET", "path": "/tests/recent?project=my-app&limit=10"},
            ],
            "retrieval": [
                {
                    "method": "POST",
                    "path": "/retrieve",
                    "body": {
                        "project": "my-app",
                        "query": "input validation rules before saving",
                        "top_k": 5,
                    },
                },
                {
                    "method": "POST",
                    "path": "/context/brief",
                    "body": {
                        "project": "my-app",
                        "recent_limit": 12,
                    },
                },
            ],
            "graph": [
                {
                    "method": "POST",
                    "path": "/graph/subgraph",
                    "body": {
                        "project": "my-app",
                        "max_nodes": 250,
                        "max_rels": 700,
                    },
                },
                {"method": "GET", "path": "/graph/summary?project=my-app&top=12"},
                {"method": "GET", "path": "/graph/terminal?project=my-app&top=12"},
                {"method": "GET", "path": "/graph/visualize?project=my-app"},
                {"method": "GET", "path": "/graph/cypher?project=my-app"},
            ],
        },
    }


@app.get("/graph/cypher")
def graph_cypher(project: str, authorization: str | None = Header(default=None)):
    """Ready-to-run Cypher snippets for Neo4j Explore to render connected graph."""
    _check_auth(authorization)
    q_overview = (
        "MATCH (p:Project {name:$project}) "
        "OPTIONAL MATCH (p)-[:HAS_SUMMARY|HAS_SRS|HAS_FIGMA|HAS_TEST|HAS_FEATURE]->(n) "
        "OPTIONAL MATCH (n)-[r]->(m) "
        "RETURN p,n,r,m LIMIT 500"
    )
    q_figma = (
        "MATCH (p:Project {name:$project})-[:HAS_FIGMA]->(fs:FigmaScreen)-[:HAS_ELEMENT]->(el:UIElement) "
        "OPTIONAL MATCH (el)-[r:SAME_AS_UI|NEXT_UI]->(el2:UIElement) "
        "RETURN fs,el,r,el2 LIMIT 500"
    )
    q_tests = (
        "MATCH (p:Project {name:$project})-[:HAS_TEST]->(t:TestCase)-[:HAS_RUN]->(run:TestRun) "
        "OPTIONAL MATCH (t)-[:COVERS_FEATURE]->(f:FeatureArea) "
        "RETURN t,run,f LIMIT 300"
    )
    return {
        "project": project,
        "queries": {
            "overview": q_overview,
            "figma_links": q_figma,
            "test_links": q_tests,
        },
    }
