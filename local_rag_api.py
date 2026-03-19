from datetime import datetime, timezone
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
import os

from fastapi import FastAPI, Header, HTTPException
from neo4j import GraphDatabase
from pydantic import BaseModel, Field

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "raihanrashid")
RAG_API_KEY = os.getenv("RAG_API_KEY", "")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


@asynccontextmanager
async def lifespan(_: FastAPI):
    with driver.session() as session:
        session.run("CREATE CONSTRAINT project_name IF NOT EXISTS FOR (p:Project) REQUIRE p.name IS UNIQUE")
        session.run("CREATE CONSTRAINT srs_id IF NOT EXISTS FOR (s:SRS) REQUIRE s.id IS UNIQUE")
        session.run("CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
        session.run("CREATE CONSTRAINT testcase_id IF NOT EXISTS FOR (t:TestCase) REQUIRE t.id IS UNIQUE")
        session.run("CREATE CONSTRAINT testrun_id IF NOT EXISTS FOR (r:TestRun) REQUIRE r.id IS UNIQUE")
    yield


app = FastAPI(title="Local Neo4j RAG API", lifespan=lifespan)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _split_text(text: str, chunk_chars: int = 1200, overlap: int = 120) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []

    chunks: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + chunk_chars, n)
        chunk = text[i:j].strip()
        if chunk:
            chunks.append(chunk)
        if j >= n:
            break
        i = max(j - overlap, i + 1)
    return chunks


class RetrieveRequest(BaseModel):
    project: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    top_k: int = 5


class IngestSRSRequest(BaseModel):
    project: str = Field(..., min_length=1)
    source_path: str = Field(..., min_length=1)
    srs_text: str | None = None
    chunk_chars: int = 1200


class LogTestRequest(BaseModel):
    project: str = Field(..., min_length=1)
    test_case_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    verdict: Literal["pass", "failed"]
    notes: str = ""
    area: str = "general"


def _check_auth(authorization: str | None):
    if not RAG_API_KEY:
        return
    expected = f"Bearer {RAG_API_KEY}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/health")
def health():
    return {"status": "ok", "neo4j_uri": NEO4J_URI}


@app.post("/ingest/srs")
def ingest_srs(req: IngestSRSRequest, authorization: str | None = Header(default=None)):
    _check_auth(authorization)

    text = req.srs_text
    if not text:
        src = Path(req.source_path)
        if not src.exists():
            raise HTTPException(status_code=404, detail=f"SRS file not found: {req.source_path}")
        text = src.read_text(encoding="utf-8", errors="ignore")

    chunks = _split_text(text, chunk_chars=req.chunk_chars)
    now = _utc_now()
    srs_id = f"{req.project}::srs::{Path(req.source_path).name}"

    with driver.session() as session:
        session.run(
            """
            MERGE (p:Project {name:$project})
            ON CREATE SET p.created_at = $now
            SET p.updated_at = $now
            MERGE (s:SRS {id:$srs_id})
            SET s.project = $project,
                s.source_path = $source_path,
                s.text = $text,
                s.updated_at = $now
            MERGE (p)-[:HAS_SRS]->(s)
            WITH s
            OPTIONAL MATCH (s)-[r:HAS_CHUNK]->(old:Chunk)
            DELETE r, old
            """,
            project=req.project,
            srs_id=srs_id,
            source_path=req.source_path,
            text=text,
            now=now,
        )

        for idx, ch in enumerate(chunks):
            session.run(
                """
                MATCH (s:SRS {id:$srs_id})
                MERGE (c:Chunk {id:$chunk_id})
                SET c.project = $project,
                    c.source = 'srs',
                    c.order = $idx,
                    c.text = $text,
                    c.updated_at = $now
                MERGE (s)-[:HAS_CHUNK]->(c)
                """,
                srs_id=srs_id,
                chunk_id=f"{srs_id}::chunk::{idx}",
                project=req.project,
                idx=idx,
                text=ch,
                now=now,
            )

    return {"status": "ok", "project": req.project, "srs_id": srs_id, "chunks_written": len(chunks)}


@app.post("/tests/log")
def log_test(req: LogTestRequest, authorization: str | None = Header(default=None)):
    _check_auth(authorization)
    now = _utc_now()
    run_id = f"{req.project}::{req.test_case_id}::run::{now}"

    with driver.session() as session:
        session.run(
            """
            MERGE (p:Project {name:$project})
            ON CREATE SET p.created_at = $now
            SET p.updated_at = $now
            MERGE (t:TestCase {id:$test_case_id})
            SET t.project = $project,
                t.title = $title,
                t.area = $area,
                t.last_verdict = $verdict,
                t.last_notes = $notes,
                t.last_run_at = $now,
                t.updated_at = $now
            MERGE (p)-[:HAS_TEST]->(t)
            MERGE (r:TestRun {id:$run_id})
            SET r.project = $project,
                r.verdict = $verdict,
                r.notes = $notes,
                r.created_at = $now
            MERGE (t)-[:HAS_RUN]->(r)
            """,
            project=req.project,
            test_case_id=req.test_case_id,
            title=req.title,
            area=req.area,
            verdict=req.verdict,
            notes=req.notes,
            now=now,
            run_id=run_id,
        )

    return {"status": "ok", "project": req.project, "test_case_id": req.test_case_id, "run_id": run_id}


@app.get("/tests/recent")
def tests_recent(project: str, limit: int = 20, authorization: str | None = Header(default=None)):
    _check_auth(authorization)
    with driver.session() as session:
        rows = session.run(
            """
            MATCH (p:Project {name:$project})-[:HAS_TEST]->(t:TestCase)
            RETURN t.id AS id, t.title AS title, t.last_verdict AS verdict, t.last_notes AS notes, t.last_run_at AS ts
            ORDER BY t.last_run_at DESC
            LIMIT $limit
            """,
            project=project,
            limit=limit,
        )
        tests = [dict(r) for r in rows]
    return {"project": project, "tests": tests}


@app.post("/retrieve")
def retrieve(req: RetrieveRequest, authorization: str | None = Header(default=None)):
    _check_auth(authorization)

    with driver.session() as session:
        srs_rows = session.run(
            """
            MATCH (p:Project {name:$project})-[:HAS_SRS]->(:SRS)-[:HAS_CHUNK]->(c:Chunk)
            WHERE c.text IS NOT NULL AND toLower(c.text) CONTAINS toLower($q)
            RETURN c.text AS text
            LIMIT $top_k
            """,
            project=req.project,
            q=req.query,
            top_k=req.top_k,
        )
        srs_chunks = [r["text"] for r in srs_rows if r.get("text")]

        if not srs_chunks:
            fallback_rows = session.run(
                """
                MATCH (p:Project {name:$project})-[:HAS_SRS]->(:SRS)-[:HAS_CHUNK]->(c:Chunk)
                RETURN c.text AS text
                ORDER BY c.order ASC
                LIMIT $top_k
                """,
                project=req.project,
                top_k=req.top_k,
            )
            srs_chunks = [r["text"] for r in fallback_rows if r.get("text")]

        test_rows = session.run(
            """
            MATCH (p:Project {name:$project})-[:HAS_TEST]->(t:TestCase)
            RETURN t.id AS id, t.title AS title, t.last_verdict AS verdict, t.last_notes AS notes, t.last_run_at AS ts
            ORDER BY t.last_run_at DESC
            LIMIT $top_k
            """,
            project=req.project,
            top_k=req.top_k,
        )
        recent_tests = [dict(r) for r in test_rows]

    history_lines = [
        f"- [{t.get('verdict', 'unknown')}] {t.get('id', '')}: {t.get('title', '')}"
        + (f" | notes: {t.get('notes', '')}" if t.get("notes") else "")
        for t in recent_tests
    ]

    context_parts = []
    if srs_chunks:
        context_parts.append("SRS snippets:\n" + "\n\n".join(srs_chunks))
    if history_lines:
        context_parts.append("Recent executed tests:\n" + "\n".join(history_lines))

    context = "\n\n".join(context_parts).strip()

    return {
        "project": req.project,
        "context": context,
        "chunks": srs_chunks,
        "recent_tests": recent_tests,
    }
