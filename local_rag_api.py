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
        session.run("CREATE CONSTRAINT figma_screen_id IF NOT EXISTS FOR (fs:FigmaScreen) REQUIRE fs.id IS UNIQUE")
        session.run("CREATE CONSTRAINT figma_element_id IF NOT EXISTS FOR (fe:UIElement) REQUIRE fe.id IS UNIQUE")
        session.run("CREATE CONSTRAINT feature_key IF NOT EXISTS FOR (f:FeatureArea) REQUIRE f.key IS UNIQUE")
        session.run("CREATE CONSTRAINT summary_id IF NOT EXISTS FOR (s:Summary) REQUIRE s.id IS UNIQUE")
    yield


app = FastAPI(title="Local Neo4j RAG API", lifespan=lifespan)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _split_text(text: str, chunk_chars: int = 1200, overlap: int = 120) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []

    # Semantic-ish SRS chunking: group FR lines first (better retrieval granularity)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    req_lines = [ln for ln in lines if re.match(r"^FR-\d+", ln, flags=re.IGNORECASE)]
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

def _all_text_in_subtree(node: dict, max_depth: int = 5, _depth: int = 0) -> list[str]:
    """Collect all visible TEXT characters from a node's subtree."""
    if _depth > max_depth:
        return []
    texts = []
    if node.get("type") == "TEXT" and node.get("characters", "").strip():
        texts.append(node["characters"].strip())
    for child in node.get("children", []):
        texts.extend(_all_text_in_subtree(child, max_depth, _depth + 1))
    return texts


def _element_label(node: dict) -> str:
    """Best human-readable label for a Figma node."""
    texts = _all_text_in_subtree(node, max_depth=4)
    # Deduplicate while preserving order
    seen: dict[str, None] = {}
    for t in texts:
        seen[t] = None
    label = " / ".join(seen.keys())
    return label[:120] if label else node.get("name", "")


def _parse_figma_screens(figma_data: dict) -> list[dict]:
    """
    Parse a Figma file export into a list of screen dicts.

    Each screen dict:
      screen_name, figma_node_id, purpose, elements[]
        element: {kind, label, name, figma_node_id, interactive}
    """
    screens: list[dict] = []
    doc = figma_data.get("document", {})
    pages = doc.get("children", []) if isinstance(doc, dict) else []

    # Some exports can be already page-level or frame-level. Normalize candidates.
    frame_candidates = []
    if pages:
        for page in pages:
            if page.get("type") == "FRAME":
                frame_candidates.append(page)
            frame_candidates.extend([c for c in page.get("children", []) if c.get("type") == "FRAME"])
    elif figma_data.get("children"):
        frame_candidates = [c for c in figma_data.get("children", []) if c.get("type") == "FRAME"]

    if not frame_candidates and doc.get("type") == "FRAME":
        frame_candidates = [doc]

    for frame in frame_candidates:
        screen: dict = {
            "screen_name": frame.get("name", "Unknown"),
            "figma_node_id": frame.get("id", ""),
            "purpose": _infer_purpose(frame.get("name", "")),
            "elements": [],
        }
        _walk_for_elements(frame, screen["elements"], depth=0, max_depth=9)
        # Deduplicate elements by (kind, label)
        seen: set[tuple] = set()
        deduped: list[dict] = []
        for el in screen["elements"]:
            key = (el["kind"], el["label"])
            if key not in seen and el["label"]:
                seen.add(key)
                deduped.append(el)
        screen["elements"] = deduped
        screens.append(screen)

    return screens


def _infer_purpose(name: str) -> str:
    """Map screen name to a short slug for querying."""
    name_lower = name.lower()
    mapping = {
        "create contact": "create_contact",
        "contact details": "contact_details",
        "contacts list": "contact_list",
        "search": "search",
        "settings": "settings",
        "organise": "organise",
        "highlights": "highlights",
    }
    for key, val in mapping.items():
        if key in name_lower:
            return val
    return "other"


# Keywords that identify interactive UI elements by Figma layer name
_BUTTON_KW = ("button", "btn", "fab", "link -", "cta")
_INPUT_KW = ("input", "field", "textarea", "textfield", "text field", "phone input", "search bar")
_NAV_KW = ("bottom navigation", "bottomnavbar", "bottom nav", "tab bar", "tabbar")
_TOGGLE_KW = ("toggle", "switch", "checkbox", "radio")
_DROPDOWN_KW = ("dropdown", "select", "picker", "spinner")


def _walk_for_elements(node: dict, out: list, depth: int, max_depth: int):
    if depth > max_depth or node.get("visible") is False:
        return

    ntype = node.get("type", "")
    name = node.get("name", "")
    name_lower = name.lower()

    element: dict | None = None

    if ntype == "FRAME":
        if any(kw in name_lower for kw in _BUTTON_KW):
            label = _element_label(node)
            element = {"kind": "button", "label": label, "name": name, "interactive": True}

        elif any(kw in name_lower for kw in _INPUT_KW):
            label = _element_label(node)
            element = {"kind": "input", "label": label or name, "name": name, "interactive": True}

        elif any(kw in name_lower for kw in _NAV_KW):
            # For nav bars extract the individual tabs as separate elements
            tabs = [
                c.get("characters", "").strip()
                for c in _iter_text_nodes(node)
                if c.get("characters", "").strip()
            ]
            tab_label = " | ".join(dict.fromkeys(tabs)) if tabs else name
            element = {"kind": "navigation", "label": tab_label, "name": name, "interactive": True}

        elif any(kw in name_lower for kw in _TOGGLE_KW):
            label = _element_label(node)
            element = {"kind": "control", "label": label or name, "name": name, "interactive": True}

        elif any(kw in name_lower for kw in _DROPDOWN_KW):
            label = _element_label(node)
            element = {"kind": "dropdown", "label": label or name, "name": name, "interactive": True}

        elif "section" in name_lower or "header" in name_lower or "heading" in name_lower:
            texts = _all_text_in_subtree(node, max_depth=2)
            if texts:
                element = {"kind": "section", "label": texts[0], "name": name, "interactive": False}

    # Recurse into children regardless of whether we captured this node
    for child in node.get("children", []):
        _walk_for_elements(child, out, depth + 1, max_depth)

    if element:
        element["figma_node_id"] = node.get("id", "")
        out.append(element)


def _iter_text_nodes(node: dict, _depth: int = 0):
    """Yield all TEXT nodes in subtree."""
    if _depth > 6:
        return
    if node.get("type") == "TEXT":
        yield node
    for child in node.get("children", []):
        yield from _iter_text_nodes(child, _depth + 1)


# ── Request models ────────────────────────────────────────────────────────────

class IngestFigmaRequest(BaseModel):
    project: str = Field(..., min_length=1)
    source_path: str = Field(..., min_length=1)
    figma_json: str | None = None


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


class LogTestRequest(BaseModel):
    project: str = Field(..., min_length=1)
    test_case_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    verdict: Literal["pass", "failed"]
    notes: str = ""
    area: str = "general"


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
        for idx, ch in enumerate(chunks):
            session.run(
                """
                MATCH (s:SRS {id:$srs_id})
                MERGE (c:Chunk {id:$chunk_id})
                SET c.project = $project, c.source = 'srs', c.order = $idx,
                    c.text = $text, c.updated_at = $now
                MERGE (s)-[:HAS_CHUNK]->(c)
                """,
                srs_id=srs_id, chunk_id=f"{srs_id}::chunk::{idx}",
                project=req.project, idx=idx, text=ch, now=now,
            )

    return {"status": "ok", "project": req.project, "srs_id": srs_id, "chunks_written": len(chunks)}


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

    screens = _parse_figma_screens(figma_data)
    if not screens:
        raise HTTPException(status_code=400, detail="No screens found in Figma JSON")

    now = _utc_now()
    figma_name = figma_data.get("name", Path(req.source_path).stem)
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
            screen_id = f"{req.project}::figma::{screen['figma_node_id']}"

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
                figma_node_id=screen["figma_node_id"],
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
                    figma_node_id=el.get("figma_node_id", ""),
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

    return {
        "status": "ok",
        "project": req.project,
        "figma_name": figma_name,
        "screens_written": len(screens),
        "elements_written": total_elements,
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

    return {"status": "ok", "project": req.project, "test_case_id": req.test_case_id, "run_id": run_id}


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


@app.post("/retrieve")
def retrieve(req: RetrieveRequest, authorization: str | None = Header(default=None)):
    """
    Retrieve SRS context + recent test history.
    Figma is NOT included here — query /figma/elements per screen separately.
    """
    _check_auth(authorization)
    tokens = _query_tokens(req.query)

    with driver.session() as session:
        # token-overlap scoring for semantically better chunk selection
        srs_rows = session.run(
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
            LIMIT $top_k
            """,
            project=req.project,
            tokens=tokens,
            top_k=req.top_k,
        )
        srs_chunks = [r["text"] for r in srs_rows if r.get("text")]

        if not srs_chunks:
            fallback = session.run(
                """
                MATCH (p:Project {name:$project})-[:HAS_SRS]->(:SRS)-[:HAS_CHUNK]->(c:Chunk)
                RETURN c.text AS text ORDER BY c.order ASC LIMIT $top_k
                """,
                project=req.project, top_k=req.top_k,
            )
            srs_chunks = [r["text"] for r in fallback if r.get("text")]

        # de-duplicate near-identical chunks
        deduped: list[str] = []
        seen = set()
        for c in srs_chunks:
            key = re.sub(r"\s+", " ", (c or "").lower()).strip()
            if key and key not in seen:
                seen.add(key)
                deduped.append(c)
        srs_chunks = deduped[: max(1, req.top_k)]

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

    context_parts = []
    if srs_chunks:
        context_parts.append("SRS requirements:\n" + "\n\n".join(srs_chunks))
    if req.include_history and history_lines:
        context_parts.append("Recent test history:\n" + "\n".join(history_lines))

    return {
        "project": req.project,
        "context": "\n\n".join(context_parts).strip(),
        "chunks": srs_chunks,
        "recent_tests": recent_tests,
    }


@app.get("/graph/stats")
def graph_stats(project: str, authorization: str | None = Header(default=None)):
    """Quick relationship diagnostics for knowledge-graph integrity checks."""
    _check_auth(authorization)
    with driver.session() as session:
        row = session.run(
            """
            MATCH (p:Project {name:$project})
            OPTIONAL MATCH (p)-[:HAS_SRS]->(s:SRS)
            OPTIONAL MATCH (s)-[:HAS_CHUNK]->(c:Chunk)
            OPTIONAL MATCH (p)-[:HAS_SUMMARY]->(sum:Summary)
            OPTIONAL MATCH (p)-[:HAS_FIGMA]->(fs:FigmaScreen)
            OPTIONAL MATCH (fs)-[:HAS_ELEMENT]->(el:UIElement)
            OPTIONAL MATCH (fs)-[:RELATED_SCREEN]-(:FigmaScreen)
                 OPTIONAL MATCH (el)-[:SAME_AS_UI]-(:UIElement)
                 OPTIONAL MATCH (el)-[:NEXT_UI]->(:UIElement)
             OPTIONAL MATCH (p)-[:HAS_FEATURE]->(f_all:FeatureArea)
            OPTIONAL MATCH (p)-[:HAS_TEST]->(t:TestCase)
            OPTIONAL MATCH (t)-[:HAS_RUN]->(r:TestRun)
             OPTIONAL MATCH (t)-[:COVERS_FEATURE]->(f_cov:FeatureArea)
            RETURN count(DISTINCT s) AS srs_count,
                   count(DISTINCT c) AS chunk_count,
                     count(DISTINCT sum) AS summary_count,
                   count(DISTINCT fs) AS figma_screen_count,
                   count(DISTINCT el) AS figma_element_count,
                 count(DISTINCT f_all) AS feature_count,
                   count(DISTINCT t) AS test_case_count,
                   count(DISTINCT r) AS test_run_count,
                 count(DISTINCT f_cov) AS covered_feature_count,
                   count(DISTINCT fs) + count(DISTINCT el) AS figma_nodes_estimate
            """,
            project=project,
        ).single()
    return {"project": project, **dict(row or {})}


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
                {"method": "GET", "path": "/graph/stats?project=contacts-app"},
            ],
            "ingest": [
                {
                    "method": "POST",
                    "path": "/ingest/srs",
                    "body": {
                        "project": "contacts-app",
                        "source_path": "./SRS1.txt",
                    },
                },
                {
                    "method": "POST",
                    "path": "/ingest/figma",
                    "body": {
                        "project": "contacts-app",
                        "source_path": "./GENERATED_JSON.json",
                    },
                },
            ],
            "tests": [
                {
                    "method": "POST",
                    "path": "/demo/tests/seed",
                    "body": {
                        "project": "contacts-app",
                        "area": "create_contact",
                        "count": 6,
                        "verdict_pattern": "alternating",
                    },
                },
                {"method": "GET", "path": "/tests/recent?project=contacts-app&limit=10"},
            ],
            "retrieval": [
                {
                    "method": "POST",
                    "path": "/retrieve",
                    "body": {
                        "project": "contacts-app",
                        "query": "create contact validation",
                        "top_k": 5,
                    },
                },
                {
                    "method": "POST",
                    "path": "/context/brief",
                    "body": {
                        "project": "contacts-app",
                        "recent_limit": 12,
                    },
                },
            ],
            "graph": [
                {
                    "method": "POST",
                    "path": "/graph/subgraph",
                    "body": {
                        "project": "contacts-app",
                        "max_nodes": 250,
                        "max_rels": 700,
                    },
                },
                {"method": "GET", "path": "/graph/summary?project=contacts-app&top=12"},
                {"method": "GET", "path": "/graph/terminal?project=contacts-app&top=12"},
                {"method": "GET", "path": "/graph/visualize?project=contacts-app"},
                {"method": "GET", "path": "/graph/cypher?project=contacts-app"},
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
