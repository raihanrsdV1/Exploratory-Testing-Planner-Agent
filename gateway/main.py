"""
Exploratory Testing Planner — Agent Gateway (FastAPI router).

This file is intentionally thin: it defines the HTTP surface (routes + OpenAPI
docs) and delegates all logic to the modular `planner` package:

    planner.config            env config + auth
    planner.model_client      LLM backends
    planner.rag_client        knowledge-graph HTTP client
    planner.textutil          pure JSON/similarity/query helpers
    planner.coverage          live coverage map + directives
    planner.context_builders  prompt context formatting
    planner.prompts           exploratory-testing prompt builders
    planner.schemas           request models
    planner.pipeline          orchestration

Nothing here is specific to any single app — all domain knowledge comes from the
ingested SRS/UI knowledge graph at request time.
"""

from pathlib import Path

import requests
from fastapi import FastAPI, Header
from fastapi.responses import HTMLResponse, Response

from observability import setup_logging
from observability.middleware import RequestLoggingMiddleware
from observability.metrics import get_metrics

from planner import config, model_client, pipeline
from planner.schemas import (
    ChatRequest,
    IngestDefectsRequest,
    IngestFigmaRequest,
    IngestSRSRequest,
    LogVerdictRequest,
    NextTestCaseRequest,
    ResetProjectRequest,
    SessionEndRequest,
    SessionStartRequest,
)

setup_logging()

app = FastAPI(
    title="Exploratory Testing Planner — Agent Gateway",
    description=(
        "Orchestrates an adaptive exploratory QA testing session for any app.\n\n"
        "**How it works:**\n"
        "1. Ingest the app's SRS (any format) and Figma export into a Neo4j knowledge graph.\n"
        "2. Call `/agent/next-testcase` — the planner retrieves targeted context (semantic + keyword "
        "hybrid) and generates a structured JSON test case, guided by live coverage state and "
        "exploratory-testing heuristics.\n"
        "3. The executor (e.g. Droidrun) runs the test on a real device.\n"
        "4. Call `/agent/log-verdict-and-next` — the verdict is logged and the next test case "
        "is generated, adapting toward failures and unexplored areas automatically.\n\n"
        "**Authentication:** Set `GATEWAY_API_KEY` in `.env` to require "
        "`Authorization: Bearer <key>` on every request. Leave blank to disable auth."
    ),
    version="2.0.0",
    openapi_tags=[
        {"name": "system",  "description": "Health checks and backend diagnostics."},
        {"name": "ingest",  "description": "Load SRS and Figma design files into the knowledge graph."},
        {"name": "project", "description": "Manage project-level data (reset slices, lifecycle)."},
        {"name": "agent",   "description": "Core exploratory test generation and coverage tracking."},
        {"name": "chat",    "description": "RAG-backed free-form Q&A against the project knowledge graph."},
    ],
)

app.add_middleware(RequestLoggingMiddleware)

@app.get("/metrics", tags=["system"], summary="Get system metrics")
def metrics():
    return get_metrics()


_DASH_REACT = Path(__file__).resolve().parent.parent / "dashboard-react" / "dist" / "index.html"
_DASH_VANILLA = Path(__file__).resolve().parent.parent / "dashboard" / "index.html"


@app.get("/dashboard", include_in_schema=False)
def dashboard_page():
    """Serve the operator dashboard (React single-file build; vanilla fallback)."""
    path = _DASH_REACT if _DASH_REACT.exists() else _DASH_VANILLA
    return HTMLResponse(path.read_text(encoding="utf-8"))


@app.get(
    "/dashboard/data",
    tags=["system"],
    summary="Aggregated dashboard data",
    description="Read-only aggregate of graph stats, live coverage, recent tests, and the active model backend — polled by the dashboard.",
)
def dashboard_data(project: str):
    return pipeline.dashboard_data(project)


# Dashboard-polling / plumbing noise to drop from the planner log stream so only
# real planner reasoning (node_enter/node_exit/llm_call/retrieval) shows.
_PLANNER_DROP = (
    "path=/dashboard", "path=/health", "path=/metrics", "path=/favicon",
    "endpoint=/graph/stats", "endpoint=/coverage/requirements",
    "endpoint=/appmodel/graph", "endpoint=/figma/overview",
)


def _filter_planner(lines: list[str]) -> list[str]:
    out = []
    for l in lines:
        if any(d in l for d in _PLANNER_DROP):
            continue
        if ("request_started" in l or "request_done" in l) and "path=/agent" not in l:
            continue
        out.append(l)
    return out


@app.get("/dashboard/logs", include_in_schema=False)
def dashboard_logs(lines: int = 250, source: str = "mobilerun"):
    """Tail of a live log so the dashboard shows what's happening in real time.

    source=mobilerun -> the device agent's thinking/actions (logs/mobilerun.log)
    source=planner   -> the planner's retrieval/generation reasoning (logs/gateway.log,
                        polling noise filtered out).
    """
    fname = {"mobilerun": "mobilerun.log", "planner": "gateway.log"}.get(source, "mobilerun.log")
    log_path = Path(__file__).resolve().parent.parent / "logs" / fname
    if not log_path.exists():
        return {"exists": False, "source": source, "lines": []}
    try:
        content = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        content = []
    if source == "planner":
        content = _filter_planner(content)
    n = max(1, min(lines, 1000))
    return {"exists": True, "source": source, "total": len(content), "lines": content[-n:]}


@app.get("/dashboard/screenshot", include_in_schema=False)
def dashboard_screenshot(project: str, state_id: str):
    """Proxy a Live App Model state screenshot from the RAG API (same-origin for the dashboard)."""
    try:
        r = requests.get(
            f"{config.RAG_API_URL}/liveui/screenshot",
            params={"project": project, "state_id": state_id},
            timeout=15,
        )
        if r.status_code != 200:
            return Response(status_code=r.status_code)
        return Response(content=r.content, media_type="image/png")
    except requests.RequestException:
        return Response(status_code=502)


@app.get(
    "/health",
    summary="Gateway health check",
    description="Returns the operational status of the gateway and identifies the active model backend (ngrok, OpenRouter, or Gemini) with its configuration.",
    tags=["system"],
    response_description="Status 'ok' with RAG API URL and active model backend details.",
)
def health():
    return {
        "status": "ok",
        "rag_api": config.RAG_API_URL,
        "model_api": config.MODEL_API_URL,
        "model": model_client.backend_info(),
    }


@app.post(
    "/srs/ingest",
    summary="Ingest SRS document",
    description=(
        "Loads a Software Requirements Specification into the Neo4j knowledge graph. The document is "
        "loaded format-agnostically (PDF/DOCX/HTML/MD/txt/...), chunked and embedded for semantic "
        "retrieval, optionally summarised by the LLM, and structurally extracted into a requirement "
        "entity graph (Requirement/Entity/ValidationRule nodes).\n\n"
        "**Re-ingesting replaces** the existing SRS for this project.\n\n"
        "**Tip:** Set `use_model_summary=true` (default) for best test quality; the summary is stored "
        "once here so it doesn't consume tokens on every generation call."
    ),
    tags=["ingest"],
    response_description="Ingest result including chunk/embedding counts, requirement counts, and summary metadata.",
    responses={
        400: {"description": "Path traversal detected in `source_path`."},
        404: {"description": "SRS file not found at `source_path`."},
        413: {"description": "SRS text or file exceeds the 500,000 character limit."},
        415: {"description": "Unsupported document format and no converter installed."},
        503: {"description": "Model summarization failed (only raised when `require_model_summary=true`)."},
    },
)
def ingest_srs(req: IngestSRSRequest, authorization: str | None = Header(default=None)):
    return pipeline.ingest_srs(req, authorization)


@app.post(
    "/figma/ingest",
    summary="Ingest Figma design file",
    description=(
        "Parses a Figma export JSON into a canonical, design-tool-agnostic UI IR and stores all screens, "
        "interactive UI elements, and navigation transitions in the Neo4j knowledge graph. Screen "
        "feature-areas are derived dynamically (optionally via the LLM) — there is no hardcoded per-app "
        "mapping.\n\n"
        "**Re-ingesting replaces** all existing Figma data for this project.\n\n"
        "The planner uses this data to reference exact UI element labels, bias coverage toward "
        "unexplored screens, and reason about navigation flows."
    ),
    tags=["ingest"],
    response_description="Ingest result with screen count, element count, transitions, and classification source.",
    responses={
        400: {"description": "Path traversal in `source_path`, invalid JSON, or no screens found."},
        404: {"description": "Figma JSON file not found at `source_path`."},
    },
)
def ingest_figma(req: IngestFigmaRequest, authorization: str | None = Header(default=None)):
    return pipeline.ingest_figma(req, authorization)


@app.post(
    "/defects/ingest",
    summary="Ingest defect history",
    description=(
        "Loads historical defect reports (bug DB / issue-tracker export) into the knowledge graph as a "
        "first-class source (ETA-REQ-301). Accepts a file path or inline JSON/CSV. Defects are linked to "
        "the feature areas and screens they affect, clustered by similarity, and used to bias test "
        "generation toward historically fragile functionality."
    ),
    tags=["ingest"],
    response_description="Ingest result with defect/similarity/area-scoring counts.",
)
def ingest_defects(req: IngestDefectsRequest, authorization: str | None = Header(default=None)):
    return pipeline.ingest_defects(req, authorization)


@app.post("/session/start", tags=["agent"], summary="Start an exploratory session")
def session_start(req: SessionStartRequest, authorization: str | None = Header(default=None)):
    return pipeline.session_start(req, authorization)


@app.get("/session/context", tags=["agent"], summary="Get current session context")
def session_context(project: str, authorization: str | None = Header(default=None)):
    return pipeline.session_context(project, authorization)


@app.get("/session/live", tags=["agent"], summary="Live execution status (WP9)")
def session_live(project: str, authorization: str | None = Header(default=None)):
    return pipeline.session_live(project, authorization)


@app.get("/metrics/trends", tags=["system"], summary="Gets-smarter trend series (WP9)")
def metrics_trends(project: str, authorization: str | None = Header(default=None)):
    return pipeline.metrics_trends(project, authorization)


@app.post("/session/end", tags=["agent"], summary="End an exploratory session")
def session_end(req: SessionEndRequest, authorization: str | None = Header(default=None)):
    return pipeline.session_end(req, authorization)


@app.post(
    "/project/reset",
    summary="Reset project data slices",
    description=(
        "Selectively deletes data from the project's knowledge graph across three independent slices: "
        "test history (`delete_tests`), SRS + requirement graph (`delete_srs`), and Figma UI "
        "(`delete_figma`). The most common use is `delete_tests=true` to start a fresh testing session "
        "without re-ingesting the knowledge base."
    ),
    tags=["project"],
    response_description="Confirmation of which slices were deleted.",
    responses={
        401: {"description": "Invalid or missing Authorization header."},
        503: {"description": "RAG API unavailable."},
    },
)
def reset_project(req: ResetProjectRequest, authorization: str | None = Header(default=None)):
    return pipeline.reset_project(req, authorization)


@app.post(
    "/agent/next-testcase",
    summary="Generate the next exploratory test case",
    description=(
        "Core endpoint. Runs the full multi-stage planner pipeline and returns the next test case.\n\n"
        "**Pipeline stages:** (1) context bootstrap, (2) coverage analysis, (3) iterative hybrid "
        "retrieval loop (semantic + keyword over SRS, UI elements, transitions), (4) exploratory test "
        "generation, (5) semantic-duplicate guard with rotated retry.\n\n"
        "**Response includes:** the test case JSON, coverage snapshot, retrieval statistics, and "
        "(if `debug_trace=true`) full prompt/response transcripts for every planning round."
    ),
    tags=["agent"],
    response_description=(
        "Generated test case with coverage state, planner trace, and retrieval statistics. "
        "Key fields: `next_testcase`, `next_testcase_json`, `coverage`, `retrieval_plan`, "
        "`planner_trace`, `finalization_mode`."
    ),
    responses={
        401: {"description": "Invalid or missing Authorization header."},
        503: {"description": "LLM backend (model) or RAG API unavailable."},
    },
)
def next_testcase(req: NextTestCaseRequest, authorization: str | None = Header(default=None)):
    return pipeline.generate_next_testcase(req, authorization)


@app.post(
    "/agent/log-verdict",
    summary="Log execution verdict",
    description=(
        "Records the execution verdict for a test case in the knowledge graph (and links covered "
        "requirements via COVERS edges).\n\n"
        "Use this to log a result without immediately generating the next test case. To do both in one "
        "call, use `/agent/log-verdict-and-next`.\n\n"
        "**Note on verdicts:** `blocked` and `skipped` are stored as `failed` internally; the original "
        "verdict is prepended to `notes`."
    ),
    tags=["agent"],
    response_description="Verdict confirmation from the knowledge graph.",
    responses={
        401: {"description": "Invalid or missing Authorization header."},
        503: {"description": "RAG API unavailable."},
    },
)
def log_verdict(req: LogVerdictRequest, authorization: str | None = Header(default=None)):
    return pipeline.log_verdict(req, authorization)


@app.post(
    "/agent/log-verdict-and-next",
    summary="Log execution verdict and get next test case",
    description=(
        "The primary endpoint for the continuous exploration loop. Atomically logs the verdict, then "
        "generates the next test case, adapting the objective based on the verdict:\n"
        "- `failed` → probe adjacent edge cases in the same area.\n"
        "- `blocked` / `skipped` → seek an alternative that avoids the blocking condition.\n"
        "- `pass` → broaden coverage toward unexplored areas.\n\n"
        "Override the adaptive objective with `next_objective`."
    ),
    tags=["agent"],
    response_description="Two-key response: `log` (verdict confirmation) and `next` (next-testcase payload).",
    responses={
        401: {"description": "Invalid or missing Authorization header."},
        503: {"description": "LLM backend or RAG API unavailable."},
    },
)
def log_verdict_and_next(req: LogVerdictRequest, authorization: str | None = Header(default=None)):
    return pipeline.log_verdict_and_next(req, authorization)


@app.get(
    "/agent/coverage",
    summary="Get exploration coverage dashboard",
    description=(
        "Returns the live exploration state for a project without generating a test case: area "
        "breakdown, hot spots (repeated failures), uncovered/exhausted areas, a prioritised "
        "exploration directive, graph-native requirement coverage (COVERS edges), and recent tests."
    ),
    tags=["agent"],
    response_description="Coverage dashboard with exploration directive, requirement coverage, and recent test history.",
    responses={
        401: {"description": "Invalid or missing Authorization header."},
        503: {"description": "RAG API unavailable."},
    },
)
def agent_coverage(project: str, authorization: str | None = Header(default=None)):
    return pipeline.agent_coverage(project, authorization)


@app.post(
    "/chat",
    summary="RAG-backed free-form Q&A",
    description=(
        "Ask any question about the app under test. The prompt is automatically augmented with the most "
        "relevant SRS context (semantic + keyword hybrid retrieval) from the knowledge graph, then sent "
        "to the active LLM backend."
    ),
    tags=["chat"],
    response_description="LLM answer with the SRS context that was retrieved and injected.",
    responses={503: {"description": "LLM backend or RAG API unavailable."}},
)
def chat(req: ChatRequest, authorization: str | None = Header(default=None)):
    return pipeline.chat(req, authorization)
