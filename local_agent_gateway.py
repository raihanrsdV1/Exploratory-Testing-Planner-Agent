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

from fastapi import FastAPI, Header

from planner import config, model_client, pipeline
from planner.schemas import (
    ChatRequest,
    IngestFigmaRequest,
    IngestSRSRequest,
    LogVerdictRequest,
    NextTestCaseRequest,
    ResetProjectRequest,
)

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
