import os
import json
import re
from pathlib import Path

import requests
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

try:
    from google import genai
    from google.genai import types
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

from dotenv import load_dotenv
load_dotenv()

# Local services (on your device)
RAG_API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:9010").rstrip("/")
RAG_API_KEY = os.getenv("RAG_API_KEY", "")

# Model API — ngrok/local server OR OpenRouter OR Gemini
MODEL_BACKEND = os.getenv("MODEL_BACKEND", "ngrok").strip().lower()  # "ngrok", "openrouter", "gemini"
MODEL_API_URL = os.getenv("MODEL_API_URL", "https://d7bf-34-90-11-71.ngrok-free.app").rstrip("/")

# Gemini config (used when MODEL_BACKEND=gemini)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
PLANNER_GEMINI_MODEL = os.getenv("PLANNER_GEMINI_MODEL", "gemini-2.5-pro")

# OpenRouter config (used when MODEL_BACKEND=openrouter)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "qwen/qwen3.6-plus:free")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

GATEWAY_API_KEY = os.getenv("GATEWAY_API_KEY", "")
APP_NAME = os.getenv("APP_NAME", "the app under test")

app = FastAPI(
    title="Exploratory Testing Planner — Agent Gateway",
    description=(
        "Orchestrates an adaptive exploratory QA testing session for Android apps.\n\n"
        "**How it works:**\n"
        "1. Ingest the app's SRS and Figma files into a Neo4j knowledge graph.\n"
        "2. Call `/agent/next-testcase` — the planner retrieves targeted context and generates "
        "a structured JSON test case, guided by live coverage state and testing heuristics.\n"
        "3. The executor (Droidrun) runs the test on a real device.\n"
        "4. Call `/agent/log-verdict-and-next` — the verdict is logged and the next test case "
        "is generated, adapting toward failures and unexplored areas automatically.\n\n"
        "**Authentication:** Set `GATEWAY_API_KEY` in `.env` to require "
        "`Authorization: Bearer <key>` on every request. Leave blank to disable auth."
    ),
    version="1.0.0",
    openapi_tags=[
        {"name": "system",  "description": "Health checks and backend diagnostics."},
        {"name": "ingest",  "description": "Load SRS and Figma design files into the knowledge graph."},
        {"name": "project", "description": "Manage project-level data (reset slices, lifecycle)."},
        {"name": "agent",   "description": "Core exploratory test generation and coverage tracking."},
        {"name": "chat",    "description": "RAG-backed free-form Q&A against the project knowledge graph."},
    ],
)


class ChatRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "project": "my-app",
        "prompt": "What validation rules apply to the phone number field?",
        "top_k": 3,
        "max_new_tokens": 512,
        "enable_thinking": False,
    }})

    prompt: str = Field(..., min_length=1, description="Question or instruction for the LLM, automatically augmented with relevant SRS context retrieved from the knowledge graph.")
    project: str = Field(default="default", description="Project identifier used to scope knowledge-graph retrieval.")
    top_k: int = Field(default=3, ge=1, le=20, description="Number of SRS chunks to retrieve as RAG context.")
    max_new_tokens: int = Field(default=2048, ge=64, le=8192, description="Maximum tokens in the LLM response.")
    enable_thinking: bool = Field(default=False, description="Enable extended chain-of-thought reasoning (supported models only).")


class NextTestCaseRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "project": "my-app",
        "app_name": "contacts app",
        "objective": "generate next high-value non-duplicate test case",
        "top_k": 8,
        "max_new_tokens": 700,
        "enable_thinking": False,
        "debug_trace": False,
        "max_retrieval_rounds": 3,
    }})

    project: str = Field(..., min_length=1, description="Project identifier scoping the Neo4j knowledge graph.")
    app_name: str = Field(default=APP_NAME, description="Display name of the app under test, injected into every LLM prompt.")
    objective: str = Field(
        default="generate the next best high-value non-duplicate test case",
        description="High-level exploration goal for this round. The planner adapts automatically based on live coverage state — override only for targeted sessions.",
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Max SRS chunks retrieved per knowledge-graph query during the retrieval planning loop.")
    max_new_tokens: int = Field(default=2048, ge=64, le=8192, description="Token budget for the final test case generation call.")
    enable_thinking: bool = Field(default=False, description="Enable extended reasoning traces (supported models only). Increases latency.")
    debug_trace: bool = Field(default=False, description="Include full debug trace in the response: prompt texts, raw model output, and retrieved context blocks for every planning round.")
    max_retrieval_rounds: int = Field(default=3, ge=1, le=6, description="Maximum planning/retrieval rounds before the gateway forces test case generation. Higher values gather more context but increase latency.")


class LogVerdictRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "project": "my-app",
        "app_name": "contacts app",
        "test_case_id": "TC-003",
        "title": "Verify email field rejects invalid format on Create Contact screen",
        "verdict": "failed",
        "notes": "App accepted 'user@' without a TLD — no validation error was shown.",
        "area": "create_contact",
        "next_objective": "",
        "top_k": 8,
        "max_new_tokens": 700,
        "enable_thinking": False,
        "debug_trace": False,
    }})

    project: str = Field(..., min_length=1, description="Project identifier.")
    app_name: str = Field(default=APP_NAME, description="Display name of the app under test.")
    test_case_id: str = Field(..., min_length=1, description="ID of the executed test case (e.g. 'TC-003'). Must match the value returned by the planner.")
    title: str = Field(..., min_length=1, description="Title of the executed test case.")
    verdict: str = Field(
        ...,
        pattern="^(pass|failed|blocked|skipped)$",
        description=(
            "Execution outcome. Accepted values:\n"
            "- `pass` — test ran successfully, expected result met.\n"
            "- `failed` — test ran but expected result was NOT met (bug found).\n"
            "- `blocked` — test could not run due to a prerequisite or environment issue.\n"
            "- `skipped` — test was intentionally skipped.\n\n"
            "`blocked` and `skipped` are stored as `failed` in the knowledge graph with the original verdict prepended to notes."
        ),
    )
    notes: str = Field(default="", description="Execution notes, error messages, or observations from the executor. Stored in the knowledge graph and used to inform subsequent test generation.")
    area: str = Field(default="general", description="Feature area of the test case (e.g. 'create_contact', 'search'). Used to compute the coverage map and drive the adaptive next-test objective.")
    next_objective: str = Field(default="", description="Override the adaptive objective for the next test case. Leave blank to let the planner derive the objective from the verdict and coverage state (recommended).")
    top_k: int = Field(default=5, ge=1, le=20, description="Max SRS chunks retrieved for the next test case generation.")
    max_new_tokens: int = Field(default=2048, ge=64, le=8192, description="Token budget for the next test case generation call.")
    enable_thinking: bool = Field(default=False, description="Enable extended reasoning for the next test generation call.")
    debug_trace: bool = Field(default=False, description="Include full debug trace in the next test case response.")


class IngestSRSRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "project": "my-app",
        "source_path": "./data/inputs/SRS1.txt",
        "chunk_chars": 1200,
        "use_model_summary": True,
        "require_model_summary": True,
    }})

    project: str = Field(..., min_length=1, description="Project identifier. Created automatically if it does not exist.")
    source_path: str = Field(..., min_length=1, description="Local file path to the SRS document (.txt, .md, etc.). Path traversal ('..') is rejected. If `srs_text` is provided this field is used as a label only.")
    srs_text: str | None = Field(default=None, description="Inline SRS text. If provided, the file at `source_path` is not read. Maximum 500,000 characters.")
    chunk_chars: int = Field(default=1200, ge=200, le=5000, description="Target character size for each SRS chunk stored in the knowledge graph. Smaller chunks improve retrieval precision; larger chunks improve coherence.")
    use_model_summary: bool = Field(default=True, description="Generate a planner-friendly SRS summary via the LLM before ingesting chunks. The summary is used in every test generation call for global context.")
    require_model_summary: bool = Field(default=True, description="If true, the request returns 503 when model summarization fails. If false, falls back to a rule-based keyword summary.")


def _summarize_srs_with_model(srs_text: str, max_new_tokens: int = 1000) -> str:
    """Create a planner-friendly SRS summary from full source text using the model backend."""
    prompt = (
        "You are an expert software requirements analyst.\n"
        "Read the full SRS below and produce a concise but complete summary for QA test planning.\n\n"
        "Output STRICT plain text (no markdown, no code fences) with this structure:\n"
        "Document: <name or inferred title>\n"
        "Functional requirements summary:\n"
        "- ...\n"
        "- ...\n"
        "Non-functional requirements summary:\n"
        "- ...\n"
        "- ...\n"
        "Validation and constraints:\n"
        "- ...\n"
        "Coverage priorities for next tests:\n"
        "- ...\n\n"
        "Rules:\n"
        "- Preserve critical requirement IDs when available (FR-#, NFR-#).\n"
        "- Keep it compact but include all high-impact behaviors and validations.\n"
        "- If non-functional requirements are not explicitly listed, state that clearly.\n\n"
        "SRS:\n"
        f"{(srs_text or '').strip()}"
    )
    model_data = _call_model(prompt, max_new_tokens, False)
    summary = (model_data.get("answer", "") or "").strip()
    return _extract_json_text(summary) if summary.startswith("{") else summary


class IngestFigmaRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "project": "my-app",
        "source_path": "./data/inputs/GENERATED_JSON.json",
    }})

    project: str = Field(..., min_length=1, description="Project identifier. Created automatically if it does not exist.")
    source_path: str = Field(..., min_length=1, description="Local path to the exported Figma JSON file. Path traversal ('..') is rejected. If `figma_json` is provided this field is used as a label only.")
    figma_json: str | None = Field(default=None, description="Inline Figma export JSON string. Accepts raw JSON or a markdown code-fenced JSON block.")


class ResetProjectRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "project": "my-app",
        "delete_tests": True,
        "delete_srs": False,
        "delete_figma": False,
    }})

    project: str = Field(..., min_length=1, description="Project to reset.")
    delete_tests: bool = Field(default=True, description="Delete all logged test cases and test runs. The next test generation starts from scratch with no history.")
    delete_srs: bool = Field(default=False, description="Delete ingested SRS chunks and summary. Re-ingest via `/srs/ingest` before the next test generation.")
    delete_figma: bool = Field(default=False, description="Delete ingested Figma screens and UI elements. Re-ingest via `/figma/ingest` before the next test generation.")


# ── Auth & helpers ────────────────────────────────────────────────────────────

def _check_gateway_auth(authorization: str | None):
    if not GATEWAY_API_KEY:
        return
    if authorization != f"Bearer {GATEWAY_API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")


def _rag_headers() -> dict:
    return {"Authorization": f"Bearer {RAG_API_KEY}"} if RAG_API_KEY else {}


def _rag_get(path: str, params: dict | None = None) -> dict:
    try:
        resp = requests.get(
            f"{RAG_API_URL}{path}", params=params,
            headers=_rag_headers(), timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"RAG backend unavailable: {e}")


def _rag_post(path: str, body: dict) -> dict:
    try:
        resp = requests.post(
            f"{RAG_API_URL}{path}", json=body,
            headers=_rag_headers(), timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"RAG backend unavailable: {e}")


def _call_model(prompt: str, max_new_tokens: int, enable_thinking: bool) -> dict:
    """
    Call the LLM. Supports multiple backends controlled by MODEL_BACKEND env var:
      - "ngrok"      → custom /generate endpoint
      - "openrouter" → OpenAI-compatible chat completions via OpenRouter
      - "gemini"     → Native Google GenAI SDK
    """
    if MODEL_BACKEND == "gemini":
        return _call_gemini(prompt, max_new_tokens, enable_thinking)
    elif MODEL_BACKEND == "openrouter":
        return _call_openrouter(prompt, max_new_tokens, enable_thinking)
    else:
        return _call_ngrok(prompt, max_new_tokens, enable_thinking)


def _call_gemini(prompt: str, max_new_tokens: int, enable_thinking: bool) -> dict:
    """Call Google's Gemini models via the GenAI SDK."""
    if not HAS_GEMINI:
        raise HTTPException(status_code=500, detail="google-genai package not installed.")
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY not set in .env")

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=PLANNER_GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=max_new_tokens,
                temperature=0.7,
                system_instruction=(
                    "You are a senior QA engineer generating exploratory test cases. "
                    "Always respond with valid JSON when asked for test cases. "
                    "Follow the exact output format specified in the user prompt."
                )
            ),
        )
        return {"answer": response.text, "thinking": ""}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model backend (Gemini) unavailable: {e}")


def _call_ngrok(prompt: str, max_new_tokens: int, enable_thinking: bool) -> dict:
    try:
        resp = requests.post(
            f"{MODEL_API_URL}/generate",
            json={"prompt": prompt, "max_new_tokens": max_new_tokens, "enable_thinking": enable_thinking},
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Model backend (ngrok) unavailable: {e}")

def _call_openrouter(prompt: str, max_new_tokens: int, enable_thinking: bool) -> dict:
    """Call OpenRouter's OpenAI-compatible chat completions API."""
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=503, detail="OPENROUTER_API_KEY not set in .env")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://qa-planner-agent.local",
        "X-Title": "QA Planner Agent",
    }

    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior QA engineer generating exploratory test cases. "
                "Always respond with valid JSON when asked for test cases. "
                "Follow the exact output format specified in the user prompt."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": 0.7,
    }

    try:
        resp = requests.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()

        # Extract answer from OpenAI-compatible response
        answer = ""
        thinking = ""
        if "choices" in data and len(data["choices"]) > 0:
            message = data["choices"][0].get("message", {})
            answer = message.get("content", "")
            # Some models return thinking in a separate field
            thinking = message.get("reasoning", "") or message.get("thinking", "")

        return {"answer": answer, "thinking": thinking}

    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Model backend (OpenRouter) unavailable: {e}")


# ── Knowledge graph queries ───────────────────────────────────────────────────

def _get_srs_and_history(project: str, query: str, top_k: int) -> dict:
    """Fetch SRS chunks + test history from the knowledge graph."""
    return _rag_post("/retrieve", {"project": project, "query": query, "top_k": top_k, "include_history": False})


def _get_figma_screens(project: str) -> list[dict]:
    """List all Figma screens (name, purpose, element counts) — lightweight index."""
    data = _rag_get("/figma/screens", {"project": project})
    return data.get("screens", [])


def _get_screen_elements(project: str, screen_name: str) -> dict[str, list[str]]:
    """Get interactive elements for one screen, grouped by kind. Compact for prompt."""
    data = _rag_get("/figma/elements", {"project": project, "screen_name": screen_name, "interactive_only": "true"})
    return data.get("elements", {})


def _get_figma_overview(project: str) -> list[dict]:
    data = _rag_get("/figma/overview", {"project": project, "top_labels": 4})
    return data.get("screens", [])


def _get_figma_transitions(project: str, screen_name: str | None = None) -> list[dict]:
    params = {"project": project, "limit": 80}
    if screen_name:
        params["screen_name"] = screen_name
    data = _rag_get("/figma/transitions", params)
    return data.get("transitions", [])


def _pick_relevant_screens(screens: list[dict], done_areas: list[str], recent_tests: list[dict]) -> list[str]:
    """
    Choose which Figma screens to pull element detail for.
    Logic: find screens not well-covered by recent tests, bias toward uncovered areas.
    Returns up to 2 screen names.
    """
    if not screens:
        return []

    # Build set of purposes already tested
    tested_areas = set(
        str(t.get("area", "")).lower().replace(" ", "_") for t in recent_tests
    )
    tested_areas.update(a.lower().replace(" ", "_") for a in done_areas)

    # Prefer screens whose purpose hasn't been tested yet
    untested = [s for s in screens if s.get("purpose", "other") not in tested_areas]
    chosen = untested if untested else screens

    # Sort by interactive_count desc — more interactive = more testable
    chosen = sorted(chosen, key=lambda s: s.get("interactive_count", 0), reverse=True)
    return [s["screen_name"] for s in chosen[:2]]


def _build_figma_context(project: str, screen_names: list[str]) -> str:
    """
    Fetch elements for chosen screens and format as compact context block.
    Example output:
      [Screen: Create Contact (No Icons)]
        buttons: Save, Add phone, Email, Birthday
        inputs: First name, Surname, Company, Add phone, Notes
    """
    if not screen_names:
        return ""

    lines = []
    for name in screen_names:
        elements = _get_screen_elements(project, name)
        if not elements:
            continue
        lines.append(f"[Screen: {name}]")
        for kind, labels in elements.items():
            lines.append(f"  {kind}s: {', '.join(labels[:10])}")
    return "\n".join(lines)


def _build_figma_overview_context(figma_overview: list[dict]) -> str:
    if not figma_overview:
        return ""
    lines = ["Available screens and key UI elements:"]
    for s in figma_overview:
        lines.append(
            f"- {s.get('screen_name','?')} (purpose={s.get('purpose','other')}, interactive={s.get('interactive_count',0)})"
        )
        if s.get("buttons"):
            lines.append("  buttons: " + ", ".join(s.get("buttons", [])[:4]))
        if s.get("inputs"):
            lines.append("  inputs: " + ", ".join(s.get("inputs", [])[:4]))
        if s.get("nav"):
            lines.append("  navigation: " + ", ".join(s.get("nav", [])[:3]))
    return "\n".join(lines)


def _build_figma_overview_generalized(figma_overview: list[dict]) -> str:
    """Generalized UI context for planning (less label-heavy, less bias)."""
    if not figma_overview:
        return "No screens available"
    lines = [f"Total screens: {len(figma_overview)}", "Screens by purpose:"]
    for s in figma_overview:
        lines.append(
            f"- {s.get('screen_name','?')} (purpose={s.get('purpose','other')}, interactive={s.get('interactive_count',0)})"
        )
    return "\n".join(lines)


def _recent_tests_exact(recent_tests: list[dict], limit: int = 50) -> str:
    if not recent_tests:
        return "none"
    items = []
    for t in recent_tests[:limit]:
        items.append(
            f"{t.get('id','?')}|{t.get('verdict','?')}|{t.get('area','general')}|{t.get('title','')[:120]}"
        )
    return "; ".join(items)


def _build_figma_flow_context(transitions: list[dict], top_n: int = 12) -> str:
    if not transitions:
        return ""
    lines = ["Known UI transitions (inferred from labels):"]
    for t in transitions[:top_n]:
        lines.append(
            f"- {t.get('from_screen','?')} --[{t.get('via_element','?')}]--> {t.get('to_screen','?')}"
        )
    return "\n".join(lines)


def _screen_index_compact(screen_index: list[dict], limit: int = 7) -> str:
    if not screen_index:
        return "[]"
    ordered = sorted(
        screen_index,
        key=lambda s: (
            -int(s.get("interactive_count", 0) or 0),
            str(s.get("screen_name", "")).lower(),
        ),
    )
    parts = []
    for s in ordered[:limit]:
        parts.append(
            f"{s.get('screen_name','?')}|purpose={s.get('purpose','other')}|interactive={s.get('interactive_count',0)}"
        )
    return "[" + "; ".join(parts) + "]"


def _compact_note(text: str, limit: int = 260) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    return t[:limit] + (" ..." if len(t) > limit else "")


# ── Coverage & exploration state ──────────────────────────────────────────────

MAX_SRS_CHARS = 500_000  # ~500 KB hard cap on ingested SRS text


def _compute_coverage_map(recent_tests: list[dict], figma_screens: list[dict]) -> dict:
    """
    Derive a live exploration coverage map from test history and known screens.
    Used to guide the planner toward untested areas and away from exhausted ones.
    """
    area_stats: dict[str, dict] = {}
    for t in recent_tests:
        area = re.sub(r"\s+", "_", str(t.get("area", "general")).lower().strip()) or "general"
        if area not in area_stats:
            area_stats[area] = {"total": 0, "passed": 0, "failed": 0}
        area_stats[area]["total"] += 1
        if str(t.get("verdict", "")).lower() == "pass":
            area_stats[area]["passed"] += 1
        else:
            area_stats[area]["failed"] += 1

    screen_purposes: set[str] = set()
    for s in figma_screens:
        p = str(s.get("purpose", "")).lower().strip()
        if p and p != "other":
            screen_purposes.add(p)

    tested_areas = set(area_stats.keys()) - {"general"}
    uncovered_purposes = sorted(screen_purposes - tested_areas)
    hot_spots = sorted(a for a, s in area_stats.items() if s["failed"] >= 2)
    exhausted = sorted(a for a, s in area_stats.items() if s["total"] >= 4 and s["failed"] == 0)
    cov_pct = round(100 * len(tested_areas) / len(screen_purposes)) if screen_purposes else 0

    return {
        "area_stats": area_stats,
        "uncovered_purposes": uncovered_purposes,
        "hot_spots": hot_spots,
        "exhausted_areas": exhausted,
        "total_tests": len(recent_tests),
        "total_areas_tested": len(tested_areas),
        "total_areas_available": len(screen_purposes),
        "coverage_pct": cov_pct,
    }


def _build_coverage_block(coverage_map: dict) -> str:
    """Format the coverage map as a prompt-ready text block."""
    stats = coverage_map.get("area_stats", {})
    total = coverage_map.get("total_tests", 0)
    if not stats:
        return "No tests executed yet — start of exploration session. Begin with broad coverage."
    lines = [f"Total tests executed: {total}"]
    lines.append("Per-area breakdown:")
    for area, s in sorted(stats.items(), key=lambda x: (-x[1]["failed"], -x[1]["total"])):
        flag = " ⚠ REPEATED FAILURES" if s["failed"] >= 2 else (" ✓ stable" if s["failed"] == 0 and s["total"] >= 3 else "")
        lines.append(f"  {area}: {s['total']} tests ({s['passed']} pass / {s['failed']} fail){flag}")
    unc = coverage_map.get("uncovered_purposes", [])
    if unc:
        lines.append(f"Completely untested areas: {', '.join(unc)}")
    cov = coverage_map.get("coverage_pct", 0)
    avail = coverage_map.get("total_areas_available", 0)
    tested = coverage_map.get("total_areas_tested", 0)
    lines.append(f"Overall coverage: {tested}/{avail} known areas ({cov}%)")
    return "\n".join(lines)


def _build_exploration_directive(coverage_map: dict, recent_tests: list[dict]) -> str:
    """
    Build a prioritised exploration directive so the model knows exactly what
    to test next based on live coverage state.
    """
    lines: list[str] = []
    hot_spots = coverage_map.get("hot_spots", [])
    uncovered = coverage_map.get("uncovered_purposes", [])
    exhausted = coverage_map.get("exhausted_areas", [])

    if hot_spots:
        lines.append(
            f"[PRIORITY 1 — INVESTIGATE] Areas with repeated failures need deeper edge-case coverage: "
            f"{', '.join(hot_spots[:3])}"
        )
    if uncovered:
        lines.append(
            f"[PRIORITY 2 — EXPAND] Areas with ZERO test coverage yet: "
            f"{', '.join(uncovered[:5])}"
        )

    last_areas = [str(t.get("area", "")).lower().strip() for t in recent_tests[:4] if t.get("area")]
    if len(last_areas) >= 3 and len(set(last_areas)) == 1:
        last_verdicts = [str(t.get("verdict", "")).lower() for t in recent_tests[:4]]
        if all(v == "pass" for v in last_verdicts):
            lines.append(
                f"[PRIORITY 3 — PIVOT] Last {len(last_areas)} consecutive tests all PASSED in "
                f"'{last_areas[0]}'. This area is likely stable — move to a different area now."
            )
        else:
            lines.append(
                f"[PRIORITY 3 — VARY ANGLE] Still in '{last_areas[0]}' with mixed results. "
                "Try a different test type: boundary values, invalid input, or state transitions."
            )

    if exhausted:
        lines.append(
            f"[DEPRIORITIZE] Well-covered stable areas (4+ tests, 0 failures — avoid repeating): "
            f"{', '.join(exhausted[:3])}"
        )

    if not lines:
        lines.append("[OPEN EXPLORATION] No specific signal. Aim for breadth first, then depth.")

    return "\n".join(lines)


def _get_brief_context(project: str) -> dict:
    return _rag_post("/context/brief", {"project": project, "recent_limit": 100})


def _planner_prompt_for_retrieval(brief: dict, app_name: str, objective: str) -> str:
    return (
        f"You are planning retrieval for QA testcase generation in {app_name}.\n"
        "Given compact project context, output retrieval plan JSON only.\n\n"
        "Context summary:\n"
        f"SRS summary:\n{brief.get('srs_summary','')}\n\n"
        f"Figma summary:\n{brief.get('figma_summary','')}\n\n"
        f"Screen index: {brief.get('screen_index', [])}\n\n"
        f"Recent tests: {brief.get('recent_tests', [])}\n\n"
        f"Objective: {objective}\n\n"
        "Return STRICT JSON:\n"
        "{\n"
        "  \"focus_queries\": [\"...\", \"...\"],\n"
        "  \"target_screens\": [\"...\", \"...\"],\n"
        "  \"reason\": \"short reason\"\n"
        "}\n"
        "Constraints: max 2 focus_queries, max 2 target_screens, keep concise."
    )


def _planner_prompt_for_action(
    brief: dict,
    app_name: str,
    objective: str,
    retrieval_round: int,
    max_rounds: int,
    collected_queries: list[str],
    collected_screens: list[str],
    context_chars: int,
    figma_overview: list[dict],
    retrieved_notes: list[str],
    coverage_map: dict | None = None,
) -> str:
    retrieved_notes_str = "\n".join(f"- {n}" for n in retrieved_notes[-6:]) if retrieved_notes else "- none yet"
    srs_summary = brief.get("srs_summary", "") if isinstance(brief, dict) else ""
    srs_full = str(srs_summary).strip() if srs_summary else "(none)"

    figma_summary = brief.get("figma_summary", "") if isinstance(brief, dict) else ""
    figma_full = str(figma_summary).strip() if figma_summary else "(none)"

    figma_overview_general = _build_figma_overview_generalized(figma_overview)
    recent_tests = brief.get("recent_tests", []) if isinstance(brief, dict) else []
    recent_tests_exact = _recent_tests_exact(recent_tests)

    cmap = coverage_map or {}
    hot_spots = cmap.get("hot_spots", [])
    uncovered = cmap.get("uncovered_purposes", [])
    coverage_hint = (
        f"Coverage hint — hot spots (repeated failures): {hot_spots[:3] or 'none'}; "
        f"unexplored areas: {uncovered[:4] or 'none'}; "
        f"overall coverage: {cmap.get('coverage_pct', '?')}%"
    )

    if retrieval_round <= 1:
        global_context_block = (
            "Full global context (round 1):\n"
            f"SRS summary:\n{srs_full}\n\n"
            f"Figma summary:\n{figma_full}\n\n"
            f"Screen index (compact): {_screen_index_compact(brief.get('screen_index', []) if isinstance(brief, dict) else [])}\n"
            f"Figma UI overview (generalized):\n{figma_overview_general}\n"
            f"Recent tests (exact): {recent_tests_exact}\n"
            f"Exploration coverage: {coverage_hint}\n\n"
        )
    else:
        global_context_block = (
            "Global memo (do not re-ask same broad context):\n"
            f"Screen index (compact): {_screen_index_compact(brief.get('screen_index', []) if isinstance(brief, dict) else [])}\n"
            f"Recent tests (exact): {recent_tests_exact}\n"
            f"Exploration coverage: {coverage_hint}\n"
            "Use Retrieved context so far to refine, not restart.\n\n"
        )

    response_link_line = (
        "This prompt is a response to your previous retrieval request and includes the requested DB context.\n\n"
        if retrieval_round > 1
        else ""
    )

    return (
        f"You are a retrieval planner for exploratory QA test generation in {app_name}.\n"
        "You interact with a knowledge database (SRS + Figma KG + test history) to gather context.\n"
        "Your goal: retrieve context that will help generate a test case targeting real defects — "
        "prioritise unexplored areas and hot spots with repeated failures.\n"
        "Decide your NEXT ACTION only.\n\n"
        f"Objective: {objective}\n"
        f"Retrieval round: {retrieval_round}/{max_rounds}\n"
        f"Collected queries so far: {collected_queries}\n"
        f"Collected screens so far: {collected_screens}\n"
        f"Collected context size (chars): {context_chars}\n\n"
        f"{response_link_line}"
        "Retrieved context so far (continuation from earlier requests):\n"
        f"{retrieved_notes_str}\n\n"
        f"{global_context_block}"
        "Return STRICT JSON only with this schema:\n"
        "{\n"
        "  \"action\": \"retrieve\" | \"produce_testcase\",\n"
        "  \"retrieval_requests\": [{\"source\":\"srs|figma_ui|figma_flow\", \"query\":\"...\", \"screen\":\"optional\"}],\n"
        "  \"focus_queries\": [\"...\", \"...\"],\n"
        "  \"target_screens\": [\"...\", \"...\"],\n"
        "  \"reason\": \"short reason\"\n"
        "}\n"
        "Rules:\n"
        "- If more context is needed, set action=retrieve and provide explicit retrieval_requests (max 3).\n"
        "- Prefer retrieving context for HOT SPOTS and UNEXPLORED areas (see coverage hint above).\n"
        "- Use source=srs for business rules, validation constraints, and error conditions.\n"
        "- Use source=figma_ui for screen elements and control availability.\n"
        "- Use source=figma_flow for navigation/button-to-screen behaviour.\n"
        "- For source=srs, query must be a keyword expression, e.g. \"validation AND (email OR phone)\".\n"
        "- Avoid re-requesting context you already have unless refining.\n"
        "- If context is sufficient to write a targeted test, set action=produce_testcase.\n"
        "- Never output markdown or text outside JSON."
    )


def _to_query_expr(text: str, fallback: str) -> str:
    raw = (text or "").strip()
    if not raw or raw.lower() == "none":
        raw = fallback
    if "?" not in raw and (" and " in raw.lower() or " or " in raw.lower()):
        return raw

    toks = re.findall(r"[a-zA-Z0-9_]+", raw.lower())
    stop = {"what", "how", "when", "where", "why", "is", "are", "the", "a", "an", "for", "to", "of", "in", "on"}
    keep = []
    for t in toks:
        if len(t) < 3 or t in stop:
            continue
        if t not in keep:
            keep.append(t)
    if not keep:
        return fallback
    return " AND ".join(keep[:8])


def _parse_retrieval_plan(raw: str, fallback_screens: list[str]) -> dict:
    plan = _parse_testcase(raw)
    if not isinstance(plan, dict):
        return {"focus_queries": [], "target_screens": fallback_screens[:2], "reason": "fallback"}

    fq = plan.get("focus_queries", [])
    ts = plan.get("target_screens", [])
    if not isinstance(fq, list):
        fq = []
    if not isinstance(ts, list):
        ts = []

    cleaned_fq = [str(x).strip() for x in fq if str(x).strip()][:2]
    cleaned_ts = [str(x).strip() for x in ts if str(x).strip()][:2]
    if not cleaned_ts:
        cleaned_ts = fallback_screens[:2]

    return {
        "focus_queries": cleaned_fq,
        "target_screens": cleaned_ts,
        "reason": str(plan.get("reason", "")).strip(),
    }


def _parse_action(raw: str, fallback_screens: list[str]) -> dict:
    data = _parse_testcase(raw)
    if not isinstance(data, dict):
        return {
            "action": "retrieve",
            "focus_queries": [],
            "target_screens": fallback_screens[:2],
            "reason": "fallback",
        }

    action = str(data.get("action", "retrieve")).strip().lower()
    if action not in {"retrieve", "produce_testcase"}:
        action = "retrieve"

    fq = data.get("focus_queries", [])
    ts = data.get("target_screens", [])
    if not isinstance(fq, list):
        fq = []
    if not isinstance(ts, list):
        ts = []

    cleaned_fq = [str(x).strip() for x in fq if str(x).strip()][:2]
    cleaned_ts = [str(x).strip() for x in ts if str(x).strip()][:2]
    if not cleaned_ts:
        cleaned_ts = fallback_screens[:2]

    return {
        "action": action,
        "retrieval_requests": data.get("retrieval_requests", []) if isinstance(data.get("retrieval_requests", []), list) else [],
        "focus_queries": cleaned_fq,
        "target_screens": cleaned_ts,
        "reason": str(data.get("reason", "")).strip(),
    }


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(
    app_name: str,
    objective: str,
    srs_context: str,
    figma_overview_context: str,
    figma_context: str,
    figma_flow_context: str,
    done_titles: list[str],
    failed_titles: list[str],
    coverage_map: dict | None = None,
    recent_tests: list[dict] | None = None,
) -> str:
    cmap = coverage_map or {}
    rtests = recent_tests or []
    coverage_block = _build_coverage_block(cmap)
    directive = _build_exploration_directive(cmap, rtests)

    parts = [
        f"You are an expert exploratory software tester conducting an adaptive testing session on {app_name}.",
        f"Session objective: {objective}",
        "Your mission is to DISCOVER BUGS, not confirm expected behaviour.",
        "Think adversarially: where would a user most likely hit a real defect?",
        "Generate exactly ONE test case optimised to uncover a defect not yet found.",
        "",
        "## Live Coverage State",
        coverage_block,
        "",
        "## Exploration Directive — follow this priority order",
        directive,
        "",
    ]

    if srs_context:
        parts += [
            "## Business Rules & Requirements (from SRS)",
            "(Violations of these rules are bugs — verify that these constraints are actually enforced by the app)",
            srs_context,
            "",
        ]

    if figma_overview_context:
        parts += [
            "## App Screens & UI Structure",
            figma_overview_context,
            "",
        ]

    if figma_context:
        parts += [
            "## Interactive Elements on Relevant Screens",
            "(Use EXACT element names from the list below in your test steps — do not invent labels)",
            figma_context,
            "",
        ]

    if figma_flow_context:
        parts += [
            "## Screen Navigation Transitions",
            figma_flow_context,
            "",
        ]

    history_block = "\n".join(f"- {t}" for t in done_titles[:40]) or "- none"
    failed_block = "\n".join(f"- {t}" for t in failed_titles[:20]) or "- none"

    parts += [
        "## Executed Tests — semantic duplicates are FORBIDDEN",
        history_block,
        "",
        "## Known Failures — probe adjacent cases and variants around these",
        failed_block,
        "",
        "## Exploratory Testing Heuristics — apply at least one",
        "- BOUNDARY: Test at the edges of valid input ranges (max length, empty, zero, one-off, overflow).",
        "- INVALID INPUT: Submit malformed, null, or unexpected-type data. Does the app fail gracefully?",
        "- STATE TRANSITION: Perform an action and verify the app reaches the correct subsequent state.",
        "- INTERRUPTION: Start an action, navigate away mid-flow, then return — is data/state preserved?",
        "- COMBINATION: Test interactions between two features that may have unexpected side-effects.",
        "- RECOVERY: After an error or warning, can the user correct and retry without restarting?",
        "- PERMISSION / ACCESS: Test behaviours requiring specific preconditions or data to be present.",
        "",
        "## Strict Decision Policy",
        "1. FORBIDDEN: Any test semantically similar to a title in the executed list above.",
        "2. PRIORITY: Follow the Exploration Directive — hot spots > new areas > breadth > exhausted areas.",
        "3. PREFER negative, boundary, and state-transition tests over happy-path positive tests.",
        "4. Steps MUST reference actual UI element names from the Figma context — no invented labels.",
        "5. The 'rationale' field MUST name the specific defect class or risk this test is designed to expose.",
        "6. The 'area' field MUST align with the Exploration Directive — do not default to the easiest area.",
        "",
        "## Output — STRICT JSON only. No markdown fences, no text outside the JSON object.",
        '{"test_case_id":"...","title":"...","screen":"...","preconditions":[...],"steps":[...],'
        '"expected_result":"...","priority":"high|medium|low","area":"...",'
        '"test_type":"positive|negative|boundary|state_transition|recovery|combination",'
        '"rationale":"what specific bug or risk this test is designed to expose"}',
    ]

    return "\n".join(parts)


# ── JSON utils ────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", (text or "").lower())).strip()


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(_normalize(a).split()), set(_normalize(b).split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _is_similar_to_existing(title: str, existing: list[str], threshold: float = 0.72) -> bool:
    return any(_jaccard(title, t) >= threshold for t in existing)


def _next_testcase_id(recent_tests: list[dict], default_prefix: str = "TC") -> str:
    max_n = 0
    for t in recent_tests or []:
        rid = str(t.get("id", "") or "")
        m = re.search(r"(\d+)$", rid)
        if m:
            try:
                max_n = max(max_n, int(m.group(1)))
            except Exception:
                pass
    return f"{default_prefix}-{max_n + 1:03d}"


def _extract_json_text(raw: str) -> str:
    text = (raw or "").strip()
    # Strip markdown fences
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:] if lines[0].startswith("```") else lines
        lines = lines[:-1] if lines and lines[-1].strip() == "```" else lines
        text = "\n".join(lines).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text


def _parse_testcase(raw: str) -> dict:
    try:
        obj = json.loads(_extract_json_text(raw))
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {"raw": raw}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    summary="Gateway health check",
    description="Returns the operational status of the gateway and identifies the active model backend (ngrok, OpenRouter, or Gemini) with its configuration.",
    tags=["system"],
    response_description="Status 'ok' with RAG API URL and active model backend details.",
)
def health():
    model_info = {
        "backend": MODEL_BACKEND,
    }
    if MODEL_BACKEND == "gemini":
        model_info["model"] = PLANNER_GEMINI_MODEL
        model_info["api"] = "Google GenAI API"
    elif MODEL_BACKEND == "openrouter":
        model_info["model"] = OPENROUTER_MODEL
        model_info["api"] = OPENROUTER_BASE_URL
    else:
        model_info["api"] = MODEL_API_URL
    return {"status": "ok", "rag_api": RAG_API_URL, "model_api": MODEL_API_URL, "model": model_info}


@app.post(
    "/srs/ingest",
    summary="Ingest SRS document",
    description=(
        "Loads a Software Requirements Specification into the Neo4j knowledge graph. "
        "The document is chunked and optionally summarised by the LLM to produce a compact "
        "planner-friendly summary used in every subsequent test generation call.\n\n"
        "**Re-ingesting replaces** the existing SRS for this project. "
        "Re-ingest whenever the requirements change.\n\n"
        "**Tip:** Set `use_model_summary=true` (default) for best test quality. "
        "The model summary is stored once here so it doesn't consume tokens on every generation call."
    ),
    tags=["ingest"],
    response_description="Ingest result including chunk count, SRS ID, and summary metadata.",
    responses={
        400: {"description": "Path traversal detected in `source_path`."},
        404: {"description": "SRS file not found at `source_path`."},
        413: {"description": "SRS text or file exceeds the 500,000 character limit."},
        503: {"description": "Model summarization failed (only raised when `require_model_summary=true`)."},
    },
)
def ingest_srs(req: IngestSRSRequest, authorization: str | None = Header(default=None)):
    _check_gateway_auth(authorization)

    if any(part == ".." for part in Path(req.source_path).parts):
        raise HTTPException(status_code=400, detail="Path traversal not allowed in source_path")

    srs_text = req.srs_text
    if srs_text and len(srs_text) > MAX_SRS_CHARS:
        raise HTTPException(status_code=413, detail=f"srs_text exceeds {MAX_SRS_CHARS:,} character limit")

    if not srs_text:
        src = Path(req.source_path)
        if not src.exists():
            raise HTTPException(status_code=404, detail=f"SRS file not found: {req.source_path}")
        srs_text = src.read_text(encoding="utf-8", errors="ignore")
        if len(srs_text) > MAX_SRS_CHARS:
            raise HTTPException(status_code=413, detail=f"SRS file exceeds {MAX_SRS_CHARS:,} character limit")

    srs_summary = ""
    summary_source = "fallback"
    if req.use_model_summary and (srs_text or "").strip():
        try:
            srs_summary = _summarize_srs_with_model(srs_text)
            if srs_summary:
                summary_source = "model"
        except Exception as e:
            if req.require_model_summary:
                raise HTTPException(status_code=503, detail=f"SRS summarization failed: {e}")
            srs_summary = ""

    payload = {
        "project": req.project,
        "source_path": req.source_path,
        "srs_text": srs_text,
        "chunk_chars": req.chunk_chars,
        "srs_summary": srs_summary or None,
    }
    out = _rag_post("/ingest/srs", payload)
    out["srs_summary_source"] = summary_source
    out["srs_summary_chars"] = len(srs_summary or "")
    return out


@app.post(
    "/figma/ingest",
    summary="Ingest Figma design file",
    description=(
        "Parses a Figma export JSON and stores all screens, interactive UI elements (buttons, inputs, "
        "navigation, toggles, dropdowns), and inferred screen-to-screen navigation transitions in the "
        "Neo4j knowledge graph.\n\n"
        "**Re-ingesting replaces** all existing Figma data for this project.\n\n"
        "The planner uses this data to:\n"
        "- Generate test steps referencing exact UI element labels.\n"
        "- Identify which screens are unexplored and bias coverage toward them.\n"
        "- Infer navigation flows (button → screen) for state-transition tests."
    ),
    tags=["ingest"],
    response_description="Ingest result with screen count, element count, and Figma source name.",
    responses={
        400: {"description": "Path traversal in `source_path`, invalid JSON, or no screens found."},
        404: {"description": "Figma JSON file not found at `source_path`."},
    },
)
def ingest_figma(req: IngestFigmaRequest, authorization: str | None = Header(default=None)):
    _check_gateway_auth(authorization)
    if any(part == ".." for part in Path(req.source_path).parts):
        raise HTTPException(status_code=400, detail="Path traversal not allowed in source_path")
    resp = requests.post(
        f"{RAG_API_URL}/ingest/figma",
        json=req.model_dump(),
        headers=_rag_headers(),
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


@app.post(
    "/project/reset",
    summary="Reset project data slices",
    description=(
        "Selectively deletes data from the project's knowledge graph. "
        "Operates on three independent slices:\n\n"
        "| Slice | Flag | Effect |\n"
        "|-------|------|--------|\n"
        "| Test history | `delete_tests=true` | Clears all verdicts and runs. Next generation starts fresh. |\n"
        "| SRS | `delete_srs=true` | Removes all chunks and summary. Re-ingest required. |\n"
        "| Figma | `delete_figma=true` | Removes all screens and elements. Re-ingest required. |\n\n"
        "The most common use case is `delete_tests=true, delete_srs=false, delete_figma=false` "
        "to start a new testing session without re-ingesting the knowledge base."
    ),
    tags=["project"],
    response_description="Confirmation of which slices were deleted.",
    responses={
        401: {"description": "Invalid or missing Authorization header."},
        503: {"description": "RAG API unavailable."},
    },
)
def reset_project(req: ResetProjectRequest, authorization: str | None = Header(default=None)):
    _check_gateway_auth(authorization)
    return _rag_post("/project/reset", req.model_dump())


@app.post(
    "/agent/next-testcase",
    summary="Generate the next exploratory test case",
    description=(
        "Core endpoint. Runs the full multi-stage planner pipeline and returns the next test case.\n\n"
        "**Pipeline stages:**\n\n"
        "1. **Context bootstrap** — Loads SRS/Figma summaries, screen index, and the last 100 test "
        "results from the knowledge graph.\n\n"
        "2. **Coverage analysis** — Computes a live coverage map: which feature areas have repeated "
        "failures (hot spots), which have zero tests (uncovered), and which are over-tested (exhausted).\n\n"
        "3. **Iterative retrieval loop** — The planner LLM runs up to `max_retrieval_rounds` rounds. "
        "Each round it issues targeted queries against the knowledge graph (SRS business rules, "
        "Figma UI elements, screen-to-screen transitions) and decides whether to retrieve more "
        "context or proceed to generation.\n\n"
        "4. **Test generation** — The LLM generates a structured JSON test case, guided by the "
        "coverage map, exploration directive, and testing heuristics (boundary, invalid input, "
        "state transition, interruption, etc.).\n\n"
        "5. **Deduplication** — If the result is semantically too close to an existing test, "
        "a retry is triggered with a rotated screen focus.\n\n"
        "**Response includes:** the test case JSON, coverage snapshot, retrieval statistics, "
        "and (if `debug_trace=true`) full prompt/response transcripts for every planning round."
    ),
    tags=["agent"],
    response_description=(
        "Generated test case with coverage state, planner trace, and retrieval statistics. "
        "Key fields: `next_testcase` (parsed JSON), `next_testcase_json` (raw string for logging), "
        "`coverage`, `retrieval_plan`, `planner_trace`, `finalization_mode`."
    ),
    responses={
        401: {"description": "Invalid or missing Authorization header."},
        503: {"description": "LLM backend (model) or RAG API unavailable."},
    },
)
def next_testcase(req: NextTestCaseRequest, authorization: str | None = Header(default=None)):
    _check_gateway_auth(authorization)

    # Stage 1: get compact global context (summaries + recent tests + screen index)
    brief = _get_brief_context(req.project)
    recent_tests = brief.get("recent_tests", []) if isinstance(brief, dict) else []
    done_titles = [str(t.get("title", "")).strip() for t in recent_tests if t.get("title")]
    failed_titles = [
        str(t.get("title", "")).strip()
        for t in recent_tests
        if t.get("title") and str(t.get("verdict", "")).lower() == "failed"
    ]
    done_areas = [str(t.get("area", "")) for t in recent_tests if t.get("area")]
    figma_screens = brief.get("screen_index", []) if isinstance(brief, dict) else []
    figma_overview = _get_figma_overview(req.project)
    fallback_screens = _pick_relevant_screens(figma_screens, done_areas, recent_tests)

    # Compute live coverage map — drives exploration directives throughout generation
    coverage_map = _compute_coverage_map(recent_tests, figma_screens)

    # Stage 2: iterative retrieval planning loop
    max_retrieval_rounds = max(1, min(req.max_retrieval_rounds, 6))
    planner_trace: list[dict] = []
    collected_queries: list[str] = []
    selected_screens: list[str] = []
    srs_context_blocks: list[str] = []
    figma_ui_blocks: list[str] = []
    flow_context_blocks: list[str] = []
    last_round_retrieved_notes: list[str] = []
    agent_signaled_ready = False
    finalization_mode = "max_retrieval_rounds_fallback"

    debug_trace: dict = {
        "planner_rounds": [],
        "retrieved_blocks": [],
    }

    for round_no in range(1, max_retrieval_rounds + 1):
        action_prompt = _planner_prompt_for_action(
            brief=brief,
            app_name=req.app_name,
            objective=req.objective,
            retrieval_round=round_no,
            max_rounds=max_retrieval_rounds,
            collected_queries=collected_queries,
            collected_screens=selected_screens,
            context_chars=len("\n\n".join(srs_context_blocks)),
            figma_overview=figma_overview,
            retrieved_notes=last_round_retrieved_notes,
            coverage_map=coverage_map,
        )
        action_model = _call_model(action_prompt, max(320, min(req.max_new_tokens, 700)), False)
        action = _parse_action(action_model.get("answer", ""), fallback_screens)

        if req.debug_trace:
            debug_trace["planner_rounds"].append(
                {
                    "round": round_no,
                    "prompt": action_prompt,
                    "model_answer_raw": action_model.get("answer", ""),
                    "parsed_action": action,
                }
            )

        round_trace = {
            "round": round_no,
            "action": action.get("action", "retrieve"),
            "retrieval_requests": action.get("retrieval_requests", []),
            "focus_queries": action.get("focus_queries", []),
            "target_screens": action.get("target_screens", []),
            "reason": action.get("reason", ""),
        }

        if action["action"] == "produce_testcase":
            planner_trace.append(round_trace)
            agent_signaled_ready = True
            finalization_mode = "agent_signaled_sufficient_context"
            break

        requests_spec = action.get("retrieval_requests", [])
        if not requests_spec:
            # backward-compatible fallback
            requests_spec = [{"source": "srs", "query": q} for q in (action.get("focus_queries", []) or [req.objective])[:2]]
            for s in (action.get("target_screens", []) or fallback_screens[:2])[:2]:
                requests_spec.append({"source": "figma_ui", "screen": s})

        round_retrieved_notes: list[str] = []

        for rr in requests_spec[:3]:
            source = str(rr.get("source", "srs")).strip().lower()
            query_raw = rr.get("query")
            query = str(query_raw).strip() if query_raw is not None else ""
            screen = str(rr.get("screen", "")).strip()

            if source == "srs":
                q = _to_query_expr(query, req.objective)
                if q not in collected_queries:
                    collected_queries.append(q)
                data = _get_srs_and_history(req.project, q, top_k=min(req.top_k, 2))
                block = data.get("context", "")
                if block:
                    srs_context_blocks.append(block)
                    round_retrieved_notes.append(f"srs | query={q} | {_compact_note(block)}")
                    if req.debug_trace:
                        debug_trace["retrieved_blocks"].append({"round": round_no, "source": source, "query": q, "context": block})

            elif source == "figma_ui":
                s = screen or (action.get("target_screens", []) or fallback_screens[:1])[0] if (action.get("target_screens", []) or fallback_screens) else ""
                if s and s not in selected_screens:
                    selected_screens.append(s)
                if s:
                    elements = _get_screen_elements(req.project, s)
                    ui_lines = [f"[Screen: {s}]"]
                    for kind, labels in elements.items():
                        ui_lines.append(f"  {kind}s: {', '.join(labels[:10])}")
                    ui_block = "\n".join(ui_lines)
                    if ui_block.strip() != f"[Screen: {s}]":
                        figma_ui_blocks.append(ui_block)
                        note = f"figma_ui | screen={s} | {_compact_note(ui_block)}"
                        round_retrieved_notes.append(note)
                        if req.debug_trace:
                            debug_trace["retrieved_blocks"].append({"round": round_no, "source": source, "screen": s, "context": ui_block})

            elif source == "figma_flow":
                trans = _get_figma_transitions(req.project, screen_name=screen if screen else None)
                if trans:
                    flow_block = _build_figma_flow_context(trans, top_n=10)
                    if flow_block:
                        flow_context_blocks.append(flow_block)
                        round_retrieved_notes.append(f"figma_flow | screen={screen or '*'} | {_compact_note(flow_block)}")
                        if req.debug_trace:
                            debug_trace["retrieved_blocks"].append({"round": round_no, "source": source, "screen": screen, "context": flow_block})

        round_trace["retrieved_context_chars"] = len("\n\n".join(srs_context_blocks))
        planner_trace.append(round_trace)
        last_round_retrieved_notes = round_retrieved_notes

        if round_no > 1 and not round_retrieved_notes:
            finalization_mode = "no_new_context_early_finalize"
            break

        # safety: avoid overloading prompt size
        if len("\n\n".join(srs_context_blocks)) > 9000:
            break

    # Fallback: ensure at least one retrieval happened
    if not srs_context_blocks:
        fallback_query = req.objective
        data = _get_srs_and_history(req.project, fallback_query, top_k=min(req.top_k, 3))
        block = data.get("context", "")
        if block:
            srs_context_blocks.append(block)
            last_round_retrieved_notes = [f"srs | query={fallback_query} | {_compact_note(block)}"]
        if fallback_screens:
            selected_screens = fallback_screens[:2]

    srs_context = "\n\n".join(dict.fromkeys(srs_context_blocks))[:8000]
    figma_overview_context = _build_figma_overview_context(figma_overview)
    figma_context = "\n\n".join(dict.fromkeys(figma_ui_blocks))[:2400] if figma_ui_blocks else _build_figma_context(req.project, selected_screens[:3])
    figma_flow_context = "\n\n".join(dict.fromkeys(flow_context_blocks))[:2500]

    retrieval_plan = {
        "focus_queries": collected_queries[:2],
        "target_screens": selected_screens[:2],
        "reason": planner_trace[-1].get("reason", "") if planner_trace else "fallback",
    }

    # Stage 4: final testcase generation using targeted context
    prompt = _build_prompt(
        app_name=req.app_name,
        objective=req.objective,
        srs_context=srs_context,
        figma_overview_context=figma_overview_context,
        figma_context=figma_context,
        figma_flow_context=figma_flow_context,
        done_titles=done_titles,
        failed_titles=failed_titles,
        coverage_map=coverage_map,
        recent_tests=recent_tests,
    )

    # 4. Call model
    model_data = _call_model(prompt, req.max_new_tokens, req.enable_thinking)
    raw_answer = model_data.get("answer", "")
    parsed = _parse_testcase(raw_answer)

    if req.debug_trace:
        debug_trace["final_generation"] = {
            "prompt": prompt,
            "model_answer_raw": raw_answer,
            "model_thinking": model_data.get("thinking", ""),
        }

    # 5. Duplicate check — retry with different focus if too similar
    candidate_title = str(parsed.get("title", "")) if isinstance(parsed, dict) else ""
    blocked_titles = list(dict.fromkeys((done_titles or []) + (failed_titles or [])))
    if candidate_title and _is_similar_to_existing(candidate_title, blocked_titles, threshold=0.60):
        # Rotate to next uncovered screens
        already_picked = set(selected_screens)
        alt_screens = [s["screen_name"] for s in figma_screens if s["screen_name"] not in already_picked][:2]
        alt_figma_context = _build_figma_context(req.project, alt_screens) if alt_screens else figma_context

        blocked = "\n".join(f"- {t}" for t in blocked_titles[:20]) or "- none"

        retry_prompt = _build_prompt(
            app_name=req.app_name,
            objective=req.objective + " (RETRY — choose a DISTINCT test case; the previous suggestion was too similar to an already-executed test)",
            srs_context=srs_context,
            figma_overview_context=figma_overview_context,
            figma_context=alt_figma_context,
            figma_flow_context=figma_flow_context,
            done_titles=done_titles,
            failed_titles=failed_titles,
            coverage_map=coverage_map,
            recent_tests=recent_tests,
        ) + "\n\nBlocked titles (semantic overlap with any of these is FORBIDDEN):\n" + blocked
        model_data = _call_model(retry_prompt, req.max_new_tokens, req.enable_thinking)
        raw_answer = model_data.get("answer", "")
        parsed = _parse_testcase(raw_answer)
        if req.debug_trace:
            debug_trace["final_retry"] = {
                "prompt": retry_prompt,
                "model_answer_raw": raw_answer,
                "model_thinking": model_data.get("thinking", ""),
            }

    # 6. Enforce stable unique external testcase ID for downstream logging.
    if isinstance(parsed, dict):
        parsed["test_case_id"] = _next_testcase_id(recent_tests)
        raw_answer = json.dumps(parsed, ensure_ascii=False, indent=2)

    # 7. Auto-log the generated test case so coverage and recent_tests grow on every call,
    #    even when the caller only uses /agent/next-testcase (no executor in the loop).
    #    Only log if this TC-ID hasn't already been recorded (avoids double-counting when
    #    /agent/log-verdict-and-next later records the real verdict).
    if isinstance(parsed, dict) and parsed.get("test_case_id"):
        existing_ids = {str(t.get("id", "")) for t in recent_tests}
        if parsed["test_case_id"] not in existing_ids:
            try:
                _rag_post("/tests/log", {
                    "project": req.project,
                    "test_case_id": parsed["test_case_id"],
                    "title": parsed.get("title", ""),
                    "verdict": "pass",
                    "notes": "[GENERATED] Awaiting execution.",
                    "area": parsed.get("area", "general"),
                })
            except Exception:
                pass

    out = {
        "project": req.project,
        "retrieval_plan": retrieval_plan,
        "planner_trace": planner_trace,
        "finalization_mode": finalization_mode,
        "agent_signaled_ready": agent_signaled_ready,
        "retrieved_context_stats": {
            "retrieval_rounds_executed": len(planner_trace),
            "queries_used": collected_queries[:2],
            "screens_used": selected_screens[:3],
            "srs_context_chars": len(srs_context),
            "figma_overview_chars": len(figma_overview_context),
            "figma_context_chars": len(figma_context),
            "figma_flow_chars": len(figma_flow_context),
        },
        "target_screens": selected_screens[:3],
        "next_testcase_json": raw_answer,
        "next_testcase": parsed,
        "recent_tests_count": len(recent_tests),
        "failed_tests_count": len(failed_titles),
        "thinking": model_data.get("thinking", ""),
        "coverage": {
            "total_tests": coverage_map.get("total_tests", 0),
            "coverage_pct": coverage_map.get("coverage_pct", 0),
            "uncovered_areas": coverage_map.get("uncovered_purposes", []),
            "hot_spots": coverage_map.get("hot_spots", []),
            "exhausted_areas": coverage_map.get("exhausted_areas", []),
            "exploration_directive": _build_exploration_directive(coverage_map, recent_tests),
        },
    }

    if req.debug_trace:
        out["debug_trace"] = debug_trace

    return out


@app.post(
    "/agent/log-verdict",
    summary="Log execution verdict",
    description=(
        "Records the execution verdict for a test case in the knowledge graph.\n\n"
        "Use this when you want to log a result without immediately generating the next test case. "
        "To log a verdict **and** get the next test case in one call, use `/agent/log-verdict-and-next` instead.\n\n"
        "**Note on verdicts:** `blocked` and `skipped` are stored as `failed` internally. "
        "The original verdict is prepended to `notes` so it remains visible in test history."
    ),
    tags=["agent"],
    response_description="Verdict confirmation from the knowledge graph.",
    responses={
        401: {"description": "Invalid or missing Authorization header."},
        503: {"description": "RAG API unavailable."},
    },
)
def log_verdict(req: LogVerdictRequest, authorization: str | None = Header(default=None)):
    _check_gateway_auth(authorization)

    rag_verdict = req.verdict if req.verdict in {"pass", "failed"} else "failed"
    rag_notes = req.notes
    if req.verdict in {"blocked", "skipped"} and req.notes:
        rag_notes = f"[{req.verdict.upper()}] {req.notes}"
    elif req.verdict in {"blocked", "skipped"}:
        rag_notes = f"[{req.verdict.upper()}] Test was {req.verdict}."

    return _rag_post("/tests/log", {
        "project": req.project,
        "test_case_id": req.test_case_id,
        "title": req.title,
        "verdict": rag_verdict,
        "notes": rag_notes,
        "area": req.area,
    })


@app.post(
    "/agent/log-verdict-and-next",
    summary="Log execution verdict and get next test case",
    description=(
        "The primary endpoint for the continuous exploration loop. Atomically:\n\n"
        "1. **Logs the verdict** for the just-executed test case into the knowledge graph, "
        "making it visible to future test generation calls.\n\n"
        "2. **Generates the next test case**, adapting the objective based on the verdict:\n"
        "   - `failed` → focuses on the same area to map adjacent edge cases around the failure.\n"
        "   - `blocked` / `skipped` → seeks an alternative test that avoids the blocking condition.\n"
        "   - `pass` → broadens coverage toward unexplored areas per the current coverage map.\n\n"
        "Override the adaptive objective with `next_objective` for explicit session control.\n\n"
        "**Note on verdicts:** `blocked` and `skipped` are stored as `failed` internally "
        "(the RAG layer uses a binary pass/fail model). The original verdict is prepended to `notes` "
        "so it remains visible in test history."
    ),
    tags=["agent"],
    response_description=(
        "Two-key response: `log` — verdict confirmation from the knowledge graph; "
        "`next` — full next-testcase payload (same schema as `/agent/next-testcase`)."
    ),
    responses={
        401: {"description": "Invalid or missing Authorization header."},
        503: {"description": "LLM backend or RAG API unavailable."},
    },
)
def log_verdict_and_next(req: LogVerdictRequest, authorization: str | None = Header(default=None)):
    _check_gateway_auth(authorization)

    log_data = log_verdict(req, authorization=authorization)

    if req.next_objective.strip():
        next_objective = req.next_objective.strip()
    elif req.verdict == "failed":
        next_objective = (
            f"generate the next best test case targeting the area '{req.area}' "
            "where the last test failed — map out adjacent edge cases"
        )
    elif req.verdict in {"blocked", "skipped"}:
        next_objective = (
            f"generate an alternative test case for the area '{req.area}' "
            "that avoids the blocking condition"
        )
    else:
        next_objective = "generate the next best test case to broaden overall coverage"

    next_req = NextTestCaseRequest(
        project=req.project,
        app_name=req.app_name,
        objective=next_objective,
        top_k=req.top_k,
        max_new_tokens=req.max_new_tokens,
        enable_thinking=req.enable_thinking,
        debug_trace=req.debug_trace,
    )
    next_data = next_testcase(next_req, authorization=authorization)

    return {"log": log_data, "next": next_data}


@app.get(
    "/agent/coverage",
    summary="Get exploration coverage dashboard",
    description=(
        "Returns the live exploration state for a project without generating a test case. "
        "Use this to monitor session progress, identify where to focus, or decide when to stop.\n\n"
        "**Response fields:**\n\n"
        "| Field | Description |\n"
        "|-------|-------------|\n"
        "| `summary` | Totals: test count, coverage %, areas tested vs available. |\n"
        "| `area_breakdown` | Per-area pass/fail counts. |\n"
        "| `uncovered_areas` | Feature areas derived from Figma with zero tests. |\n"
        "| `hot_spots` | Areas with ≥2 failures — highest priority for deeper exploration. |\n"
        "| `exhausted_areas` | Areas with ≥4 tests and 0 failures — diminishing returns. |\n"
        "| `exploration_directive` | Prioritised plain-text next-action recommendation. |\n"
        "| `recent_tests` | Last 20 executed tests with verdicts. |"
    ),
    tags=["agent"],
    response_description="Coverage dashboard with exploration directive and recent test history.",
    responses={
        401: {"description": "Invalid or missing Authorization header."},
        503: {"description": "RAG API unavailable."},
    },
)
def agent_coverage(project: str, authorization: str | None = Header(default=None)):
    _check_gateway_auth(authorization)
    brief = _get_brief_context(project)
    recent_tests = brief.get("recent_tests", []) if isinstance(brief, dict) else []
    figma_screens = brief.get("screen_index", []) if isinstance(brief, dict) else []
    figma_overview = _get_figma_overview(project)

    coverage_map = _compute_coverage_map(recent_tests, figma_screens)
    directive = _build_exploration_directive(coverage_map, recent_tests)

    return {
        "project": project,
        "summary": {
            "total_tests": coverage_map["total_tests"],
            "coverage_pct": coverage_map["coverage_pct"],
            "areas_tested": coverage_map["total_areas_tested"],
            "areas_available": coverage_map["total_areas_available"],
        },
        "area_breakdown": coverage_map["area_stats"],
        "uncovered_areas": coverage_map["uncovered_purposes"],
        "hot_spots": coverage_map["hot_spots"],
        "exhausted_areas": coverage_map["exhausted_areas"],
        "exploration_directive": directive,
        "figma_screen_count": len(figma_overview),
        "recent_tests": recent_tests[:20],
    }


@app.post(
    "/chat",
    summary="RAG-backed free-form Q&A",
    description=(
        "Ask any question about the app under test. The prompt is automatically augmented with "
        "the most semantically relevant SRS chunks retrieved from the knowledge graph, then sent "
        "to the active LLM backend.\n\n"
        "Useful for:\n"
        "- Ad-hoc requirement lookups ('What does FR-12 say about email validation?').\n"
        "- Test planning discussions ('What edge cases should I cover for the Save button?').\n"
        "- Exploring the ingested SRS without running a full test generation pipeline."
    ),
    tags=["chat"],
    response_description="LLM answer with the SRS context chunks that were retrieved and injected.",
    responses={
        503: {"description": "LLM backend or RAG API unavailable."},
    },
)
def chat(req: ChatRequest, authorization: str | None = Header(default=None)):
    _check_gateway_auth(authorization)

    srs_context = ""
    try:
        rag_data = _get_srs_and_history(req.project, req.prompt, req.top_k)
        srs_context = rag_data.get("context", "")
    except Exception:
        pass

    prompt = f"Context:\n{srs_context}\n\nQuestion:\n{req.prompt}" if srs_context else req.prompt
    model_data = _call_model(prompt, req.max_new_tokens, req.enable_thinking)

    return {
        "prompt": req.prompt,
        "context": srs_context,
        "answer": model_data.get("answer", ""),
        "thinking": model_data.get("thinking", ""),
    }
