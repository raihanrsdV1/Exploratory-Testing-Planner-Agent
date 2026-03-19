import os
import json
import re

import requests
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

# Local services (on your device)
RAG_API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:9000").rstrip("/")
RAG_API_KEY = os.getenv("RAG_API_KEY", "")

# Notebook model API tunnel URL
MODEL_API_URL = os.getenv("MODEL_API_URL", "https://c4e0-34-13-217-185.ngrok-free.app").rstrip("/")
if not MODEL_API_URL:
    raise RuntimeError("Set MODEL_API_URL to notebook public URL, e.g. https://xxxx.ngrok-free.app")

GATEWAY_API_KEY = os.getenv("GATEWAY_API_KEY", "")

app = FastAPI(title="Local Agent Gateway")


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    top_k: int = 3
    max_new_tokens: int = 512
    enable_thinking: bool = False


class NextTestCaseRequest(BaseModel):
    project: str = Field(..., min_length=1)
    app_name: str = "contacts app"
    objective: str = "generate the next best test case"
    top_k: int = 6
    max_new_tokens: int = 700
    enable_thinking: bool = False


class LogVerdictRequest(BaseModel):
    project: str = Field(..., min_length=1)
    app_name: str = "contacts app"
    test_case_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    verdict: str = Field(..., pattern="^(pass|failed)$")
    notes: str = ""
    area: str = "general"
    top_k: int = 6
    max_new_tokens: int = 700
    enable_thinking: bool = False


class IngestSRSRequest(BaseModel):
    project: str = Field(..., min_length=1)
    source_path: str = Field(..., min_length=1)
    srs_text: str | None = None
    chunk_chars: int = 1200


def _check_gateway_auth(authorization: str | None):
    if not GATEWAY_API_KEY:
        return
    expected = f"Bearer {GATEWAY_API_KEY}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _rag_headers():
    if not RAG_API_KEY:
        return {}
    return {"Authorization": f"Bearer {RAG_API_KEY}"}


def _retrieve_context(project: str, query: str, top_k: int) -> dict:
    try:
        resp = requests.post(
            f"{RAG_API_URL}/retrieve",
            json={"project": project, "query": query, "top_k": top_k},
            headers=_rag_headers(),
            timeout=45,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"RAG backend unavailable: {e}")


def _call_model(prompt: str, max_new_tokens: int, enable_thinking: bool) -> dict:
    try:
        model_resp = requests.post(
            f"{MODEL_API_URL}/generate",
            json={
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "enable_thinking": enable_thinking,
            },
            timeout=180,
        )
        model_resp.raise_for_status()
        return model_resp.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Model backend unavailable: {e}")


def _next_testcase_prompt(app_name: str, objective: str, context: str) -> str:
    return (
        "You are a senior QA test designer. "
        f"System under test: {app_name}.\n"
        "Task: Propose exactly ONE next test case that is not a duplicate of previously executed tests.\n"
        f"Objective: {objective}.\n\n"
        "Use this context (SRS + prior tests + verdicts):\n"
        f"{context}\n\n"
        "Return STRICT JSON only with keys: "
        "test_case_id, title, preconditions, steps, expected_result, priority, area, rationale.\n"
        "Requirements:\n"
        "- preconditions: array of strings\n"
        "- steps: array of clear numbered actions\n"
        "- expected_result: concise and measurable\n"
        "- priority: one of [high, medium, low]\n"
        "- area: feature area in the app\n"
        "- rationale: explain why this is the best next test after previous verdict history\n"
        "No markdown. JSON only."
    )


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", (text or "").lower())).strip()


def _jaccard(a: str, b: str) -> float:
    sa = set(_normalize(a).split())
    sb = set(_normalize(b).split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _is_similar_to_existing(candidate_title: str, existing_titles: list[str], threshold: float = 0.72) -> bool:
    for t in existing_titles:
        if _jaccard(candidate_title, t) >= threshold:
            return True
    return False


def _extract_json_text(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _parse_testcase(raw: str) -> dict:
    cleaned = _extract_json_text(raw)
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {"raw": raw}


@app.get("/health")
def health():
    return {"status": "ok", "rag_api": RAG_API_URL, "model_api": MODEL_API_URL}


@app.post("/srs/ingest")
def ingest_srs(req: IngestSRSRequest, authorization: str | None = Header(default=None)):
    _check_gateway_auth(authorization)
    resp = requests.post(
        f"{RAG_API_URL}/ingest/srs",
        json=req.model_dump(),
        headers=_rag_headers(),
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


@app.post("/agent/next-testcase")
def next_testcase(req: NextTestCaseRequest, authorization: str | None = Header(default=None)):
    _check_gateway_auth(authorization)

    retrieval_query = (
        f"Find SRS details and recent executed tests for {req.app_name}. "
        "Focus on coverage gaps and what should be tested next."
    )
    rag_data = _retrieve_context(req.project, retrieval_query, req.top_k)
    context = rag_data.get("context", "")
    recent_tests = rag_data.get("recent_tests", []) if isinstance(rag_data, dict) else []

    done_titles = [str(t.get("title", "")).strip() for t in recent_tests if str(t.get("title", "")).strip()]
    failed_titles = [
        str(t.get("title", "")).strip()
        for t in recent_tests
        if str(t.get("title", "")).strip() and str(t.get("verdict", "")).lower() == "failed"
    ]

    history_block = "\n".join([f"- {x}" for x in done_titles[:30]])
    failed_block = "\n".join([f"- {x}" for x in failed_titles[:20]])

    base_prompt = _next_testcase_prompt(req.app_name, req.objective, context)
    base_prompt += (
        "\n\nAlready executed test titles (must avoid semantic duplicates):\n"
        f"{history_block if history_block else '- none'}\n\n"
        "Failed tests to prioritize follow-up/adjacent coverage:\n"
        f"{failed_block if failed_block else '- none'}\n\n"
        "Decision policy:\n"
        "1) Do NOT propose a testcase semantically similar to executed titles.\n"
        "2) Prefer generating next testcase that is adjacent to failed behavior or closes a coverage gap.\n"
        "3) Keep area diverse over time; avoid repeating same validation pattern in consecutive turns."
    )

    model_data = _call_model(base_prompt, req.max_new_tokens, req.enable_thinking)
    raw_answer = model_data.get("answer", "")
    parsed = _parse_testcase(raw_answer)

    candidate_title = str(parsed.get("title", "")) if isinstance(parsed, dict) else ""
    if candidate_title and _is_similar_to_existing(candidate_title, done_titles):
        retry_prompt = (
            base_prompt
            + "\n\nYour previous proposal was too similar to existing tests. "
              "Generate a DIFFERENT testcase in a different feature area or scenario. "
              "Must still satisfy JSON schema."
        )
        model_data = _call_model(retry_prompt, req.max_new_tokens, req.enable_thinking)
        raw_answer = model_data.get("answer", "")
        parsed = _parse_testcase(raw_answer)

    return {
        "project": req.project,
        "context": context,
        "next_testcase_json": raw_answer,
        "next_testcase": parsed,
        "recent_tests_count": len(recent_tests),
        "failed_tests_count": len(failed_titles),
        "thinking": model_data.get("thinking", ""),
    }


@app.post("/agent/log-verdict-and-next")
def log_verdict_and_next(req: LogVerdictRequest, authorization: str | None = Header(default=None)):
    _check_gateway_auth(authorization)

    log_resp = requests.post(
        f"{RAG_API_URL}/tests/log",
        json={
            "project": req.project,
            "test_case_id": req.test_case_id,
            "title": req.title,
            "verdict": req.verdict,
            "notes": req.notes,
            "area": req.area,
        },
        headers=_rag_headers(),
        timeout=45,
    )
    log_resp.raise_for_status()
    log_data = log_resp.json()

    next_req = NextTestCaseRequest(
        project=req.project,
        app_name=req.app_name,
        objective="generate the next best test case after latest verdict",
        top_k=req.top_k,
        max_new_tokens=req.max_new_tokens,
        enable_thinking=req.enable_thinking,
    )
    next_data = next_testcase(next_req, authorization=authorization)

    return {
        "log": log_data,
        "next": next_data,
    }


@app.post("/chat")
def chat(req: ChatRequest, authorization: str | None = Header(default=None)):
    _check_gateway_auth(authorization)

    context = ""
    if req.prompt.strip():
        try:
            rag_data = _retrieve_context("default", req.prompt, req.top_k)
            context = rag_data.get("context", "")
        except Exception:
            context = ""

    prompt_with_context = (
        f"Context:\n{context}\n\nUser question:\n{req.prompt}" if context else req.prompt
    )

    model_data = _call_model(prompt_with_context, req.max_new_tokens, req.enable_thinking)

    return {
        "prompt": req.prompt,
        "context": context,
        "answer": model_data.get("answer", ""),
        "thinking": model_data.get("thinking", ""),
    }
