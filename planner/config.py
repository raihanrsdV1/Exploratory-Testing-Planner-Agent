"""Environment configuration and gateway authentication."""

from __future__ import annotations

import os

import settings as _s

from fastapi import HTTPException
from dotenv import load_dotenv

load_dotenv()

# Local RAG / knowledge-graph service (on-device).
RAG_API_URL = (os.getenv("RAG_API_URL") or "http://127.0.0.1:9010").rstrip("/")
RAG_API_KEY = os.getenv("RAG_API_KEY", "")

# Model backend: "ngrok" (custom /generate), "openrouter", or "gemini".
MODEL_BACKEND = (os.getenv("MODEL_BACKEND") or "ngrok").strip().lower()
MODEL_API_URL = (os.getenv("MODEL_API_URL") or "http://127.0.0.1:8000").rstrip("/")

# Gemini (MODEL_BACKEND=gemini).
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
PLANNER_GEMINI_MODEL = os.getenv("PLANNER_GEMINI_MODEL", "gemini-2.5-pro")

# OpenRouter (MODEL_BACKEND=openrouter).
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "qwen/qwen3.6-plus:free")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# Gateway-level API key (optional). When set, every request needs Authorization: Bearer <key>.
GATEWAY_API_KEY = os.getenv("GATEWAY_API_KEY", "")

# Default display name for the app under test. Purely cosmetic — injected into
# prompts so the model has a label; carries no app-specific behaviour.
APP_NAME = os.getenv("APP_NAME", "the app under test")

# Hard cap on ingested SRS text size (~500 KB).
MAX_SRS_CHARS = 500_000

# Explore/exploit balance for a run (ETA: defect-focused depth vs. coverage breadth):
#   exploit  — drill into areas that have already broken (defect-prone depth)
#   explore  — push into untested areas (coverage breadth)
#   balanced — investigate failures first, then expand (default; today's behaviour)
EXPLORATION_MODE = _s.EXPLORATION_MODE


# ── Prompt budget (planner/budget.py) ────────────────────────────────────────
# One global ceiling for the generation prompt, replacing the old scattered
# per-block caps. Blocks are filled priority-first, so the budget is spent on the
# oracle (business rules), the UI controls, and what past tests proved, before it
# is spent on lower-value context. Raise for richer prompts, lower to cut cost;
# every call pays this, so it is the main cost dial in the system.
PROMPT_BUDGET_TOKENS = _s.PROMPT_BUDGET_TOKENS


def check_gateway_auth(authorization: str | None) -> None:
    """Raise 401 unless the gateway API key matches (no-op when key is unset)."""
    if not GATEWAY_API_KEY:
        return
    if authorization != f"Bearer {GATEWAY_API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")
