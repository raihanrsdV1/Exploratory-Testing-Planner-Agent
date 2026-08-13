"""
Single source of truth for every tunable in the system.

Nothing else should call ``os.getenv`` for a project setting. Import from here:

    from settings import EXECUTOR_MAX_STEPS, PROJECT

Values come from ``.env`` (loaded once, here) with the defaults below. Keeping
them in one file is what makes it possible to change a knob without hunting the
same magic number through six modules — and to see, in one screen, exactly how
the system is configured.

Grouped by the component that consumes them.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

# Load .env exactly once, from the project root, regardless of the caller's cwd.
_ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_ROOT, ".env"))


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", ""}


def _str(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


PROJECT_ROOT = _ROOT

# ── Datastore ────────────────────────────────────────────────────────────────
NEO4J_URI = _str("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = _str("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = _str("NEO4J_PASSWORD", "")

# ── Service endpoints + auth ─────────────────────────────────────────────────
RAG_API_URL = _str("RAG_API_URL", "http://127.0.0.1:9010")
RAG_URL = _str("RAG_URL", RAG_API_URL)
GATEWAY_URL = _str("GATEWAY_URL", "http://127.0.0.1:9100")
RAG_API_KEY = _str("RAG_API_KEY")
GATEWAY_API_KEY = _str("GATEWAY_API_KEY")

# ── Project under test ───────────────────────────────────────────────────────
PROJECT = _str("PROJECT", "contacts-app")
PROJECT_NAME = _str("PROJECT_NAME", PROJECT)
APP_NAME = _str("APP_NAME", "the app under test")
TARGET_APP_PACKAGE = _str("TARGET_APP_PACKAGE", "com.android.contacts")
SRS_PATH = _str("SRS_PATH", "./data/inputs/Sample-Contacts-App-SRS.txt")
FIGMA_PATH = _str("FIGMA_PATH", "./data/inputs/GENERATED_JSON.json")

# ── Planner model backend ────────────────────────────────────────────────────
MODEL_BACKEND = _str("MODEL_BACKEND", "openrouter").lower()
MODEL_API_URL = _str("MODEL_API_URL")
PLANNER_GEMINI_MODEL = _str("PLANNER_GEMINI_MODEL", "gemini-2.5-pro")
OPENROUTER_API_KEY = _str("OPENROUTER_API_KEY")
OPENROUTER_MODEL = _str("OPENROUTER_MODEL", "qwen/qwen3.7-flash")
OPENROUTER_BASE_URL = _str("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
GEMINI_API_KEY = _str("GEMINI_API_KEY")

# ── Tiered models ────────────────────────────────────────────────────────────
# Business logic is extracted ONCE per document but every later decision rests on
# it, so it is worth spending a strong model there and a cheap one on the loop.
# Leave EXTRACTION_MODEL empty to reuse OPENROUTER_MODEL.
EXTRACTION_MODEL = _str("EXTRACTION_MODEL", "")
# Sample the extraction N times and keep the best (self-consistency). N=1 disables.
EXTRACTION_SAMPLES = _int("EXTRACTION_SAMPLES", 1)
# Extraction token budget. Reasoning models bill their scratchpad against this,
# and some endpoints refuse to disable reasoning at all — at 4000 the JSON answer
# was truncated and most sections yielded nothing, so the "stronger" model
# produced FEWER requirements than the cheap one.
EXTRACTION_MAX_TOKENS = _int("EXTRACTION_MAX_TOKENS", 0)  # 0 = uncapped
# Model that judges which sample is best; falls back to EXTRACTION_MODEL.
EXTRACTION_JUDGE_MODEL = _str("EXTRACTION_JUDGE_MODEL", "")

# ── Embeddings (semantic retrieval + dedup) ──────────────────────────────────
EMBEDDING_BACKEND = _str("EMBEDDING_BACKEND", "auto").lower()
EMBEDDING_MODEL = _str("EMBEDDING_MODEL")
GEMINI_EMBED_MODEL = _str("GEMINI_EMBED_MODEL", "gemini-embedding-2")

# ── Planner behaviour ────────────────────────────────────────────────────────
TOP_K = _int("TOP_K", 8)
# One ceiling for the whole generation prompt; blocks are filled priority-first
# (see planner/budget.py). This is the main cost dial — every call pays it.
PROMPT_BUDGET_TOKENS = _int("PROMPT_BUDGET_TOKENS", 50_000)
MAX_RETRIEVAL_ROUNDS = _int("MAX_RETRIEVAL_ROUNDS", 3)
# Generation token ceiling. Reasoning models spend 3-7k tokens on their scratchpad
# before the JSON answer; too low a cap truncates it and the response fails to parse.
GENERATION_MAX_TOKENS = _int("GENERATION_MAX_TOKENS", 12000)
# exploit | explore | balanced
EXPLORATION_MODE = _str("EXPLORATION_MODE", "balanced").lower()
if EXPLORATION_MODE not in {"exploit", "explore", "balanced"}:
    EXPLORATION_MODE = "balanced"
# Older knowledge counts for less; halves every N days.
KNOWLEDGE_HALF_LIFE_DAYS = _int("KNOWLEDGE_HALF_LIFE_DAYS", 90)

# ── Knowledge sources: what the planner is allowed to use ────────────────────
# Turn a source off when it is inaccurate rather than deleting its data — a stale
# design file does not just fail to help, it invents screens and controls that do
# not exist, and the planner then writes tests for them.
#
#   srs        requirements + validation rules extracted from the spec
#   figma      the design/UI guide (often drifts from the shipped app)
#   live_ui    the Live App Model: screens actually observed on the device
#   defects    historical defect reports
#   navtree    learned navigation paths
ENABLED_SOURCES = tuple(
    x.strip() for x in _str(
        "ENABLED_SOURCES", "srs,live_ui,defects,navtree"
    ).split(",") if x.strip()
)
# When both a design guide and live observations exist, the device wins: it is
# ground truth, the design file is an intention.
LIVE_UI_OVERRIDES_FIGMA = _bool("LIVE_UI_OVERRIDES_FIGMA", True)

# ── App-boundary policy ──────────────────────────────────────────────────────
# Leaving the app under test is legitimate for some flows (a file picker for VCF
# import, account settings for sync), so excursions are allowed — but drifting
# for many consecutive steps means the agent is lost, not working.
TARGET_APP_ONLY = _bool("TARGET_APP_ONLY", False)
# Packages the agent may legitimately visit mid-test (system pickers etc.).
COMPANION_PACKAGES = tuple(
    x.strip() for x in _str(
        "COMPANION_PACKAGES",
        "com.android.documentsui,com.google.android.documentsui,"
        "com.android.settings,com.android.providers.media",
    ).split(",") if x.strip()
)
# Consecutive out-of-app observations before the run is treated as drifted.
MAX_FOREIGN_STEPS = _int("MAX_FOREIGN_STEPS", 12)

# ── Executor (device runs) ───────────────────────────────────────────────────
EXECUTOR_LLM_PROVIDER = _str("EXECUTOR_LLM_PROVIDER", "OpenRouter")
EXECUTOR_LLM_MODEL = _str("EXECUTOR_LLM_MODEL", "qwen/qwen3-vl-32b-instruct")
EXECUTOR_ROUNDS = _int("EXECUTOR_ROUNDS", 2)
# Step budget per test. Set generously: when the agent runs out of steps the run
# is scored STEP_LIMIT_EXCEEDED and excluded from defect metrics, so a tight
# budget silently shrinks the sample rather than revealing anything about the app.
EXECUTOR_MAX_STEPS = _int("EXECUTOR_MAX_STEPS", 50)
EXECUTOR_TIMEOUT = _int("EXECUTOR_TIMEOUT", 420)
EXECUTOR_MAX_TOKENS = _int("EXECUTOR_MAX_TOKENS", 4000)
EXECUTOR_CONTEXT_WINDOW = _int("EXECUTOR_CONTEXT_WINDOW", 128_000)
SELF_HEAL = _bool("SELF_HEAL", True)
# Show the device agent screenshots as well as the accessibility tree. Needs a
# vision-capable model. Worth it for verdict quality and for Compose/Flutter
# screens that expose almost no structural control names.
EXECUTOR_VISION = _bool("EXECUTOR_VISION", True)

# ── Live App Model (screen identity) ─────────────────────────────────────────
# Structural Jaccard at/above this merges an observation into an existing state.
STATE_MERGE_THRESHOLD = _float("STATE_MERGE_THRESHOLD", 0.9)
# Hamming distance on the 8x8 average hash, used only when the a11y tree is thin.
PHASH_MATCH_DISTANCE = _int("PHASH_MATCH_DISTANCE", 6)
# A screen observed mid-render shows a strict SUBSET of its controls (the nav bar,
# search bar and app bar paint late). Jaccard punishes it for elements it never had
# a chance to render — 0.52 for two captures of the same list — so the same screen
# forked into two states. Containment (|A∩B| / min(|A|,|B|)) is 1.0 for a partial
# capture and stays low (0.36) for genuinely different screens.
STATE_CONTAINMENT_THRESHOLD = _float("STATE_CONTAINMENT_THRESHOLD", 0.95)
# Guard against a trivially small capture being contained in everything.
STATE_MIN_CONTROLS_FOR_CONTAINMENT = _int("STATE_MIN_CONTROLS_FOR_CONTAINMENT", 8)

# ── Autonomous crawler ───────────────────────────────────────────────────────
CRAWL_ROUNDS = _int("CRAWL_ROUNDS", 3)
CRAWL_MAX_STEPS = _int("CRAWL_MAX_STEPS", 25)
CRAWL_GOAL = _str("CRAWL_GOAL")

# ── Simulator (device-free loop) ─────────────────────────────────────────────
SIM_ROUNDS = _int("SIM_ROUNDS", 3)
SIM_FAIL_EVERY = _int("SIM_FAIL_EVERY", 5)
SIM_OUTPUT_FILE = _str("SIM_OUTPUT_FILE")

# ── Debug / reset toggles ────────────────────────────────────────────────────
DEBUG_TRACE = _bool("DEBUG_TRACE", True)
RESET_TESTS_FIRST = _bool("RESET_TESTS_FIRST", False)
# Start every batch from an identical graph: leftover tests skew dedup, coverage
# and risk, and leftover UIStates credit this run with a map it did not build.
CLEAN_SLATE = _bool("CLEAN_SLATE", True)
# Reset the APP itself before each test, not just the graph. Without it contacts
# created by earlier tests accumulate (30 by the sixth run), so any test whose
# setup assumes a clean app becomes permanently unsatisfiable, and leftover form
# state leaks between tests.
#   pm_clear  — wipe app data entirely (true clean state, slowest)
#   relaunch  — force-stop and reopen (keeps data, clears navigation/form state)
#   none      — leave the device alone
DEVICE_RESET = _str("DEVICE_RESET", "pm_clear").lower()
# WHEN the reset happens.
#   suite — once before the whole run (default). State then accumulates across
#           tests the way it does for a human exploratory tester, so bugs that
#           only appear in a messy app (stale form data, growing lists) stay
#           discoverable. Resetting between every test makes that class of bug
#           structurally impossible to find.
#   test  — before every test: maximum reproducibility, but each data-dependent
#           test must rebuild its own fixtures and accumulation bugs vanish.
DEVICE_RESET_SCOPE = _str("DEVICE_RESET_SCOPE", "suite").lower()
# Packages cleared alongside the app. On Android the app's DATA usually lives in a
# separate provider: clearing com.google.android.contacts leaves every contact in
# place because they belong to com.android.providers.contacts. Clearing the app
# alone looked successful while resetting nothing.
DATA_PROVIDER_PACKAGES = tuple(
    x.strip() for x in _str("DATA_PROVIDER_PACKAGES", "com.android.providers.contacts").split(",")
    if x.strip()
)
RESET_ALL_FIRST = _bool("RESET_ALL_FIRST", False)

# ── Logging / observability ──────────────────────────────────────────────────
LOG_LEVEL = _str("LOG_LEVEL", "INFO").upper()
LOG_FILE = _str("LOG_FILE", "logs/app.jsonl")
LOGTAIL_SOURCE_TOKEN = _str("LOGTAIL_SOURCE_TOKEN")


def summary() -> dict:
    """Effective configuration, for logging at startup or on the dashboard."""
    return {
        "project": PROJECT,
        "app_package": TARGET_APP_PACKAGE,
        "model_backend": MODEL_BACKEND,
        "prompt_budget_tokens": PROMPT_BUDGET_TOKENS,
        "executor_max_steps": EXECUTOR_MAX_STEPS,
        "executor_timeout": EXECUTOR_TIMEOUT,
        "executor_rounds": EXECUTOR_ROUNDS,
        "executor_model": EXECUTOR_LLM_MODEL,
        "executor_vision": EXECUTOR_VISION,
        "state_merge_threshold": STATE_MERGE_THRESHOLD,
        "exploration_mode": EXPLORATION_MODE,
        "enabled_sources": list(ENABLED_SOURCES),
        "extraction_model": EXTRACTION_MODEL or OPENROUTER_MODEL,
        "extraction_samples": EXTRACTION_SAMPLES,
        "target_app_only": TARGET_APP_ONLY,
    }
