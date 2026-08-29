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
PROJECT = _str("PROJECT", "default")
PROJECT_NAME = _str("PROJECT_NAME", PROJECT)
APP_NAME = _str("APP_NAME", "the app under test")
# No default: a package default means an unconfigured run silently targets
# whichever app the default happens to name, instead of failing loudly.
TARGET_APP_PACKAGE = _str("TARGET_APP_PACKAGE", "")
# The launcher activity, so the app can be started deterministically instead of
# being searched for by name. Find it with:
#   adb shell cmd package resolve-activity --brief <package> | tail -1
TARGET_APP_ACTIVITY = _str("TARGET_APP_ACTIVITY", "")
# Every name the launcher icon might carry, comma-separated. The label is in the
# app's own locale — ShobarKhamar ships as "সবার খামার" — so an English-only
# lookup misses, and an agent that cannot find the app concludes it is NOT
# INSTALLED and goes to the Play Store to install it. That actually happened:
# the agent left the app, hit a Google sign-in wall, and uninstalled the app
# under test on its way past.
TARGET_APP_LABELS = tuple(
    x.strip() for x in _str("TARGET_APP_LABELS", "").split(",") if x.strip()
)


def app_identity_block() -> str:
    """Unambiguous instructions for opening the app under test.

    The package name is authoritative; labels are hints only.
    """
    lines = [
        f"The app under test is the package '{TARGET_APP_PACKAGE}'. It is ALREADY "
        f"INSTALLED on this device."
    ]
    if TARGET_APP_ACTIVITY:
        lines.append(f"Its launcher activity is '{TARGET_APP_ACTIVITY}'.")
    if TARGET_APP_LABELS:
        shown = " or ".join(f"'{x}'" for x in TARGET_APP_LABELS)
        lines.append(
            f"Its launcher icon may be labelled {shown} — the label is in the "
            f"app's own language, so do not rely on it."
        )
    lines.append(
        "Open it BY PACKAGE NAME. If a lookup by display name fails, that means "
        "your search term was wrong, NOT that the app is missing."
    )
    if TARGET_APP_ONLY:
        lines.append(
            f"Stay inside '{TARGET_APP_PACKAGE}' for the whole test. Never open "
            f"the Play Store, a browser, or system settings, and never install, "
            f"update or uninstall anything. If you believe the app is gone, "
            f"re-open it by package name and continue. Report failure rather "
            f"than leaving the app to look for it."
        )
    return "\n".join(lines)
SRS_PATH = _str("SRS_PATH", "")   # no default: see FIGMA_PATH below
# Defaults to nothing on purpose. A design file belongs to ONE application, so a
# default path silently attaches whichever file happens to be in the repo to
# whatever project is being ingested — and `FIGMA_PATH=` cannot clear it, because
# _str() falls back to the default on an empty value. Set it explicitly per project.
FIGMA_PATH = _str("FIGMA_PATH", "")

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
# it, so it is worth spending a stronger model there and a cheap one on the loop.
# Leave EXTRACTION_MODEL empty to reuse OPENROUTER_MODEL.
#
# COST: extraction is by far the most expensive thing this system does. One pass
# is ~(sections x samples) + 1 uncapped calls, so on the flagship tier it cost
# ~$1.80 per document — roughly 200 test executions. A large non-flagship model
# with a capped output does the same job for a few cents; spend the difference on
# running more tests, not on re-reading the same document.
EXTRACTION_MODEL = _str("EXTRACTION_MODEL", "")
# Sample the extraction N times and keep the best (self-consistency). N=1 disables.
# Each extra sample re-runs EVERY section, so cost scales linearly with N.
EXTRACTION_SAMPLES = _int("EXTRACTION_SAMPLES", 1)
# Extraction token budget. Reasoning models bill their scratchpad against this,
# and some endpoints refuse to disable reasoning at all — at 4000 the JSON answer
# was truncated and most sections yielded nothing, so the "stronger" model
# produced FEWER requirements than the cheap one. 0 = uncapped, which is safe on
# a non-reasoning model but is what made the flagship tier so expensive.
EXTRACTION_MAX_TOKENS = _int("EXTRACTION_MAX_TOKENS", 8000)
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

# ── Session identity: pre-provisioned credentials ────────────────────────────
# Some apps gate every interesting flow behind a login the agent cannot perform
# by itself: the OTP goes to a phone we do not control, seller signup needs NID
# documents and admin approval, and each successful registration consumes a phone
# number that cannot be reused. For those apps the account is supplied here,
# already created and approved, and the agent signs in rather than signs up.
#
# APP_LOGIN_ROLE is what the planner is told it is testing as, so it can reason
# about which flows are even reachable (a buyer cannot publish a product).
APP_LOGIN_ROLE = _str("APP_LOGIN_ROLE", "")
APP_LOGIN_IDENTIFIER = _str("APP_LOGIN_IDENTIFIER", "")   # phone number / email
APP_LOGIN_SECRET = _str("APP_LOGIN_SECRET", "")           # password
# Free text for anything the identifier/secret pair does not capture: which
# button to press, an OTP that is fixed for test numbers, a PIN, a second factor.
APP_LOGIN_HINT = _str("APP_LOGIN_HINT", "")

# What the signed-in account ALREADY HAS. The planner knows the account's role
# but not its state, so it generates tests for a state that does not exist: a
# "register a new farm" test against an account whose farm is already
# registered has no entry point to reach, and the agent cycles hunting for a
# screen the app will never show. Describe the account as a tester would find
# it, in the app's own vocabulary.
APP_ACCOUNT_STATE = _str("APP_ACCOUNT_STATE", "")


def verification_block() -> str:
    """How to establish that a test's outcome actually happened.

    An agent that acts first and looks afterwards cannot tell "the list is
    empty" from "the page did not load", so it repeats the action forever. In a
    real run it tapped one control eight times with identical reasoning, against
    a list screen that shows nothing at all when empty — no "no items yet"
    message — and the test it was running (abandoning a wizard creates no
    partial record) had in fact already PASSED. It had the answer on screen and
    no way to recognise it.
    """
    return (
        "ESTABLISH A BASELINE FIRST: before performing the action under test, "
        "visit the screen where its result will be visible and note what is "
        "there. You cannot judge 'nothing was created' unless you know what the "
        "screen looked like beforehand.\n"
        "JUDGING AN EMPTY RESULT: a list that renders nothing may be genuinely "
        "empty rather than broken — many apps show no message for an empty "
        "list. If you reach a screen twice and it shows the same thing both "
        "times, treat that as the answer, not as a failure to load. Do NOT "
        "repeat an action that produced no change; decide what the unchanged "
        "screen means and report it.\n"
        "REPORT WHAT YOU SAW: state the observed outcome and whether it matches "
        "the expected result. 'The list was empty, so no record was created' is "
        "a complete and valid result."
    )


def device_input_block() -> str:
    """Text-entry guidance for the device agent.

    The driver's clear-before-type cannot verify itself on every framework: it
    reads the focused field's length, and when that is unreadable the check is
    skipped and it reports success having cleared nothing. The typed text is
    then inserted at the cursor instead of replacing the value, so a field
    holding "Trust Dairy Farm 1" became "Trust Dairy  FarTest FTest Farm 1rm 1"
    after two attempts. The agent cannot detect this unless it looks.
    """
    return (
        "TEXT ENTRY: typing into a field does NOT reliably replace what is already "
        "there — it may be inserted in the middle of the existing value. After "
        "typing into any field, READ THE FIELD BACK and confirm it contains "
        "exactly what you intended and nothing else. If it does not, tap the "
        "field, delete its contents one character at a time until it is empty, "
        "then type again. Never retype into a field you have not verified is "
        "empty; repeating the same entry corrupts the value further."
    )


def app_login_block() -> str:
    """Credentials as a prompt fragment for the DEVICE agent, or '' if unset.

    Only the executor gets the secret — it is the only component that types it.
    The planner gets `app_session_block()` instead, which names the role without
    the password, so a credential never enters the much larger, much more widely
    logged planning prompt.
    """
    if not APP_LOGIN_IDENTIFIER:
        return ""
    lines = ["If the app shows a login screen, sign in with these credentials "
             "instead of registering a new account:"]
    lines.append(f"  identifier / phone: {APP_LOGIN_IDENTIFIER}")
    if APP_LOGIN_SECRET:
        lines.append(f"  password: {APP_LOGIN_SECRET}")
    if APP_LOGIN_HINT:
        lines.append(f"  note: {APP_LOGIN_HINT}")
    lines.append("Never create a new account and never change or reset this "
                 "account's password — both would lock the suite out.")
    return "\n".join(lines)


def app_session_block() -> str:
    """Role context for the PLANNER — no secret."""
    if not APP_LOGIN_ROLE:
        return ""
    lines = [f"The device is already signed in as a '{APP_LOGIN_ROLE}'. Generate tests "
             f"reachable by that role; do not assume permissions it lacks."]
    if APP_ACCOUNT_STATE:
        lines.append(
            f"Current account state: {APP_ACCOUNT_STATE} Do NOT write a test that "
            f"requires creating something this account already has — there is no "
            f"entry point for it and the test cannot run. Test changing, viewing "
            f"or validating what exists instead."
        )
    return " ".join(lines)


# ── Out-of-scope areas ───────────────────────────────────────────────────────
# Requirements the agent must not attempt even though they exist in the spec,
# because completing them needs something outside the agent's reach: an SMS code,
# an identity document, an admin approval, or a phone number that is spent on
# first use. These are reported as deliberately-untested, NOT as failures, so
# they never contaminate the defect count.
OUT_OF_SCOPE = tuple(
    x.strip() for x in _str("OUT_OF_SCOPE", "").split(",") if x.strip()
)

# Preconditions the tester can only OBSERVE, never establish. The planner emits
# these for tests that do not need them ("no existing records" on a field
# validation test), and the device agent then abandons the run after two actions
# without exercising anything. Such a precondition is worse than none.
#
# App-agnostic by default: only phrases that are unachievable for ANY app belong
# here. Anything naming a specific domain object belongs in the project's .env,
# because a phrase that is unachievable in one app is ordinary setup in another.
UNACHIEVABLE_PRECONDITIONS = tuple(
    x.strip().lower() for x in _str(
        "UNACHIEVABLE_PRECONDITIONS",
        "empty database,clean state,database is empty,fresh install,"
        "no data exists,sim card,cloud account",
    ).split(",") if x.strip()
)

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
# Containment scores a small control set inside a large one as 1.0, so a nearly
# empty screen is "contained" in every screen. Cap how different the two sizes
# may be before a containment match is allowed.
STATE_CONTAINMENT_MAX_RATIO = _float("STATE_CONTAINMENT_MAX_RATIO", 2.0)
# Same-widgets-different-values match. A control's content-description is its
# label on a button but its VALUE on a form field ("Select the type of animal"
# becomes "Cow"), so a multi-step form fragments into a new state per step — one
# Add-Cattle wizard produced nine. Comparing skeletons (resource_id + class)
# asks "is this the same set of widgets?" while the full-key floor keeps two
# genuinely different screens with similar widget types apart.
STATE_SKELETON_THRESHOLD = _float("STATE_SKELETON_THRESHOLD", 0.95)
STATE_SKELETON_MIN_FULL = _float("STATE_SKELETON_MIN_FULL", 0.60)

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
# Whether the clean slate also deletes the LEARNED APP MAP (UIState nodes,
# transitions, navigation memory). It should not, by default.
#
# Test results are outcomes and must be wiped for a clean measurement. The app
# map is knowledge ABOUT THE APP — what its screens are called and how they
# connect — and deleting it makes every campaign start blind. Round one then has
# no screen names, so the planner invents them ("Animal Registration Multi-step
# Flow" for a screen actually called "Add Cattle") and the agent burns its step
# budget hunting for screens that do not exist. It also makes REQ-302/303
# untestable: navigation memory cannot pay off across runs it is deleted between.
CLEAN_SLATE_APPMODEL = _bool("CLEAN_SLATE_APPMODEL", False)
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
    x.strip() for x in _str("DATA_PROVIDER_PACKAGES", "").split(",")
    if x.strip()
)
RESET_ALL_FIRST = _bool("RESET_ALL_FIRST", False)

# Test media re-seeded onto the device after every reset. `pm clear` deletes the
# app's MediaStore contributions, so images seeded once by hand vanish at the
# first clean slate and every image-dependent test then fails for want of a file
# to pick — which reads as an app defect and is not one.
DEVICE_FIXTURE_DIR = _str("DEVICE_FIXTURE_DIR", os.path.join(_ROOT, "data", "fixtures", "media"))
DEVICE_FIXTURE_DEST = _str("DEVICE_FIXTURE_DEST", "/sdcard/Pictures")


# ── Failure attribution taxonomy ─────────────────────────────────────────────
# The single most important classification in the system: it decides whether a
# failed run is evidence about the APP, a limitation of OUR agent, or an
# environment problem — and those must never be summed. Defined once here
# because it was previously duplicated between the executor and the analysis
# script, the two copies drifted, and NAVIGATION_LIVELOCK ended up counted as an
# agent fault in one and as "unclassified" in the other. The reporting copy was
# the one missing it, so autonomy read 100% when it was 67%.
APP_FAULT = frozenset({"ASSERTION_FAILURE", "CRASH", "APP_UNRESPONSIVE"})
AGENT_FAULT = frozenset({"TIMEOUT", "ELEMENT_NOT_FOUND", "NAVIGATION_FAILURE",
                         "NAVIGATION_LIVELOCK"})
ENV_FAULT = frozenset({"PRECONDITION_NOT_MET", "PERMISSION_DENIED",
                       "STEP_LIMIT_EXCEEDED"})

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
