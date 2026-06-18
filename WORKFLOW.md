# QA Agent System — Full Workflow Guide

## Quick Start

```bash
# 1. Make sure Neo4j Desktop is running (start it from the app)
# 2. Start everything — services, ingest, and executor
./start.sh

# 3. Watch the executor live
tail -f logs/simulation_result.txt

# 4. When done
./start.sh --stop
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        YOUR MACHINE                                  │
│                                                                      │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐   │
│  │  RAG API     │    │  Agent Gateway   │    │  Executor Runner │   │
│  │ :9010        │◄───│  :9100           │◄───│ executor_runner  │   │
│  │ (Neo4j +     │    │  (Planner Logic) │    │ .py              │   │
│  │  Embeddings) │    └────────┬─────────┘    └────────┬─────────┘   │
│  └──────┬───────┘             │                       │             │
│         │                     │ LLM calls             │ ADB         │
│         ▼                     ▼                       ▼             │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐   │
│  │  Neo4j       │    │  OpenRouter API  │    │  Android Device  │   │
│  │  (Graph DB)  │    │  (Cloud LLM)     │    │  (Real Phone/    │   │
│  └──────────────┘    └──────────────────┘    │   Emulator)      │   │
│                                              └──────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

There are **3 services** and **1 executor**:

| Component | Port | What it does |
|---|---|---|
| **RAG API** | 9010 | Stores and retrieves SRS/Figma knowledge from Neo4j. Runs local embeddings (fastembed). |
| **Agent Gateway** | 9100 | Orchestrates the planner. Calls the LLM, manages the retrieval loop, generates test cases. |
| **Neo4j** | 7687 | Graph database holding requirements, UI screens, and test history. |
| **Executor** | (no port) | Reads test cases from the Gateway and runs them on a real Android device via Droidrun + ADB. |

---

## The Full Loop — Step by Step

### Phase 1: Startup & Ingestion (`./start.sh`)

When you run `./start.sh`, it does four things in order:

**1. Start RAG API** — boots a FastAPI server (`local_rag_api.py`) that connects to Neo4j.

**2. Start Agent Gateway** — boots a second FastAPI server (`local_agent_gateway.py`) with the planner logic.

**3. Ingest Knowledge** (`ingest_all.py`) — this is the critical data loading step:

```
ingest_all.py
    │
    ├── [1] Reset project  →  DELETE all old tests/SRS/Figma nodes from Neo4j
    │
    ├── [2] Ingest SRS     →  POST /srs/ingest (via Gateway)
    │       │
    │       ├── Load SRS1.txt from ./data/inputs/
    │       ├── Split into chunks (~700 chars each)
    │       ├── Embed all chunks with fastembed (local, no rate limits)
    │       ├── Extract 100 requirements (FR/NFR) via regex
    │       └── Write to Neo4j: SRS nodes, Chunk nodes, Requirement nodes
    │
    ├── [3] Ingest Figma   →  POST /figma/ingest (via Gateway)
    │       │
    │       ├── Load GENERATED_JSON.json from ./data/inputs/
    │       ├── Parse 7 screens, 107 UI elements
    │       ├── Derive screen purposes from name slugs
    │       └── Write to Neo4j: FigmaScreen nodes, UIElement nodes
    │
    └── [4] Stats check    →  Confirm everything is in the graph
```

**4. Start Executor** — launches `executor_runner.py` as a background process.

---

### Phase 2: The Planner Loop (Agent Gateway)

Every time the executor asks "what should I test next?", the Gateway runs this multi-stage pipeline:

```
POST /next (Gateway)
    │
    ├── Stage 1: Global Context
    │       Read brief context from Neo4j:
    │       - SRS summary (what the app does)
    │       - Figma screen index (what screens exist)
    │       - Recent tests + verdicts (what's been done)
    │       - Coverage map (which areas are under-tested)
    │
    ├── Stage 2: Iterative Retrieval Loop (up to 3 rounds)
    │       For each round:
    │       ├── Ask LLM: "given what you know, what do you need to retrieve?"
    │       │     → LLM returns: {action: "retrieve", queries: [...], screens: [...]}
    │       ├── Execute retrieval against Neo4j:
    │       │     - Semantic (vector) search over SRS chunks
    │       │     - Keyword hybrid search
    │       │     - Figma UI element lookup by screen name
    │       │     - Figma navigation transitions
    │       └── If LLM says "produce_testcase" → exit loop early
    │
    ├── Stage 3: Test Case Generation
    │       Build a rich prompt with:
    │       - Retrieved SRS context (~8000 chars)
    │       - Figma UI context (interactive elements per screen)
    │       - Coverage directive ("avoid these areas, focus on these")
    │       - List of already-executed tests (to avoid duplicates)
    │       → LLM generates a JSON test case
    │
    ├── Stage 4: Duplicate Check
    │       Compare new test title against all existing tests (similarity threshold 60%)
    │       If too similar → retry with alternate Figma screens
    │
    └── Stage 5: Auto-log
            Write new test case to Neo4j immediately with verdict="pass" (pending)
            → Coverage map updates automatically for next iteration
```

The test case JSON looks like this:

```json
{
  "title": "Add contact with duplicate phone number",
  "area": "contacts_creation",
  "objective": "Verify duplicate phone number detection",
  "steps": [
    "Open the Contacts app",
    "Tap the '+' button to add a new contact",
    "Enter a name that already exists",
    "Enter a phone number already in the address book",
    "Tap Save"
  ],
  "expected_result": "App warns user about duplicate and offers merge/cancel",
  "requirement_ids": ["FR-12", "FR-15"]
}
```

---

### Phase 3: The Executor Loop (`executor_runner.py`)

The executor is a continuous loop that runs on your machine while an Android device is connected via ADB:

```
executor_runner.py (infinite loop)
    │
    ├── 1. GET /next from Gateway
    │       → Receives a test case JSON (see above)
    │
    ├── 2. Translate to Droidrun goal
    │       Combine: title + steps + expected_result
    │       → "Navigate to contacts app. Tap '+'. Enter name 'John'. ..."
    │
    ├── 3. Execute on Android device (Droidrun)
    │       Droidrun uses its own LLM (Gemini 2.5 Pro) to:
    │       ├── Observe screen state via ADB screencap
    │       ├── Decide next action (tap, type, scroll, swipe)
    │       ├── Execute action via ADB
    │       └── Repeat until goal achieved or timeout (120s)
    │
    ├── 4. Interpret result
    │       Droidrun returns: {success: true/false, reason: "..."}
    │       → Map to verdict: "pass" or "failed"
    │
    ├── 5. POST /verdict+next to Gateway
    │       ├── Log verdict to Neo4j (updates test history)
    │       └── Request next test case in one call
    │
    └── 6. Sleep 5s → repeat from step 1
```

---

## Monitoring

### Watch the executor live
```bash
tail -f logs/simulation_result.txt
```

### Check graph stats
```bash
curl http://127.0.0.1:9010/graph/stats?project=contacts-app | python3 -m json.tool
```

### Interactive QA shell (manual test requests)
```bash
venv/bin/python test_loop_client.py
```

### View all logs
```bash
tail -f logs/rag_api.log      # RAG API / Neo4j operations
tail -f logs/gateway.log      # Gateway / planner operations
tail -f logs/simulation_result.txt  # Executor / Droidrun output
```

---

## Data Flow Diagram

```
SRS1.txt ──────────► Gateway ──► RAG API ──► Neo4j
GENERATED_JSON.json ──┘   (ingest)           │
                                             │
                    ┌────────────────────────┘
                    │  (retrieve context)
                    ▼
             Agent Gateway
             (planner logic)
                    │
                    │  (prompt)
                    ▼
             OpenRouter LLM ──► test case JSON
                    │
                    │  (HTTP)
                    ▼
           executor_runner.py
                    │
                    │  (ADB)
                    ▼
           Android Device ──► pass/fail verdict
                    │
                    │  (HTTP)
                    ▼
             Agent Gateway ──► Neo4j (test log)
                    │
                    └──► next test case (loop continues)
```

---

## Key Files

| File | Role |
|---|---|
| `start.sh` | Master startup script |
| `local_rag_api.py` | RAG API server (Neo4j CRUD + embeddings) |
| `local_agent_gateway.py` | Gateway HTTP routes |
| `planner/pipeline.py` | Core planner orchestration logic |
| `planner/model_client.py` | LLM backend (OpenRouter/Gemini/ngrok) |
| `planner/rag_client.py` | HTTP client for RAG API |
| `planner/prompts.py` | Prompt builders |
| `embeddings.py` | Embedding backend (fastembed/gemini/sentence-transformers) |
| `executor_runner.py` | Droidrun-based Android test executor |
| `ingest_all.py` | One-shot data ingestion script |
| `data/inputs/SRS1.txt` | Software Requirements Specification |
| `data/inputs/GENERATED_JSON.json` | Figma UI screen/element data |
| `.env` | All configuration (keys, URLs, project name) |

---

## Configuration (`.env` key settings)

```env
PROJECT=contacts-app          # Name of the app under test
APP_NAME=Samsung Contacts     # Display name used in prompts

MODEL_BACKEND=openrouter      # LLM for planning: openrouter | gemini | ngrok
EMBEDDING_BACKEND=fastembed   # Embeddings: fastembed (local) | gemini | auto

OPENROUTER_API_KEY=sk-or-... # Your OpenRouter API key
OPENROUTER_MODEL=minimax/minimax-m3  # Model to use for planning

GEMINI_API_KEY=AIza...        # Used by the Executor (Droidrun) for device control
EXECUTOR_LLM_PROVIDER=GoogleGenAI
EXECUTOR_LLM_MODEL=gemini-2.5-pro

TARGET_APP_PACKAGE=com.android.contacts  # Android package to test
EXECUTOR_TIMEOUT=120          # Max seconds per test case
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| RAG API won't start | Wrong Neo4j password | Update `NEO4J_PASSWORD` in `.env` |
| `embedding failed: 429` | Gemini free tier rate limit | Set `EMBEDDING_BACKEND=fastembed` in `.env` |
| Figma ingest timeout | LLM screen classification too slow | Already fixed — now fails fast and falls back |
| Executor not running | No ADB device found | Connect Android phone, enable USB debugging, run `adb devices` |
| `No module named 'fastembed'` | Wrong Python venv | Run `venv/bin/python3 -m pip install fastembed` |
