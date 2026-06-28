# Exploratory QA Test-Case Planner (GraphRAG + Vector Retrieval)

An adaptive, **app-agnostic** exploratory-testing agent. It ingests any app's
requirements (SRS) and UI design (Figma) into a Neo4j **knowledge graph with
vector embeddings**, then generates one high-value exploratory test case at a
time — learning from each verdict to steer toward undiscovered defects.

The loop:

1. Generate the next exploratory test case (hybrid semantic + keyword retrieval).
2. Run it (real device via Droidrun, or the simulator).
3. Log the verdict (`pass` / `failed`).
4. Generate the next, informed by SRS rules, UI elements, coverage gaps, and failure history.

---

## What's new (v2)

This is a major upgrade from the original keyword-RAG prototype:

| Area | Before | Now |
|---|---|---|
| **SRS ingestion** | `.txt` only, regex `FR-\d+` parsing | **Any format** (PDF/DOCX/HTML/MD/txt…) via `ingestion/document_loader.py` |
| **Knowledge storage** | flat `SRS → Chunk` keyword bag | **GraphRAG entity graph**: `Requirement`/`Entity`/`ValidationRule` + cross-linked chunks |
| **Retrieval** | `CONTAINS` keyword only | **Hybrid**: native Neo4j vector search + keyword, fused (RRF) + graph hop |
| **UI ingestion** | hardcoded contacts-app screen map | **Generic** canonical UI IR; purposes derived dynamically (optionally LLM-classified) |
| **Coverage** | free-text `area` strings | **Graph-native** `TestCase -[:COVERS]-> Requirement` edges |
| **Gateway code** | one ~1800-line file | modular **`planner/`** package; gateway is a thin router |
| **App specificity** | contacts-app baked in | **fully app-agnostic** — all domain knowledge comes from the ingested graph at runtime |

See `System_Architecture.md` for the multi-stage retrieval design.

---

## Architecture

```
                 ┌─────────────────────────────┐
   any SRS  ───▶ │  ingestion/ (adapters)      │
   Figma    ───▶ │  document_loader · extractor│
                 │  · ui_normalizer            │
                 └──────────────┬──────────────┘
                                │ canonical IR + extraction
                                ▼
  ┌────────────────┐    RAG   ┌──────────────────────────┐
  │ planner/       │◀────────▶│ rag_api/main.py           │
  │ (LangGraph)    │          │  Neo4j graph + vectors    │
  └───────┬────────┘          │  (embeddings.py)          │
          │ LLM                └──────────────────────────┘
          ▼
  ┌────────────────┐
  │ model backend  │  OpenRouter · Gemini · ngrok/Qwen · local litellm
  └────────────────┘
          ▲
          │ test cases / verdicts
  ┌───────┴────────┐
  │ executor       │  clients/executor_runner.py (Droidrun) · clients/simulator_runner.py (demo)
  └────────────────┘
```

- **`rag_api/main.py`** (port 9010) — Neo4j-backed knowledge graph: ingest, hybrid retrieval, coverage, graph endpoints. Auto-creates vector indexes on startup.
- **`gateway/main.py`** (port 9100) — thin FastAPI router over the **`planner/`** package. Uses **LangGraph** to orchestrate the iterative retrieval + test-generation loop.
- **Model backend** — pluggable (see [Model backend setup](#model-backend-setup)).

---

## Repository layout

```
rag_api/                  Neo4j knowledge-graph API (ingest, retrieve, coverage, graph)
gateway/                  Thin FastAPI router → planner.langgraph_agent
observability/            Logging, metrics, and tracing middleware (outputs to logs/)

planner/                  Core AI Logic
  langgraph_agent.py      LangGraph state machine for exploration & planning
  config.py               Configuration constants
  model_client.py         LLM integration (OpenRouter/Gemini/etc)
  rag_client.py           RAG API client
  textutil.py             JSON parsing utilities
  coverage.py             Coverage tracking logic
  prompts.py              Agent prompts
  schemas.py              Data models

ingestion/                Format-agnostic ingestion pipeline
  document_loader.py      any document → text
  extractor.py            text → Requirement/Entity/ValidationRule (LLM + rule fallback)
  ui_normalizer.py        Figma export → canonical UI IR (no hardcoded purposes)

clients/                  Execution scripts
  executor_runner.py      Real Android device executor (via Droidrun)
  simulator_runner.py     Simulated loop (no device) for demos/testing
  test_loop_client.py     Minimal interactive loop client

scripts/
  ingest_all.py           One-shot ingest helper (reset → SRS → Figma → stats)

start.sh                  One-command local startup / stop
requirements.txt          Local Python dependencies
docs/                     Architecture diagrams and documentation
data/inputs/              Sample SRS + Figma export
```

---

## Prerequisites

- Python 3.10+
- Neo4j 5.13+ (vector index support; 2025/2026 builds fine) — Desktop or Docker
- A model backend (OpenRouter key, Gemini key, or a running Qwen/local server)
- **Appium** installed and running (`appium` server)
- **Android Studio** with a running emulator (e.g., Pixel 9a) or a physical device connected via ADB

---

## Steps to Run

### 1) Install

```bash
pip install -r requirements.txt
```

This includes `markitdown`/`pypdf` (multi-format docs) and `fastembed` (ONNX embeddings — no torch). Optional upgrades (`docling`, `unstructured`, `sentence-transformers`) are auto-detected if installed.

---

### 2) Start Device Environment (Appium & Emulator)

Before running the agent, you must have your device environment ready:
1. Open **Android Studio** and launch your Android emulator (e.g., Pixel 9a).
2. Start the **Appium** server in a separate terminal:
   ```bash
   appium
   ```

---

### 3) Start Neo4j

See `neo4j_setup.md`. Quick Docker:

```bash
docker run --name neo4j-qa -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your-password -d neo4j:5
```

---

### 4) Configure `.env`

Create `.env` in the project root:

```ini
# ── Neo4j ──────────────────────────────────────────────
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password

# ── Model backend: "openrouter" | "gemini" | "ngrok" ───
MODEL_BACKEND=openrouter

# Local services
RAG_API_URL=http://127.0.0.1:9010

# Optional API keys to protect the services (leave blank to disable)
RAG_API_KEY=
GATEWAY_API_KEY=

# Embeddings (defaults are fine): auto | fastembed | gemini | sentence_transformers | none
EMBEDDING_BACKEND=auto
```

Then add the keys for your chosen backend (next section).

---

### Model backend setup

The gateway selects a backend via `MODEL_BACKEND`. All three speak the same
internal contract, so the rest of the pipeline is identical.

#### Option A — OpenRouter (recommended for hosted LLMs)

**Step 1 — Get an API key.** Sign in at <https://openrouter.ai>, open
<https://openrouter.ai/keys>, click **Create Key**, and copy it (it starts with `sk-or-...`).

**Step 2 — Put the key in `.env`.** Open the **`.env` file in the project root**
(the same file you created in step 3 above — create it if missing) and add these
lines. The API key goes **only** in `.env` — never in code or committed to git
(`.env` is git-ignored):

```ini
# ── .env (project root) ──
MODEL_BACKEND=openrouter
OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx   # ← paste your key here
OPENROUTER_MODEL=qwen/qwen-2.5-72b-instruct                 # ← the model the agent uses
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1            # default; rarely changed
```

**Step 3 — Choose the agent's model.** `OPENROUTER_MODEL` is the LLM that powers
the planner agent (retrieval planning, test-case generation, extraction). Pick
any model id from <https://openrouter.ai/models> and paste its **exact slug**
(the `vendor/model-name` shown on the model page) as `OPENROUTER_MODEL`. Examples:

| `OPENROUTER_MODEL` | Notes |
|---|---|
| `qwen/qwen-2.5-72b-instruct` | strong, balanced default |
| `deepseek/deepseek-chat` | strong reasoning, low cost |
| `google/gemini-2.0-flash-001` | fast, cheap |
| `anthropic/claude-3.5-sonnet` | high quality |
| `meta-llama/llama-3.1-70b-instruct:free` | free tier (`:free` suffix, rate-limited) |

**Step 4 — Apply it.** Save `.env` and **(re)start the gateway** so it picks up
the change:

```bash
uvicorn gateway.main:app --host 0.0.0.0 --port 9100 --reload
curl http://127.0.0.1:9100/health     # "model" should show backend=openrouter + your model
```

To switch models later, just edit `OPENROUTER_MODEL` in `.env` and restart the gateway.

#### Option B — Google Gemini

```ini
MODEL_BACKEND=gemini
GEMINI_API_KEY=...
PLANNER_GEMINI_MODEL=gemini-2.5-pro
```

#### Option C — Self-hosted / notebook (Qwen via `planner.ipynb`, or local `planner.py`)

The gateway's `ngrok` backend just POSTs to a `/generate` endpoint, so it works with **any** server implementing that contract:

```ini
MODEL_BACKEND=ngrok
MODEL_API_URL=https://xxxx.ngrok-free.app   # or http://127.0.0.1:8000
```

- **Qwen notebook:** run `planner.ipynb` cells (load model → start FastAPI → open ngrok on port 8000), then paste the public URL into `MODEL_API_URL`.
- **Local generic server (`planner.py`):** a small [litellm](https://docs.litellm.ai/)-based server that exposes the same `/generate` and routes to any provider. Useful to run OpenRouter/OpenAI/etc. behind the `ngrok` contract:

  ```ini
  # planner.py reads these (note: different vars from the gateway's native backend)
  LLM_PROVIDER=openrouter
  LLM_MODEL=openrouter/qwen/qwen-2.5-72b-instruct
  LLM_API_KEY=sk-or-...
  ```
  ```bash
  pip install litellm
  python planner.py                 # serves /generate on :8000
  # then set MODEL_BACKEND=ngrok, MODEL_API_URL=http://127.0.0.1:8000
  ```

> Tip: `MODEL_BACKEND=openrouter` (Option A) is the simplest path for OpenRouter — `planner.py` is only needed if you want a single uniform local model endpoint.

---

### 5) Start the services

Two terminals (or use `./start.sh`):

```bash
# Terminal 1 — knowledge graph
uvicorn rag_api.main:app --host 0.0.0.0 --port 9010 --reload
# Terminal 2
MODEL_API_URL=https://xxxx.ngrok-free.app \
uvicorn gateway.main:app --host 0.0.0.0 --port 9100 --reload
```

Health checks:

```bash
curl http://127.0.0.1:9010/health     # {"status":"ok", ...}
curl http://127.0.0.1:9100/health     # shows the active model backend
```

`./start.sh` starts both, ingests the sample data, and runs the executor.
`./start.sh --no-ingest` starts services only; `./start.sh --stop` kills them.

---

### 6) Ingest knowledge

*(If you have already ingested the knowledge graph for your project, you can skip this step).*

Pick any `project` name — it scopes everything in the graph.

```bash
# SRS — any format (.txt/.md/.pdf/.docx/.html). Builds chunks+embeddings AND the requirement graph.
curl -X POST http://127.0.0.1:9100/srs/ingest \
  -H 'Content-Type: application/json' \
  -d '{"project":"my-app","source_path":"./data/inputs/SRS1.txt"}'

# Figma — canonical UI IR, dynamic feature-area classification
curl -X POST http://127.0.0.1:9100/figma/ingest \
  -H 'Content-Type: application/json' \
  -d '{"project":"my-app","source_path":"./data/inputs/GENERATED_JSON.json"}'
```

The defaults use the LLM for the SRS summary, entity extraction, and screen
classification. To ingest **without a model** (deterministic rule-based fallback):

```bash
curl -X POST http://127.0.0.1:9100/srs/ingest -H 'Content-Type: application/json' -d '{
  "project":"my-app","source_path":"./data/inputs/SRS1.txt",
  "use_model_summary":false,"require_model_summary":false,"extract_entities":true
}'
curl -X POST http://127.0.0.1:9100/figma/ingest -H 'Content-Type: application/json' -d '{
  "project":"my-app","source_path":"./data/inputs/GENERATED_JSON.json",
  "use_model_classification":false
}'
```

One-shot helper (reset → SRS → Figma → stats):

```bash
PROJECT=my-app python scripts/ingest_all.py
```

Verify:

```bash
curl "http://127.0.0.1:9010/graph/stats?project=my-app"
# requirement_count, validation_rule_count, embedded_chunk_count, figma_screen_count, ...
```

---

### 7) Generate, execute, adapt

```bash
# Next exploratory test case
curl -X POST http://127.0.0.1:9100/agent/next-testcase \
  -H 'Content-Type: application/json' \
  -d '{"project":"my-app","objective":"generate the next high-value exploratory test case"}'

# Log a verdict and get the next test in one call
curl -X POST http://127.0.0.1:9100/agent/log-verdict-and-next \
  -H 'Content-Type: application/json' \
  -d '{
    "project":"my-app","test_case_id":"TC-001",
    "title":"Verify the entry form rejects malformed input",
    "verdict":"failed","notes":"accepted an invalid value",
    "area":"data_entry","requirement_ids":["FR-7"]
  }'
```

Run the loop automatically:

```bash
python clients/executor_runner.py     # real Android device via Droidrun
python clients/simulator_runner.py    # simulated verdicts (no device) — good for demos
```

---

## 7) Coverage & graph inspection

```bash
# Exploration dashboard (areas, hot spots, gaps, requirement coverage)
curl "http://127.0.0.1:9100/agent/coverage?project=my-app" | python3 -m json.tool

# Graph-native requirement coverage (COVERS edges)
curl "http://127.0.0.1:9010/coverage/requirements?project=my-app" | python3 -m json.tool

# Compact graph views
curl "http://127.0.0.1:9010/graph/summary?project=my-app&top=12" | python3 -m json.tool
curl "http://127.0.0.1:9010/graph/terminal?project=my-app&top=12"
curl "http://127.0.0.1:9010/graph/visualize?project=my-app" | python3 -m json.tool   # Cytoscape
curl "http://127.0.0.1:9010/graph/cypher?project=my-app"     # ready-to-run Neo4j queries
```

Free-form Q&A against the graph (RAG):

```bash
curl -X POST http://127.0.0.1:9100/chat -H 'Content-Type: application/json' \
  -d '{"project":"my-app","prompt":"What validation rules apply before saving?"}'
```

---

## Observability & Tracing

The system includes a dedicated `observability/` package that wraps critical components. 
All requests to the RAG API, interactions with LLMs, and generation latencies are automatically tracked.

Logs are aggregated in JSON Lines format in the `logs/` directory for easy parsing:
- `logs/app.jsonl`: Contains `node_enter`, `node_exit`, `llm_call`, and `rag_call` events, complete with latency timings and token estimations.
- `logs/gateway.log` & `logs/rag_api.log`: Standard application logs.

---

## Knowledge graph model

**Nodes:** `Project`, `SRS`, `Chunk` (+`embedding`), `Requirement` (+`embedding`),
`Entity`, `ValidationRule`, `FigmaScreen`, `UIElement`, `FeatureArea`, `TestCase`,
`TestRun`, `Summary`.

**Key relationships:**

```
(Project)-[:HAS_SRS]->(SRS)-[:HAS_CHUNK]->(Chunk)
(Project)-[:HAS_REQUIREMENT]->(Requirement)-[:HAS_RULE]->(ValidationRule)
(Requirement)-[:INVOLVES]->(Entity)
(Requirement)-[:IN_FEATURE]->(FeatureArea)
(Project)-[:HAS_FIGMA]->(FigmaScreen)-[:HAS_ELEMENT]->(UIElement)
(UIElement)-[:NAVIGATES_TO]->(FigmaScreen)        # real prototype links when present
(Project)-[:HAS_TEST]->(TestCase)-[:HAS_RUN]->(TestRun)
(TestCase)-[:COVERS]->(Requirement)               # graph-native coverage
```

Vector indexes `chunk_embedding` and `requirement_embedding` are created
automatically at RAG-API startup (when embeddings are enabled).

---

## Embeddings

`embeddings.py` is pluggable via `EMBEDDING_BACKEND`:

- `auto` (default) — picks `fastembed` if installed, else `sentence_transformers`, else `gemini`, else `none`.
- `fastembed` — ONNX `BAAI/bge-small-en-v1.5` (384-d), no torch. Recommended on-device.
- `gemini` — Google `text-embedding-004` (needs `GEMINI_API_KEY`).
- `none` — disables vectors; retrieval gracefully falls back to keyword only.

Override the model with `EMBEDDING_MODEL`.

---

## Troubleshooting

- **Model backend unavailable (503):** check `MODEL_BACKEND` and its key/URL. For `ngrok`, the tunnel URL changes on restart — update `MODEL_API_URL`.
- **Neo4j auth error:** fix `NEO4J_PASSWORD` in `.env`. An empty `NEO4J_URI` falls back to the local default.
- **`retrieval_mode: fallback_ordered`:** no embeddings available or nothing ingested yet — install `fastembed` and re-ingest to enable semantic search.
- **Figma ingest "no screens":** ensure the export is valid Figma JSON (raw or fenced).
- **Vector index not created:** needs Neo4j 5.13+. Older servers still work (keyword-only retrieval).

---

## Security

- `.env`, `logs/`, `__pycache__/`, and `*.log` are git-ignored. Never commit API keys, ngrok tokens, or DB credentials.
- Set `RAG_API_KEY` / `GATEWAY_API_KEY` to require `Authorization: Bearer <key>` on the services.

---

## Quick start (minimal)

```bash
pip install -r requirements.txt
# .env: NEO4J_* + MODEL_BACKEND=openrouter + OPENROUTER_API_KEY + OPENROUTER_MODEL

uvicorn rag_api.main:app       --port 9010 --reload     # terminal 1
uvicorn gateway.main:app --port 9100 --reload     # terminal 2

PROJECT=my-app python scripts/ingest_all.py             # terminal 3
curl -X POST http://127.0.0.1:9100/agent/next-testcase \
  -H 'Content-Type: application/json' -d '{"project":"my-app"}'
```
