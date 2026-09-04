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

> **New here? Start with [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)** — a tested,
> end-to-end setup walkthrough (including the on-device portal setup that the sections below
> predate) plus a troubleshooting table. The rest of this README is reference material.

---

## Current status — ShobarKhamar campaign (29 Aug 2026)

The agent is running against **ShobarKhamar** (`com.tirzokpvt.shobarkhamar`, Flutter,
build 2.1.3), a livestock marketplace, on an Android 14 emulator. Latest 10-round
campaign, signed in as a farmer/seller:

| Metric | Value | Note |
|---|---|---|
| Pass | 3 / 10 | app behaved as expected |
| Candidate defect | 1 | **unvalidated** — needs human confirmation |
| Agent failure | 5 | our agent could not complete |
| Environment | 1 | step budget |
| **Autonomy** | **50%** | the honest weak point |
| **Requirement coverage** | **14 / 106** | was 5/106, and 2/33 on the previous app |
| App model | 25 states | package + activity populated on all |
| Cost | ~1¢ per test, ~$0.02 per SRS ingestion | audited against the billing API |

Artefacts per run: `logs/batch_<project>_<ts>.csv` (route, steps, verdict,
attribution, cited requirement ids) and `logs/review_sheet.csv` (blank rubric
columns for manual scoring).

### What the campaign found in the app

Unvalidated — these are candidate defects pending manual review:

- **No validation feedback on an empty mandatory field.** Saving Farm Info with an
  empty Farm Name produced no message naming the missing field (`FR-FARM-04`).
- **No success confirmation.** A successful farm update showed no feedback (`FR-FARM-06`).
- **No empty state on list screens.** "My Products" renders nothing when empty
  rather than saying so, which is indistinguishable from a failed load (`FR-NAV-07`).
- **Device registration race.** On a cold start the app POSTs
  `/api/device/device_installations` with an empty `fcm_token`, receives
  `400 fcm_token is required`, and shows *"Device registration failed. Check your
  connection"* — with the network working. It succeeds ~40s later once the token
  arrives. The error message misattributes a client sequencing bug to the network.
- **Language selection does not apply until a full restart.**

The build points at `https://test.sobarkhamar.com` (staging), so writes do not
reach production users.

---

## Fixes made this cycle

Every item below was a real defect found by running the system end to end, not by
reading code. Grouped by what they broke.

### Silent failures (the dominant class)

| Defect | Effect |
|---|---|
| `_state_signature` returned `""` on failure | empty strings compare equal → `is_livelocked()` returned **True for every test in a suite**, autonomy 0%, nothing logged. Now returns a unique sentinel + CRITICAL degradation |
| `_record_observations` `return []` on import error | disabled the **entire Live App Model** silently. Now CRITICAL |
| navtree retrieval `except: pass` | learned routes silently unused (REQ-302 stopped paying off) |
| `DERIVED_FROM` `except: pass` | requirements orphaned from source text |
| **Degradations were per-process** | executor and API are separate processes, so every executor-side degradation was invisible to the dashboard and batch report, which printed *"0 fallbacks, trustworthy"* regardless. Now a shared JSONL sink |
| `_degrade_once` recorded only the first occurrence | told us *that* something degraded, never *how often*. Now counted, events sampled at 1/2/5/10/25/… |

### Device channel

| Defect | Effect |
|---|---|
| **On-device portal was never installed** | `portal_mode="auto"` silently fell back to ADB for every run ever done. `adb input text` drops characters and its clear-before-type cannot verify itself, so a field holding `Trust Dairy Farm 1` became `Trust Dairy  FarTest FTest Farm 1rm 1` while the tool logged success |
| No portal assertion | `driver.portal_available` returned `False` for months and nothing read it. Now asserted at connect, CRITICAL if absent, with automatic re-setup |
| `get_ui_tree()` shape mismatch | mobilerun's own normalizer expects a list + `package`; the driver returns a root dict + `packageName`. Result: **0 nodes, package None** on every observation. Adapter took the same screen from 0 → 24 usable nodes |
| `uiautomator dump` not retried | fails while the UI animates; one attempt dropped us to the sparse payload. Now 3 attempts |
| `pm clear` deleted its own test fixtures | seeded gallery images vanished at each clean slate, so every image-dependent test failed for want of a file. Now re-seeded as part of reset |

### App-agnosticism (the system is meant to run on *any* Android app)

| Defect | Effect |
|---|---|
| Contacts vocabulary in `prompts.py` rule 5b | *"create a contact named X"*, *"no existing contacts"*, *"contacts exist on SIM"* shipped in **every generation prompt for every app** |
| `_UNACHIEVABLE_PRECONDITION` hardcoded Contacts phrases | now `settings.UNACHIEVABLE_PRECONDITIONS`, generic default |
| App-naming defaults (`TARGET_APP_PACKAGE`, `SRS_PATH`, `FIGMA_PATH`, `PROJECT`) | an unconfigured run silently targeted the wrong app instead of failing. All now empty |
| `FIGMA_PATH=` could not be cleared | `_str()` falls back to the default on an empty value, so the Contacts design file kept loading |
| `ENABLED_SOURCES` not enforced | a **disabled** Figma source still fed the generation prompt — the Contacts design file leaked into a livestock project and the first generated test was about a Contacts List |
| `TARGET_APP_ONLY` was dead config | defined, never read. The agent left the app, opened the Play Store, and **uninstalled the app under test** |

### State identity and attribution

| Defect | Effect |
|---|---|
| Value-bearing `content_description` treated as identity | a dropdown's placeholder becoming its value made every wizard step a new screen — one Add Cattle form produced **9 states**. Skeleton matching (`resource_id + class`) fixes it: **19 correct merges, 0 false** on measured data |
| Containment had no size guard (audit A-5) | a 1-control blank screen is "contained" in every screen. `STATE_CONTAINMENT_MAX_RATIO` now caps it |
| Screen labels prefixed with the activity | single-activity apps (all Flutter) got `Main · ` on everything; these labels are handed to the device agent as screen names, so it hunted for captions no app displays |
| Attribution taxonomy duplicated | the reporting copy omitted `NAVIGATION_LIVELOCK`, so **autonomy read 100% when it was 67%**. Now one definition in `settings.py`, asserted by object identity in tests |
| Livelock measured variety, not periodicity | `A→B→C` repeated escapes a distinct-count check. Now detects periods 2–6 |
| Every stall blamed on our agent | a frozen screen under repeated **taps** is the app refusing; under repeated **swipes** it is us failing to scroll elsewhere. `classify_stuck()` now needs positive action evidence before claiming `APP_UNRESPONSIVE` |
| **`delete_appmodel: True` on every campaign** | we deleted the learned app map at the start of every run, so round 1 always flew blind and invented screen names. This also made REQ-302/303 ("every cycle makes the agent smarter") untestable. Split into `CLEAN_SLATE` (results) and `CLEAN_SLATE_APPMODEL` (knowledge, default **False**) |

### New capabilities

- `APP_LOGIN_*` — credentials for apps gated behind a login the agent cannot perform
  (OTP, NID upload, admin approval). The **secret goes only to the executor**; the
  planner receives the role name, so a credential never enters the 50k-token
  planning prompt.
- `APP_ACCOUNT_STATE` — what the account already has, so the planner stops writing
  "register a new farm" tests for an account that already has one.
- `OUT_OF_SCOPE` — requirements the agent must not attempt, **filtered out of the
  citable-requirements list** so coverage pressure cannot override the constraint.
- `TARGET_APP_ACTIVITY` / `TARGET_APP_LABELS` — open by package, not by display
  name (ShobarKhamar's launcher label is Bengali: সবার খামার).
- `DEVICE_FIXTURE_DIR` — test media re-seeded after every device reset.

---

## Known issues

Ordered by how much they limit what we can claim.

1. **Autonomy is 50%.** Half of runs are lost to our agent, mostly on multi-step
   forms. The agent does not consider that a blocking required field may be
   *above* it or in an *earlier wizard step* — it scrolled down 11 times at the
   bottom of a form looking for fields that were at the top.
2. **`uiautomator` and the portal fight.** `FATAL EXCEPTION: UiAutomationService`
   appears in logcat — two accessibility clients registering at once. Not the app
   crashing; our own tooling. Likely cause of `STEP_LIMIT_EXCEEDED` runs.
3. **No ground truth.** Precision and recall are uncomputable. Only coverage and
   autonomy are defensible today. Needs a seeded-defect build.
4. **State-identity thresholds are tuned on one app's 18 states.** Evidence, not
   proof. No labelled answer for how many distinct screens the app actually has.
5. **`135 of 169` navtree nodes marked "avoid"** — that records thrashing, not
   learning. Should fall as the map stabilises.
6. **Test IDs restart at TC-001 every campaign.** Fine while results are wiped;
   breaks the moment we compare campaigns.
7. **The graph stores only title + verdict for a test** — `steps`, `screen` and
   `expected_result` are dispatched but never persisted, so the review sheet
   cannot show what a test actually did.
8. **Requirement citation is inconsistent** — some tests cite ids in the structured
   field, some only in prose.
9. **Defect intelligence (ETA-REQ-301) has never run on real data.** Built, dormant.
10. **Cross-application transfer (ETA-REQ-304) unvalidated** — every campaign so far
    has run against one app.

---

## Tests

```bash
./venv/bin/python tests/run_all.py     # every module, non-zero exit on failure
```

| Module | Checks | Guards |
|---|---|---|
| `test_app_state.py` | 11 | structural signature, scroll/theme invariance, thin-tree fallback |
| `test_state_identity.py` | 9 | skeleton matching, size-ratio guard, over-merge protection |
| `test_livelock.py` | 21 | unusable-signal sentinel, cyclic periods 2–6, stuck attribution |
| `test_observation.py` | 11 | driver-tree adapter, package/activity recovery, fallback path |
| `test_config_guards.py` | 27 | no app-naming defaults, single taxonomy, cross-process degradations |

Every check corresponds to a defect that reached a real run. `test_config_guards`
greps `settings.py` itself for app-naming defaults and asserts taxonomy **object
identity**, so both classes of regression fail the suite rather than silently
returning.

---

## Remaining work

### A. Finish the ShobarKhamar campaign

- [ ] **Full 25-round campaign** once autonomy is above ~75%. At ~1¢/test and
      ~4 min/test this is ~$0.25 and ~100 minutes.
- [ ] **Per-role campaigns.** Five accounts exist (seller / buyer / feed /
      medicine / machinery). Only farmer-seller has been exercised. Switch
      `APP_LOGIN_ROLE` + `APP_LOGIN_IDENTIFIER` together and update
      `APP_ACCOUNT_STATE` to match.
- [ ] **Role-boundary tests** (`FR-SELL-12/13`) — a seller must see only its own
      listings and must not reach another role's screens. Needs two accounts.
- [ ] **Manual review** of `logs/review_sheet.csv` — confirm which candidate
      defects are real. Nothing downstream is trustworthy until this happens once.

### B. Agent capability

- [ ] Teach the agent that a blocking field may be **above** or in an **earlier
      wizard step** (cause of most current agent failures).
- [ ] Resolve the uiautomator/portal accessibility conflict.
- [ ] Persist `steps` / `screen` / `expected_result` to the graph.
- [ ] Run-scoped test IDs.
- [ ] Make requirement citation mandatory in the output contract when the
      requirements block is non-empty.

### C. Measurement

- [ ] **Seeded-defect build** — 8–12 known defects with a ground-truth list. Three
      arms (full agent / memory disabled / random baseline) gives precision,
      recall, F1 and an ablation for the "gets smarter" claim. Under $1 of model
      spend.
- [ ] **Hand-label one app's screens** to price the identity thresholds instead of
      guessing them.
- [ ] Ingest a real defect export to activate ETA-REQ-301.
- [ ] Second application to validate cross-app transfer (ETA-REQ-304).

### D. Reporting

- [ ] Correct `docs/SAMSUNG_PROGRESS_REPORT.pdf`: the *"zero silent fallbacks"*
      claim read a counter that could not see the executor process, and
      *"no source is a hard dependency"* was marked met on a subset test that never
      checked whether a **disabled** source stops contributing.


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
| **UI knowledge** | requires a Figma export | **Live App Model** — the agent builds its own UI state graph from the running device (works with zero Figma/SRS) |
| **Learning** | static; only verdicts stored | execution→learning loop: every run feeds observed states + transitions back into the graph |

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
  │ executors      │  clients/executor_runner.py (Android, Droidrun)
  │                │  web_player/runner.py      (Website, Playwright)
  │                │  clients/simulator_runner.py (demo, no device)
  └────────────────┘
```

Two **players**, one brain. The planner, the knowledge graph and the gateway are
platform-agnostic; only the driving of the target differs. Web runs are tagged
`platform="web"` (a WP6 dimension the graph already understands), so Android and
website results share one project without polluting each other's retrieval.

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
  sources/                Pluggable knowledge sources (srs, figma_ui, figma_flow, live_ui)

ingestion/                Format-agnostic ingestion pipeline
  document_loader.py      any document → text
  extractor.py            text → Requirement/Entity/ValidationRule (LLM + rule fallback)
  ui_normalizer.py        Figma export → canonical UI IR (no hardcoded purposes)
  app_state.py            Live App Model: UI-state signature + dedup (structural + visual fallback)

dashboard-react/          React (Vite) live monitoring UI — built + served at /dashboard
dashboard/                Dependency-free fallback dashboard (single HTML file)

clients/                  Execution scripts
  executor_runner.py      Real Android device executor (via Droidrun)
  simulator_runner.py     Simulated loop (no device) for demos/testing
  test_loop_client.py     Minimal interactive loop client

targets/                  Target profiles — WHAT is under test (see targets/README.md)
  run.py                  One entry point:  py -m targets.run <profile>
  schema.py               Profile shape + validation (UI-ready: returns errors)
  loader.py  env.py       Load/save profiles · profile -> settings.py variables
  profiles/*.json         One file per app or site (wikipedia, contacts-app, ...)

web_player/               Website executor (Playwright) — see web_player/README.md
  runner.py               CLI batch loop:  py -m web_player.runner
  agent.py                observe -> decide -> act loop
  snapshot.py             page -> compact, ref-addressable observation
  actions.py              action vocabulary + guardrail enforcement
  oracles.py              console / page-error / HTTP signals (free defect oracles)
  failures.py             web failure taxonomy + self-healing strategies
  browser.py  goal.py  gateway.py  llm.py

scripts/
  ingest_all.py           One-shot ingest helper (reset → SRS → Figma → stats)

start.sh                  Bring up the whole stack (Neo4j + emulator + services)
stop.sh                   Tear it all down (services + emulator + Neo4j)
requirements.txt          Local Python dependencies
docs/                     Architecture diagrams and documentation
data/inputs/              Sample SRS + Figma export
```

---

## Prerequisites

- **Python 3.13 specifically** (Python 3.14 will fail to install downstream dependencies like `arize-phoenix`)
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

`./start.sh` brings up the **whole stack idempotently** — local Neo4j, the Android
emulator (boots it + enables the mobilerun accessibility service), the RAG API, and
the Gateway — **without touching your data**. Opt in with flags:

- `./start.sh --ingest` — also reset + ingest SRS/Figma (**destructive**: wipes tests + app model).
- `./start.sh --with-executor` — also start the executor test loop.
- `./start.sh --build` — (re)build the React dashboard first.
- `./start.sh --no-neo4j` / `--no-emulator` — skip those (e.g. managed elsewhere / physical device).

`./stop.sh` tears it all down (services + emulator + Neo4j); `./stop.sh --services-only`,
`--keep-emulator`, and `--keep-neo4j` scope the shutdown. Machine-specific paths
(Neo4j DBMS dir, emulator AVD) auto-detect but can be overridden in `.env` — see `.env.example`.

---

### 6) Ingest knowledge

*(If you have already ingested the knowledge graph for your project, you can skip this step).*

Pick any `project` name — it scopes everything in the graph.

```bash
# SRS — any format (.txt/.md/.pdf/.docx/.html). Builds chunks+embeddings AND the requirement graph.
curl -X POST http://127.0.0.1:9100/srs/ingest \
  -H 'Content-Type: application/json' \
  -d '{"project":"my-app","source_path":"./data/inputs/Sample-Contacts-App-SRS.txt"}'

# Figma — canonical UI IR, dynamic feature-area classification
curl -X POST http://127.0.0.1:9100/figma/ingest \
  -H 'Content-Type: application/json' \
  -d '{"project":"my-app","source_path":"./data/inputs/GENERATED_JSON.json"}'
```

For windows powershell:

```bash
# SRS — any format (.txt/.md/.pdf/.docx/.html). Builds chunks+embeddings AND the requirement graph.
curl.exe -X POST "http://127.0.0.1:9100/srs/ingest" `
    -H "Content-Type: application/json" `
    -d '{"project":"my-app","source_path":"./data/inputs/Sample-Contacts-App-SRS.txt"}'

# Figma — canonical UI IR, dynamic feature-area classification
curl.exe -X POST "http://127.0.0.1:9100/figma/ingest" `
  -H "Content-Type: application/json" `
  --data-raw '{\"project\":\"my-app\",\"source_path\":\"./data/inputs/GENERATED_JSON.json\"}'
```

The defaults use the LLM for the SRS summary, entity extraction, and screen
classification. To ingest **without a model** (deterministic rule-based fallback):

```bash
curl -X POST http://127.0.0.1:9100/srs/ingest -H 'Content-Type: application/json' -d '{
  "project":"my-app","source_path":"./data/inputs/Sample-Contacts-App-SRS.txt",
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

## Live App Model (self-built UI state graph)

When no Figma guide (or SRS) exists — the common case for a generic Android app —
the agent **builds its own map of the app** by observing the running device. Every
executor step's accessibility tree becomes a `UIState` node and each navigation a
`TRANSITIONS_TO` edge, so the graph grows as the agent explores and is reused to
plan future tests.

The hard part is **state identity** — deciding whether a screen is new or one
already seen. This is done structurally, not visually:

- Each state's **signature** is a hash of its *structural skeleton* — the set of
  controls by `(resource_id, class, content_description, clickable)` plus
  `package`/`activity` and whether a dialog is open.
- The volatile free **text is dropped**, so scrolling a list to different data is
  the **same** state, and a **light→dark theme switch is the same state** (the
  view hierarchy is identical; only pixels change).
- Genuinely different screens (a new activity, an open dialog, an empty vs.
  populated list) get **new** signatures.
- Exact-signature is the fast path; a structural **Jaccard** near-match tolerates
  minor chrome; and for apps with thin accessibility trees (Compose/games/WebView)
  a **perceptual screenshot hash** is the visual fallback.

This mirrors mobilerun's own guarded-macro state matching, and is unit-tested on
the scroll / theme / navigate cases in `tests/test_app_state.py`
(`./venv/bin/python tests/test_app_state.py`).

**How it's built:** `clients/executor_runner.py` reads mobilerun's per-step
trajectory (accessibility trees + screenshots) after each run and POSTs each
observation to `POST /liveui/observe`, which dedupes it into the graph. Screenshots
are stored under `data/appmodel/<project>/` and served via `/liveui/screenshot`.
The map is exposed at `GET /appmodel/graph` and surfaced to the planner via the
`live_ui` knowledge source. Abstraction logic lives in `ingestion/app_state.py`.

---

## Dashboard (live monitoring)

An operator dashboard is served by the gateway at
**<http://127.0.0.1:9100/dashboard?project=my-app>**. It polls every few seconds
and shows: test cases and verdicts, bugs found, coverage by area, the business
policies (validation rules) the LLM extracted from the SRS, the exploration
directive, an **interactive Live App Model graph** (draggable state nodes; click a
state to see its screenshot), and a **live mobilerun log stream** so you can watch
what the agent is thinking/doing on the device in real time.

The dashboard is a **React (Vite) app** in `dashboard-react/`, built to a single
self-contained file (`npm --prefix dashboard-react install && npm --prefix
dashboard-react run build`) that the gateway serves at `/dashboard` (falling back
to the dependency-free `dashboard/index.html` if the build is absent). Data comes
from `GET /dashboard/data`, logs from `GET /dashboard/logs`, and state screenshots
from `GET /dashboard/screenshot`.

**Where are the live agent logs?** mobilerun logs to the `mobilerun` logger; the
executor tees it (plus its own logs) to `logs/mobilerun.log`, which the dashboard
streams. Tail it directly with `tail -f logs/mobilerun.log`.

---

## Knowledge graph model

**Nodes:** `Project`, `SRS`, `Chunk` (+`embedding`), `Requirement` (+`embedding`),
`Entity`, `ValidationRule`, `FigmaScreen`, `UIElement`, `FeatureArea`, `TestCase`,
`TestRun`, `Summary`, `UIState` (Live App Model).

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
(Project)-[:HAS_STATE]->(UIState)-[:TRANSITIONS_TO {action}]->(UIState)   # Live App Model (self-built)
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

uvicorn rag_api.main:app --port 9010 --reload           # terminal 1
uvicorn gateway.main:app --port 9100 --reload           # terminal 2

PROJECT=my-app python scripts/ingest_all.py             # terminal 3
curl -X POST http://127.0.0.1:9100/agent/next-testcase \
  -H 'Content-Type: application/json' -d '{"project":"my-app"}'
```

Powershell runnable:

```bash
pip install -r requirements.txt
uvicorn rag_api.main:app --port 9010 --reload           # terminal 1
uvicorn gateway.main:app --port 9100 --reload           # terminal 2
$env:PROJECT="my-app"; python .\scripts\ingest_all.py   # terminal 3
Invoke-RestMethod -Uri "http://127.0.0.1:9100/agent/next-testcase" -Method POST -ContentType "application/json" -Body '{"project":"my-app"}'
```