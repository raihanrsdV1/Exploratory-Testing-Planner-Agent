# Plan: Evolving the Exploratory Testing Agent into an Autonomous, Learning Test Engineer

> Implementation roadmap responding to Samsung's ETA-TR-2026-001 (`data/Extended_planning.txt`) and the
> cover-email correction. Written as a handoff spec for coding agents. Build order and work packages are in Part D.

## Context — why this work

Samsung's 29-June roadmap (`data/Extended_planning.txt`, ETA-TR-2026-001) asks us to evolve the agent from a **static-knowledge test generator** into an **autonomously learning, self-optimizing exploratory test engineer** ("every execution cycle must make the agent smarter than the last"). Their cover email adds one hard correction that is *not* just another feature: **no single knowledge source may be a hard dependency.** SRS, UI guide, and defect data must be independently usable, and the system must degrade gracefully to whatever is available.

Two decisions from the team reshape the roadmap's framing:

1. **Build the full roadmap (all 8 ETA-REQs, 301–308).** Nothing is dropped; this plan sequences them for dependency-correctness rather than shrinking scope.
2. **It must run on *any* generic Android app, not be hardcoded to Contacts.** This is the deepest constraint. A random Android app typically has **no SRS, no Figma guide, and no defect history** — so the *only guaranteed knowledge source is the live app UI itself* (mobilerun's on-device accessibility tree / screenshots) plus whatever the agent learns by exploring. Graceful degradation must therefore bottom out at **"explore the live device and build the app model from scratch."** This is stronger than the email's SRS-optional framing and is treated as a first-class source in this plan (the "Live-UI source").

Other locked decisions: defect data will be **synthetic/sample** initially (real Samsung exports drop in later against the same schema); the agentic redesign is **surgical + close the execution→learning loop** (keep endpoints backward-compatible, make sources pluggable, feed the executor's real trajectory back into the graph; defer a full tool-calling rewrite).

Intended outcome: a demoable agent that (a) generates high-value tests from whatever sources exist, including zero-doc apps; (b) ingests defect history and biases toward defect-prone areas; (c) remembers successful navigation paths and reuses them; (d) accumulates experiential memory (execution logs, error patterns, strategy effectiveness, coverage, sessions); (e) partitions knowledge by profile/platform/application; and (f) can *prove* it is getting smarter via an evaluation harness.

---

## Part A — Critique of the current architecture (code-accurate)

The foundation is solid — a genuine hybrid-retrieval GraphRAG loop — but it has structural gaps that block the roadmap's vision. Anchored to real code:

1. **It is a document store, not an experiential memory.** Ingestion writes SRS/Figma once; the only write-back from running tests is `TestCase`/`TestRun` verdict + notes (`rag_api/main.py::/tests/log`). There is **zero learning loop into retrieval** — no defect density, no navigation memory, no strategy scores. The roadmap's entire thesis is unimplemented today.

2. **The "bug oracle" is the SRS, so SRS-optional is not truly supported.** `planner/prompts.py::build_testcase_prompt` already guards blocks with `if srs_context:`, so it *structurally* survives a missing SRS. But *what counts as a bug* comes almost entirely from the `## Business Rules & Requirements (from SRS)` block ("Violations of these rules are bugs"), and the retrieval planner's source enum is hardcoded to `srs|figma_ui|figma_flow` (`planner/prompts.py::planner_prompt_for_action`, lines ~162, ~170–174). Remove the SRS and the agent keeps running but loses its notion of correctness — it falls back to generic heuristics on UI structure. **Graceful degradation requires a pluggable bug-oracle abstraction, not just skipping a prompt block.**

3. **The "GraphRAG entity graph" is half-built.** `ingestion/extractor.py` asks the LLM for `requirements/entities/validation_rules`, but on our live ingest the LLM extraction **silently fell back to regex** (`extraction_source: fallback`, `entities_written: 0`). The regex fallback only understands `FR-\d+`-style tags and emits **no entities** at all (`rule_based_extraction` returns `entities: []`). On the real hierarchical/prose SRS (`data/inputs/Sample-Contacts-App-SRS.txt`: `3.2.1.1.1`, "DETAILED DESCRIPTION", TODO markers, iOS/Android variants) the entity/validation quality collapses. The `(Requirement)-[:INVOLVES]->(Entity)` layer is effectively vestigial.

4. **UI retrieval is not semantic, which is fatal for zero-doc apps.** SRS retrieval is genuinely hybrid (vector + keyword + RRF + graph-hop, `rag_api/main.py::/retrieve`), but Figma retrieval is exact-screen-name lookup (`/figma/elements`, `planner/rag_client.py::get_screen_elements`). When the UI is the *primary* source (any generic app), the agent can only find a screen if it already knows its name. UI labels/purposes are embedded nowhere.

5. **The executor throws away everything it learns.** `clients/executor_runner.py::execute_test_on_device` runs mobilerun and keeps only `success/reason/steps(count)`, then logs a verdict string. mobilerun actually emits a per-step trace (`ExecutorResultEvent{action, outcome, error}`) and writes a `Trajectory` to disk — the exact data NavTree (302) and ExecutionLog (303) need. **The single richest data source for experiential learning already exists and is discarded.**

6. **Coverage and risk are ephemeral.** `planner/pipeline.py::agent_coverage` + `planner/coverage.py` recompute coverage in-memory each call; nothing is persisted, so there is no history, no trend, no regression-risk signal.

7. **Dedup is token-overlap only.** `planner/textutil.py::is_similar_to_existing` uses Jaccard ≥ 0.72. Semantically identical tests phrased differently slip through. Embeddings already exist (`rag_api/embeddings.py`), so this is a cheap upgrade (ETA-REQ-307.3).

8. **The "agent" is a fixed pipeline, not an agent.** `planner/langgraph_agent.py` is a 3-round retrieve→generate StateGraph whose only decision is retrieve-vs-produce, parsed from JSON pseudo-tool-calls. It has no feedback edge from real execution and no self-evaluation. Per the team's decision we keep this structure (surgical), but we **close the loop** so execution results re-enter the graph and thus the next generation.

9. **App-agnostic in prompt, not in ingestion.** Prompts avoid hardcoding, but ingestion assumes a Figma export exists (`ingestion/ui_normalizer.py` is Figma-shaped) and `.env` bakes in `contacts-app`/`com.android.contacts`. For "any Android app" we need a **live-UI ingestion adapter** and fully project-parameterized runs.

---

## Part B — Systems-level decisions (my calls; override any you disagree with)

- **B1. Introduce a pluggable "Knowledge Source" registry.** Every source (SRS, UI-guide, Live-UI, Defects, NavTree, ExecutionLog, StrategyMemory, ErrorPatterns) implements a small interface: `summary_block(project, dims)`, `retrieve(project, query, dims, k)`, `prompt_block(context)`. The planner iterates registered sources instead of the hardcoded `srs|figma_ui|figma_flow` enum. This makes the email's correction and every roadmap source additive rather than invasive. *Decision: this registry is WP0 and is the backbone of the whole plan.*

- **B2. A source-independent "bug oracle."** Define correctness signals with a priority cascade: SRS validation rules → defect history patterns → UI-guide affordances/heuristics → generic exploratory heuristics (always available). The generation prompt composes whichever oracle blocks exist. No source is required; the generic heuristic tier guarantees the agent always has *some* notion of "what would be a bug."

- **B3. Live-UI is a first-class source (new).** Add an ingestion adapter that snapshots the running app via mobilerun's accessibility tree/screenshots and normalizes it into the **same canonical UI IR** that `ingestion/ui_normalizer.py` produces from Figma. This is what makes zero-doc generic Android apps work, and it doubles as the seed for the NavTree. *Decision: elevate this above where the roadmap implies it; it is the load-bearing source for the team's generic-app requirement.*

- **B4. Persist what has history or is expensive; recompute what is cheap.** Persist as nodes: `Defect`, `NavTreeNode`, `ExecutionLog`, `ErrorPattern`, `StrategyMemory`, `Session`, `regression_risk_score` on `FeatureArea`. Keep the coverage *percentage* computed on demand, but persist the `CoverageHeatmap` snapshot (303.4) so trends/anomalies (308) have a time series. Avoids turning cheap derivations into stale nodes.

- **B5. Close the execution→learning loop via mobilerun's event stream, not the final result.** `executor_runner.py` must subscribe to `ExecutorResultEvent`s (or read the `Trajectory` writer output) to capture the real action/screen sequence, then POST to `/execution/log` and `/navtree/record-path`. *Open spike:* confirm the exact trajectory schema (`mobilerun/agent/trajectory/writer.py`, `config.logging.save_trajectory`) before building NavTree.

- **B6. Backward compatibility is mandatory.** All new endpoints are additive; existing routes keep their contracts (ETA-NFR-003). New request params (`profile`, `platform`, `application`, dedup mode) are optional with today's behavior as default.

- **B7. Context budgeter before the prompt bloats.** As 8 new `## blocks` land in `build_testcase_prompt`, add a token budget that ranks/truncates source blocks by relevance (risk score, coverage gap, dimensional match) so generation quality doesn't degrade from prompt stuffing.

- **B8. Evaluation harness is not optional for a capstone.** Build a defect-discovery / coverage-growth measurement (ETA-REQ-307 elevated) early enough to *prove* the "gets smarter" claim across sessions. Without it there is no evidence the roadmap succeeded.

- **B9. Dimensions: build the full model, validate on the `application` axis first.** Per the team, the near-term proof is "many generic Android apps," so `profile=mobile, platform=android` is fixed while `application` varies freely. Build `Profile`/`Platform`/`Application` nodes, dimensional filtering, and cross-dimensional transfer (304.4) as the roadmap specifies, but the first validation is app→app generalization, not watch/TV/tizen.

- **B10. Harden extraction for real/absent docs.** Chunk long SRS before LLM extraction, add JSON-repair + per-chunk retry, and make entities actually populate. When no SRS exists, the entity graph is simply empty and the pipeline leans on Live-UI + heuristics (B2). *Note: `SRS1.txt` (the clean `FR-N` sample) is deprecated; the real target is the hierarchical/prose `data/inputs/Sample-Contacts-App-SRS.txt`, which the current regex fallback parses badly — so this hardening is required, not optional.*

- **B11. The Live App Model — a self-built, screenshot-grounded UI state graph (unifies WP1/WP3/WP4).** Figma is *one optional way* to seed the UI graph, never the only one. The agent builds its own map of the app by observing the running screen: each distinct screen configuration is a `UIState` node (with its screenshot), transitions between states are labeled edges, and **the agent decides "new state vs. already-known" on every observation** via a structural accessibility-tree signature (primary) + perceptual screenshot hash + vision/text embedding (tie-breakers). New states and the paths that reached them are remembered; revisits just increment counters; multiple edges into a state capture alternative paths. The map updates continuously during testing and decays stale states after app updates. *Decision: this is the answer to "what's the point of exploring with no Figma/SRS" — the Live App Model is the always-available knowledge source, and Figma/SRS enrich it when present.*

- **B12. An operator dashboard is a first-class deliverable.** Without a live view of which test is executing, the verdict stream, coverage, defect-prone areas, and the evolving app-model graph, the system is unobservable in operation ("flying blind"). Today there are only raw logs + a `/metrics` endpoint. A dedicated dashboard WP (WP9) is added rather than left implicit in the evaluation harness.

---

## Part C — Target architecture (one picture)

```
 SOURCES (pluggable registry, B1)          KNOWLEDGE GRAPH (Neo4j)         AGENT LOOP (LangGraph, surgical)
 ─ SRS (any format)        ─┐                                             1 bootstrap brief  (all source summaries,
 ─ UI guide (Figma)         │   ingest ┌─────────────────────────┐            dimension-filtered)
 ─ Live-UI (device, NEW) ───┼────────► │  docs · entities · UI    │        2 iterative retrieve (registry sources,
 ─ Defects (synthetic→real)─┤          │  defects · navtree ·     │◄──────►   incl. defects + learned nav paths)
 ─ ExecutionLog (NEW) ──────┤ write-   │  exec logs · strategies  │        3 generate testcase (bug-oracle cascade
 ─ NavTree (NEW) ───────────┘  back    │  coverage · sessions ·   │            + risk + strategy + target-env blocks)
     ▲ (B5 loop)                       │  Profile/Platform/App    │        4 semantic dedup (embeddings)
     │                                 └─────────────────────────┘        5 output
 executor_runner.py ── mobilerun event stream ──────────────────────────────────┘
   (captures real trajectory → ExecutionLog + NavTree + StrategyMemory outcomes)
```

Graceful-degradation contract (B2): each stage composes only the blocks whose sources exist; the generic-heuristic oracle tier and the Live-UI source guarantee the loop always produces a meaningful test, even for a zero-doc app.

---

## Part D — Phased work packages (sequenced for dependency-correctness)

Each WP is sized for handoff to a coding agent: goal, mapped ETA-REQ, files to touch, schema/endpoints, acceptance criteria, and verification. Sequence differs from the roadmap's Phase 1–6 only to respect data dependencies; **all 8 ETA-REQs are included.**

### WP0 — Source registry + bug-oracle + graceful degradation *(foundation; the email's correction)*
- **Goal:** No source is a hard dependency; adding a source is additive.
- **Touch:** `planner/` new `sources/` package (registry + base interface); refactor `planner/langgraph_agent.py` (bootstrap/retrieve/generate nodes) and `planner/prompts.py` (`planner_prompt_for_action` source enum → registry-driven; `build_testcase_prompt` → oracle-cascade blocks). `planner/pipeline.py` wiring.
- **Acceptance:** Ingest only Figma (no SRS) → `/agent/next-testcase` still returns a valid, UI-grounded test. Ingest nothing but point at a live app (after WP1) → still returns a test from heuristics. Existing full-source behavior unchanged.
- **Verify:** Run `/agent/next-testcase` for a project with each source subset {SRS+UI, UI-only, none}; assert non-empty `next_testcase.steps` and that removed-source blocks vanish from the debug prompt.

### WP1 — Live App Model: screenshot-grounded UI state graph *(the core enabler for zero-doc apps; unifies live-UI ingestion + incremental new-state detection)*
- **Why:** With no Figma and no SRS, the only knowledge is what the app *shows*. Over-relying on Figma is disastrous (it's often absent or stale). The agent must **build and continuously grow its own app map** by looking at the running screen — and be smart enough to know when it has reached a **screen it has never seen** (so it records the new state and the path that reached it) versus one already in the map. This is model-based GUI exploration; it is the substrate the NavTree (WP4) walks over.
- **Data model — the Live App Model (see B11):**
  - `UIState` node = one distinct screen configuration. Props: `id`, `signature` (see below), `screenshot_ref`, `caption`, `element_summary`, `visit_count`, `first_seen`, `last_seen`, `embedding` (for semantic retrieval), optional `activity`/`package`.
  - `(:Project)-[:HAS_STATE]->(:UIState)`
  - `(:UIState)-[:TRANSITIONS_TO {action, element, visit_count, success_count}]->(:UIState)` — **multiple outgoing edges model the alternative paths** through the app.
  - `(:UIState)-[:REALIZES]->(:FigmaScreen)` — when a Figma guide also exists, link the live state to its design screen (Figma becomes *one* seed of the same graph, not the only source).
- **New-vs-known state detection (the "is this new?" decision — B11):** on each observation compute a **structural signature** first = normalized hash of the visible element set (types + labels + hierarchy) from mobilerun's accessibility tree, with volatile content (list data, timestamps, counters) stripped. Match against existing `UIState.signature`. Tie-break/confirm with a **perceptual screenshot hash (pHash)** for visual similarity, and fall back to a **vision/text embedding cosine** when structural signatures are ambiguous. Above threshold → known state (increment `visit_count`, refresh `last_seen`); below → **new** `UIState` (store screenshot + caption + embedding) and record the `TRANSITIONS_TO` edge that reached it.
- **Touch:** new `ingestion/live_ui.py` (accessibility tree + screenshot → `UIState` + signature; reuse the canonical UI IR from `ingestion/ui_normalizer.py` for the element list); new endpoints `POST /liveui/observe` (submit one observed state → returns `{state_id, is_new}`), `GET /appmodel/graph` (the state/transition map, Cytoscape-shaped like `/graph/visualize`); screenshot artifact storage (filesystem/blob, referenced on the node); `planner/sources/liveui.py` source so the state graph is retrievable by intent. Parameterize `.env`/runs so `PROJECT`/`TARGET_APP_PACKAGE` are per-app, not baked to contacts. Also add a lightweight **autonomous crawl mode** (tap/scroll systematically to map the app before goal-directed testing) as `clients/crawl_runner.py`.
- **Continuous update + decay:** the executor (WP3) calls `/liveui/observe` on every step, so the map densifies during normal testing, not just during a crawl. Apply `knowledge_half_life` so states that disappear after an app update fade out (B4).
- **Acceptance:** Point at any installed Android app (no Figma, no SRS) → a `UIState` graph builds; re-observing the same screen does **not** create a duplicate; reaching a genuinely new screen **does** add a node + transition; `/agent/next-testcase` produces a test grounded in real observed states.
- **Verify:** Crawl a non-contacts app (e.g. Settings); assert node count stabilizes on revisits and grows on new screens; confirm two paths to the same state produce two `TRANSITIONS_TO` edges into one `UIState` (not two states).

### WP2 — Defect intelligence *(ETA-REQ-301; synthetic data)*
- **Goal:** Defect history as a first-class source and bug-oracle tier.
- **Touch:** `rag_api/main.py` (new `Defect` nodes + relationships per §4; `POST /ingest/defects`, `GET /defects/summary`, `GET /defects/prone-areas`; defect-density into `/retrieve` scoring); `planner/sources/defects.py`; `## Defect History Context` block in `build_testcase_prompt`; `Summary{kind='defects'}`; seed `data/inputs/defects_sample.json` (synthetic contacts + a generic app).
- **Acceptance (301 AC):** defects queryable; retrieval biases toward defect-prone areas; `/context/brief` includes defect summary; prompt carries defect context when relevant.
- **Verify:** Ingest sample defects concentrated in one area → generated tests skew to that area vs. a no-defect baseline.

### WP3 — Execution-trace capture *(ETA-REQ-303.1; prerequisite for all experiential learning — pulled earlier than roadmap)*
- **Goal:** Persist the executor's real trajectory + rich verdict.
- **Touch:** `clients/executor_runner.py` (subscribe to mobilerun event stream / read Trajectory — see B5 spike); `ExecutionLog` node + `POST /execution/log`; `(:ExecutionLog)-[:FOR_TEST]->(:TestCase)`.
- **Acceptance:** every run writes an `ExecutionLog` with steps attempted/completed, verdict, error_type/message, environment snapshot.
- **Verify:** Run a 2-round loop; assert one `ExecutionLog` per round with a non-empty action/screen sequence.

### WP4 — Navigation memory *(ETA-REQ-302)*
- **Goal:** Persist shortest successful paths; reuse them; avoid failed ones.
- **Touch:** `NavTreeNode` model + `POST /navtree/record-path`, `GET /navtree/retrieve-path`, `GET /navtree/failed-paths`; shortest-path merge + `avoid` flag (success/visit ratio < 0.3); `planner/sources/navtree.py` → `## Learned Navigation Path` + `## Known Failed Navigation Paths` blocks; cross-test sub-path reuse. Feed from WP3 traces.
- **Acceptance (302 AC):** re-execution uses stored shortest path (retrievable API); failed paths avoided; tree grows without duplication; path retrieval < 200ms.
- **Verify:** Run the same test twice; second run's prompt contains the learned path; confirm fewer exploratory steps.

### WP5 — Experiential learning *(ETA-REQ-303.2–303.6)*
- **Goal:** Error patterns, strategy memory, coverage heatmap, sessions, knowledge decay.
- **Touch:** `ErrorPattern`, `StrategyMemory`, `CoverageHeatmap`, `Session` nodes; `GET /execution/error-patterns`, `GET /coverage/heatmap`, `POST /session/start|end`, `GET /session/context`; persist `CoverageHeatmap` (replaces ephemeral compute in `planner/coverage.py`); `knowledge_half_life` (default 90d) weighting in retrieval scoring; strategy-effectiveness update when a test finds a defect.
- **Acceptance (303 AC):** logs queryable; error patterns surfaced in planner; strategy scores bias generation; heatmap drives prioritization; session continuity across calls; decay applied.
- **Verify:** Across a multi-test session, show coverage heatmap growth and a strategy score changing after a defect-finding test.

### WP6 — Multi-dimensional KG *(ETA-REQ-304; validate on `application` axis first, B9)*
- **Goal:** Partition + filter by profile/platform/application; cross-dimensional transfer.
- **Touch:** `Profile`/`Platform`/`Application` nodes + `TARGETS_*`/`VALID_FOR_*`/`TESTS_APPLICATION` edges; optional `profile|platform|application` params on `/ingest/*`, `/retrieve`, `/agent/next-testcase` (default = today's single dimension); dimension-filtered queries everywhere via the registry; `MAY_APPLY_TO {transfer_confidence}`; `## Target Environment` prompt block.
- **Acceptance (304 AC):** dimensional filtering works; retrieval returns only in-dimension context; transfer suggests tests for untested combos; per-dimension SRS/Figma ingest independently.
- **Verify:** Ingest two apps; generate with `application` filter and confirm no cross-app leakage; show a transferred test suggestion.

### WP7 — Self-healing + regression risk *(ETA-REQ-305, 306)*
- **Goal:** Adaptive recovery on failure; risk-weighted prioritization.
- **Touch:** failure classification (`NAVIGATION_FAILURE|ELEMENT_NOT_FOUND|ASSERTION_FAILURE|TIMEOUT|CRASH|PERMISSION_DENIED`) in `executor_runner.py`; per-category retry (alt NavTree path, wait/retry, log-as-defect, extend timeout, restart+checkpoint) → logged in `ExecutionLog`; `## Previous Failure Context` retry block; `regression_risk_score` on `FeatureArea` from defect density + fail ratio + recency + nav instability; `GET /risk/scores`; `## Regression Risk Assessment` block.
- **Acceptance (305/306 AC):** failures classified + recovery attempted + outcome logged; risk scores stored, biasing generation, exposed via API.
- **Verify:** Inject a recoverable failure and confirm a retry attempt + logged outcome; confirm high-risk areas get more generation attention.

### WP8 — Quality metrics, semantic dedup, anomaly detection *(ETA-REQ-307, 308)*
- **Goal:** Prove-it feedback loop + emerging-issue detection. (Pull **307.3 semantic dedup** forward to WP0/WP1 as a quick win — embeddings already exist.)
- **Touch:** `planner/textutil.py::is_similar_to_existing` → embedding-cosine dedup (keep Jaccard as fast pre-filter); per-test `defect_discovery_rate`/`execution_stability`/`coverage_contribution`; strategy-score feedback; `AnomalyAlert` node + engine over `ExecutionLog` (failure-rate spikes, time regressions, new error types, path instability) + `GET /anomalies`; anomalies surfaced to generation.
- **Acceptance (307/308 AC):** semantic dedup catches reworded duplicates; effectiveness metrics stored; anomalies detected and drive investigation tests.
- **Verify:** Feed a reworded duplicate → rejected; synthesize a failure spike → `AnomalyAlert` appears and biases next test.

### WP9 — Operator dashboard & live observability *(so we are not flying blind; new, B12)*
- **Goal:** A single live view of what the agent is doing and how it is trending.
- **Panels:** current session + which test case is executing now; the verdict stream (pass/fail per round, live); coverage heatmap and defect-prone areas; the **Live App Model graph** (`UIState` nodes + `TRANSITIONS_TO` edges, with screenshots on hover) so you can *watch the map grow*; recent `ExecutionLog` timeline; anomaly alerts; the Part F "gets smarter" trend charts.
- **Touch:** new `dashboard/` served by the gateway (or a small static SPA) reading existing read endpoints (`/agent/coverage`, `/graph/visualize`, `/appmodel/graph`, `/tests/recent`, `/anomalies`, `/risk/scores`) + a new `GET /session/live` (current execution status) and an SSE/websocket stream off the executor's event loop; reuse the `observability/` logs (`logs/app.jsonl`) as the event source. Keep it self-contained (can be published as an Artifact for demos).
- **Acceptance:** During a running `executor_runner.py` loop, the dashboard shows the active test, updates verdicts live, and renders the app-model graph growing as new states are discovered.
- **Verify:** Start a loop; watch the dashboard reflect each round and each newly-discovered `UIState` in near-real-time.

---

## Part E — Consolidated data model, API surface, prompt-injection map

- **New nodes** (full list in `data/Extended_planning.txt` §4): `UIState` (Live App Model, B11) + `TRANSITIONS_TO`/`HAS_STATE`/`REALIZES` edges, `Defect`, `NavTreeNode`, `ExecutionLog`, `ErrorPattern`, `StrategyMemory`, `CoverageHeatmap`, `Session`, `Profile`, `Platform`, `Application`, `AnomalyAlert`; enhance `FeatureArea.regression_risk_score`, `TestCase.targets_defect_area`. The `NavTreeNode` path layer (WP4) is a walk over the `UIState` graph — a NavTree step's target should link to the `UIState` it lands on (`LEADS_TO_SCREEN` generalizes to `LEADS_TO_STATE`). Add constraints/vector-indexes in the `rag_api/main.py` lifespan block alongside the existing ones.
- **New endpoints** (all additive, backward-compatible): `/liveui/observe` + `/appmodel/graph` (Live App Model, new), `/ingest/defects`, `/defects/summary`, `/defects/prone-areas`, `/navtree/record-path|retrieve-path|failed-paths`, `/execution/log`, `/execution/error-patterns`, `/coverage/heatmap`, `/session/start|context|end|live`, `/risk/scores`, `/anomalies`; `+dimension params` on `/ingest/srs`, `/ingest/figma`, `/retrieve`, `/agent/next-testcase`.
- **Prompt-injection points** (all in `planner/prompts.py::build_testcase_prompt`, composed by the WP0 registry, gated by source presence): `## Defect History Context`, `## Learned Navigation Path`, `## Known Failed Navigation Paths`, `## Target Environment`, `## Regression Risk Assessment`, `## Strategy Suggestions`, `## Previous Failure Context`. Retrieval-planner sources extend `planner_prompt_for_action`'s enum via the registry (no hardcoded list).
- **Config:** new `.env`/`planner/config.py` keys — `KNOWLEDGE_HALF_LIFE_DAYS=90`, `DEDUP_MODE=semantic`, `NAV_AVOID_THRESHOLD=0.3`, dimensional defaults; keep everything optional with current defaults.

---

## Part F — Evaluation harness (prove "gets smarter") *(build alongside WP2–WP5)*

- **Metrics:** defect-discovery rate per session, coverage growth curve (from `CoverageHeatmap` snapshots), redundant-test rate (dedup rejections), avg steps-to-complete (should fall as NavTree fills), strategy-effectiveness trend.
- **Method:** seed synthetic defects with known ground-truth locations; run N-round sessions with and without each learning source enabled (ablation); chart the deltas.
- **Artifact:** `scripts/evaluate.py` + a small dashboard summarizing per-session trends. This is the capstone's evidence.

---

## Part G — Risks, spikes, and sequencing notes

- **Spike (blocks WP3/WP4):** confirm mobilerun's trajectory/event schema (`mobilerun/agent/trajectory/writer.py`, `ExecutorResultEvent`, `config.logging.save_trajectory`). Everything experiential depends on capturing the real path; do this spike first.
- **Extraction fragility (WP0/WP2):** hierarchical/prose SRS + JSON-mode flakiness on qwen-flash. Chunk + repair + retry; consider a stronger model for extraction only.
- **Prompt bloat (WP5+):** enforce the B7 context budgeter before all blocks co-exist.
- **Cross-dimensional transfer (WP6, 304.4)** is the most speculative requirement; keep `transfer_confidence` conservative and human-reviewable to avoid polluting the graph with wrong transfers.
- **State-signature tuning (WP1):** the new-vs-known threshold is the highest-leverage knob — too strict fragments one screen into many states, too loose collapses distinct screens. Make it configurable and validate on a real app crawl. This is the open decision to confirm before WP1 coding: structural-signature-primary (recommended, cheap, robust) vs. vision-embedding-primary (semantic, needs a multimodal model per observation).
- **Recommended build order:** Spike → WP0(+semantic dedup, ✅ core done) → **WP1 (Live App Model)** → WP2 → WP3 → WP4 → WP5 → WP6 → WP7 → WP8, with **WP9 (dashboard)** built incrementally from WP1 onward (it just reads existing endpoints) and the evaluation harness (Part F) growing from WP2. WP1 is now the critical path for the generic-app goal; WP0/WP2/WP3 can parallelize once the Part E schema is frozen.

## Verification (end-to-end, per WP)

Environment is already up locally (Neo4j Desktop "test" @7687, RAG :9010, Gateway :9100, emulator + mobilerun). For each WP: (1) ingest the relevant source subset; (2) hit the new endpoint(s) and assert graph writes via `/graph/stats` and Neo4j Browser; (3) run `/agent/next-testcase` and inspect the debug prompt for the expected `## blocks`; (4) run a real `clients/executor_runner.py` loop and confirm the execution→learning write-back; (5) run `scripts/evaluate.py` to confirm the smarter-over-time deltas.
