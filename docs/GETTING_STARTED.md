# Getting Started

A complete, tested walkthrough: from a fresh clone to watching the agent find bugs on a real
Android emulator. Follow it top to bottom the first time — the device setup in step 4 is the
part that most often goes wrong, and its failure message is misleading.

## What this tool does

It is an **exploratory testing agent**. You give it whatever knowledge you have about an app
(a requirements doc, a Figma export, past defect reports — any subset, or none at all), and it
runs a loop:

1. **Planner** picks the single highest-value next test case and writes it as JSON.
2. **Executor** drives a real Android app to carry that test out.
3. **Everything the run reveals is written back** into a Neo4j knowledge graph — the verdict,
   the screens visited, the actions taken, why it failed.
4. The next test case is generated from that enriched graph.

The graph *is* the agent's memory. Each LLM call is stateless, so the quality of what is in the
graph determines the quality of the next test. (See [Two things newcomers get
wrong](#two-things-newcomers-get-wrong).)

---

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.13** | 3.14 fails to build dependencies. Use [uv](https://docs.astral.sh/uv/) so an OS upgrade can't delete your interpreter — a Homebrew `python@3.13` can vanish on upgrade and orphan the venv. |
| **Neo4j 5.13+** | Desktop or Docker. Vector-index support is needed for semantic retrieval. |
| **Android Studio** | For the emulator (e.g. Pixel 9a). A physical device over ADB works too. |
| **An LLM API key** | OpenRouter (recommended) or Gemini. |
| **Node 18+** | Needed for the full dashboard. `dashboard-react/dist/` is git-ignored, so build it once: `npm --prefix dashboard-react install && npm --prefix dashboard-react run build`. Without it the gateway serves the simpler committed fallback (`dashboard/index.html`), which lacks the planner-trace and per-run step panels. |

**Appium is *not* required.** Older docs mention it; the executor talks to the device through
mobilerun's on-device portal over ADB. You can ignore Appium entirely.

## 2. Install

```bash
uv python install 3.13
uv venv venv --python 3.13          # must be "venv" — start.sh looks for ./venv
source venv/bin/activate
uv pip install -r requirements.txt "droidrun==0.6.8"
```

Pin `droidrun==0.6.8`: the code targets that API, and newer releases have broken it before.
Sanity check:

```bash
python -c "from mobilerun import MobileAgent; print('mobilerun ok')"
python tests/test_app_state.py      # expect 11/11 passed
```

## 3. Configure `.env`

Copy `.env.example` to `.env` and fill in the values that matter:

```ini
# ── Knowledge graph ──
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password

# ── Planner model (also used for SRS/Figma ingestion) ──
MODEL_BACKEND=openrouter
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=deepseek/deepseek-v4-flash-0731

# ── Device-agent model (drives the emulator; fires on every step, so keep it cheap) ──
EXECUTOR_LLM_PROVIDER=OpenRouter
EXECUTOR_LLM_MODEL=qwen/qwen3.5-flash-02-23

# ── Project + inputs ──
PROJECT=contacts-app
SRS_PATH=./data/inputs/Sample-Contacts-App-SRS.txt
FIGMA_PATH=./data/inputs/GENERATED_JSON.json
TARGET_APP_PACKAGE=com.google.android.contacts

# ── Run budget ──
EXECUTOR_ROUNDS=10          # test cases per invocation
EXECUTOR_MAX_STEPS=45       # device actions allowed per test
EXECUTOR_TIMEOUT=450        # seconds of wall clock per test
```

Two things worth knowing:

- **Use a strong model for the planner.** It writes strict JSON and does the reasoning. A weak
  model produces malformed output and shallow SRS extraction. `EXECUTOR_LLM_MODEL` is a
  separate, cheaper model because it fires on every device step.
- **`TARGET_APP_PACKAGE` must actually be installed.** On recent emulator images the Contacts
  app is `com.google.android.contacts`, *not* `com.android.contacts`. Check with
  `adb shell pm list packages | grep contact`.

## 4. Set up the device (the step that bites people)

Start your emulator in Android Studio, then confirm ADB sees it:

```bash
adb devices        # expect: emulator-5554   device
```

Now install mobilerun's on-device **portal** — the app that reads the screen. Without it, the
executor fails with a *"Parse Error: Failed to parse state data from ContentProvider"* that
looks like data corruption but actually means "the portal isn't there."

```bash
./venv/bin/mobilerun setup

# setup installs the APK but often cannot enable these itself — do it explicitly:
adb shell settings put secure enabled_accessibility_services \
  com.mobilerun.portal/com.mobilerun.portal.service.MobilerunAccessibilityService
adb shell settings put secure accessibility_enabled 1
adb shell ime enable com.mobilerun.portal/.input.MobilerunKeyboardIME
adb shell ime set com.mobilerun.portal/.input.MobilerunKeyboardIME

./venv/bin/mobilerun ping     # must print "You're good to go!"
```

> **This can silently undo itself.** If the emulator restores a Quick Boot snapshot taken
> before the portal was installed, the portal, the accessibility setting and the keyboard all
> disappear. Either **Cold Boot** the AVD and re-save its snapshot *after* running setup, or
> re-run the block above after each emulator restart. `mobilerun doctor` reports exactly which
> piece is missing.

Also note: the keyboard step is not optional. Test cases type text, and without mobilerun's IME
typing silently does nothing.

## 5. Start the services

Two terminals, so you can see each service's logs:

```bash
# Terminal 1 — knowledge graph API
./venv/bin/python -m uvicorn rag_api.main:app --host 0.0.0.0 --port 9010

# Terminal 2 — planner gateway
./venv/bin/python -m uvicorn gateway.main:app --host 0.0.0.0 --port 9100
```

```bash
curl http://127.0.0.1:9010/health     # also proves Neo4j is reachable
curl http://127.0.0.1:9100/health     # shows the active model
```

> **Restarting a service properly.** `kill` does not always release the port in time; the new
> process then dies with `[Errno 48] address already in use` while the **old code keeps
> serving**, which produces baffling 404s on endpoints you just added. Always confirm the port
> is free:
> ```bash
> for pid in $(lsof -ti :9100); do kill -TERM $pid; done; sleep 3
> [ -n "$(lsof -ti :9100)" ] && kill -9 $(lsof -ti :9100)
> ```

## 6. Ingest what you know about the app

```bash
./venv/bin/python scripts/ingest_all.py       # SRS + Figma, paths from .env

# optional: synthetic defect history, which biases testing toward fragile areas
curl -X POST http://127.0.0.1:9100/defects/ingest -H 'Content-Type: application/json' \
  -d '{"project":"contacts-app","source_path":"./data/inputs/defects_sample.json"}'

curl "http://127.0.0.1:9010/graph/stats?project=contacts-app"
```

Expect non-zero `requirement_count`, `validation_rule_count` and `figma_screen_count`. Counts
vary between runs because extraction is LLM-driven — that is normal, not a bug.

**None of this is mandatory.** With no SRS and no Figma, the agent falls back to exploratory
heuristics plus the app model it builds by observing the live device. That path is the whole
point of the design: it works on an app you have no documentation for.

> ⚠️ `scripts/ingest_all.py` begins with a project reset, and `POST /project/reset` deletes
> **opt-out**, not opt-in: `{"project":"x","delete_tests":true}` also wipes the SRS and Figma,
> because those flags default to `true`. To clear only test history, spell every flag out:
> ```bash
> curl -X POST http://127.0.0.1:9010/project/reset -H 'Content-Type: application/json' \
>   -d '{"project":"contacts-app","delete_tests":true,"delete_srs":false,"delete_figma":false}'
> ```

## 7. Run the loop

```bash
./venv/bin/python clients/executor_runner.py
```

Each round: the planner generates a test, the emulator executes it, the verdict and trajectory
are written back. Open the dashboard alongside it:

**<http://127.0.0.1:9100/dashboard?project=contacts-app>**

Other entry points:

| Command | Use |
|---|---|
| `clients/simulator_runner.py` | Full loop with fake verdicts — no device. Good for demos. |
| `clients/crawl_runner.py` | Explore an app with no goal, just to build the app model first. |
| `scripts/verify_enhancements.py` | LLM-free integration check of every graph endpoint. |

### Choosing depth or breadth: `EXPLORATION_MODE`

A long session pulls in two directions — dig deeper into the areas that keep breaking, or
spread out to cover ground nobody has touched. Set the balance per run in `.env`:

| Mode | Behaviour | Use when |
|---|---|---|
| `exploit` | Known-fragile areas first; prefers a new angle on a broken area over opening a new one. | Hardening a release; you have real defect history. |
| `explore` | Untested areas first; only revisits fragile areas once nothing untested is left. | A new app, or coverage looks lopsided. |
| `balanced` | Investigate failures, then expand. **Default.** | General exploratory sessions. |

The mode reorders the Exploration Directive and states itself in the prompt, so you can see
which one is active in the dashboard's **Exploration Directive** panel. **Restart the gateway
after changing it** — it is read at startup.

Regardless of mode, two safety rails always apply: an area whose last 3+ tests all passed
triggers a `[PIVOT]` away from it, and areas with 4+ tests and zero failures are
`[DEPRIORITIZE]`d — so `exploit` cannot get stuck grinding a stable area forever.

## 8. Reading the dashboard

- **Live App Model** — the map of screens the agent built by watching the device. Scrolling a
  list or switching to dark mode does *not* create a duplicate state; a new activity or an open
  dialog does. Click a node for its screenshot.
- **Planner Execution Trace** — per generation: which LangGraph nodes ran, how long each took,
  LLM latency and token cost. Start here when generation is slow or returns nothing.
- **Execution Paths** — each run's real route through the app. **`▸ show device steps`** expands
  every device action with the agent's reasoning for it and the final explanation of the
  verdict. This is the fastest way to understand *why* a test failed.
- **Live Logs** — device agent and planner side by side, both filtered to meaningful events.
- **SRS Knowledge** — the business policies extracted from the requirements doc, and which
  requirements no test covers yet.

## Two things newcomers get wrong

**1. `failed` means "this test found something", not "the tool broke."**
This is an exploratory tester: a failing verdict is a *success*. It feeds strategy memory,
defect-discovery metrics and regression-risk scores. Verdicts you will see:

| Verdict | Meaning |
|---|---|
| `planned` | Generated, not yet executed. Excluded from coverage and pass rate. |
| `pass` | Executed; the app behaved as the test expected. |
| `failed` | Executed; the app did **not** behave as expected — a candidate defect. |

Two failure categories are deliberately *not* counted as discoveries, because the run never
observed the app misbehaving: `PRECONDITION_NOT_MET` (setup never completed) and
`STEP_LIMIT_EXCEEDED` (ran out of steps). Both still appear in the execution log.

**2. The agent's intelligence is whatever is in the graph.**
Every LLM call is stateless — there is no chat history. Before each generation the planner
re-reads the graph and writes the relevant history into a fresh prompt. So a wrong verdict in
the graph is indistinguishable from truth on every later call. If you interrupt a run
mid-test, that test stays `planned` rather than silently becoming a pass.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Parse Error: Failed to parse state data from ContentProvider` | The portal is missing, or its accessibility service is off. Re-run step 4; `mobilerun doctor` names the missing piece. |
| Typing does nothing on device | mobilerun's IME is not the active keyboard — the `ime enable/set` lines in step 4. |
| `Planner returned empty test case` | The model's JSON was truncated. Raise `max_new_tokens`, or use a stronger planner model. |
| Every run ends "Reached max step count" | The device model's output cap is truncating its tool calls. Check `EXECUTOR_MAX_TOKENS` (default 4000). |
| A 404 on an endpoint you know exists | A stale process is still serving the port. See the restart note in step 5. |
| `404 ... /srs/ingest` from `ingest_all.py` | Usually a stale gateway; also check `SRS_PATH` points at a file that exists. |
| `requirement_count: 0` after ingest | Ingest failed or hit the wrong project. Re-run and read the full response body. |
| `Could not get usage: Unsupported provider: OpenRouter_LLM` | Harmless. mobilerun can't tally tokens for OpenRouter. Ignore. |
| Dashboard shows stale numbers | It polls every 4s; check the dot in the header and the gateway's health. |

## Where things live

```
rag_api/          Neo4j knowledge-graph API (:9010) — ingest, retrieval, learning, metrics
gateway/          Thin FastAPI router (:9100) + dashboard endpoints
planner/          The agent: LangGraph loop, prompts, knowledge sources, coverage
  sources/        Pluggable knowledge sources (srs, figma_ui, figma_flow, live_ui, defects, navtree)
ingestion/        Document loading, SRS extraction, UI normalisation, app-state signatures
clients/          executor_runner (device), simulator_runner (no device), crawl_runner (mapping)
dashboard-react/  Dashboard source; `npm run build` emits the single file the gateway serves
data/inputs/      Sample SRS, Figma export, synthetic defects
logs/             app.jsonl (structured), mobilerun.log (device), trajectories/ (per-run steps)
docs/             Architecture, workflow, and the implementation plan
```

For architecture and the work-package history, see [NEXTGEN_IMPLEMENTATION_PLAN.md](NEXTGEN_IMPLEMENTATION_PLAN.md)
and [WORKFLOW.md](WORKFLOW.md).
