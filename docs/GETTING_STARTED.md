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

The system is **app-agnostic** — this repo has run real campaigns against two unrelated apps
(a Contacts app, and a Bengali-language livestock marketplace) side by side in the same Neo4j
instance, kept apart purely by a `project` name. Nothing about the code assumes any specific app.

---

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.13** | 3.14 fails to build dependencies. Use [uv](https://docs.astral.sh/uv/) so an OS upgrade can't delete your interpreter — a Homebrew `python@3.13` can vanish on upgrade and orphan the venv. |
| **Neo4j 5.13+, Enterprise edition** | Desktop or Docker. Vector-index support is needed for semantic retrieval; Enterprise's multi-database feature is handy for testing a restore without touching real data (see [Backing up and restoring](#backing-up-and-restoring)). |
| **Android Studio** | For the emulator. A physical device over ADB works too — see [Sideloading an app from a physical device](#sideloading-an-app-from-a-physical-device) if the app isn't on the Play Store under your test account. |
| **An LLM API key** | OpenRouter (recommended) or Gemini. |
| **Node 18+** | Needed for the full dashboard. `dashboard-react/dist/` is git-ignored, so build it once: `npm --prefix dashboard-react install && npm --prefix dashboard-react run build`. Without it the gateway serves the simpler committed fallback (`dashboard/index.html`), which lacks the planner-trace, per-run step, and learning-intelligence panels. **Rebuild it any time `dashboard-react/src/` is newer than `dashboard-react/dist/index.html`** — a stale build silently hides whatever panel was added most recently. |

**Appium is *not* required.** The executor talks to the device through mobilerun's on-device
portal over ADB. You can ignore Appium entirely, and leaving it running alongside this stack is
harmless but does nothing for it.

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
python tests/run_all.py      # every test module; expect 79/79 passed across 5 modules
```

## 3. Configure `.env`

**`settings.py` (project root) is the single source of truth for every tunable** — every other
module imports from it (`from settings import EXECUTOR_MAX_STEPS, ...`) instead of calling
`os.getenv` directly. Read it once; it documents *why* each default is what it is, not just what
it is. `.env.example` mirrors it. Copy `.env.example` to `.env` and fill in at least this much:

```ini
# ── Knowledge graph ──
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password

# ── Planner model backend ──
MODEL_BACKEND=openrouter
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=qwen/qwen3.8-flash

# ── Device-agent model (drives the emulator; fires on every step) ──
EXECUTOR_LLM_PROVIDER=OpenRouter
EXECUTOR_LLM_MODEL=qwen/qwen3.7-flash
EXECUTOR_VISION=1              # send screenshots too; needs a vision-capable model

# ── The app under test — NO defaults; an unconfigured run fails loudly, not silently ──
PROJECT=your-app-name
SRS_PATH=./data/inputs/your-app-SRS.txt
FIGMA_PATH=                    # leave empty if you have no design file
TARGET_APP_PACKAGE=com.example.yourapp

# ── Run budget ──
EXECUTOR_ROUNDS=10             # test cases per invocation
EXECUTOR_MAX_STEPS=50          # device actions allowed per test
EXECUTOR_TIMEOUT=420           # seconds of wall clock per test
```

Things worth knowing:

- **Model tiering.** Three separate model knobs, each for a different job:
  - `OPENROUTER_MODEL` — the planner's per-round model. Needs to write strict JSON reliably; a
    weak model produces malformed output that fails to parse.
  - `EXTRACTION_MODEL` — used **once per document**, for structured SRS extraction. Leave empty
    to reuse `OPENROUTER_MODEL`. Since every later decision rests on what this extracts, a
    stronger model here is usually worth the one-time cost — but see the extraction-fallback
    gotcha below before assuming "stronger" always means "better output."
  - `EXECUTOR_LLM_MODEL` — the device agent. Fires on every UI action, so cost adds up fast;
    keep it cheap. If `EXECUTOR_VISION=1`, it must support image input (check the model's
    listing on OpenRouter — "flash"-tier names are not a reliable signal either way).
  - `EXTRACTION_SAMPLES` (default `1`) — extract the SRS N times and keep the best (self-consistency).
    A single attempt has a real, non-hypothetical chance of failing outright on a subset of
    sections and silently degrading to a regex fallback (see below); `3` is a cheap insurance
    policy against that on a document that matters.
- **`TARGET_APP_PACKAGE` must actually be installed**, and there's no fallback default —
  `settings.py` deliberately leaves this blank so a misconfigured run fails instead of quietly
  testing the wrong app. Check with `adb shell pm list packages | grep <name>`.
- **A non-English or unusually-labelled launcher icon** will make the agent conclude the app
  is "not installed" and try to reinstall/replace it — set `TARGET_APP_ACTIVITY` (get it with
  `adb shell cmd package resolve-activity --brief <package> | tail -1`) and `TARGET_APP_LABELS`
  (comma-separated, every label the icon might show) so it opens the app by package name instead
  of hunting for a name match.
- **Apps gated behind a login the agent cannot complete itself** (OTP to a phone you don't
  control, ID upload, admin approval) need a pre-provisioned, already-approved test account:
  `APP_LOGIN_ROLE`, `APP_LOGIN_IDENTIFIER`, `APP_LOGIN_SECRET`, `APP_LOGIN_HINT` (free text —
  which button to tap, whether OTP is skipped for test numbers), and `APP_ACCOUNT_STATE`
  (describe what the account *already has*, so the planner doesn't write a "register a farm"
  test for an account whose farm already exists — there's no UI entry point for that, and the
  agent will burn its step budget hunting for a screen that will never appear).
  `OUT_OF_SCOPE` (comma-separated phrases) keeps the planner from generating tests for
  flows that account provably cannot reach.
- **`PROMPT_BUDGET_TOKENS`** (default `50000`) is one global ceiling for the whole generation
  prompt, filled priority-first (business rules and UI controls first; background context last).
  This is the main cost dial in the system — every planner call pays it.

The full annotated list — device-reset policy, out-of-app drift detection, state-identity
thresholds, crawler settings — lives in `settings.py` itself; skim it once.

## 4. Set up the device (the step that bites people)

Start your emulator, then confirm ADB sees it:

```bash
adb devices        # expect: emulator-5554   device
```

`./start.sh` (see step 5) now handles the on-device portal setup automatically on every run —
it re-applies the accessibility-service and keyboard settings unconditionally, which self-heals
the most common failure mode below without you doing anything. The manual version, for when you
need to understand or repair it by hand:

```bash
./venv/bin/mobilerun setup

adb shell settings put secure enabled_accessibility_services \
  com.mobilerun.portal/com.mobilerun.portal.service.MobilerunAccessibilityService
adb shell settings put secure accessibility_enabled 1
adb shell ime enable com.mobilerun.portal/.input.MobilerunKeyboardIME
adb shell ime set com.mobilerun.portal/.input.MobilerunKeyboardIME

./venv/bin/mobilerun ping     # must print "You're good to go!"
```

> **This can silently undo itself.** If the emulator restores a Quick Boot snapshot taken
> before the portal was installed, the portal, the accessibility setting and the keyboard all
> disappear. Without the portal, the executor fails with a *"Parse Error: Failed to parse state
> data from ContentProvider"* that reads like data corruption but just means "the portal isn't
> there" — or worse, silently falls back to raw ADB text input, which drops characters and
> cannot verify it actually cleared a field before typing into it. Either **Cold Boot** the AVD
> and re-save its snapshot *after* running setup, or rely on `start.sh` re-asserting it every run.
> `mobilerun doctor` names exactly which piece is missing if something still seems off.

### Sideloading an app from a physical device

If the app under test isn't easily installable on the emulator (region-locked, not on the Play
Store under your test account, etc.), pull it off a real phone you control and push it to the
emulator:

```bash
adb devices -l                                          # confirm both phone and emulator show up
adb -s <phone-serial> shell pm path <package.name>       # find the APK path(s) on the phone
adb -s <phone-serial> pull <path-from-above> ./app.apk
adb -s emulator-5554 install ./app.apk                   # add install-multiple if pm path listed several files (split APK)
adb -s emulator-5554 shell monkey -p <package.name> -c android.intent.category.LAUNCHER 1
```

A split APK (several paths from `pm path`, common for a Play-Store install) needs
`install-multiple` with every file, not `install`. One real risk: some apps refuse to behave, or
block certain features, if they detect they weren't installed via the Play Store (integrity
checks) — this is usually fine for an ordinary app, but you won't know for certain until you try.

## 5. Start the services

```bash
./start.sh --build
```

This one command brings up Neo4j (if not already running), the emulator (if not already
connected), re-asserts the device portal, builds the dashboard, and starts the RAG API and
gateway — skipping anything already healthy rather than restarting it. Flags:

| Flag | Effect |
|---|---|
| *(none)* | Infrastructure only — no data changes. |
| `--ingest` | Also **reset and re-ingest** SRS/Figma. **Destructive** — see the warning in step 6. |
| `--with-executor` | Also start the executor test loop. |
| `--build` | (Re)build the React dashboard first. |
| `--no-neo4j` / `--no-emulator` | Skip that component (e.g. Neo4j managed elsewhere, or a physical device). |
| `--stop` | Delegates to `./stop.sh`. |

```bash
curl http://127.0.0.1:9010/health     # also proves Neo4j is reachable
curl http://127.0.0.1:9100/health     # shows the active planner model
```

To bring it back down: `./stop.sh` (stops services **+ emulator + Neo4j** by default —
`--keep-emulator` or `--services-only` if you don't want that).

> **Restarting a service properly, if you do it by hand.** `kill` does not always release the
> port before a fresh process tries to bind it — the new process can die with
> `[Errno 48] address already in use` while `start.sh` reports "port already in use, reusing
> existing process" and the **old process (possibly already dead, possibly running stale code)**
> keeps being treated as healthy. If a service seems to be serving stale behavior after a config
> change, confirm what's actually listening before trusting a "reusing existing process" message:
> ```bash
> lsof -i :9100 -sTCP:LISTEN     # shows the real PID and start time
> for pid in $(lsof -ti :9100); do kill -TERM $pid; done; sleep 3
> [ -n "$(lsof -ti :9100)" ] && kill -9 $(lsof -ti :9100)
> ```

## 6. Ingest what you know about the app

```bash
./venv/bin/python scripts/ingest_all.py       # SRS + Figma, paths from .env

# optional: defect history, which biases testing toward historically fragile areas
curl -X POST http://127.0.0.1:9100/defects/ingest -H 'Content-Type: application/json' \
  -d '{"project":"your-app-name","source_path":"./data/inputs/your-defects.json"}'

curl "http://127.0.0.1:9010/graph/stats?project=your-app-name"
```

Expect non-zero `requirement_count`, `validation_rule_count`, and (if you have one)
`figma_screen_count`. Counts vary between runs because extraction is LLM-driven — that alone is
normal, not a bug.

**None of this is mandatory.** With no SRS and no Figma, the agent falls back to exploratory
heuristics plus the app model it builds by observing the live device. That path is the whole
point of the design: it works on an app you have no documentation for.

> ⚠️ **`ingest_all.py` always resets tests, SRS, *and* Figma together — every time, unconditionally.**
> It is not "refresh the SRS"; it is "wipe this project's tests and knowledge and rebuild from
> the source files." Real, in-progress test history has been lost this way. If you only want to
> refresh the SRS/Figma without touching test history, don't use `ingest_all.py` — call the
> endpoint directly with the flags spelled out:
>
> ```bash
> curl -X POST http://127.0.0.1:9010/project/reset -H 'Content-Type: application/json' \
>   -d '{"project":"your-app-name","delete_tests":false,"delete_srs":true,"delete_figma":true}'
> # then re-run just the /srs/ingest (and /figma/ingest) calls yourself
> ```
>
> Also note `POST /project/reset` deletes **opt-out, not opt-in**: every flag defaults to `true`,
> so `{"project":"x","delete_tests":true}` silently also wipes SRS and Figma. Always spell out
> every flag explicitly for a partial reset.
>
> ⚠️ **A single extraction attempt can silently collapse to a near-empty result.** If enough of
> the SRS's per-section extraction calls come back with malformed JSON (happens intermittently,
> not just on weak models), the whole pass falls back to a crude regex extractor — a
> dramatically weaker mechanism, not just "a worse model." The signature is `validation_rule_count`
> **and** `entity_count` both near zero at once. Check `logs/degradations.jsonl` for
> `srs_extraction_fallback` (severity `critical`) to confirm this happened, and just re-run the
> ingest — the failure is intermittent, and `EXTRACTION_SAMPLES=3` makes it far less likely to
> recur, since the best of several independent attempts is kept.

## 7. Run the loop

```bash
./venv/bin/python clients/executor_runner.py
```

Each round: the planner generates a test, the emulator executes it, the verdict and trajectory
are written back. Open the dashboard alongside it:

**<http://127.0.0.1:9100/dashboard?project=your-app-name>**

Other entry points:

| Command | Use |
|---|---|
| `clients/simulator_runner.py` | Full loop with fake verdicts — no device. Good for demos. |
| `clients/crawl_runner.py` | Explore an app with no goal, just to build the app model first. |
| `scripts/verify_enhancements.py` | LLM-free integration check of every graph endpoint. |
| `scripts/dump_prompt.py` | Rebuild and print the exact generation prompt — no LLM call, free — with a per-block size breakdown. Useful when a test looks wrong and you want to see exactly what the planner was told. |
| `scripts/analyze_batch.py` | Post-campaign report: attribution-split outcomes, learning-layer state, degradations. |

### Choosing depth or breadth: `EXPLORATION_MODE`

A long session pulls in two directions — dig deeper into the areas that keep breaking, or
spread out to cover ground nobody has touched. Set the balance per run in `.env`:

| Mode | Behaviour | Use when |
|---|---|---|
| `exploit` | Known-fragile areas first; prefers a new angle on a broken area over opening a new one. | Hardening a release; you have real defect history. |
| `explore` | Untested areas first; only revisits fragile areas once nothing untested is left. | A new app, or coverage looks lopsided. |
| `balanced` | Investigate failures, then expand. **Default.** | General exploratory sessions. |

The mode reorders the Exploration Directive and states itself in the prompt. **Restart the
gateway after changing it** — it is read at startup.

Regardless of mode, two safety rails always apply: an area whose last 3+ tests all passed
triggers a `[PIVOT]` away from it, and areas with 4+ tests and zero failures are
`[DEPRIORITIZE]`d — so `exploit` cannot get stuck grinding a stable area forever.

### Failure attribution: whose fault was it?

Every failed run is classified into exactly one of three buckets — defined once in `settings.py`
so this can never drift between components again:

| Bucket | Categories | Meaning |
|---|---|---|
| `APP_FAULT` | `ASSERTION_FAILURE`, `CRASH`, `APP_UNRESPONSIVE` | The app genuinely misbehaved — a real candidate defect. |
| `AGENT_FAULT` | `TIMEOUT`, `ELEMENT_NOT_FOUND`, `NAVIGATION_FAILURE`, `NAVIGATION_LIVELOCK` | Our agent couldn't complete the test; says nothing about the app. |
| `ENV_FAULT` | `PRECONDITION_NOT_MET`, `PERMISSION_DENIED`, `STEP_LIMIT_EXCEEDED` | Setup/environment problem; the app was never really exercised. |

Only `APP_FAULT` counts toward defect-discovery metrics and strategy reinforcement — mixing these
buckets is how "autonomy" and "bugs found" numbers become meaningless.

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
- **Learning Intelligence** (defects, risk, anomalies, strategy memory, nav-tree, requirement
  coverage) — these are the signals that steer generation; the panel exists specifically so an
  empty/dormant module (e.g. no defect history ingested) is *visible* as empty rather than
  silently contributing nothing.
- **Degradation banner** — appears at the top when the run fell back to weaker behavior anywhere
  (extraction fallback, a dropped source, a missing device portal). If it says results should not
  be trusted, believe it before drawing conclusions from that run's data.

## Backing up and restoring

Given how easy it is to lose test history (see the `ingest_all.py` warning above) or to want to
try a risky change, two scripts exist for this:

```bash
./venv/bin/python scripts/backup_neo4j.py                 # full, unscoped dump -> data/backups/neo4j_backup_<ts>.json
./venv/bin/python scripts/restore_neo4j.py <backup.json> --wipe-first
```

`backup_neo4j.py` dumps every node and relationship across **every** project in one file — it is
not project-scoped. Before trusting a restore against real data, verify it against a throwaway
database first (Neo4j Enterprise supports multiple databases):

```bash
# one-off setup, in a Python shell or script using the neo4j driver against database="system":
#   CREATE DATABASE restoretest IF NOT EXISTS
./venv/bin/python scripts/restore_neo4j.py <backup.json> --database restoretest --wipe-first
# then check counts/spot-check content in restoretest before ever running --wipe-first against
# your real database
```

## Multiple projects, one Neo4j

Every node is tagged with a `project` property, and every query filters on it — this is the
standard multi-tenant pattern (the same one relational databases use with a `tenant_id` column),
not something specific to this app. Two (or more) completely unrelated apps can have their SRS,
tests, and app models sitting in the same Neo4j instance with zero risk of one leaking into the
other's generation prompts. Switch which one the running services act on purely via `.env`'s
`PROJECT=` value; nothing else needs to change.

## Two things newcomers get wrong

**1. `failed` means "this test found something", not "the tool broke."**
This is an exploratory tester: a failing verdict is a *success*. It feeds strategy memory,
defect-discovery metrics and regression-risk scores. Verdicts you will see:

| Verdict | Meaning |
|---|---|
| `planned` | Generated, not yet executed. Excluded from coverage and pass rate. |
| `pass` | Executed; the app behaved as the test expected. |
| `failed` | Executed; the app did **not** behave as expected — a candidate defect (only if attributed `APP_FAULT` — see the table above). |

**2. The agent's intelligence is whatever is in the graph.**
Every LLM call is stateless — there is no chat history. Before each generation the planner
re-reads the graph and writes the relevant history into a fresh prompt. So a wrong verdict in
the graph is indistinguishable from truth on every later call. If you interrupt a run
mid-test, that test stays `planned` rather than silently becoming a pass.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Parse Error: Failed to parse state data from ContentProvider` | The portal is missing, or its accessibility service is off. Re-run step 4's manual block; `mobilerun doctor` names the missing piece. |
| Typing does nothing, or corrupts existing field text | mobilerun's IME isn't active, and the driver silently fell back to raw ADB input (which can't verify it cleared a field before typing). Re-run the manual portal setup in step 4. |
| `Planner returned empty test case` | The model's JSON was truncated. Raise `GENERATION_MAX_TOKENS`, or use a stronger planner model. |
| Every run ends "Reached max step count" | The device model's output cap is truncating its tool calls, or the model genuinely needs more steps. Check `EXECUTOR_MAX_TOKENS`; this verdict is `STEP_LIMIT_EXCEEDED` (`ENV_FAULT`) and is excluded from defect metrics on purpose. |
| `503 ... 429 Client Error: Too Many Requests` mid-campaign, killing the whole process | OpenRouter rate-limiting on the specific model/provider route, not your account or code — checkable via `GET https://openrouter.ai/api/v1/key`. Usually clears within a minute; it can recur under the combined load of the executor's rapid per-step calls and the planner's own calls on the same model family. The process dies on this uncaught error — just restart it; it resumes from the graph's current state. |
| `requirement_count` and `validation_rule_count`/`entity_count` all near zero after ingest | Extraction silently fell back to regex. See the ingest warning in step 6. |
| A 404/stale behavior on an endpoint you know exists or just changed | A stale process is still serving the port — see the restart note in step 5. |
| `Could not get usage: Unsupported provider: OpenRouter_LLM` | Harmless. mobilerun can't tally tokens for OpenRouter. Ignore. |
| Dashboard shows stale numbers | It polls every few seconds; check the dot in the header and the gateway's health. |

## Where things live

```
settings.py       Single source of truth for every tunable — read this before .env
rag_api/          Neo4j knowledge-graph API (:9010) — ingest, retrieval, learning, metrics
gateway/          Thin FastAPI router (:9100) + dashboard endpoints
planner/          The agent: LangGraph loop, prompts, knowledge sources, coverage, prompt budget
  sources/        Pluggable knowledge sources (srs, figma_ui, figma_flow, live_ui, defects, navtree)
ingestion/        Document loading, SRS extraction, UI normalisation, app-state signatures
observability/    Structured logging + degradations.py (silent-fallback tracking)
clients/          executor_runner (device), simulator_runner (no device), crawl_runner (mapping)
dashboard-react/  Dashboard source; `npm run build` emits the single file the gateway serves
scripts/          ingest_all, backup/restore_neo4j, dump_prompt, analyze_batch, verify_enhancements
data/inputs/      Sample SRS + design files per app
data/fixtures/    Test media (images) re-seeded onto the device after every reset
data/backups/     Full Neo4j JSON dumps from scripts/backup_neo4j.py
tests/            79 checks across 5 modules — `python tests/run_all.py`
logs/             app.jsonl (structured), mobilerun.log (device), degradations.jsonl, trajectories/
docs/             This guide, architecture diagrams, and the implementation plan
start.sh/stop.sh  Bring up / tear down the whole local stack (Neo4j, emulator, services)
```

For architecture and the work-package history, see [NEXTGEN_IMPLEMENTATION_PLAN.md](NEXTGEN_IMPLEMENTATION_PLAN.md)
and [WORKFLOW.md](WORKFLOW.md). For exactly what the planner sends the model on each call, see
[PLANNER_PROMPT_ANATOMY.md](PLANNER_PROMPT_ANATOMY.md) or run `scripts/dump_prompt.py`.
