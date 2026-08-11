# Practical Improvement Roadmap — Get It Working, Then Make It Better

Goal: a working, reliable exploratory-testing pipeline — not a research paper. Tasks are
ordered by **value per effort**. Do them top to bottom; each has a concrete **verify** step
so you know when it's done.

> Background/deeper "why" and references live in [`research-agentic-exploration.md`](research-agentic-exploration.md).
> This file is the one to actually work from.

---

## Phase 0 — Get it running again (blocking; nothing works until this is done)

The venv's Python interpreter is dead (Homebrew removed the `python@3.13` it was built on),
so **nothing in the project can execute right now**. Fix the environment first.

| # | Task | Verify |
|---|---|---|
| 0.1 | Rebuild the venv on a stable Python 3.13 (recommend `uv`): `uv python install 3.13` → `uv venv --python 3.13` → `uv pip install -r requirements.txt` | `./venv/bin/python -c "import mobilerun, langgraph, neo4j; print('ok')"` prints `ok` |
| 0.2 | Start Neo4j + both services (`rag_api` :9010, `gateway` :9100) | both `/health` endpoints return `status: ok`; gateway shows the model backend |
| 0.3 | **Validate the planner loop with the simulator first** (`clients/simulator_runner.py`) — no device needed. This isolates planner/RAG bugs from device/Droidrun bugs. | ingest sample SRS+Figma → one `next-testcase` returns a JSON test with non-empty `steps` |
| 0.4 | Only then bring up the real-device path: emulator + `appium` + `adb`, run `clients/executor_runner.py` for 1 round | `preflight()` passes and one test executes on the emulator with a logged verdict |

**Do not** start changing code until 0.3 passes — you need a running baseline to test fixes against.

---

## Phase 1 — Fix the bugs that make the system lie to itself (correctness)

These are small, high-impact, and don't need the device. Do them right after Phase 0.

### 1.1 Stop pre-logging generated tests as `pass` — *the big one*
Every generated test is written to Neo4j with `verdict:"pass"` **before it runs**
([`planner/langgraph_agent.py:401-415`](../planner/langgraph_agent.py#L401-L415)). This inflates
`coverage_pct` and fills `recent_tests` with fiction — and coverage is exactly what steers the
*next* test. So the "adaptive" loop is partly reacting to results that never happened.
- **Fix:** either don't pre-log at all (let the executor log the only verdict), or log with a
  distinct `verdict:"planned"` that `coverage.compute_coverage_map` **excludes** until a real run lands.
- **Verify:** generate 3 tests without executing → coverage/`recent_tests` do **not** count them as passed.

### 1.2 Generalize the hardcoded app name
[`clients/executor_runner.py:234`](../clients/executor_runner.py#L234) hardcodes
`"Open the Contacts app on this device."` — wrong for any non-contacts app despite the
"app-agnostic" claim. Use `APP_NAME` / `TARGET_APP_PACKAGE`.
- **Verify:** set `APP_NAME=Calculator` → the Droidrun goal text references Calculator, not Contacts.

### 1.3 Kill the latent `NameError`
[`planner/langgraph_agent.py:407`](../planner/langgraph_agent.py#L407) uses `time.time()` but
`time` is never imported. It's in an effectively-dead fallback branch — either `import time` or
delete the dead branch.
- **Verify:** `grep -n "import time" planner/langgraph_agent.py` (or the branch is gone).

### 1.4 Pin the dependency + fix the stale docstring
[`requirements.txt:7`](../requirements.txt#L7) has `droidrun` unpinned; a future `pip install`
could pull a breaking version. Pin `droidrun==0.6.8`. Also fix the "Droidrun v0.5.7 API"
docstring in `executor_runner.py` (it's actually 0.6.8-compatible).
- **Verify:** `pip install -r requirements.txt` on a clean venv still gives 0.6.8.

---

## Phase 2 — Cheap quality wins (better tests, still no big new infra)

### 2.1 Semantic dedup instead of title Jaccard
[`planner/textutil.py:61`](../planner/textutil.py#L61) blocks duplicates by word-overlap on
*titles* — too crude (misses reworded duplicates, wrongly blocks distinct-but-similar tests).
You already ship `fastembed`. Embed `title + steps` and dedup on cosine similarity instead.
- **Verify:** two reworded-but-identical tests get flagged; two genuinely different boundary
  tests on the same field do not.

### 2.2 Steer by uncovered *requirements*, not `area` strings
You already compute graph-native requirement coverage (`/coverage/requirements`, `COVERS` edges)
but the exploration directive uses free-text `area` strings instead. Feed the list of
**uncovered requirements** into the next objective.
- **Verify:** after covering requirement FR-3, the directive stops targeting it and moves to an uncovered one.

### 2.3 Lightweight independent oracle
Right now Droidrun both drives the test and judges pass/fail from its own goal interpretation
([`executor_runner.py:353`](../clients/executor_runner.py#L353)) — a buggy app can still "succeed."
Have the planner emit a **concrete, checkable expected condition** (e.g. "a Toast 'Saved' appears"
or "resourceId `contact_row` count increased"), and verify it from the post-action a11y tree
independently of Droidrun's self-report.
- **Verify:** on a deliberately broken flow, the oracle reports `failed` even when Droidrun says `success`.

---

## Phase 3 — Give the planner runtime context (your "auto-Figma" idea)

This is the big practical upgrade and directly solves "detailed Figma is hectic." Build the app
map automatically from what MobileRun already sees, and feed it to the planner so it stops
planning blind against design-time specs.

### 3.1 ScreenGraph, after-run first
New module `planner/screen_graph.py`.
- **Node = a screen**, keyed by `(activity_name, hash(sorted resourceId/text of the a11y list))`
  (the a11y list is already flat — cheap to hash).
- **Edge = the action** that changed the signature, from `action_history` (`{"action":..., ...}`)
  + `action_outcomes`.
- **Source (after-run):** read `agent.trajectory.ui_states` (per-step a11y) zipped with
  `shared_state.action_history`. Requires `config.logging.save_trajectory != "none"` — set it in
  `execute_test_on_device`.
- **Verify:** run one session, dump the graph — nodes match the screens you saw, edges match the taps.

### 3.2 Optional: image context via one-shot captions
MobileRun already captures screenshots. When a **new** screen appears, run a VLM once to store a
short caption on the node ("Login: email + password fields, 'Sign in' button"). Keep the PNG path
so you can go fully multimodal later if a backend supports it. Cheap, model-agnostic, and adds the
visual info the a11y tree misses (layout, empty/error states).
- **Verify:** each screen node has a caption; planner prompt includes them.

### 3.3 Feed the graph into the planner
Add `ScreenGraph.to_prompt_text()` as a new context block in
`langgraph_agent.bootstrap_context` / `generate_testcase`. Now the planner references screens it
has *actually reached* and can spot screens the SRS implies but you've never visited.
- **Verify:** generated steps reference real, observed element labels; the planner flags an unreached spec'd screen.

---

## Phase 4 — Only if Phase 3 works well (optional, incremental)

- **Live loop:** instead of after-run, stream `RecordUIStateEvent` + `ExecutorResultEvent` during
  the run and replan on *new screen* or *failure* (not every step — too slow). Turns the planner
  into a real-time explorer. See the research doc.
- **Simple novelty bias:** in the directive, prefer screens/actions with low visit counts
  (`1/√count`) so it naturally pushes into unexplored territory. No ML needed to start.

---

## Knowing whether you're actually improving (do this early)

You don't need a research benchmark, but you need *some* repeatable number or you're guessing.
Minimal version: run N rounds on the **same emulator snapshot + same app**, and log to a CSV:

- unique screens reached (from ScreenGraph)
- crashes / ANRs seen
- pass/fail counts
- requirement coverage %

Re-run it before/after each change. If a change doesn't move these, it didn't help. `submission.csv`
already exists — this can slot in next to it.

---

## Suggested first working session

1. Phase 0.1–0.3 — get the simulator loop green.
2. Phase 1.1 + 1.3 — stop the fake-pass logging, kill the NameError.
3. Re-run the simulator loop — confirm coverage now reflects reality.
4. Then bring up the device (0.4) and tackle Phase 2 onward.

Stop after each task and run its **verify** — that's how you keep the system working while changing it.
