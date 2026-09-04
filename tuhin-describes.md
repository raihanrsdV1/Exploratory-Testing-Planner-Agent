# Session summary — tuhin-working

Six tasks worked on this session: a target-configuration UI, a Playwright web-agent
improvement, sharper focus on failed regions, a fix for the 50-step limit, and
routing execution notes through real RAG retrieval instead of dumping them into
the planner's context. (The 2-minute demo video was out of scope for this tool.)

All changes are backed by the existing test suite plus new tests —
`./venv/bin/python tests/run_all.py` — **7/7 modules, 219/219 checks passing**.

---

## 1. Target-configuration UI

A new **Targets** tab in the React dashboard (`dashboard-react/src/Targets.jsx`),
switchable from the existing **Dashboard** tab via a tab bar in the header.

- List every profile in `targets/profiles/*.json`, including invalid ones (with
  their validation errors, so a broken profile can still be opened and fixed).
- A form generated from `targets/schema.py`'s shape: name/kind/project/display
  name, the web or android block depending on `kind`, login credentials,
  knowledge paths (SRS/Figma/defects), run budget (rounds/max_steps/timeout/
  clean_slate/self_heal), and an optional model override.
- **Validate** button — checks without saving (inline error list).
- **Save** button — `PUT`s the profile; the server re-validates and refuses to
  write an invalid one (422 with the error list).
- **Run** button — launches `py -m targets.run <name>` and polls its status
  (pid / running / exit code) until it finishes.

New backend: `gateway/targets_api.py` (business logic) wired into
`gateway/main.py` as `GET /targets`, `GET /targets/{name}`,
`POST /targets/validate`, `PUT /targets/{name}`, `POST /targets/{name}/run`,
`GET /targets/{name}/run-status`.

**Why a run is a subprocess, never in-process:** `targets/run.py` itself warns
that `settings.py` is read once at first import and is process-global. The
gateway has already imported `settings` at its own startup, so running a
profile in-process would silently leak that profile's config into the
gateway's own process — and therefore into every other request. `Popen`-ing
`python -m targets.run <name>` is exactly what a human does at the CLI, kept
that way on purpose.

Verified end-to-end against the live gateway/rag_api: listed the real
profiles, loaded and saved a scratch one, launched an actual run, watched it
fail cleanly at the "Playwright not installed" preflight check, then cleaned
up the scratch artifacts.

---

## 2. Playwright web-agent: a second livelock guard ("wandering")

`web_player/agent.py` already detected the *exact-repeat* case: the same
action against an unchanged page, 4 times in a row → `NAVIGATION_LIVELOCK`.

It did **not** catch *wandering*: a different action every turn (e.g.
scrolling further and further) while the page itself never changes at all —
so a run could burn its entire step budget unnoticed and get reported as
`ASSERTION_FAILURE` (counted as a discovered bug) or `STEP_LIMIT_EXCEEDED`.

Added a second, content-only signature (`_content_signature`, ignoring the
action) tracked alongside the existing one. `WEB_STALL_STEPS` (new setting,
default 6) consecutive steps with no change to the page — regardless of what
action was tried — now ends the run early, correctly classified as
`NAVIGATION_LIVELOCK` (an agent fault, not a defect).

Tested in `tests/test_web_player.py` with a fake chat client that replies
with a different action every turn against a page that never changes —
confirms the new guard (not the old one) fires, at the configured step, with
the right classification. 93 → 97 checks.

---

## 3. Focus on failed regions

Three layers, per what was actually missing (confirmed by reading the code
first, not guessing):

- **Cross-run prioritization** (`planner/coverage.py`) already investigated
  hot spots before expanding into new areas — that part was working.
- **Regression risk scores were computed but never actually competed for
  priority.** `rag_api/risk.py`'s `regression_risk_score` was rendered into
  the prompt as a separate informational block, parallel to but disconnected
  from the `[PRIORITY]` ordering `build_exploration_directive` produces. Now
  the top risk-ranked areas (minus ones already flagged as session hot spots)
  get their own `[RISK]` priority line, so a defect-dense area surfaces even
  before it has failed a test *this* session.
- **Within-run drill-down**, added narrowly: `settings.verification_block()`
  (shared by both players) now tells the agent to confirm a suspected defect
  with **one** more targeted check on the same screen before reporting it —
  deliberately bounded so it can't turn into the wandering behaviour the
  guard above exists to stop.

New tests in `tests/test_planner_coverage.py` cover the `[RISK]` line
appearing/not-repeating/degrading cleanly.

---

## 4. Why the 50-step limit gets hit

Diagnosed both players before changing anything:

- **Web:** the exact-repeat guard was too narrow — fixed by #2 above.
- **Android:** `clients/executor_runner.py` has **no** in-loop stuck
  detection at all today. It used to — a real livelock detector with cycle
  detection, content-vs-screen discrimination, `APP_UNRESPONSIVE` vs
  `NAVIGATION_LIVELOCK` classification (`tests/test_livelock.py`, 103 checks)
  — but it was **deliberately removed** on 2026-08-30: it introduced a real
  concurrency bug where the "cancelled" agent kept running alongside the next
  test. That history is recorded in `docs/PLANNER_IMPROVEMENTS_FUTURE.md`,
  which also names the next step: since executor-side cancellation is a known
  trap, steer the *planner* away from writing tests too large for a screen
  instead.

Implemented exactly that (proposals #1+#2 from that doc, built together as
suggested):

- `rag_api/learning.py`: `mine_agent_difficulty()` — screens where recent
  runs stalled/timed out (2+ occurrences, `AGENT_FAULT ∪ {STEP_LIMIT_EXCEEDED}`
  from the shared taxonomy in `settings.py`), plus typical step cost per
  feature area from real execution history.
- New endpoint: `GET /execution/agent-difficulty`.
- New prompt block, **"Known Agent Difficulty (steer test design, not defect
  evidence)"** — deliberately kept separate from anything that reads as app
  defect evidence, and from `planner/coverage.py`'s `NON_INFORMATIVE_ERRORS`
  filter (which is correct to exclude these from the bug oracle — this reuses
  the same exclusion, for a different purpose).

---

## 5. Investigator output → RAG (was overwhelming the planner's context)

Traced the actual path before changing anything: execution notes were
already reasonably bounded (truncated to 1200 chars, capped at 25) — but they
reached the planner via `/context/brief`, a plain `LIMIT 100` Cypher scan,
**not** real retrieval. The "investigator" (executor/web player) writes a
compact note; it just wasn't being *ranked* by relevance to what the planner
is about to test next, only by recency.

- `TestRun` notes are now embedded at write time (`rag_api/main.py`'s
  `/tests/log`), using the same `rag_api/embeddings.py` backend already used
  for SRS chunks and requirements.
- New vector index `testrun_embedding`, created the same way as
  `chunk_embedding`/`requirement_embedding`.
- New endpoint `POST /execution/notes/retrieve` — semantic search over past
  failure notes, filtered to `verdict='failed'`.
- `planner/context_builders.build_failure_context()` now tries semantic
  retrieval (ranked by relevance to the session objective) first, falling
  back to the original recency view when embeddings are disabled or the call
  fails — so behaviour is unchanged for a project running
  `EMBEDDING_BACKEND=none`.

---

## Files touched

**New:**
- `dashboard-react/src/Targets.jsx`
- `gateway/targets_api.py`
- `tests/test_planner_coverage.py`

**Modified:**
- `dashboard-react/src/App.jsx`, `dashboard-react/src/styles.css`
- `gateway/main.py`
- `planner/context_builders.py`, `planner/coverage.py`,
  `planner/langgraph_agent.py`, `planner/prompts.py`, `planner/rag_client.py`
- `rag_api/learning.py`, `rag_api/main.py`, `rag_api/schemas.py`
- `settings.py`
- `tests/test_web_player.py`
- `web_player/agent.py`

## One thing still to do

`gateway` and `rag_api` are running without `--reload` and need a restart to
pick up everything except the dashboard HTML (that part is served fresh per
request and is already live). Not restarted automatically — that's your
call, since both processes may be mid-use.
