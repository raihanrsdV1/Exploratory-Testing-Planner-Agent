# Capability inventory — what the system actually does, and where

Compiled 20 Sep 2026 by reading the code and `docs/` on every branch. Purpose: stop the
report under-claiming. Each row says **where the capability lives** and **whether the report
mentions it**, so Chapters 4 and 5 can be written from fact rather than memory.

## Branch state (20 Sep 2026)

| branch | vs `main` | newest | holds |
|---|---|---|---|
| `origin/main` | — | 12 Sep | integration point; everything below unless stated |
| `origin/kmazd-v2` | **5 ahead**, 35 behind | **20 Sep** | Azad: findings lifecycle, proposal gate, 9-tool planner, rewritten `docs/` |
| `origin/experimental` | 8 ahead, 0 behind | 13 Sep | Jonayed: web reliability + exploration graph + file upload (superset of `jonayed`) |
| `origin/jonayed` | 6 ahead, 0 behind | 12 Sep | subset of `experimental` |
| `origin/Niloy_44-working` | 7 ahead, 0 behind | 12 Sep | Niloy: 337-line security/code review; planner-trace tail fix |

All three of Jonayed's/Niloy's branches fast-forward onto `main`. `kmazd-v2` is a one-way
feeder: our work flows to `main`, `main` never flows back, so `kmazd-v2` has **no**
`targets/`, `web_player/` or `report_api.py`.

> **Report rule:** only claim what is on `main` **or** `kmazd-v2` as delivered. Work that
> exists only on `experimental`/`Niloy_44-working` is not yet part of the system.

## The executor is **mobilerun**, not DroidRun

`from mobilerun import MobileAgent, AndroidDriver, load_llm, MobileConfig, AgentConfig`
plus `mobilerun_core_cli.portal`. Version 0.6.8. Only a stale function name
(`build_droidrun_goal`) and one `requirements.txt` line still say `droidrun`.
The on-device companion is `com.mobilerun.portal`: it supplies both the accessibility
service the executor drives **and** an input method that makes text entry reliable. Without
it the executor falls back to shell input, which corrupts multi-field forms.

## Planner — two modes behind one contract (`PLANNER_MODE`)

`pipeline` is **still the default**; `tools` is the replacement, not yet proven over a long
campaign. Same output contract, so the executor is unaffected and the two are A/B-comparable
by one environment variable.

**Tool-calling planner** (`planner/agent_loop.py`, `planner/tools.py`) — **nine** tools:
`search_requirements`, `list_untested_requirements`, `get_screen`, `list_screens`,
`findings_summary`, `list_findings`, `list_open_questions`, `get_coverage`, `get_nav_path`.
A tool whose knowledge source is disabled or empty is **never registered**, so the model
cannot be misled by an empty answer. Round shape: 7–8 turns, 12–18 tool calls, ~67 s.
`MAX_TURNS` 12, `FORCE_PROPOSE_AFTER` 6 (after which `tool_choice` is pinned to
`propose_test_case`, because a model on `auto` investigates to the ceiling and never commits).

**`propose_test_case` — the validating gate.** Server-side, before anything reaches a device:
screen hint names a real observed `UIState`; every requirement id exists; not a semantic
duplicate (Jaccard + embedding cosine); not in `OUT_OF_SCOPE`; preconditions creatable through
the UI. Each rejection returns an actionable reason. `MAX_REJECTIONS` 3, then best effort is
accepted and a `planner_proposal_unvalidated` degradation is recorded, so a round never
returns nothing.
**Measured:** under `pipeline` the named screen matched a real observed screen **~16%** of
the time; under `tools`, **1 rejection across 8 test cases**.

**Open questions.** `UNVERIFIED`, `SUSPECTED_DEFECT`, `SPEC_VIOLATION` findings start *open*.
The planner may target one via `addresses`, spending one of its attempts. Slots are reserved
half-and-half between started and fresh questions; within the started pool, most-attempted
first. At `MAX_FINDING_ATTEMPTS` (3) a question auto-closes **inconclusive** — a result for a
human, not an endless retry. A later run that answers it flips it to *resolved*.

**Three feedback channels, deliberately distinct:** findings (about the app, durable, survive
`CLEAN_SLATE`) · coverage (aggregate, per campaign) · recent runs (about our attempt, single
run). The last two newest runs get a full interpretation; older ones are one-liners so a
*pattern* is visible, with a warning line that fires only on real repetition.

## Capabilities the report was missing

| capability | code | on |
|---|---|---|
| mobilerun portal bootstrap (`ensure_portal_ready`) | `clients/executor_runner.py` | main |
| Vision: screenshot to executor **and** planner; pHash for state identity | `EXECUTOR_VISION=True`, `langgraph_agent.py:414`, `PHASH_MATCH_DISTANCE=6` | main |
| 9-tool planner + validating gate + open questions | `planner/agent_loop.py`, `tools.py`, `proposal.py` | kmazd-v2 |
| SRS drift detection on re-ingest | `GET /srs/drift` | main |
| Business-logic rules with source-chunk provenance | `GET /business-logic/rules` | main |
| Best-of-N extraction with a judge model | `EXTRACTION_SAMPLES`, `EXTRACTION_JUDGE_MODEL` | main |
| Simulator runner (loop with no device) | `clients/simulator_runner.py` | main |
| CAPTCHA pause-and-notify handoff | `web_player/agent.py` | main |
| Single-run concurrency lock | `web_player/runlock.py` | main |
| Knowledge upload + per-target ingest from dashboard | `gateway/targets_api.py` | main |
| Graph visualise / terminal / cypher endpoints | `rag_api/main.py` | main |
| `/chat` endpoint | `gateway/main.py` | main |
| `GET /campaigns` preserving aggregates across resets | `rag_api/main.py` | kmazd-v2 |
| Web evidence reviewer, API route registry, passive exploration graph, file upload | `web_player/review.py`, `api_registry.py`, `exploration/`, `fixtures.py` | **experimental only — do not claim** |

## Real findings the agent has produced (for Chapters 5 and 7)

Target: `shobarkhamar` (`com.tirzokpvt.shobarkhamar`, Flutter, livestock marketplace),
Android 14 emulator, signed in as farmer/seller, **no Figma export** — SRS plus its own
observations only. `times_seen` counts *independent* re-derivations across runs.

Candidate defects, none human-confirmed yet:
- Language dropdown opens only near the chevron; tapping the button's text area does nothing — **3×**
- System BACK on the Menu screen exits the app to the OS home screen — 1×
- BACK from the Disease List context exits the app entirely — 1×
- Selecting English leaves the UI in Bengali on subsequent screens — 1×

`AGENT_DIFFICULTY` findings (our own weaknesses, deliberately kept out of the bug oracle, and
the evidence base for the "map of where an interface is hard to use" argument):
misidentifies the `ব্লগ` bottom-nav tab (3×) · repeats a menu open/close cycle ~7× with no
progress · reaches the right screen then navigates away (4×) · taps inert controls repeatedly
rather than concluding they are inert.

## Known limitations, as the team records them

1. Autonomy **~50%**; most losses are our agent on multi-step forms, failing to scroll back up
   to a required field above its position.
2. `uiautomator` and the mobilerun portal conflict — two accessibility clients registering at
   once. Our tooling, not the app.
3. State-identity thresholds tuned on one app's states. Evidence, not proof.
4. Test ids restart at TC-001 each campaign, which breaks campaign-over-campaign comparison.
5. The graph stores title + verdict per test, not steps/screen/expected result.
6. ETA-REQ-301 (defect intelligence) never run on real data.
7. ETA-REQ-304 (cross-application transfer) unvalidated.
8. `docs/reports/SAMSUNG_PROGRESS_REPORT.pdf` contains a known-wrong "zero silent fallbacks"
   claim — a counter that could not see the executor process.

## The two experiments that would close the measurement gap

**A. Campaign-over-campaign** (a few dollars, an afternoon). Two campaigns back to back on one
app, graph kept. Measure whether campaign 2 cites more real screens, hits more untested
requirements, produces fewer `UNVERIFIED` findings. Now possible because findings survive
`CLEAN_SLATE` and `GET /campaigns` preserves aggregates past the reset.

**B. Seeded-defect build** (under $1). 8–12 known defects with a ground-truth list; three arms:
full agent / memory disabled / random baseline. Yields precision, recall, F1 and an ablation
for the "gets smarter" claim. `PLANNER_MODE` makes the planner arm a flag flip.

## Authoritative docs to read before writing Chapters 4 and 5

On `kmazd-v2`, rewritten 20 Sep: `docs/WORKFLOW.md` (read first) · `docs/PLANNER.md` (the tool
loop, nine tools, the gate) · `docs/INVESTIGATOR.md` (364 lines: taxonomy, dedup, config traps)
· `docs/start/System_Architecture.md` (319 lines) · `docs/ROADMAP.md` (status, findings,
limitations) · `docs/sus/PLANNER_PROMPT_ANATOMY.md` (the literal prompt bytes, both modes).
