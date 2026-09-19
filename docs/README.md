# Documentation index

Read in this order.

## Start here

| doc | what it answers |
|---|---|
| **[WORKFLOW.md](WORKFLOW.md)** | How the whole system works, end to end. **Read this first.** |
| **[GETTING_STARTED.md](start/GETTING_STARTED.md)** | First-time setup: Neo4j, emulator, on-device portal, `.env`, troubleshooting |
| **[ROADMAP.md](ROADMAP.md)** | Current status, what the agent has found, what to build next, known limitations |

## The three agents, in depth

| doc | covers |
|---|---|
| [System_Architecture.md](start/System_Architecture.md) | The **pipeline** planner's LangGraph state machine (`PLANNER_MODE=pipeline`, the default) |
| **[PLANNER.md](PLANNER.md)** | **How the planner works in detail** — the tool loop, the 9 tools, the validating proposal gate, open questions, and the pipeline planner it replaces |
| [PLANNER_REDESIGN.md](PLANNER_REDESIGN.md) | The **tool-calling** planner (`PLANNER_MODE=tools`) — why the old retrieval loop was blind, and the validating proposal gate |
| [PLANNER_PROMPT_ANATOMY.md](sus/PLANNER_PROMPT_ANATOMY.md) | What the pipeline planner actually sends to the LLM, block by block, with measurements |
| [INVESTIGATOR.md](INVESTIGATOR.md) | How a device trajectory becomes durable findings: the taxonomy, dedup, config traps |

## Reference and background — `docs/sus/`

Moved here as *suspected redundant or outdated*. Assessed 20 Sep 2026:

| doc | verdict |
|---|---|
| [NEXTGEN_IMPLEMENTATION_PLAN.md](sus/NEXTGEN_IMPLEMENTATION_PLAN.md) | **Keep — not redundant.** `WP1`–`WP9` and `REQ-301`–`308` appear **160 times** in the code (`REQ-303` ×20, `WP6` ×19, `REQ-301` ×19 …). This is the only decoder for them; delete it and those comments become unreadable. Historical as a *plan*, load-bearing as a *glossary*. |
| [PLANNER_PROMPT_ANATOMY.md](sus/PLANNER_PROMPT_ANATOMY.md) | **Updated 20 Sep 2026 — no longer stale.** Now covers both planner modes with live measurements from this project. Companion to [PLANNER.md](PLANNER.md). |
| [research-agentic-exploration.md](sus/research-agentic-exploration.md) | **Keep.** Thesis positioning, and §3 is the offline eval-harness argument — the measurement gap in [ROADMAP.md](ROADMAP.md) §4, which is the biggest outstanding item on the project. Still the "why" behind the work. |

## Setup and architecture — `docs/start/`

| doc | covers |
|---|---|
| [GETTING_STARTED.md](start/GETTING_STARTED.md) | First-time setup: Neo4j, emulator, on-device portal, `.env`, troubleshooting |
| [System_Architecture.md](start/System_Architecture.md) | Technical reference: **how the planner, executor and investigator work together**, components, both planners, the graph schema |
| [neo4j_setup.md](start/neo4j_setup.md) | Neo4j install options and connection settings |

## Reports (deliverables, not maintained docs)

`SAMSUNG_PROGRESS_REPORT.{pdf,html}` · `SYSTEM_REVIEW.{pdf,html}` · `USER_GUIDE.pdf` ·
`Presentation-Exploratory-Automated-Testing.pdf`

⚠️ `SAMSUNG_PROGRESS_REPORT` contains claims now known to be wrong — its "zero silent fallbacks"
line read a counter that could not see the executor process. See [ROADMAP.md](ROADMAP.md) §5.

---

### Removed on 6 Sep 2026 (recoverable via `git checkout <commit> -- <path>`)

| file | why |
|---|---|
| `improvement.md` | Analysed the 3-round retrieval loop and its lack of memory. Superseded by [PLANNER_REDESIGN.md](PLANNER_REDESIGN.md) §1, which diagnoses the same defect with measurements |
| `roadmap.md` | Phase 0 was a dead venv on one machine; later phases (runtime context) are built. Superseded by [ROADMAP.md](ROADMAP.md) |
| `screengraph-guide.md` | Implementation guide for the ScreenGraph — now built as the Live App Model. Contained verified mobilerun 0.6.8 API field shapes if you ever need them back |
| `PLANNER_IMPROVEMENTS_FUTURE.md` | Folded into [ROADMAP.md](ROADMAP.md) §3 |
