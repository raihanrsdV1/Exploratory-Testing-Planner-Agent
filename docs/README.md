# Documentation index

Read in this order.

## Start here

| doc | what it answers |
|---|---|
| **[WORKFLOW.md](WORKFLOW.md)** | How the whole system works, end to end. **Read this first.** |
| **[GETTING_STARTED.md](GETTING_STARTED.md)** | First-time setup: Neo4j, emulator, on-device portal, `.env`, troubleshooting |
| **[ROADMAP.md](ROADMAP.md)** | Current status, what the agent has found, what to build next, known limitations |

## The three agents, in depth

| doc | covers |
|---|---|
| [System_Architecture.md](System_Architecture.md) | The **pipeline** planner's LangGraph state machine (`PLANNER_MODE=pipeline`, the default) |
| [PLANNER_REDESIGN.md](PLANNER_REDESIGN.md) | The **tool-calling** planner (`PLANNER_MODE=tools`) — why the old retrieval loop was blind, and the validating proposal gate |
| [PLANNER_PROMPT_ANATOMY.md](PLANNER_PROMPT_ANATOMY.md) | What the pipeline planner actually sends to the LLM, block by block, with measurements |
| [INVESTIGATOR.md](INVESTIGATOR.md) | How a device trajectory becomes durable findings: the taxonomy, dedup, config traps |

## Reference and background

| doc | covers |
|---|---|
| [neo4j_setup.md](neo4j_setup.md) | Neo4j install options and connection settings |
| [NEXTGEN_IMPLEMENTATION_PLAN.md](NEXTGEN_IMPLEMENTATION_PLAN.md) | **Historical.** Samsung's ETA-REQ-301–308 roadmap. Keep it — `WP*` / `REQ-30*` appear ~160 times in the code and this is the decoder |
| [research-agentic-exploration.md](research-agentic-exploration.md) | Thesis-level framing: live replanning, novelty-driven exploration, and the eval-harness argument |
| `architecture_flowchart.mmd`, `final_updated_flow_chart.png` | Architecture diagrams |

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
