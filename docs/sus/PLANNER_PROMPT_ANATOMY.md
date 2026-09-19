# Planner Prompt Anatomy

Exactly what the planner sends to the model, byte by byte, in both planner modes. For *how* the
planner works, read [PLANNER.md](../PLANNER.md) first — this file is the measurement companion
to it.

Measured 20 Sep 2026 against the live `shobarkhamar` campaign (49 findings, 14 open questions,
52 observed screens, 106 requirements).

Every LLM call is **stateless**. Nothing is remembered between planning rounds; the prompt is
rebuilt from the Neo4j graph each time. "Learning" means the graph grew, so the next prompt is
assembled from better facts — not that the model recalls anything.

---

## The two modes have completely different anatomies

| | `pipeline` (default) | `tools` |
|---|---|---|
| shape | one large assembled prompt | small seed + tool results in a conversation |
| model reads retrieved content | **no** — only one-line notes per round | yes, in full |
| size driver | ~15 blocks fitted to a token budget | what the model chooses to fetch |
| calls per test case | 4 (3 small + 1 large) | 7–8 turns, 12–18 tool calls |

`PLANNER_MODE` selects which runs.

---

# Part 1 — `tools` mode

## What is fixed, per round

| Component | Chars | Notes |
|---|---:|---|
| System prompt | 1,696 | role, the rules that matter most, tool-use discipline |
| 9 tool schemas | 4,295 | names, descriptions, parameter shapes |
| of which `propose_test_case` | 2,598 | the validated terminal tool, largest by far |
| **Seed message** | **4,564** | see below |
| **Fixed total** | **~10,555** | ≈ 2,600 tokens before a single tool is called |

## The seed message, block by block

```
   71 ch  Session objective
  999 ch  Session constraints (role, account state, OUT_OF_SCOPE)
  998 ch  ## Your last 3 run(s) — of 3 executed this campaign
  196 ch  Exploration directive
 1696 ch  ## Open questions (5 shown of 14)
  605 ch  ## Execution budget
-------
 4564 ch  TOTAL
```

Two of these blocks exist because **placement beat wording** in testing:

- **Open questions (1,696 ch, the largest block).** With only a *count* here and the list behind
  `list_open_questions`, the planner called the tool on turn 1 and still opened a brand-new area.
  Moving the list inline changed the behaviour immediately. Shows 5 of 14 — the rest are a tool
  call away, and the header says so.
- **Recent runs (998 ch).** Two runs get a full interpretation; the rest are one-liners so a
  *pattern* is visible (`2 of the last 3 runs exhausted the step budget`) rather than left to be
  inferred from one sample.

## Tool results — measured live

| Tool | Chars | Cap |
|---|---:|---|
| `search_requirements` | 2,993 | 3,000 |
| `list_untested_requirements` | 2,693 | 3,000 |
| `list_open_questions` | 2,493 | 2,500 |
| `list_findings` | 2,267 | 2,500 |
| `findings_summary` | 1,574 | 2,500 |
| `list_screens` | 1,455 | 2,000 |
| `get_coverage` | 1,343 | 2,000 |
| `get_screen` | 151 | 1,200 |

**There is no global prompt budget in this mode.** Each tool caps its own output, because the
model pulls rather than being pushed everything that fits. A typical round of 12–18 calls adds
roughly 15–25k characters of *content the model actually asked for*.

### The one that scales differently

`findings_summary` is bounded by **screen × kind × status combinations**, not by finding count —
so it stays roughly flat as the graph grows:

| findings in graph | `list_findings` shows | `findings_summary` shows |
|---:|---|---|
| 28 | 8 (29%) in 2,477 ch | all 28 in ~1,210 ch |
| 300 | 8 (2.7%) | all 300 in ~2,000 ch |

That is why the rollup exists. `list_findings` also appends
`[showing 8 of 49 … narrow with screen= or group=]` so truncation is never silent.

## Total per planning round

```
fixed        ~10,555 ch   (system + schemas + seed)
tool results ~15,000–25,000 ch  across 12–18 calls
------
             ~25,000–36,000 ch ≈ 6–9k tokens, spread over 7–8 calls
```

Against a 1,000,000-token window this is under 1%. The system is **cap-limited, not
context-limited**, and deliberately so.

---

# Part 2 — `pipeline` mode (still the default)

One large prompt assembled from ~15 candidate blocks, fitted highest-priority-first into
`PROMPT_BUDGET_TOKENS` (50,000) by `planner/budget.py`.

| priority | blocks | dropped first? |
|---|---|---|
| 0 | requirements, SRS context, UI context | never |
| 1 | what previous runs established, executed titles | third |
| 2 | defect history, risk, anomalies, nav path, failed nav, strategy | second |
| 3 | UI overview, transitions, failed titles | first |

Measured generation prompts on this project ranged **15,530 → 51,002 chars**. The largest was
51,002, of which **35,233 (69%) was the "what previous runs established" block** when it still
inlined the investigator's prose reports. That block is now built from **findings** and sits at
roughly 2,000 chars — the single biggest reduction in this design.

### What still grows here

Only two blocks scale with session length, both capped:

```
prompt ≈ fixed + bounded + 90 × min(tests, 120) + (findings block, ~2,000)
```

When a cap binds the excess is **dropped, not summarised**: past 120 tests the planner can
regenerate an old test because it left the dedup list. Aggregates (coverage, risk, strategy,
error patterns) are computed over the whole database, so the *shape* of history survives — only
per-test detail is lost.

---

## Reproducing these numbers

```bash
# pipeline mode: dump the assembled prompt and a per-block table
./venv/bin/python scripts/dump_prompt.py

# tools mode: every LLM call of one generation, input and output, in order
less logs/planner/TC-001.txt
```

Note `logs/planner/*.txt` truncates each logged message at 4,000 chars, so for exact seed sizes
rebuild it with `agent_loop._seed_user_message(...)` rather than reading the log.

---

## Gaps — data in the graph that still never reaches a prompt

| Available | Why it matters |
|---|---|
| Screenshots (`data/appmodel/<project>/*.png`) | `get_screen` returns text only. Pure-Compose screens expose no control names structurally, so they are effectively invisible. Pipeline mode *does* attach one screenshot at generation. |
| Per-step device actions (`logs/trajectories/*/trajectory.json`) | The investigator reads them; the planner never sees which interactions provably worked. |
| `Entity` nodes, defect `root_cause_category`, per-test effectiveness | Computed and stored, unused in generation. |
| Role of the observer | Nothing records which role saw a screen or finding — see [ROADMAP.md](../ROADMAP.md). |
