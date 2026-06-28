# Planner Improvement Plan — Exploratory Testing Agent

## Current Architecture (What You Have)

The planner is a **3-round RAG retrieval loop**. For every test case request:

1. **Load global summary** → SRS summary, Figma overview, recent test history
2. **Retrieval planning loop (max 3 rounds)** → LLM decides what to fetch (SRS chunks, Figma UI elements, Figma flow edges), Gateway fetches it from Neo4j, repeat
3. **Final test case generation** → one prompt using all collected context, outputs a JSON test case
4. **Duplicate check** → if too similar, retry with alternate screens

**The core weakness**: The planner treats every round as *"generate one test case"*, discards all state after, and starts fresh next call. It has no memory of *why* it went where it did, no model of the app's state, and no exploration strategy — it just tries not to repeat titles.

---

## What "Exploratory Testing" Actually Requires

True exploratory testing is about **discovering unknown behaviour** by reasoning about what hasn't been tested yet, what failed, and what the current app state implies. It needs:

| Capability | Current Planner | What's Needed |
|---|---|---|
| App state awareness | ❌ None | Track which screens were visited, which paths were taken |
| Strategic coverage | ❌ Jaccard title dedup only | Explicit model of tested/untested areas |
| Failure-driven pivoting | ⚠️ Loosely (adds "failed tests" to prompt) | Structured failure analysis and targeted follow-up |
| Multi-step exploration chains | ❌ None | Generate sequences of tests that explore one area deeply |
| Self-critique | ❌ None | Agent evaluates whether its own test is novel and meaningful |

---

## Proposed Improvements (Modern Agentic Techniques)

### 1. Coverage Graph — Replace Title Dedup with Area-Level Coverage Map

**Current problem:** The planner avoids duplicates by checking Jaccard similarity of test case *titles*. This is purely surface-level. Two tests called "Verify Login" and "Test Authentication" are flagged as similar, but "Test Login with Empty Password" and "Test Login with Very Long Password" are both allowed even though they test the same boundary.

**What to do:** Maintain a **Coverage Map** in Neo4j — a structured record of `(screen, action_type, input_class)` tuples that have been tested. Before generating a test, the LLM receives the coverage map and must produce a test that fills a *gap* in it.

```
Coverage Map Example:
  - create_contact | tap('Save') | empty_required_field → tested (FAILED)
  - create_contact | tap('Save') | valid_all_fields → tested (PASS)
  - create_contact | input('Email') | invalid_format → NOT TESTED ← pick this
```

This moves from "don't repeat titles" to "don't repeat behaviours".

---

### 2. ReAct-style Agent Loop — Give the Planner a Real Action Space

**Current problem:** The planner's action space is binary: `retrieve` or `produce_testcase`. After retrieval, it has no way to reason about what it discovered or reflect on whether the context is actually sufficient.

**What to do:** Expand the action space to a proper **ReAct (Reason + Act)** loop with explicit Thought, Action, Observation steps:

```
Available Actions:
  - retrieve_srs(query)        → fetch SRS rules for a domain
  - retrieve_ui(screen_name)   → fetch interactive elements of a screen
  - retrieve_flow(screen_name) → fetch navigation edges from a screen
  - retrieve_coverage(area)    → fetch what has/hasn't been tested in an area
  - inspect_failure(test_id)   → fetch full notes from a failed test case
  - produce_test(...)          → finalize and output test case
```

Adding `retrieve_coverage` and `inspect_failure` is the key upgrade — they let the agent actively interrogate the test history rather than just reading a summary.

**Concrete implementation:** These are all RAG API calls. The gateway already calls `_get_srs_and_history`, `_get_screen_elements`, and `_get_figma_transitions`. You just need to add two new RAG endpoints:
- `GET /coverage/map?project=X&area=Y` → returns structured coverage tuples
- `GET /tests/failure-detail?project=X&test_id=Y` → returns full notes of a failed run

---

### 3. Chain-of-Thought Rationale — Force the LLM to Justify its Test Choice

**Current problem:** The LLM is asked to "produce a test case" and outputs a JSON blob. There is no step where it explains *why* this test is valuable, *what gap* it fills, or *what risk* it covers. This makes the output brittle and hard to audit.

**What to do:** Before generating the final JSON, add a required **rationale step** where the LLM must produce a short structured reasoning:

```json
{
  "coverage_gap_identified": "Email input field has never been tested with Unicode characters",
  "risk_assessed": "High — email is used for account recovery; silent failure would be critical",
  "test_strategy": "Boundary value analysis on email field character encoding",
  "test_case": { ... }
}
```

This forces the model to think before it writes, and gives you a human-readable audit trail for every generated test.

---

### 4. Exploration Strategy Modes — Add a Strategy Parameter

**Current problem:** Every call to `/agent/next-testcase` uses the same objective string. The planner has no concept of "where am I in the testing session" or "should I go deep or broad right now".

**What to do:** Add an `exploration_strategy` parameter with three modes:

| Mode | Behaviour | When to Use |
|---|---|---|
| `breadth_first` | Maximize screen/area coverage — test one case per area before repeating | Early in a session |
| `depth_first` | Drill into failing areas — generate multiple edge cases for any failed screen | When failures are found |
| `boundary_probe` | Target boundary conditions — empty inputs, max-length, invalid types | When core flows pass |

The gateway passes the strategy to the LLM via the system prompt, shaping which gaps it prioritizes.

---

### 5. Session Memory — Persist Planning State Between Calls

**Current problem:** Every `/next-testcase` call is completely stateless. The planner throws away all retrieval context and reasoning after generating a test. The next call starts from scratch.

**What to do:** Add a lightweight **session object** stored in Neo4j:

```json
{
  "session_id": "sess-001",
  "project": "contacts-app",
  "strategy": "depth_first",
  "current_focus_area": "create_contact",
  "consecutive_tests_in_area": 2,
  "areas_completed": ["contact_list", "search"],
  "pending_failure_follow_ups": ["TC-012: Email validation crash"]
}
```

The gateway reads this session at the start of each call and uses it to maintain strategic continuity. For example: if it's been in `create_contact` for 3 tests and all passed, it auto-rotates to the next uncovered area.

---

## Implementation Priority

Given your existing stack, I'd recommend implementing in this order:

1. **Quick win (1-2 hours):** Add `exploration_strategy` param + update the prompt to use it — zero new infrastructure needed
2. **Medium (3-4 hours):** Add `inspect_failure` action + RAG endpoint — directly improves the agent's failure-driven pivoting
3. **Bigger lift (1-2 days):** Coverage Map in Neo4j + `retrieve_coverage` action — replaces Jaccard dedup with real semantic coverage tracking
4. **Polish (2-3 hours):** Chain-of-Thought rationale field in the output JSON — improves auditability

> [!NOTE]  
> Session Memory (#5) can be implemented very lightly by storing a `session` JSON blob on the `Project` node in Neo4j and reading it at the start of each `/next-testcase` call. No new infrastructure needed.
