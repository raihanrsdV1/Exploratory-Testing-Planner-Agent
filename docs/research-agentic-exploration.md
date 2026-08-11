# Research & Design Notes — Live Agentic Exploration, Novelty-Driven Testing, and an Eval Harness

Deep-dive on three "drastic, thesis-worthy" directions for the Exploratory Testing Planner:

1. **Close the loop live** — the planner replans mid-run from a growing *screen graph*, so planner and executor co-evolve the app map.
2. **Novelty / coverage-driven policy** — a principled exploration objective (intrinsic novelty reward over screen signatures) replacing heuristic free-text `area` strings.
3. **Offline eval harness** — a known-buggy sample app with labelled defects, measuring bug-discovery rate and coverage-over-time, so any of the above can be *proven* to help.

Everything here is grounded in the current codebase (`planner/langgraph_agent.py`, `rag_api/main.py`, `clients/executor_runner.py`) and in the MobileRun/DroidRun 0.6.8 internals we verified (`shared_state`, per-step `RecordUIStateEvent`, `trajectory.ui_states`).

---

## 0. Thesis positioning — read this first

Before building, be clear-eyed about prior art, because **three published systems are already very close to what you have**:

| System | What it is | Overlap with your design |
|---|---|---|
| **AutoDroid** (MobiCom '24) | LLM + **UI Transition Graph (UTG)** memory built by offline random exploration; functionality-aware UI representation feeds the LLM | Almost identical to "auto-build a screen graph and feed it to an LLM." Your ScreenGraph ≈ their UTG. |
| **GPTDroid** (ICSE '24) | Frames GUI testing as an LLM Q&A loop with **functionality-aware memory**; asks "is this a new functionality?" | Same feedback loop + novelty check you're proposing in idea #2. |
| **DroidAgent** (ICST '24) | Autonomous **Planner + Actor** LLM agents that set their own *intents* and pursue them | Same planner/executor split you have (planner = you, actor = MobileRun). |

**So what is your novelty?** The honest, defensible differentiator is that all of the above are **spec-free black-box explorers** — they discover the app from scratch. **Yours is spec-grounded**: you ingest an **SRS + design (Figma) into a GraphRAG knowledge graph** and use it as *both* an oracle ("what should this screen do?") and a coverage target ("which requirements have I not exercised?"). Frame the thesis as:

> **Requirement-grounded agentic exploratory testing** — coupling an a-priori specification graph (SRS/design) with an a-posteriori runtime screen graph, and steering exploration by the *gap* between them.

That gap — "requirements/screens the spec implies but the running app never reached" — is simultaneously your exploration objective *and* a bug detector (spec'd feature that's missing or unreachable). No prior system above has the spec side. Keep hammering this; it's what makes the work more than "AutoDroid with a different LLM."

---

## 1. Close the loop live — co-evolving planner + executor

### 1.1 The idea

Today the flow is **stateless and design-time only**: `/agent/next-testcase` plans purely from SRS+Figma, MobileRun executes, a verdict is logged, repeat. The planner never sees the *actual* runtime UI. "Live" means: as MobileRun acts, stream its per-step observations into a growing **ScreenGraph**, and let the planner (re)decide the next objective from that live map — mirroring MobileRun's own internal manager→executor loop, but at *your* planner's level and grounded in the spec graph.

### 1.2 What the raw material actually is (verified against MobileRun 0.6.8)

- `MobileAgentState` (`shared_state`) holds only the **current** screen's `a11y_tree`, plus a **flat unordered** `visited_activities: set` and a **linear** `action_history`/`action_outcomes`. There is **no** per-step screen history and **no** graph. You must build the graph yourself.
- Per-step a11y *is* available two ways:
  - **Live:** subscribe to the workflow event stream — `RecordUIStateEvent(ui_state=list[Dict])` fires every step; `ExecutorResultEvent(action, outcome, error)` carries the action that caused the transition.
  - **After-run:** `agent.trajectory.ui_states` (parallel list) + screenshots on disk, when `config.logging.save_trajectory != "none"`.
- `a11y_tree` elements are a **flat indexed list**: `{index, resourceId, className (short), checkedState, text, bounds:"l,t,r,b", children:[]}`. So a screen *signature* is a cheap hash over the flat list — no tree walking.

### 1.3 Concrete design

**Node (screen):** key = `(activity_name, stable_hash(sorted resourceId/text set of the a11y list))`. Activity alone is too coarse (one activity = many visual states); the a11y hash collapses re-visits of the *same* state into one node. This is the exact granularity problem APE (ICSE '19) and Q-testing (ISSTA '20) tackle — cite them and note your hash is the "cheap" end of that spectrum, tunable later.

**Edge (transition):** when the signature changes between step N and N+1, add `node_N --[action_N]--> node_{N+1}`, labelled with the executor's action dict (`{"action":"click","index":5}`, `open_app`, `system_button(back)`, `swipe`, …) plus `outcome` (bool). Accumulate across runs → the real navigation graph (reachable screens, which actions move between them, dead ends, loops). This is precisely DroidBot's UTG and Stoat's stochastic model, built from an LLM-driven executor instead of random/model-based input.

**Replanning trigger** (the genuinely new part): don't replan every step (too slow — each planner LLM round is 30–60 s). Replan when *interesting* things happen:
- a **new** screen signature appears (novelty — see §2),
- an action's `outcome` is `False` or an `error` is set (failure — pivot),
- the executor has taken K steps without new coverage (stuck — inject a new objective),
- MobileRun finishes a subgoal (natural checkpoint).

**Where it plugs into your code:**
- Wrap `execute_test_on_device` in `clients/executor_runner.py`: replace `result = await agent.run()` with `handler = agent.run(); async for ev in handler.stream_events(): screen_graph.observe(ev)`. `observe()` handles `RecordUIStateEvent` (new node) and `ExecutorResultEvent` (edge).
- Add `planner/screen_graph.py`: `ScreenGraph.observe(event)`, `.to_prompt_text()`, `.to_json()`, novelty queries.
- Feed `screen_graph.to_prompt_text()` into the planner prompt (new context block alongside SRS/Figma) in `langgraph_agent.generate_testcase` / `bootstrap_context`.
- Optionally persist screens/edges into Neo4j as new node types (`(RunScreen)-[:TRANSITION {action,outcome}]->(RunScreen)`) cross-linked to `FigmaScreen`/`Requirement`, so coverage lives in one graph.

### 1.4 Build order & risks

Build **after-run first** (read `trajectory.ui_states` once the run ends) to validate node/edge construction and serialization — cheap, deterministic, no async event plumbing. Then swap in the live stream. Risks: (a) latency of LLM replanning vs. MobileRun's own pace — mitigate with the trigger policy above, not per-step; (b) **partial observability** — the same visual screen can hash differently due to dynamic content (timestamps, lists); normalise the signature (drop volatile text, keep resourceIds); (c) two agents now write "the plan" — keep MobileRun as the low-level actor and your planner as the high-level intent-setter to avoid fighting (this is DroidAgent's Planner/Actor separation).

### 1.5 Key papers
- **DroidAgent** — Planner/Actor LLM agents, intent-driven (your closest live-loop analogue).
- **AutoDroid** — UTG-as-LLM-memory; validates "graph feeds the model."
- **GPTDroid** — iterative LLM↔app dialogue with functionality memory.
- **DroidBot** / **Stoat** — the classical UI-transition-graph / stochastic-model formulations your ScreenGraph re-implements.
- **APE** (ICSE '19) — model abstraction/refinement = the screen-signature-granularity problem.

---

## 2. Novelty / coverage-driven exploration policy

### 2.1 The idea

Replace the free-text `area`-string heuristics in `coverage.compute_coverage_map` / `build_exploration_directive` with a **quantified exploration objective**: prefer actions/objectives that (a) reach **novel** screen signatures, (b) cover **uncovered requirements** (graph-native `COVERS` edges you already compute but underuse), and (c) probe near **recent failures**. This turns "don't repeat titles" into a real intrinsic-motivation signal.

### 2.2 The RL/exploration foundations to cite

This is a direct application of intrinsic-motivation / novelty search:

- **Novelty search** (Lehman & Stanley, 2011) — abandon the objective; reward *behavioural novelty* alone. Conceptual backbone.
- **Count-based exploration**: **pseudo-counts** (Bellemare et al., 2016) and **`#Exploration` via state hashing** (Tang et al., 2017). The hashing paper is the tightest analogy — they hash continuous states into buckets and reward rarely-seen buckets; **your screen signature is exactly that hash**, so novelty reward = `1/√(visit_count[signature])`.
- **Curiosity / prediction error**: **ICM** (Pathak et al., 2017) and **RND** (Burda et al., 2018) — reward states a learned model predicts poorly. Heavier (needs a learned model); mention as the "learned" upgrade path over count-based.
- **Go-Explore / "First return, then explore"** (Ecoffet et al., *Nature* 2021) — *archive* promising states, deterministically **return** to an under-explored one, *then* explore. This maps beautifully to GUIs: you can **navigate back** to a specific screen via the ScreenGraph, then explore its untried actions. This is arguably the single most transferable idea for your setting.
- **Q-testing** (Pan et al., ISSTA '20) — **curiosity-driven RL for Android GUI testing**, with a neural state-comparison module at "functional scenario" granularity. This is your closest prior art on the *policy* side — position your contribution as *spec-aware* novelty (novelty weighted by requirement coverage) vs. their spec-free curiosity.

### 2.3 Concrete scoring

Define a per-candidate score the planner (or a cheap selector) maximises:

```
score(next_objective) =
      w_novelty     * expected_new_signatures          # from ScreenGraph visit counts
    + w_requirement * uncovered_requirements_reachable  # from COVERS edges + spec graph
    + w_failure     * proximity_to_recent_failures      # bias toward flaky/buggy areas
    - w_cost        * estimated_steps_to_reach
```

- `expected_new_signatures`: use Go-Explore logic — screens with untried outgoing actions (frontier) score high; fully-expanded screens score ~0.
- `uncovered_requirements_reachable`: you *already* compute `/coverage/requirements`; wire it into the directive instead of only reporting it.
- Start with fixed weights; a stretch goal is learning them (bandit/RL) — but fixed weights already beat the current string heuristics and are far easier to defend.

### 2.4 Where it plugs in
- `planner/coverage.py`: add `novelty_score(signature)` and `frontier_screens()` backed by ScreenGraph visit counts; have `build_exploration_directive` rank by the score above rather than area strings.
- Feed the top-ranked frontier target + uncovered requirements into the planner objective (the `objective` that `log_verdict_and_next` currently hand-writes as English).

---

## 3. Offline eval harness — the measuring stick

Without this you cannot claim any of §1–§2 improved anything (see CLAUDE.md §4: verifiable goals). Build it **early** — ideally right after the after-run ScreenGraph — so every later change is measurable.

### 3.1 Metrics (over wall-clock or step budget)
- **Activity/screen coverage over time** — from the ScreenGraph (unique signatures & activities vs. steps). Standard GUI-testing curve.
- **Code coverage** — instrument the app under test with **JaCoCo** (or run a coverage-enabled build) for method/line coverage; the gold-standard metric used by every tool below.
- **Unique crashes** — distinct stack traces from `logcat`/ANRs.
- **Bug-detection rate on labelled faults** — % of *known* injected/curated bugs the system triggers. This is the headline thesis number.
- **Requirement coverage** — your unique axis: % of SRS requirements exercised (`COVERS` edges). Nobody else can report this — feature it.

### 3.2 Datasets & benchmarks
- **AndroidWorld** (Google, 2024) — 116 tasks across 20 apps with **reproducible, adb-state-checked rewards**. Crucially, it *deliberately avoids LLM judges* and checks device state directly — use it as your **independent oracle** blueprint (fixes the "executor judges itself" flaw) and as a ready-made task suite.
- **"Automated Test Input Generation for Android: Are We There Yet?"** (Choudhary, Gorla, Orso, ASE '15) — the **AndroTest** benchmark + the canonical evaluation *methodology* (coverage & fault comparison vs. Monkey and tools). Copy its experimental design.
- **"Benchmarking automated GUI testing for Android against real-world bugs"** (Su et al., FSE '21, the *Themis* suite) — apps with **real, reproducible bugs** — ideal for bug-detection-rate.
- **Mutation testing** (e.g. MutAPK) — inject synthetic faults when you need more labelled bugs than a curated set provides.

### 3.3 Baselines to beat
- **Monkey** (random) — the mandatory floor.
- **DroidBot** (UTG-guided, no LLM) — shows the graph helps.
- Your own **ablations**: (i) SRS+Figma only (current system), (ii) + ScreenGraph, (iii) + novelty policy, (iv) + live loop. The ablation ladder *is* the thesis result.

### 3.4 Harness shape
Pick 3–5 apps (mix of AndroidWorld/Themis + your own SRS'd app). For each config: fresh emulator snapshot → run for a fixed budget → record coverage curve, crashes, bugs-found, requirement coverage → plot. Automate with the emulator + `adb` you already drive in `executor_runner.preflight()`.

### 3.5 Key papers
- **AndroidWorld** (reward-via-adb; oracle design).
- **"Are We There Yet?"** (AndroTest; methodology).
- **Themis / real-world-bugs benchmark** (FSE '21).
- **Sapienz** (Mao et al., ISSTA '16) & **Stoat** (FSE '17) — standard baselines & the crash-count/coverage metrics everyone reports.

---

## 4. Suggested sequencing

```
1. ScreenGraph (after-run, from trajectory.ui_states)   → verify: nodes/edges look right on one run
2. Eval harness + baselines (Monkey, current system)     → verify: reproducible coverage/bug curves
3. Novelty/coverage policy (fixed weights)               → verify: beats current system on the harness
4. Live loop (event stream + trigger-based replanning)   → verify: beats after-run on the harness
5. (stretch) Independent oracle via adb state checks; learned policy weights
```

Rationale: build the **measuring stick (step 2) before** the ambitious changes, so §1 and §2 are evaluated, not asserted. Each step has a concrete pass/fail against the harness.

---

## References

**LLM-driven GUI testing / task automation (closest prior art)**
- AutoDroid: LLM-powered Task Automation in Android — Wen et al., MobiCom 2024. https://arxiv.org/abs/2308.15272 · https://dl.acm.org/doi/10.1145/3636534.3649379
- Make LLM a Testing Expert (GPTDroid) — Liu et al., ICSE 2024. https://arxiv.org/abs/2310.15780 · https://dl.acm.org/doi/abs/10.1145/3597503.3639180
- Intent-Driven Mobile GUI Testing with Autonomous LLM Agents (DroidAgent) — Yoon et al., ICST 2024. https://arxiv.org/abs/2311.08649 · https://github.com/coinse/droidagent
- Deeply Reinforcing Android GUI Testing with Deep RL — ICSE 2024. https://dl.acm.org/doi/10.1145/3597503.3623344
- DinoDroid: Testing Android Apps Using Deep Q-Networks. https://arxiv.org/pdf/2210.06307

**Model-based / UI-transition-graph exploration (classical)**
- DroidBot: A Lightweight UI-Guided Test Input Generator for Android — Li et al., ICSE 2017. https://ylimit.github.io/static/files/DroidBot_ICSE2017.pdf · https://github.com/honeynet/droidbot
- Guided, Stochastic Model-Based GUI Testing of Android Apps (Stoat) — Su et al., FSE 2017. https://dl.acm.org/doi/10.1145/3106237.3106298 · https://github.com/tingsu/Stoat
- Practical GUI Testing via Model Abstraction and Refinement (APE) — Gu et al., ICSE 2019.
- Sapienz: Multi-objective Automated Testing for Android — Mao, Harman, Jia, ISSTA 2016.

**Curiosity-driven testing (policy prior art)**
- Reinforcement Learning Based Curiosity-Driven Testing of Android Applications (Q-testing) — Pan et al., ISSTA 2020. https://dl.acm.org/doi/10.1145/3395363.3397354 · https://github.com/anlalalu/Q-testing

**Exploration / intrinsic-motivation foundations (RL)**
- Abandoning Objectives: Evolution Through the Search for Novelty Alone — Lehman & Stanley, Evolutionary Computation 2011.
- Unifying Count-Based Exploration and Intrinsic Motivation (pseudo-counts) — Bellemare et al., NeurIPS 2016. https://arxiv.org/abs/1606.01868
- #Exploration: A Study of Count-Based Exploration for Deep RL (state hashing) — Tang et al., NeurIPS 2017. https://arxiv.org/abs/1611.04717
- Curiosity-driven Exploration by Self-supervised Prediction (ICM) — Pathak et al., ICML 2017. https://arxiv.org/abs/1705.05363
- Exploration by Random Network Distillation (RND) — Burda et al., ICLR 2019. https://arxiv.org/abs/1810.12894
- First return, then explore (Go-Explore) — Ecoffet et al., Nature 2021. https://www.nature.com/articles/s41586-020-03157-9 · https://arxiv.org/abs/2004.12919

**Benchmarks & evaluation methodology**
- AndroidWorld: A Dynamic Benchmarking Environment for Autonomous Agents — 2024. https://arxiv.org/abs/2405.14573 · https://github.com/google-research/android_world
- Automated Test Input Generation for Android: Are We There Yet? (AndroTest) — Choudhary, Gorla, Orso, ASE 2015. https://arxiv.org/abs/1503.07217
- Benchmarking Automated GUI Testing for Android Against Real-World Bugs (Themis) — Su et al., ESEC/FSE 2021. https://dl.acm.org/doi/abs/10.1145/3468264.3468620
