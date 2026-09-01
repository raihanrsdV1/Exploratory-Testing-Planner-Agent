# Planner Improvements — Future Work

Ideas for what to send the planner next, to improve test-plan quality. None of these are
implemented yet. Ordered by priority — #1 is the one most directly motivated by measured data
and the cheapest to build.

Context: current generation prompt runs ~6,600 tokens against a 50,000-token
`PROMPT_BUDGET_TOKENS` ceiling (~13% used) — there is no token-budget pressure blocking any of
this. See [PLANNER_PROMPT_ANATOMY.md](PLANNER_PROMPT_ANATOMY.md) for the full current prompt
breakdown, or run `scripts/dump_prompt.py` to re-measure live.

---

## 1. Tell the planner which screens are hard for *our agent* — not app defects, agent capability

**The gap:** nothing currently feeds "the agent keeps getting stuck here" to the planner. In
fact the opposite happens on purpose: `planner/coverage.py`'s `NON_INFORMATIVE_ERRORS` set
deliberately filters `STEP_LIMIT_EXCEEDED` and similar categories *out* of the "known failures"
block, because they're correctly not app-defect evidence. That's the right call for the bug
oracle, but it throws away a genuinely different, useful signal in the process.

**The evidence:** a 25-round ShobarKhamar campaign (30 Aug 2026) had 8 runs end in
`NAVIGATION_LIVELOCK` (since removed as a mechanism — see below) plus 2 more that turned out to
be misclassified timeouts. Grouping all of them by *last screen reached* showed **5 of 8 (62.5%)
ended on one screen** — "Farm Info Update" (`খামারের তথ্য আপডেট`), a 24-control form. This
matches a limitation already documented in the README: the agent doesn't reliably scroll back up
to find a field above its current position on a long form. Nothing told the planner this before
generating the *next* test for that area, so it kept writing the same style of ambitious
multi-field test against a screen that structurally couldn't support it.

**Proposal:** a new prompt block, clearly separated from defect-oracle content so it can never
be mistaken for "the app is broken here":

```text
## Known Agent Difficulty (steer test design, not defect evidence)
- Farm Info Update: 5 of 8 recent runs stalled/timed out here.
  Prefer a narrower, single-field test on this screen rather than a full multi-field flow.
```

**How to build it:** query `ExecutionLog` for `error_type IN ('STEP_LIMIT_EXCEEDED',
'NAVIGATION_FAILURE', 'ELEMENT_NOT_FOUND', 'TIMEOUT')` (the `AGENT_FAULT`/budget-exhaustion
categories, per `settings.py`'s taxonomy — deliberately the same set `NON_INFORMATIVE_ERRORS`
excludes elsewhere), group by `path_labels[-1]` (last screen reached), surface screens with 2+
occurrences. Natural home: a new function in `planner/context_builders.py` alongside
`build_failure_context`, wired into `build_learned_context`'s return tuple and
`planner/prompts.py`'s block list (probably priority 2 in `planner/budget.py`, alongside risk/
anomaly/strategy — steering signal, not oracle-critical).

**Note on `NAVIGATION_LIVELOCK` specifically:** livelock detection itself was removed from
`clients/executor_runner.py` on 30 Aug 2026 (see git history / conversation — it introduced a
real concurrency bug where the "cancelled" agent kept running concurrently with the next test).
That means this exact error_type will not recur, but the underlying agent-capability gap it was
catching didn't go away — those tests now simply run to the full `EXECUTOR_TIMEOUT`/
`EXECUTOR_MAX_STEPS` ceiling before being recorded as `STEP_LIMIT_EXCEEDED` or (after the
`WorkflowTimeoutError` misclassification fix) `ASSERTION_FAILURE`. The clustering signal is
still fully collectible from those categories; this proposal doesn't depend on livelock
detection existing.

---

## 2. A natural companion: tell the planner the typical step cost per area

**The gap:** the planner has no sense of "this kind of flow tends to need 35+ steps" — it just
writes a test and hopes it fits in `EXECUTOR_MAX_STEPS` (50). A multi-field form test and a
"tap one button" test cost the same nothing to *plan*, but very different amounts to *execute*.

**Proposal:** alongside #1's difficulty block, surface typical `device_steps` per feature area
from execution history (a simple average or median over recent runs, from the same
`ExecutionLog` data). Lets the planner deliberately scope a test to fit the remaining budget
instead of accidentally writing something structurally too large.

**Why it's a companion, not separate work:** it's the same query, the same data source, and the
same prompt block as #1 — build them together.

---

## 3. Feed proven interaction steps, not just screen/control names (bigger lift)

**The gap:** the planner already sees real observed control names for a target screen (via
`planner/sources/liveui.py`, fixed earlier this session), but not *actions that provably worked*
on that screen before. It still has to invent plausible-sounding steps from a static list of
controls, which can be wrong about ordering, required intermediate taps, or which control
actually triggers what.

**The data already exists:** `logs/trajectories/<timestamp>_<hash>/trajectory.json` holds real
`ToolExecutionEvent`s — `tool_name`, `tool_args`, `success`, and a human-readable `summary` — for
every device action of every run (this is what powers the dashboard's `▸ show device steps`
panel; see `gateway/main.py`'s `/dashboard/run-steps` endpoint for the parsing pattern to reuse).

**Proposal:** for a target screen, pull the tool-call sequence from the most recent *successful*
run that reached it, and inject it as "a route that worked here before":

```text
## Proven interaction on 'Search Results' (from a successful prior run)
1. tap(index=41) -> opened search bar
2. type(index=41, text="...") -> entered query
3. (results appeared without further action)
```

**Why it's a bigger lift than #1/#2:** needs matching trajectories to the *screen* they occurred
on (trajectories are keyed by timestamp/run, not by screen — would need to cross-reference each
trajectory's tool calls against the `UIState` path recorded in the corresponding `ExecutionLog`,
or extend trajectory capture to tag each `ToolExecutionEvent` with the `UIState` id it occurred
in). Build #1 and #2 first; this is the natural follow-on once that's proven valuable.

---

## 4. Still open: no vision captioning on complex/Compose screens

**The gap:** screens with no structural control names (pure-Compose or Flutter screens whose
accessibility tree exposes nothing usable — e.g. this project's own heaviest-visited state,
`android.view.View`, several hundred visits with zero named controls) are invisible to the
planner beyond a bare control count. `EXECUTOR_VISION=1` is on and screenshots are captured and
stored (`data/appmodel/<project>/*.png` — 159 on the current ShobarKhamar graph), but nothing
ever *describes* them back into text for the (text-only) planner to read.

**Proposal (as discussed earlier in the project's life, still not built):** when a new `UIState`
is created (`is_new=True` in `/liveui/observe`) and its structural tree is "thin"
(`app_state.is_thin_tree()` already exists and is used for the perceptual-hash fallback — reuse
it as the trigger), send the screenshot to a vision-capable model once, ask for a short caption
plus visible interactive labels, and store the result as `caption`/`vision_labels` on the
`UIState` node. `planner/sources/liveui.py` would then surface real captions for screens that
currently show only "N controls, no names."

**Why lowest priority right now:** the current campaign's clearest agent-capability problem
(Farm Info Update) already has full structural control names — this gap doesn't explain that
specific failure. Worth doing when a Compose/Flutter-heavy app becomes the active project and
screens start showing up genuinely blank in the app-model graph.

**Revisit (31 Aug 2026):** re-examined this after being asked directly "the executor sees
screens to make smart decisions — why can't the planner?" Verified there is no model-capability
blocker: both `qwen/qwen3.7-flash` (executor) and `qwen/qwen3.8-flash` (planner, current
`OPENROUTER_MODEL`) are multimodal on OpenRouter (text+image+video in). The only real gap is
plumbing — [planner/model_client.py](../planner/model_client.py)'s `_call_openrouter` always
sends a plain-text `content` string (`model_client.py:134-137`), never an image content block;
nothing threads an image through `call_model()` into `planner_step`/`generate_testcase`.

Checked call volume before assuming cost was a concern: one test-case generation cycle in
[planner/langgraph_agent.py](../planner/langgraph_agent.py) is at most ~6 `planner_step`
retrieval rounds + 1-2 `generate_testcase` calls ≈ 7-8 LLM calls total — versus up to
`EXECUTOR_MAX_STEPS` (50) vision-bearing calls just to *execute* that one test case. Planning is
genuinely cheap next to execution; this de-prioritization was too conservative.

**Revised proposal — direct vision at generation time (do this before the caption-cache idea
above):** when `generate_testcase` is about to write steps for a specific target screen, attach
that screen's stored screenshot (already sitting in `data/appmodel/<project>/*.png`, populated
whenever `EXECUTOR_VISION=1`) straight into that one `call_model()` call as an image content
block, alongside the existing text prompt. Simple, always reflects the real current screen
(no staleness risk the way a cached caption has), and costs at most 1-2 image-bearing calls per
test case generated. Reserve the caption-cache approach above for screens with genuinely no
structural info at all (thin-tree Compose/Flutter screens) where there's no single "the" screen
state to attach live — a persistent caption is the better fit there. The two are complementary,
not competing: direct vision for planning against a specific known screen; cached captions for
otherwise-invisible ones.

---

## Suggested build order

1. **#1 + #2 together** — same data source, same query, cheapest, most directly tied to a
   measured 62.5%-concentrated failure mode. Natural next session's work.
2. **#3** — once #1/#2 are proven to help, invest in trajectory-to-screen matching.
3. **#4** — opportunistic, when the active project is Compose/Flutter-heavy enough that blank
   app-model states are common.
