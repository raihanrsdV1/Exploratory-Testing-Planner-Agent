# Roadmap and handoff

Where the project stands, what it has actually found, and what to do next — in priority order.
Written 6 Sep 2026. If you are new here, read [WORKFLOW.md](WORKFLOW.md) first.

---

## 1. Status

**Working end to end.** A campaign plans a test, runs it on a real Android emulator, evaluates
the trajectory, writes findings into Neo4j, and plans the next test from the grown graph.
6 test modules pass (`./venv/bin/python tests/run_all.py`).

Two planners exist side by side, selected by `PLANNER_MODE`:

| | `pipeline` (default) | `tools` |
|---|---|---|
| how it retrieves | fixed loop, model sees only one-line notes | tool calls, model reads real results |
| screen grounding | none — invented names reach the device | validated against observed screens |
| requirement ids | asked not to invent; unenforced | rejected if they don't exist |
| dedup | after generation, one blind retry | before committing, with a reason |
| cost | ~4 LLM calls | 7–8 turns, 12–15 tool calls, ~67s |

`pipeline` is still the default because the tool planner has not yet been proven better on a
campaign — see §4.

**The active project is `shobarkhamar`** (`com.tirzokpvt.shobarkhamar`, Flutter, livestock
marketplace) on an Android 14 emulator, signed in as a farmer/seller. It has **no Figma export** —
`FIGMA_PATH` is deliberately empty, so the agent runs on SRS + its own observations. That is the
realistic configuration and the one worth developing against.

---

## 2. What the agent has actually found

Candidate defects, unvalidated by a human — this is the list to confirm first:

| finding | seen | screen |
|---|---|---|
| The language dropdown only opens when tapping near the chevron (~830,590); tapping the button's text area does nothing | **3×** | Seller Tester |
| System BACK on the Menu screen exits the app to the OS home screen instead of going back | 1× | Seller Tester |
| BACK from the Disease List menu context exits the app entirely; the agent had to relaunch | 1× | Disease List |
| Selecting 'ইংরেজি' (English) leaves the UI in Bengali on subsequent screens | 1× | Main |

The BACK-button behaviour appearing independently on two different screens is the strongest
signal in the set. `times_seen` counts *independent* observations across runs — a 3× finding was
re-derived by three separate evaluations.

Earlier campaigns (in the README) also flagged: no validation feedback on an empty mandatory
field, no success confirmation on farm update, no empty-state on list screens, and a device
registration race that misreports a client sequencing bug as a network error.

**Nothing here is human-confirmed yet.** Doing that once is a prerequisite for trusting anything
downstream.

### What the agent found about *itself*

`AGENT_DIFFICULTY` findings — our own weaknesses, deliberately kept out of the bug oracle:

- misidentifies the `ব্লগ` bottom-nav tab and goes to the Disease List instead (**3×**)
- repeats a menu open/close cycle ~7 times with no state progress
- reaches the correct farm-profile screen then navigates away without completing the task (4×)
- taps controls repeatedly with no visible effect rather than concluding they are inert

These are collected and retrievable (`group=agent`) but **no prompt block consumes them yet**
under the pipeline planner. The tool planner reads them naturally via `list_findings`.

---

## 3. Next work, in priority order

### 3.1 Cold-start screen grounding ✅ **BUILT (20 Sep 2026) — opt-in, default OFF**

With an **empty** app model — a brand-new project, or one where `CLEAN_SLATE_APPMODEL` wiped the
map — `proposal.validate` had nothing to check a `screen_hint` against, so it silently skipped the
check entirely. An invented screen reached the device unchallenged on exactly the run where the
agent knows least, and the executor spent its whole step budget hunting for it.

`REQUIRE_GROUNDED_SCREEN_HINT` (`.env`, default `0`) now requires `screen_hint='unknown'` until at
least one screen has been observed; the executor explores instead and the map fills in after the
first run. The warm-state check — a named screen contradicting a **non-empty** app model — is
separate and always on.

**Default OFF by decision:** it changes planner behaviour on a fresh project, so enabling it is a
deliberate choice rather than something that starts happening silently mid-measurement.

> ⚠️ **While it is off, the bug it fixes is still live.** On a fresh app the planner can name a
> screen that does not exist and nothing will stop it. Turn it on before any campaign on a new
> app, or before the app-agnostic claim is tested — that is precisely the scenario it protects.

### 3.2 Finding lifecycle — make the agent *conclude* things ✅ **DONE (7 Sep 2026)**
Findings are flat facts. Nothing marks one as an **open question that must be closed**. If the
SRS claims a feature, we test it, and it fails, there are three possible answers — not
implemented / broken / implemented differently — and the planner currently records the failure
and wanders off.

Built: `status` + `attempts` on findings, `list_open_questions()` tool, `addresses` on
`propose_test_case`, `resolves` in the investigator contract, auto-close at 3 attempts, and
`scripts/backfill_finding_status.py` for findings created before the lifecycle. See
[INVESTIGATOR.md](INVESTIGATOR.md) §"Finding lifecycle". Verified end to end: the planner
targeted an open question and spent an attempt (1/3).

**Still open:** pipeline mode has no open-question prompt block — the mechanism works in
`PLANNER_MODE=tools` only.

**Hard constraint:** an earlier campaign produced five near-duplicate tests that each burned the
full step budget digging into an area the agent could not reach. Curiosity must split the two
cases: *reached it and it misbehaved* (probe narrower) versus *never reached it* (an agent
problem — a repeat just burns budget). The taxonomy already distinguishes these.

### 3.3 Preserve campaign evidence ✅ **DONE (7 Sep 2026)**

`CLEAN_SLATE` destroyed the previous campaign's tests and execution logs before the next one
started, so the campaign-over-campaign comparison in §4 could not be run from the graph at all —
a paired comparison had to be reconstructed from raw log files. A `CampaignSummary` is now written
immediately before the wipe (`GET /campaigns`).

**Still open:** aggregates only. Per-test comparison across campaigns would need campaign-scoped
ids (Option B) — worth doing only if the aggregates turn out to be too coarse.

### 3.4 Scaling the finding graph — rollup ✅ **DONE (20 Sep 2026)**, scoping deferred

**The problem, measured.** A finding serialises to ~320 characters against `list_findings`'
2,500-character cap, so roughly **eight reach the planner however many exist** — 8 of 28 today,
and 8 of 300 after one full 25-round campaign (the rate is ~3 findings per evaluated test). Worse,
it arrived silently: the response said `count: 7` with no hint that 21 others existed.

**Built:**

- `findings_summary()` — a rollup bounded by *screen x kind x status* rather than by finding
  count, so it stays roughly constant as the graph grows. **1,210 chars covering all 28 findings,
  against 2,477 chars covering 8.** It also answers a question the list form structurally cannot:
  *which screen has the most unresolved defects?*
- Visible truncation — `[showing 8 of 28 findings … narrow with screen= or group= to see the rest]`,
  reporting the count that actually survives the character cap, not the database limit.
- Balanced open-question slots (see 3.2).

**Deferred — "scope after choosing" (a tool-description change).** Push the planner toward
`list_findings(screen=…)` once it knows which screen matters. Would cut the survey from ~8,700 to
~2,500 characters with better coverage.

Held back deliberately, for one reason worth remembering:

> **Scoping hides cross-screen patterns.** The strongest finding in this campaign —
> *"pressing BACK exits the app"* — was recognised because it appeared independently on
> **Seller Tester** and on **Disease List**. A planner scoped to one screen sees a local quirk,
> not a systemic navigation defect. The rollup mitigates this with per-screen defect counts, but
> counts are not claims, and the textual similarity that makes such a pattern obvious is only
> visible unscoped.

Two further notes: the change is a behavioural nudge, not a guarantee (a count of open questions
in the seed changed nothing; moving the list inline changed everything — description changes are
the weakest lever available), and it must never be applied to the broad survey phase, only after
an area is chosen. **Revisit when the planner is observed making bad area choices** — that is the
signal it would fix. Until then the ~6,000 characters it saves are not a constraint being hit.

### 3.5 Exploration tunnelling ✅ **FIXED (20 Sep 2026)**

**Measured failure.** A 13-round campaign put **9 of 13 tests on one screen**, and **20 of 74
findings described a single text field** — all of them the same defect (*the farm name field
validates nothing*) restated for empty, whitespace, special characters, emoji and 150 characters.
One duplicate was force-accepted after exhausting its 3 rejections.

**Cause: a positive feedback loop with no damping term.** Three signals all pointed at the same
screen, and the structural one could never stop — `hot_spots` promoted an area on `failed >= 2`,
while the only exit (`exhausted_areas`) required `failed == 0`. **A failing area was a one-way
door.**

**Fixed in three layers:**

1. `AREA_SATURATION` (5) — an area stops being promoted past that many tests whatever its
   verdicts, and the directive marks it `[SPENT]` with the reason.
2. The recent-runs note no longer says *"probe the same behaviour"*; it points at `get_coverage`
   and `findings_summary` first.
3. **Clustering + generalisation** — findings on one screen sharing a cause (0.72 cosine, far
   looser than dedup's 0.90) are offered to the evaluator, which may collapse them into one
   claim. Members keep their detail and are linked by `GENERALISED_BY`, but leave every
   planner-facing view. Verified live: 74 → 71 findings, one claim replacing four.

**An idea checked and rejected:** "exhaust an area when it stops producing new findings". The
ninth test on that screen was *still* producing technically-new findings, so it would never have
fired. A spend cap was missing, not an information-yield check — worth remembering, because the
intuitive fix was the wrong one.

**Still open:** the duplicate force-accept. When a proposal exhausts `MAX_REJECTIONS` the best
effort is taken unvalidated, so a *known* duplicate can still be logged. Exempting duplicates
from that fallback is a small follow-up.

### 3.6 Coverage reporting *(~15 min)*
`coverage_pct` is computed from **Figma** screen purposes. With no Figma it reports `0%` while
requirement coverage says 11% — misleading in the dashboard and in any write-up. Report
requirement coverage when no design file exists.

### 3.7 Finding decay and retirement — **premature, revisit after several campaigns**
UIStates decay (90-day half-life) and strategies decay; findings never do. A fixed defect stays
in the oracle forever and the planner keeps steering around a bug that no longer exists. Real and
compounding, but two things measured on 20 Sep 2026 say it cannot be built usefully yet:

- **There is no time signal.** Every finding and every screen currently carries the same
  `last_seen` date — all the data came from two days of runs — so nothing can be distinguished as
  stale. Decay needs campaigns spaced over time before it has anything to act on.
- **`FOUND_BY` edges do not survive.** An earlier version of this section proposed using "not
  re-observed across N recent runs", but `CLEAN_SLATE`'s `DETACH DELETE` destroys those edges —
  **0 findings currently link to any run**. The workable signal is `Finding.last_seen` versus its
  screen's `UIState.last_seen`: a screen visited far more recently than its finding was
  re-observed is the retirement candidate.

Build it as **soft decay** — mark stale and deprioritise, never delete. "Not re-observed" can
simply mean "not re-tested", and silently dropping a real defect is much worse than carrying a
fixed one.

### 3.8 Typical step cost per area
The planner has no sense that "this kind of flow needs 35+ steps" and can write a test that
structurally cannot fit in `EXECUTOR_MAX_STEPS`. Same `ExecutionLog` data as the agent-difficulty
signal.

### 3.9 Proven interaction steps, not just control names
The planner sees real control names for a screen but not *actions that provably worked*. Requires
matching trajectory steps to screens — a bigger lift, worth it after 3.1–3.3 land.

### 3.10 Vision captioning for thin screens
Screens with no usable accessibility tree (pure Compose/Flutter) are invisible beyond a control
count. Screenshots are already captured in `data/appmodel/<project>/`. Direct vision at
generation time is already implemented; the remaining gap is **cached captions** for screens with
no structural info at all. Lowest priority until a Compose-heavy app is the active project.

### Deferred deliberately
- **Deleting the `pipeline` planner** — needs §4's numbers first.
- **OpenRouter provider pinning** (`provider: {order, sort}`) — real speedup available, not
  urgent while `qwen3.7-flash` is healthy.

---

## 4. Proving it works — the biggest gap

**No ground truth exists.** Precision and recall are uncomputable. Coverage and autonomy are the
only defensible metrics today, and "the agent gets smarter" is currently a *mechanism* that is
demonstrably exercised, not a measured *outcome*.

Two experiments, cheapest first:

**A. Campaign-over-campaign (a few dollars, an afternoon).** Run two campaigns back to back on
the same app, keeping the graph. Measure whether campaign 2 cites more real screens, hits more
untested requirements, and produces fewer `UNVERIFIED` findings. Now possible: findings survive
`CLEAN_SLATE`, and `GET /campaigns` preserves each campaign's aggregates past the reset that used
to destroy them.

**B. Seeded-defect build (under $1).** 8–12 known defects with a ground-truth list, three arms:
full agent / memory disabled / random baseline. Gives precision, recall, F1 and an ablation for
the "gets smarter" claim. `PLANNER_MODE` makes the planner arm a flag flip.

**A/B on the planner** is now nearly free: same executor, same investigator, one env var.

---

## 5. Known limitations

1. **Autonomy ~50%.** Roughly half of runs are lost to our own agent, mostly on multi-step forms —
   it does not reliably scroll back up to find a required field above its current position.
2. **`uiautomator` and the mobilerun portal conflict.** `FATAL EXCEPTION: UiAutomationService`
   appears in logcat — two accessibility clients registering at once. Our tooling, not the app.
3. **State-identity thresholds are tuned on one app's states.** Evidence, not proof; no labelled
   answer for how many distinct screens the app really has.
4. **Test ids restart at TC-001 each campaign.** Fine while results are wiped; breaks the moment
   you compare campaigns — which experiment A above requires.
5. **The graph stores title + verdict for a test**, not `steps`/`screen`/`expected_result`, so a
   review sheet cannot show what a test actually did.
6. **Defect intelligence (ETA-REQ-301) has never run on real data.** Built, dormant — needs a real
   defect export ingested.
7. **Cross-application transfer (ETA-REQ-304) unvalidated.** Every campaign so far ran against one
   app. Running a second app is the single best test of the app-agnostic claim.
8. **`docs/SAMSUNG_PROGRESS_REPORT.pdf` contains claims now known to be wrong** — the "zero silent
   fallbacks" line read a counter that could not see the executor process. Correct it before
   reusing that report.

---

## 6. Where the requirement ids come from

`WP1`–`WP9` and `ETA-REQ-301`–`308` appear ~160 times across the code. They are Samsung's
roadmap, and [NEXTGEN_IMPLEMENTATION_PLAN.md](sus/NEXTGEN_IMPLEMENTATION_PLAN.md) is the decoder —
keep that file even though it is historical, or those references become unreadable.
Thesis-level framing and the eval-harness argument live in
[research-agentic-exploration.md](sus/research-agentic-exploration.md).
