# Roadmap and handoff

Where the project stands, what it has actually found, and what to do next — in priority order.
Written 6 Sep 2026. If you are new here, read [WORKFLOW.md](WORKFLOW.md) first.

---

## 1. Status

**Working end to end.** A campaign plans a test, runs it on a real Android emulator, evaluates
the trajectory, writes findings into Neo4j, and plans the next test from the grown graph.
6 test modules / 104 checks pass (`./venv/bin/python tests/run_all.py`).

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

### 3.1 Cold-start screen grounding *(~20 min)* — **highest value per effort**
`proposal.validate` skips the screen check when the app model is empty (`if observed and …`), so
on a brand-new project the planner can invent a screen name and the executor burns 50 steps on
it. Zero protection on exactly the run where the agent knows least. Fix: when no screens are
observed, require `screen_hint: "unknown"`, gated by a `REQUIRE_GROUNDED_SCREEN_HINT` setting so
it can be switched off.

### 3.2 Finding lifecycle — make the agent *conclude* things *(the substantial one)*
Findings are flat facts. Nothing marks one as an **open question that must be closed**. If the
SRS claims a feature, we test it, and it fails, there are three possible answers — not
implemented / broken / implemented differently — and the planner currently records the failure
and wanders off.

Add `status: open | resolved | inconclusive` and `attempts` to findings, a `list_open_questions()`
tool, and a directive rule: *close an open question before opening a new area.* Bound it at 2–3
attempts, then mark `inconclusive` with the evidence gathered — that is a result for a human, not
a failure.

**Hard constraint:** an earlier campaign produced five near-duplicate tests that each burned the
full step budget digging into an area the agent could not reach. Curiosity must split the two
cases: *reached it and it misbehaved* (probe narrower) versus *never reached it* (an agent
problem — a repeat just burns budget). The taxonomy already distinguishes these.

### 3.3 Coverage reporting *(~15 min)*
`coverage_pct` is computed from **Figma** screen purposes. With no Figma it reports `0%` while
requirement coverage says 11% — misleading in the dashboard and in any write-up. Report
requirement coverage when no design file exists.

### 3.4 Finding decay and retirement
UIStates decay (90-day half-life) and strategies decay; findings never do. A fixed defect stays
in the oracle forever and the planner keeps steering around a bug that no longer exists. The
`FOUND_BY` edges give the raw material: a finding not re-observed across N recent runs touching
its screen is a retirement candidate.

### 3.5 Typical step cost per area
The planner has no sense that "this kind of flow needs 35+ steps" and can write a test that
structurally cannot fit in `EXECUTOR_MAX_STEPS`. Same `ExecutionLog` data as the agent-difficulty
signal.

### 3.6 Proven interaction steps, not just control names
The planner sees real control names for a screen but not *actions that provably worked*. Requires
matching trajectory steps to screens — a bigger lift, worth it after 3.1–3.3 land.

### 3.7 Vision captioning for thin screens
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
untested requirements, and produces fewer `UNVERIFIED` findings. This became possible only
recently — findings now survive `CLEAN_SLATE`, tied to the app-model slice.

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
roadmap, and [NEXTGEN_IMPLEMENTATION_PLAN.md](NEXTGEN_IMPLEMENTATION_PLAN.md) is the decoder —
keep that file even though it is historical, or those references become unreadable.
Thesis-level framing and the eval-harness argument live in
[research-agentic-exploration.md](research-agentic-exploration.md).
