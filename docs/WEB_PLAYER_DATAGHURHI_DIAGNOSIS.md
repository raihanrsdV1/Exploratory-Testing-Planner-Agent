# Why every DataGhurhi test fails

Diagnosis of the `dataghurhi-auth` target, 11 Sep 2026.

**Question asked:** every test case fails, and the failures look like our testing
agent's fault rather than defects in the site. Is that true, and why?

**Short answer:** yes, it is true, and it is mostly *not* a bug in the agent's
reasoning. Eight concrete causes are identified below, each with a specific fix.

The three that matter most:

1. **The profile contradicts its own specification**, and the planner's
   coverage-seeking makes it worse every round — 4 of 5 test cases in the fresh
   run were impossible before the run began (finding 3).
2. **A dropped connection to OpenRouter was filed as a crash in DataGhurhi**
   (finding 5). This is the only bug here that puts a false finding on the
   *site's* record.
3. **The newly-switched executor model throws away 23% of its turns** on replies
   that are not valid JSON (finding 7).

Two further items are safety rather than throughput: the agent attempted to
change the account password with no guardrail against it (finding 6).

---

## 1. Evidence base

Two sources, kept separate because they say different things:

| Source | What it covers |
|---|---|
| `logs/web_player.prev.log` | the runs you reported — 8 DataGhurhi test cases, 182 agent steps |
| `logs/web_player.log` | a fresh 5-round run executed for this report |

**A warning about the log.** `logs/web_player.log` is written by the operator
trace *and* by the test suite's stub fixtures (`shop.example.com`,
`127.0.0.1:8931`). Any count taken over the raw file is wrong. An early pass of
this analysis "found" 35 uses of an invalid `scroll_down` action; all 35 came
from a test fixture. The dispatcher's `Unknown action` error appears **0 times**
in the entire log, which proves no real run ever sent an invalid action name.
Every figure below is scoped to DataGhurhi blocks only. See finding 8.

### The fresh 5-round run commissioned for this report

`py -m targets.run dataghurhi-auth --rounds 5` — 11 Sep 2026, 586.8 s,
**1 passed / 4 failed (20%)**.

| # | Test case | Steps | Outcome | Attributed |
|---|---|---|---|---|
| TC-001 | non-existent survey slug shows an explicit error | 3 | **PASS** | — |
| TC-002 | sign-in with invalid credentials | 13 | tried to click `Logout` | `BLOCKED_BY_GUARDRAIL` |
| TC-003 | password-recovery email enumeration | 4 | tried to click `Logout` | `BLOCKED_BY_GUARDRAIL` |
| TC-004 | password change rejects identical password | ~10 (logged 0) | ConnectionReset to **openrouter.ai** | `CRASH` ← wrong, see finding 5 |
| TC-005 | email-format validation on the sign-in form | 10 | looped hunting for a login page | `NAVIGATION_LIVELOCK` |

**48 steps, 1 verdict reached.** Four of the five test cases (TC-002 through
TC-005) are `FR-AUTH` requirements that need a *signed-out* session — on a
profile that is permanently signed in and forbidden from logging out. Only
TC-001, the one public-surface test, was executable at all. That is the coverage
trap in finding 3, observed end to end.

### Outcomes of the 8 prior DataGhurhi test cases

| # | Steps | Unparseable replies | Refused actions | Outcome |
|---|---|---|---|---|
| 1 | 18 | 1 | 0 | **PASS** |
| 2 | 30 | 1 | 0 | STEP LIMIT |
| 3 | 30 | 0 | 0 | STEP LIMIT |
| 4 | 26 | 2 | 1 | LIVELOCK |
| 5 | 29 | 0 | 7 | aborted |
| 6 | 30 | 1 | 0 | STEP LIMIT |
| 7 | 0 | 0 | 0 | aborted |
| 8 | 19 | 0 | 3 | LIVELOCK |

- **182 steps, 1 verdict.** Exactly one `finish` call across every step taken.
- **Mean 26 of 30 steps.** The budget is nearly always exhausted.
- Every non-passing outcome is classified `STEP_LIMIT_EXCEEDED`,
  `NAVIGATION_LIVELOCK` or similar — all in `AGENT_FAULT` / `ENV_FAULT`, none in
  `APP_FAULT`. **So the system is already telling you the truth: these are not
  defect findings.** The attribution is working; the runs are simply not
  producing evidence about the site.

---

## 2. Finding 1 — the step budget cannot cover the tests being written

**This is the dominant cause.** `WEB_MAX_STEPS = 30`.

The planner writes test cases whose preconditions require building data first.
A representative one from your run:

> Preconditions: *Create a new survey with a file-upload question that has a
> maximum file size and allowed types. Open the survey in preview mode.*

Reaching the point where that test can *begin* costs roughly:

```
create project → name it → set research field → submit → open it
  → create survey → name it → submit → open editor
  → add section → add question → choose type → configure size limit
  → save → open preview
```

That is 15–20 successful actions before the first assertion, on a good run, with
no page-load stalls and no mistakes. With 30 steps total and ~5 lost to blank
observations (finding 2), the agent runs out of budget during *setup* and never
reaches the behaviour under test. The single passing test, TC-001, needed one
navigation and one observation — 3 steps.

**Fix.** Two options, best combined:

1. Raise `run.max_steps` to **80–100** for this profile. A CRUD application is
   not comparable to a read-only site; 30 was chosen for Wikipedia.
2. Seed the fixtures once, by hand — a project, a survey, and one question of
   each type — and rewrite the preconditions to *reference* them instead of
   creating them. This is far cheaper per run and makes results comparable
   between runs.

---

## 3. Finding 2 — one observation in six is of a blank page

**42 of 256 observations (16%) on DataGhurhi showed `0 controls`.**

Navigation uses `wait_until="domcontentloaded"`:

```python
# web_player/actions.py:188
await self.page.goto(url, timeout=..., wait_until="domcontentloaded")
# web_player/browser.py:80  — same for the per-test reset
```

DataGhurhi is a React SPA. `domcontentloaded` fires when the HTML document is
parsed — *before* React mounts anything. The agent therefore observes an empty
page, and must spend a turn on `wait` to recover. Seen live in this report's run:

```
[2/30] .../v/999-nonexistent
    think: The page indicates it may still be loading with no interactive elements found yet.
    act:   {"action": "wait", "seconds": 2}
    page:  0 controls
```

At roughly 5 wasted steps per test case out of 30, this is **~17% of the entire
budget** spent waiting for renders that the driver should have waited for.

**Fix.** After navigation, wait for the app to settle before observing — either
`wait_until="networkidle"`, or `domcontentloaded` followed by a short
`wait_for_selector` on a stable app root, or a bounded retry in
`snapshot.observe` that re-reads once when it finds zero controls. The last is
the most robust, because it also covers in-page route changes that never fire a
navigation event at all.

---

## 4. Finding 3 — the profile and the specification contradict each other

`dataghurhi-auth` is **permanently signed in**: `storage_state: "auth.json"`,
holding a JWT valid until 2026-10-11 (checked — auth is *not* broken).

It ingests `data/inputs/dataghurhi-spec.md`, which contains **54 requirements,
11 of them `FR-AUTH`** and 14 marked `[PUBLIC]` — registration, sign-in,
password policy, email validation. Both DataGhurhi profiles ingest the *same*
spec; they differ only in `storage_state` and their blocked-control lists.

Nothing tells the planner that the authenticated profile should not draw from
the public/auth requirements. So it does. TC-002 of this run:

> *Preconditions: user is NOT signed in*
> `think: the precondition requires the user to be not signed in... I need to
> verify the authentication state or find the login entry point`

The agent then looped on the user menu until the livelock guard fired. It could
not possibly succeed: it is signed in, and `logout` is in `blocked_texts`, so it
is *forbidden by the guardrails* from reaching the state the test requires.

Roughly **20% of the requirement pool is impossible-by-construction** on this
profile, and those tests consume a full budget each before failing.

### It is self-reinforcing — this is the key point

The planner is explicitly steered toward coverage gaps. From
`planner/coverage.py:143`:

```python
f"[EXPAND] Areas with ZERO test coverage yet: {', '.join(uncovered[:5])}"
```

So the loop is:

1. the `FR-AUTH` requirements can never be covered on a permanently signed-in profile
2. they therefore stay in `uncovered_purposes` forever
3. the planner is told to prioritise uncovered areas, so it writes another auth test
4. the agent burns a whole test slot discovering it cannot sign out
5. coverage does not move — go to 2

**The target does not drift toward the reachable parts of the application; it
drifts toward the unreachable ones.** In the fresh run for this report, TC-002
and TC-003 were both auth tests, both ended `BLOCKED_BY_GUARDRAIL`, and both
died trying to click `Logout`. That is 2 of the first 3 slots spent on tests
that were impossible before the run began.

**Fix.** Split the knowledge, not just the session:

- give `dataghurhi` (signed out) a spec containing the `[PUBLIC]` requirements
- give `dataghurhi-auth` a spec containing only the authenticated ones
- or, if the graph should stay unified, set the profile's `OUT_OF_SCOPE` to the
  `FR-AUTH` ids so the planner is told they cannot be reached — the mechanism
  already exists and is used by the ShobarKhamar Android campaign.

---

## 5. Finding 4 — a reasoning model can return an empty answer, and we mislabel it

Now directly relevant, because the executor model was just changed to
`qwen/qwen3.7-flash`, which is a **reasoning** model.

`web_player/llm.py:81`:

```python
return choices[0].get("message", {}).get("content", "") or ""
```

When such a model spends its whole token allowance on the hidden scratchpad, the
API returns HTTP 200 with `content: null` and `finish_reason: "length"`.
Reproduced directly against OpenRouter:

```
max_tokens=8     finish=length  content=NULL   reasoning_chars=28
max_tokens=1500  finish=stop    content='{"action":"fill",...'  reasoning_chars=1617
```

At 1500 tokens the model is fine for a small page — but `reasoning_chars` grows
with prompt size, and `WEB_LLM_MAX_TOKENS = 1500` is shared between scratchpad
and answer. When it does not fit, the client returns `""`, `parse_action("")`
yields `_error`, and the agent logs:

> `model reply was not a JSON action — reprompting`

…burning a step, with **nothing anywhere recording that the real cause was token
exhaustion**. In the prior runs this cost 5 steps; the misleading part is the
diagnosis, not the count.

**Fix.** Three small changes in `llm.py`:

1. Raise `WEB_LLM_MAX_TOKENS` to ~4000 for reasoning models.
2. When `content` is empty, fall back to `message.reasoning` before giving up —
   the JSON is often in there.
3. If `finish_reason == "length"` and content is empty, raise a descriptive
   `LLMError` naming token exhaustion instead of silently returning `""`.

Also worth noting: when probed with a loosely-worded prompt this model returned a
JSON **array** of actions using `element_id`, and on a second call used
`_action`/`_element` keys — neither matches the contract. The real system prompt
is far stricter and the observed rate of unparseable replies was low (5 in 182
steps, 2.7%), so this is a watch item, not a present cause.

---

## 6. Finding 5 — a dropped connection to OpenRouter is filed as a DataGhurhi defect

**The most damaging bug in this report**, because it is the only one that puts a
false finding on the *site's* record rather than merely wasting a run.

TC-004 of the fresh run ended:

```
❌ CRASH: ('Connection aborted.',
           ConnectionResetError(10054, 'An existing connection was forcibly closed
                                by the remote host'))
```

That connection was to **openrouter.ai**, not to DataGhurhi. It was recorded as:

| field | value |
|---|---|
| `error_type` | `CRASH` |
| `CRASH ∈ APP_FAULT` | **true** |
| `device_steps` | `0` (≈10 steps were actually taken) |
| `error_message` | *(empty)* |
| duration | 276 s |

So a network blip on our side is now in the graph as a crash discovered in
DataGhurhi, with its step count erased.

**Why it slipped through.** There are two independent gaps, and they compound:

```python
# web_player/llm.py
_RETRY_TOKENS = ("429", "500", "502", "503", "504", "too many requests",
                 "overloaded", "timed out", "timeout")
```

1. `('Connection aborted.', ConnectionResetError(...))` contains none of those
   tokens, so `_is_transient()` returns **False** and the call is **not
   retried** — although a single retry would almost certainly have succeeded.
2. A `requests.exceptions.ConnectionError` is not an `LLMError`, so the runner's
   `except LLMError` — added precisely to stop provider outages being blamed on
   the site — never sees it. It falls to the generic `except Exception`, which
   writes `error_type="CRASH"`.

An earlier fix in this project made `LLMError` an environment fault. That fix was
incomplete: it covered failures the client *raises*, not failures the underlying
HTTP library raises through it.

**Fix.**

1. Add connection-level failures to `_RETRY_TOKENS` — `connection aborted`,
   `connection reset`, `connection refused`, `remote end closed`, `max retries` —
   or better, catch `requests.exceptions.RequestException` by type rather than by
   string matching.
2. Wrap every non-HTTP transport failure in `LLMError` before it leaves
   `ChatClient.chat()`, so the runner's existing `LLM_UNAVAILABLE` path handles
   it and it lands in `ENV_FAULT`.
3. Pass the real step count into `log_execution` on the crash path instead of a
   hardcoded `0`, and make sure `error_message` is actually populated.

---

## 7. Finding 6 — a near-miss: the agent tried to change the account password

**This one is a safety issue, not a throughput issue.**

TC-004 of the fresh run was *"update the user's password... setting the new
password to the same value"*. Two of our own defects combined:

**(a) Nothing in the guardrails stops it.** `dataghurhi-auth` blocks 25 controls
— delete, logout, pay, publish, register, OTP — but **not** `Update Password`,
`Change Password`, `Save Changes` or `Submit`. On a permanently signed-in profile
against a real BUET account, a successful password change would have invalidated
the credentials and locked every future run out of the account.

**(b) Our password masking handed the model a fake value to type.**
`snapshot.py:120` renders a password field as `'*'.repeat(raw.length)`. That is
right for the log — a secret must never be written there. But the agent then
*reads it back as data*:

```
think: I need to enter the current password... I will use 'password' as a likely candidate.
act:   {"action": "fill", "ref": "e5", "text": "password"}
think: I need to fill the new password field with the same value as the current password.
act:   {"action": "fill", "ref": "e6", "text": "********"}       <-- the MASK, typed as a value
act:   {"action": "click", "ref": "e8"}                          -> "Update Password"
```

It typed **eight literal asterisks** as the new password, because that is what our
observation showed it.

**What saved the account was the site's own password policy:**

> `Password must be at least 8 characters long, include a number and a special character.`

The change was rejected. Nothing was damaged. But had the policy been laxer, the
account password would now be `********` and `auth.json` would be worthless.

**Fixes, both small and both needed:**

1. Add the account-mutating controls to `blocked_texts` for every authenticated
   profile: `update password`, `change password`, `delete account`,
   `deactivate`. Credentials-changing controls belong in the same class as
   `logout`, for exactly the same reason — they end the session the batch runs on.
2. Distinguish "masked for display" from "this is the value". Render a password
   field as `value=<hidden, 12 chars>` rather than a string of asterisks that
   looks like data, so the model cannot mistake the mask for the secret.

---

## 8. Finding 7 — the new executor model is markedly worse at the JSON contract

The executor model was changed from `minimax/minimax-m3` to
`qwen/qwen3.7-flash` just before this run, to match the team's `.env`. Measured
over DataGhurhi steps only:

| Model | Steps | Unparseable replies | Rate |
|---|---|---|---|
| `minimax/minimax-m3` (prior runs) | 163 | 5 | **3.1%** |
| `qwen/qwen3.7-flash` (this run) | 48 | 11 | **22.9%** |

Nearly one turn in four is thrown away because the reply is not a JSON action.
That is a 7× regression and, after the profile mismatch, the largest single
source of waste in this run.

It is **not** token exhaustion. Probed with the real system prompt and a
realistic 60-element page, `qwen3.7-flash` at `max_tokens=1500` returns valid
JSON using 608 completion tokens against 2141 characters of hidden reasoning —
comfortable headroom. The cause is formatting: in looser probes the same model
returned a JSON *array* of actions, and used `element_id` / `_action` / `_element`
keys instead of `ref` / `action`.

**We cannot say more than that, because the failing replies are discarded.**
`parse_action()` returns a generic `_error` and the raw text is never logged:

```python
return {"action": "_error", "reason": "no JSON action object found in the reply"}
```

**Fix.** Log the first ~200 characters of any reply that fails to parse. Without
it, a 23% failure rate is undiagnosable. Then either revert the executor model to
`minimax/minimax-m3` (the planner can stay on `qwen3.7-flash`; they are separate
settings) or relax `parse_action` to accept the first element of a JSON array and
to map the common key aliases.

---

## 9. Finding 8 — the test suite writes into the operator log

`tests/test_web_player.py` drives a stub agent that writes real trace lines into
`logs/web_player.log`. Consequences:

- any analysis of a real run is contaminated unless scoped by host
- the dashboard's **🌐 Web (Playwright)** panel interleaves fixture output with
  live runs
- it produced a false finding during this very investigation

**Fix.** Point `web_player/trace.py` at a temp file when running under test (an
env var such as `WEB_TRACE_FILE`, or monkeypatching `trace.LOG_PATH` in the
fixture).

---

## 10. What the site itself is telling us

Not agent faults — these are real, reproducible signals from DataGhurhi, caught
by the passive oracles rather than by LLM judgement:

- **A persistent `403` on the translation endpoint**, on every page, every run:
  `Translation error: AxiosError: Request failed with status code 403`. This is
  worth a bug report on its own and likely explains a non-functioning language
  toggle.
- **Accordion headers are not controls.** On `/security-settings`, "Change
  Password" and "Secret Question & Answer" are `<div class="sec-card-header">`
  with `cursor: pointer`, no `role`, no `tabindex`, no `aria-expanded`. They are
  not keyboard reachable — an accessibility defect. (The agent could not see
  them at all until the snapshot was taught to treat `cursor: pointer` as
  interactive.)
- **TC-004 in the earlier run hit `Failed to create survey.`** after a
  double-submit — possibly a real defect, possibly agent-induced. Worth a manual
  reproduction.

---

## 11. Fixes applied

All eight findings were fixed on 11 Sep 2026. Verified by `tests/run_all.py`
(**10/10 modules, 338 checks**) plus a live check against the site.

| Finding | Change | Where |
|---|---|---|
| 1 — step budget | `max_steps` 30 → **80**, `timeout` 420 → **600** | `dataghurhi-auth.json` |
| 2 — blank observations | `goto` waits for `networkidle` (falling back to `domcontentloaded`); `observe()` re-reads up to 3× when it finds nothing | `actions.py`, `browser.py`, `snapshot.py` |
| 3 — profile/spec conflict | spec split into a signed-in and a signed-out edition; 10 unreachable requirements removed from the authenticated one | `dataghurhi-auth-spec.md`, `dataghurhi-public-spec.md` |
| 4 — empty answer | `content: null` now falls back to `reasoning`, else raises a named `LLMError`; `WEB_LLM_MAX_TOKENS` 1500 → **4000** | `llm.py`, `.env` |
| 5 — transport error as app defect | transport failures are transient **by type** and normalise to `LLMError`; the crash path logs the real step count and route | `llm.py`, `agent.py`, `runner.py` |
| 6 — password near-miss | credential controls blocked; a filled password renders as *"contains 12 hidden characters"* instead of `********` | `dataghurhi-auth.json`, `snapshot.py` |
| 7 — off-contract replies | `parse_action` accepts action arrays, key aliases and `[e7]` refs; unparseable replies are now logged | `llm.py`, `agent.py` |
| 8 — log pollution | `WEB_TRACE_FILE` redirects the transcript; both test modules point at a temp file | `trace.py`, `tests/` |

### Guardrails rebalanced

Five controls were **unblocked** because they prevented legitimate testing of the
agent's own fixtures:

| Unblocked | Why it was hampering |
|---|---|
| `delete`, `remove`, `মুছে ফেলুন` | blocked "Delete question" and "Remove option" — no survey-design requirement could be exercised at all |
| `publish` | a survey must be published before any `FR-RESP-*` response requirement can be reached |
| `invite` | opening the collaborator dialog is harmless; only *sending* reaches a real person |

Ten were **added**, all irreversible or costly: `update password`,
`change password`, `reset password`, `delete account`, `deactivate account`,
`close account`, `delete project`, `delete survey`, `delete response`,
`delete all`.

The principle applied: block what cannot be undone or what reaches the outside
world; allow what the agent can create and clean up itself.

---

## 11b. Verification runs — what the fixes actually changed

Three batches were run after the fixes. Each failed for a shallower reason than
the last, and the final blocker is not code.

| | Run 2 (before model fix) | Run 3 | Run 4 |
|---|---|---|---|
| Test cases reached | 1 | 3 | 2 |
| **Passed** | 0 | 0 | **1** |
| Auth tests drawn (impossible) | 0 | 0 | 0 |
| `BLOCKED_BY_GUARDRAIL` | 0 | 0 | 0 |
| Misattributed to the site | 0 | 0 | 0 |
| Step counts recorded | — | correct | correct |

**Confirmed fixed by live evidence:**

- **The coverage trap is broken.** Before: 4 of 5 test cases were `FR-AUTH`
  requirements needing a signed-out session. After: **zero**, across all three
  runs. Every test case drawn was reachable.
- **No false defects.** Not one failure was recorded as `APP_FAULT`.
- **Accounting is honest.** `steps=29`, `steps=24` — no more hardcoded `0`.
- **A real test passed.** Run 4 TC-001 executed 18 steps and reached a genuine
  verdict on an i18n requirement: Bangla text typed into a free-text question was
  preserved verbatim across an interface language switch.

**Two further bugs found by these runs, both fixed:**

1. **Native HTML5 validation was invisible.** Run 3's TC-002 was *"submit an
   empty project title"*. DataGhurhi uses native `required` validation, whose
   message is a browser tooltip present in no DOM node. The agent deduced it —
   *"the clicks aren't producing visible changes because the HTML5 required
   validation is preventing submission"* — but could not observe it, probed six
   ways and tripped the wandering guard. **While testing validation.**
   `snapshot.py` now reads `checkValidity()` / `validationMessage`; the same form
   now reports
   `REJECTED BY THE BROWSER: "Please fill out this field."`
2. **A budget overrun was retried with the same budget.** Deterministic, so all
   attempts failed identically. The retry now doubles the allowance.

### Final run — 4 rounds, all completed

With credits restored, the whole batch ran to completion for the first time.

| # | Test case | Steps | Outcome |
|---|---|---|---|
| TC-001 | FR-RESP-02 — invalid survey slug shows an explicit error | 2 | **PASS** |
| TC-002 | draft preservation on navigating away | 21 | livelock |
| TC-003 | (survey navigation flow) | 13 | wandering |
| TC-004 | navigation away leaves the app in a sane state | 14 | **PASS** |

**2 passed / 4 — and, more importantly, no failure was the harness's fault:**

- 4 of 4 rounds completed; no batch abort, no `LLM_UNAVAILABLE`, no `CRASH`
- **0** `BLOCKED_BY_GUARDRAIL`, **0** auth tests drawn
- **0** failures attributed to `APP_FAULT` — nothing false filed against the site
- unparseable replies down to **2 of 50 steps (4%)** from 22.9%
- mean 12.5 steps per test against an 80 budget — the budget is no longer the constraint

Before/after on the same target:

| | Original run | Final run |
|---|---|---|
| Rounds completed | 5 | 4 of 4 |
| Passed | 1 (trivial, 3 steps) | **2** (2 and 14 steps) |
| Impossible auth tests | 4 of 5 | **0** |
| Blocked by guardrails | 2 | **0** |
| Misattributed to the site | 1 (`CRASH`) | **0** |
| Wasted on unparseable replies | 22.9% | **4%** |

### What is still imperfect

Both remaining failures are `NAVIGATION_LIVELOCK` — the agent looping while trying
to reach a survey through the project → survey → preview chain. Notably the *same*
draft-preservation test passed in 12 steps on an earlier run and looped for 21 on
this one, so this is variance in the agent's exploration, not a broken fix. The
guards caught it correctly in both cases and attributed it to the agent.

This is the honest remaining gap: multi-step navigation into nested resources is
where the agent still gets stuck. The pre-seeded-fixture recommendation (item 4
below) is the direct answer to it — if the survey already exists and the test
references it, the chain the agent keeps losing its way in disappears.

### The earlier blocker was billing, not code

Run 4 ended on:

```
OpenRouter 402: This request would exceed your available credits
```

```
total_credits: 5    total_usage: 5.16     <- account overspent
key limit: $2       limit_remaining: $1.21
```

Note the interaction: OpenRouter pre-authorises the **maximum** cost a request
could incur, so the escalating retry introduced above could itself provoke a 402
on a nearly-spent key. Its ceiling was therefore capped at 8000 tokens — a terse
model answers a browser step in roughly 200.

**A five-round batch cannot be demonstrated end to end until the account is
topped up.** Everything upstream of that is fixed and evidenced.

---

## 12. Recommended order of work

| # | Change | Effort | Expected effect |
|---|---|---|---|
| 1 | `max_steps` → 80–100 for DataGhurhi | one line | tests can reach their assertion |
| 2 | Wait for the SPA to settle before observing | small | recovers ~17% of the budget |
| 3 | Scope auth requirements out of the `-auth` profile | small | stops ~20% of impossible tests |
| 4 | Pre-seed fixtures; rewrite preconditions to reference them | medium | largest quality gain per run |
| 5 | Treat transport errors as `LLM_UNAVAILABLE`, and retry them | small | stops false defect records (finding 5) |
| 6 | Block `update password` / `change password`; stop masking as a value | small | protects the account (finding 6) |
| 7 | Log unparseable replies; revert the executor model | small | recovers ~23% of steps (finding 7) |
| 8 | Keep test fixtures out of the operator log | small | trustworthy logs and dashboard |

Items 1–3 are configuration and a few lines of code, and together should move
this target from "never reaches a verdict" to "reaches a verdict most of the
time". Item 4 is what makes runs *comparable*, which is what the evaluation
harness needs.

---

## 13. A note on the headline claim

"None of the testing is passing and it fails due to our agent error" is accurate,
but the framing understates how well one part is working: **the failure
attribution is correct.** The runs are being recorded as `STEP_LIMIT_EXCEEDED`
and `NAVIGATION_LIVELOCK`, which are agent/environment faults, and *not* as
`ASSERTION_FAILURE`, which would have counted as defects discovered in
DataGhurhi. A less careful system would have reported eight bugs in a site where
it had found none. The problem to solve is throughput, not honesty.
