# DataGhurhi — Exploratory Testing Campaign Report

**Target:** `https://dataghurhi.cse.buet.ac.bd` · **Dates:** 11–12 Sep 2026
**Campaigns:** 4 × 5 rounds · **Player:** web (Playwright/Chromium, headed)
**Planner model:** `deepseek/deepseek-v4-flash-0731` · **Executor model:** `qwen/qwen3.5-flash-02-23`
**Specifications:** [`dataghurhi-spec.md`](../../data/inputs/dataghurhi-spec.md) (runs 1–3), [`dataghurhi-auth-spec.md`](../../data/inputs/dataghurhi-auth-spec.md) (run 4)
**Auto-generated PDF for run 4:** [`dataghurhi-run4-report.pdf`](dataghurhi-run4-report.pdf) — **its "1 defect" is wrong; see §3.5**

---

## 1. Headline

**Twenty runs produced no confirmed defect in DataGhurhi.**

Run 4 recorded one app-attributed verdict — TC-001, a whitespace-only project name
"refused with no feedback". **That verdict was false.** DataGhurhi does give feedback: a
native browser alert, *"Please fill in all required fields"*. Our web player never listens
for native dialogs, and Playwright dismisses them automatically, so the agent never saw
it. My own reproduction used the same driver and repeated the mistake. The error was
caught by the tester trying the form by hand (§3.5).

| | Run 1 — anonymous | Run 2 — signed in | Run 3 — account probe | Run 4 — jonayed's fixes merged |
|---|---|---|---|---|
| What stopped the runs | login wall | guardrail on *Publish* | invisible modal, load race, budget | invisible modal, finding the right screen, CAPTCHA, invisible alert |
| Agent faults | 2 | 0 | 4 | 3 |
| Environment / guardrail | 3 | 5 | 1 | 1 |
| **App faults** | **0** | **0** | **0** | **1 recorded — false (our tool)** |

---

## 2. Runs 1–3 (summary)

- **Run 1 — anonymous.** The signed-out site is a login wall; my spec's public-surface
  table was derived from routes that exist, not routes that render without a session.
  The agent invented a BUET student email and tried to log in (stopped by reCAPTCHA).
- **Run 2 — signed in** via a session captured by hand. Three tests stopped at the
  blocked *Publish*; a fixed "no projects" account state made each round rebuild the
  same project.
- **Run 3 — live account state added.** No more duplicate projects, and a test
  published a survey for the first time. Four runs livelocked: the dashboard briefly
  shows "No Projects Found" while loading, and the "Create New Project" modal carries no
  dialog role, so the agent kept clicking behind it.

---

## 3. Run 4

### 3.1 What changed

jonayed's branch was merged in first. From it:
- the page settles (`networkidle` plus re-reads of a blank snapshot) before the agent looks —
  this removed run 3's "No Projects Found" misread;
- a **CAPTCHA hand-off**: a headed run pauses up to 180 s, beeping, for a person to solve it;
- a run lock, 80 steps and 600 s per test, and an **auth-only spec** (71 requirements) so
  no signed-out test can be drawn.

Kept from run 3: the live account probe, and the oracle fix (aborted requests are never
counted as server errors).

### 3.2 Results

| Test | Objective | Recorded result | Actually | Duration | Steps |
|---|---|---|---|---|---|
| TC-001 | a whitespace-only name is rejected | `ASSERTION_FAILURE` | **false defect — the app showed an alert the agent could not see (§3.5)** | 32.3s | 8 |
| TC-002 | analysis refuses categorical-only columns | `NAVIGATION_LIVELOCK` | agent could not find the screen | 232.3s | 58 |
| TC-003 | analysis refuses an unsupported data type | `NAVIGATION_LIVELOCK` | invisible modal | 260.2s | 48 |
| TC-004 | an expired survey refuses new responses | `BLOCKED_BY_CAPTCHA` | precondition never set up (§3.4) | 231.7s | 21 |
| TC-005 | preprocessing rejects a wrongly typed cell edit | `NAVIGATION_LIVELOCK` | agent oscillated between two screens | 132.9s | 34 |

169 steps in 889 s. Every probe observed the account holding 7 projects; only TC-005
created another, with a distinct name, for its own data-type test.

### 3.3 The livelock still happens — for two different reasons

| Test | Cause |
|---|---|
| TC-003 | **The invisible modal.** Clicks behind "Create New Project" were refused with the browser naming `modal-overlay-modern` as the element in the way. The agent's own reasoning shows the gap: *"previous steps failed due to modal overlay, but current page shows no visible modal."* |
| TC-002 | **Could not find the screen.** Clicked the "Test Categorical Analysis" project six times without reaching an analysis view for that survey. |
| TC-005 | **Oscillated** between "open the project" and "Quantitative Analysis", looking for the preprocessing screen, until the guard stopped it. |

TC-004 also hit a modal (the *Survey Link* dialog) mid-run, but recovered.

### 3.4 TC-004 — the CAPTCHA hand-off worked; the verdict hides that

1. The agent opened its own published survey's public form (`/v/b39-c02`) and hit reCAPTCHA.
2. The run paused, beeping: *"CAPTCHA — YOUR INPUT IS NEEDED"*.
3. After 120 s: *"solved after 120s — continuing"* — a person solved it.
4. The agent clicked **Submit**; DataGhurhi showed its success page (`/survey-success`).
   Its next thought: *"submitted successfully despite expecting it to be closed — this
   suggests a potential defect. I need to go back and verify the survey settings."* It did,
   and found the end date was not in the past — so it did **not** file a false defect.
5. It then went back to the form, met a **fresh** CAPTCHA, and finished as
   `BLOCKED_BY_CAPTCHA`.

The recorded verdict describes step 5 and erases step 4. The objective concerned an
**expired** survey and the agent never set one up; accepting the response was correct.
The honest outcome is *precondition not met*.

### 3.5 TC-001 — a false defect, and why verification missed it

**What the agent reported.** It typed a project name of only spaces, pressed *Create
Project*, saw nothing happen, and concluded the app refused silently.

**What actually happens.** DataGhurhi raises a native browser alert —
*"dataghurhi.cse.buet.ac.bd says: Please fill in all required fields"* — and creates
nothing. That is correct behaviour: the input is refused and the user is told.

**Why the agent could not see it.** A native `alert()` is drawn by the browser, not the
page, so it appears in no snapshot. Playwright dismisses any dialog nobody listens for,
and `web_player/` never listens. The message existed for a moment and vanished unread.

**Why my reproduction agreed with the agent.** I checked for new page text, validation
messages, style changes and DOM mutations — all page-level channels, and all run through
the same Playwright driver that auto-dismisses the alert. Three methods, one blind spot.
A rerun with a dialog listener attached captures the alert immediately:
`('alert', 'Please fill in all required fields')`.

**What caught it.** The tester trying the form by hand.

**The PDF** counts this as the run's one defect and describes it as a whitespace title
not being rejected. Both are wrong; the recorded verdict in the knowledge graph is still
`ASSERTION_FAILURE` and should be corrected before the PDF is regenerated.

---

## 4. Findings about DataGhurhi

All are **unconfirmed candidates**, found by investigating failed runs by hand. None came
from an agent verdict. None depends on native dialogs, so none is affected by the blind
spot in §3.5 — but each still needs a human to confirm it.

| # | Finding | Evidence | Relates to |
|---|---|---|---|
| 4.1 | **Translation requests fail with 403** on every page load. The browser sends a Google API key in the request URL and Google refuses it — consistent with a restricted or disabled key | every run, signed in and out | `FR-I18N-01/02` |
| 4.2 | **The session JWT travels in a URL query string** (`/api/surveytemplate/stream/<id>?token=…`) | live traffic and the shipped bundle | `NFR-SEC` |
| 4.3 | **A project with no surveys returns 404** (`GET /api/project/{id}/surveys` → `No surveys found for this project`) instead of an empty list | reproduced on 4 projects | `FR-PROJ-03`, `FR-DATA-03` |
| 4.4 | **The dashboard shows "No Projects Found" while still loading** | timed at ~2.5–3.0 s | `FR-DATA-03` |
| 4.5 | **The "Create New Project" modal has no dialog semantics** (no `role="dialog"`, no `aria-modal`), unlike "Create New Survey" | DOM inspected | `NFR-A11Y-01` |

**Checked and ruled out:**
- **Whitespace-only project name** — refused, with an alert. Not a defect (§3.5). A minor
  observation only: the generic *"fill in all required fields"* does not name the field,
  and the field visibly contains something (spaces). A usability nicety, not a fault.
- Global search works; duplicate project names are allowed by the spec; no crashes and no
  genuine 5xx in four campaigns.

---

## 5. Agent behaviour worth flagging

- **TC-001 — accurate about what it could see, unaware of what it could not.** Nothing
  on the page changed, so it reported nothing changed. The fault is the tool's (§6.12).
- **TC-004 — caught its own false defect** (§3.4), then let the final verdict overwrite
  real progress.
- **Run 1 — invented credentials** and attempted a login. Block `log in` on profiles
  without credentials.

---

## 6. Defects found in our own system

| # | Defect | Status |
|---|---|---|
| **6.12** | **Native browser dialogs are invisible and silently dismissed.** `web_player/` never listens for `dialog` events, so Playwright auto-dismisses every `alert`, `confirm` and `prompt`. Validation shown in an alert reads to the agent as "no response" — a **false app defect**, the most damaging misclassification this system can make. A `confirm("Are you sure?")` is silently answered **Cancel**, so any flow behind a confirmation can never be completed or tested | **not fixed** — produced the only app verdict in four campaigns |
| 6.1 | Session context leaks across projects (`targets/env.py` maps the login role only for Android; the gateway reads identity from its own environment) | worked around; **not fixed** |
| 6.2 | Web player incompatible with the goal-based planner | fixed (run 1) |
| 6.3 | Playwright never installed | fixed (run 1) |
| 6.4 | Oracle counted aborted requests as server 5xx | fixed (run 3) |
| 6.5 | Account state was a fixed snapshot | fixed (run 3) |
| 6.6 | Screenshots from one campaign overwrite another's | **not fixed**; reviewed screenshots copied to `docs/reports/assets/` |
| 6.7 | First snapshot taken before a single-page app loads | fixed (jonayed) |
| 6.8 | The snapshot cannot see a modal without ARIA dialog semantics, and lists controls behind any modal as clickable | **not fixed** — 1 of 3 livelocks in run 4 |
| 6.9 | No data fixture: analysis and preprocessing tests must find and build their own data | **not fixed** — 2 of 3 livelocks in run 4 |
| 6.10 | A test's final verdict ignores what it achieved earlier (TC-004) | **not fixed** |
| 6.11 | The PDF narrative describes defects from the test title, not the evidence | **not fixed** |

---

## 7. Housekeeping

- **Throwaway account now holds 9 projects** — #262–#268 and #278 from the agent, #279
  (`V`) from manual testing. **Survey #597 is published, public, and holds at least one
  submitted response** (TC-004). Delete or unpublish by hand.
- `auth.json` is a live session (gitignored). Delete it when done.
- **The gateway is running with DataGhurhi's session context.** Restart it plainly before
  any ShobarKhamar or contacts-app run (see `RUN_COMMANDS.txt`, section 2).

---

## 8. What to change before run 5

1. **Handle native dialogs (§6.12).** Listen for `dialog` events; put the message into the
   agent's next observation and into the browser findings; let the agent choose accept or
   dismiss, with guardrail text checked against the message. Then correct TC-001's recorded
   verdict and regenerate the PDF.
2. **Only list controls that can actually receive a click (§6.8)** — hit-test each
   element's centre with `document.elementFromPoint`, and name what covers it.
3. **Seed a data fixture once per campaign (§6.9)** so analysis and preprocessing tests
   start on the right screen.
4. **Record the furthest point a test reached (§6.10)**; classify an impossible setup as
   `PRECONDITION_NOT_MET`.
5. **Build the PDF narrative from verdict notes (§6.11)**; fix the session leak (§6.1);
   key screenshots by campaign (§6.6).

---

## 9. Honest summary

Across four campaigns the pipeline learned to sign in, build and publish surveys, and hand
a CAPTCHA to a person. It has not yet produced a confirmed defect in DataGhurhi: its only
app verdict came from a native alert it could not see.

The lesson is about verification as much as the agent. A defect reported by a tool must be
reproduced through a **different channel** from the one the tool uses — ideally a person
at the screen — because checks run through the same driver share its blind spots. That is
what went wrong here, and a manual test is what put it right.

---

### Artefacts

| Item | Path |
|---|---|
| Specifications | `data/inputs/dataghurhi-spec.md`, `data/inputs/dataghurhi-auth-spec.md` |
| Target profiles | `targets/profiles/dataghurhi.json`, `targets/profiles/dataghurhi-auth.json` |
| Campaign logs | `logs/dataghurhi_campaign.log` (1), `logs/dataghurhi_auth_campaign.log` (2), `logs/dataghurhi_auth_run3.log` (3), `logs/dataghurhi_auth_run4.log` (4) |
| Agent trace | `logs/web_player.log` (all runs, in time order) |
| Run 4 PDF | `docs/reports/dataghurhi-run4-report.pdf` — superseded on TC-001 by §3.5 |
| Preserved screenshots | `docs/reports/assets/dataghurhi-2026-09-11/` — incl. `run4-TC-001-alert-hidden-from-agent.png` |
