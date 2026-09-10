# DataGhurhi — Exploratory Testing Campaign Report

**Target:** `https://dataghurhi.cse.buet.ac.bd` · **Project slice:** `dataghurhi`
**Date:** 11 Sep 2026 · **Rounds:** 5 · **Player:** web (Playwright/Chromium, headed)
**Planner model:** `deepseek/deepseek-v4-flash-0731` · **Executor model:** `qwen/qwen3.5-flash-02-23`
**Specification:** [`data/inputs/dataghurhi-spec.md`](../../data/inputs/dataghurhi-spec.md) — 67 requirements, 8 guardrails

---

## 1. Headline

**The campaign found no defects, and it produced almost no evidence about DataGhurhi's
quality. Autonomy was 0%: all five runs were lost to our own agent or our own guardrails,
not to the application.**

This is reported first because the opposite reading — "5 failed tests, 5 bugs" — is the
single most damaging misreading available here. In this system a `failed` verdict spans
three different things, and only one of them is evidence about the app.

| Test | Attribution | Class | Duration | Steps |
|---|---|---|---|---|
| TC-001 | `NAVIGATION_LIVELOCK` | agent fault | 17.4s | 5 |
| TC-002 | `NAVIGATION_FAILURE` | agent fault | 223.5s | 60 |
| TC-003 | `BLOCKED_BY_GUARDRAIL` | environment | 3.6s | 1 |
| TC-004 | `STEP_LIMIT_EXCEEDED` | environment | 138.8s | 30 |
| TC-005 | `BLOCKED_BY_GUARDRAIL` | environment | 7.1s | 3 |

**App faults: 0 of 5.** Requirement coverage: **7 / 82**, and those seven are covered only
in the sense that a test *cited* them — none was actually exercised against the app.

---

## 2. What the agent actually did

Every run converged on the same place. The screenshot captured at the end of TC-004
(`logs/web_shots/TC-004-failed.png`) shows it exactly: the agent is sitting on the
**login page**, having typed an invented email address (`2105002@ugrad.cse.buet.ac.bd`)
and a password, with the site answering *"Please complete the reCAPTCHA verification."*

Routes reached across the whole campaign:

| Route | Visits |
|---|---|
| `/` (login wall) | 101 |
| `/visualization` | 4 |
| `/analysis` | 4 |
| `/forgot-password` | 4 |
| `/signup` | 2 |
| `/login` | 1 |

101 of ~120 observations were the same signed-out landing page.

---

## 3. Why it failed — four causes, in order of impact

### 3.1 The anonymous surface is a login wall (spec error, mine)

`data/inputs/dataghurhi-spec.md` classifies `/`, `/home`, `/about`, `/faq` and `/v/:slug`
as a **public** surface worth 19 requirements. In practice the site root *is* the sign-in
page for a signed-out visitor, and the routes the planner wanted (`/analysis`,
`/visualization`, `/preprocess`) bounce straight back to it.

The surface table in the spec was derived from the React route table in the JS bundle,
which lists routes that *exist* — not routes that *render anything without a session*.
That inference was wrong and the campaign paid for it in full.

### 3.2 Nothing enforces the PUBLIC/AUTH split the spec documents

The spec marks every requirement `[PUBLIC]` or `[AUTH]`, precisely so an unauthenticated
run would stay on reachable ground. **The planner never sees that distinction** — the
markers are prose inside requirement text, not a filter. So it generated:

- TC-001 — chart axis labels (`/visualization`, AUTH)
- TC-002 — analysis column-type refusal (`/analysis`, AUTH)
- TC-004 — analysis with only categorical columns (`/analysis`, AUTH)
- TC-003 — password change (AUTH)
- TC-005 — password recovery (public, but see 3.3)

Five for five on the authenticated surface, with no credentials configured. These tests
were impossible before the browser opened.

### 3.3 The guardrails did their job, and that ended two tests instantly

TC-003 died after **one step** and TC-005 after **three**:

> `Refused to activate 'Sign Up' — it matches the blocked control 'sign up' for this target.`
> `Refused to activate 'Send OTP' — it matches the blocked control 'send otp' for this target.`

Both refusals are correct — `OOS-02` forbids creating accounts and sending real email to
real addresses. But a password-recovery test whose only route to a verdict is *pressing
Send OTP* cannot succeed under that rule. The guardrail and the test case were in direct
contradiction, and nothing detected that before spending a round on it.

### 3.4 reCAPTCHA gates the one action that would open everything

Login is protected by reCAPTCHA, so even correct credentials would not let an automated
run through unaided. This was visible in the JS bundle before the campaign
(`recaptchaToken` on submit, `"রিক্যাপচা যাচাই সম্পন্ন করুন"` in the Bangla strings) and
is confirmed by the TC-004 screenshot.

---

## 4. What was observed about the application anyway

The browser oracles collect signals regardless of whether a test reaches a verdict, so
the campaign is not entirely empty. **All of the following are unvalidated candidates
and need a human to confirm them.**

### 4.1 The translation feature returns 403 on the live site — candidate defect

Present on **every** test, on ordinary page loads:

```
Translation error: AxiosError: Request failed with status code 403
    at Ike (https://dataghurhi.cse.buet.ac.bd/assets/…)
Failed to load resource: the server responded with a status of 403 ()
```

DataGhurhi advertises multilingual operation prominently — *"Create multilingual reports
(33 languages)"*, *"৩৩টি ভাষায় সার্ভে তৈরি করুন"* — and the client calls the Google
Translate v2 endpoint directly. A 403 there means the translation path is failing in
production for anonymous visitors.

Relates to `FR-I18N-01` / `FR-I18N-02`. Most likely an API key that is expired, out of
quota, or referrer-restricted. **Confirm manually** by opening the site and switching
language; if the UI silently stays in English, `FR-I18N-02`'s fallback requirement is met
in appearance but the feature is broken underneath.

### 4.2 A 404 on a page resource

`Failed to load resource: the server responded with a status of 404 ()` — captured but not
attributed to a specific request. Low confidence, low value without the URL.

### 4.3 No crashes, no 5xx

Across ~120 page observations, no uncaught page exception and no 5xx from the application's
own backend. `WEB_FAIL_ON_HTTP_5XX` was on for this run, so a server error would have
failed a test outright. None did. That is a genuine, if narrow, positive signal about the
signed-out surface.

---

## 5. Agent behaviour worth flagging

**The agent invented credentials and attempted to log in.** It typed
`2105002@ugrad.cse.buet.ac.bd` — a plausible BUET student address it was never given —
and submitted the login form. The attempt failed at reCAPTCHA and no account was accessed.

This was not blocked because `blocked_texts` deliberately leaves *Log in* clickable, so
that sign-in **validation** (`FR-AUTH-05`, `FR-AUTH-06`) stays testable. The tradeoff is
now visible: on a real system, repeated failed logins against invented addresses are
indistinguishable from credential stuffing in the target's own logs. Recommend blocking
`log in` as well unless credentials are supplied.

**The wandering guard worked.** TC-001 ended at 17.4s with `NAVIGATION_LIVELOCK` rather
than burning all 30 steps — the repeat-detection added to `web_player/agent.py` firing as
designed on a page that did not respond.

---

## 6. Defects found in *our own system* during this campaign

Three, all fixed or flagged in the course of the run:

1. **Session context leaked across projects.** `targets/env.py` sets `APP_LOGIN_ROLE` only
   for `android` profiles, so a web run inherited whatever `.env` held. The gateway was
   still carrying ShobarKhamar's session block (*"signed in as a 'farmer/seller'… already
   has a farm named 'Trust Dairy Farm'"*) and injected it into every generation prompt as
   *"Session Constraints — these override every other instruction."* The first test
   generated for DataGhurhi was **"Edit farm name to 255 characters"**. Worked around by
   restarting the gateway with a neutral session env; **not yet fixed in code** — the
   profile configures the runner process, but the gateway that generates test cases is
   long-lived and reads `.env`.

2. **The web player could not run at all against the current planner.** kmazd-v2's
   goal-based redesign removed `steps` from the planner's output contract (replacing it
   with `screen_hint` + `objective`), and the Android executor was migrated to match — but
   `web_player/` was not. `runner.py` hard-required `steps` and aborted every run with
   *"Planner returned an empty test case."* Migrated `goal.py` and `runner.py` to the new
   contract.

3. **Playwright was never installed**, so the web player had never successfully run on this
   machine. It is listed in `requirements.txt`; the venv has no `pip`, so it was installed
   with `uv pip install`.

---

## 7. What to change before the next campaign

Ordered by how much each unlocks.

1. **Supply credentials.** Two-thirds of the spec is `[AUTH]`. Capture a session once with
   `playwright codegen --save-storage=auth.json` and set `web.storage_state` in the
   profile — this also sidesteps reCAPTCHA, which no automated login can pass. Without
   this, no further campaign will produce meaningfully different results.

2. **Create a test survey you own**, then unblock `submit` and point at its `/v/<slug>`.
   That converts Section 1 of the spec (19 requirements — the richest, most defect-dense
   part) from untestable to testable, without touching the live research study.

3. **Make the PUBLIC/AUTH split machine-readable** so an unauthenticated run cannot draw
   an `[AUTH]` requirement. Simplest form: keep two spec files and ingest only the public
   one when no credentials are configured.

4. **Fix the session leak in code** (item 6.1) so a project's app identity travels with the
   request rather than living in the gateway's process env.

5. **Reconcile guardrails against generated tests before execution.** A test whose verdict
   depends on a blocked control should be discarded at generation time, not discovered
   3.6 seconds into a round.

---

## 8. Honest summary

This campaign tested our testing system more than it tested DataGhurhi. It surfaced three
real defects in our own pipeline and one credible candidate defect in the target (the
translation 403). As an assessment of DataGhurhi's quality it is **not usable** — 0% of
runs produced app evidence, and the spec's own public-surface assumption turned out to be
wrong.

The next run, with a captured session and a self-owned test survey, is the one whose
numbers will mean something.

---

### Artefacts

| Item | Path |
|---|---|
| Specification | `data/inputs/dataghurhi-spec.md` |
| Target profile | `targets/profiles/dataghurhi.json` |
| Campaign log | `logs/dataghurhi_campaign.log` |
| Agent trace (per step) | `logs/web_player.log` |
| Final screenshots | `logs/web_shots/TC-00{1..5}-failed.png` |
| Knowledge graph | Neo4j project slice `dataghurhi` — 82 requirements, 85 validation rules, 28 embedded chunks |
