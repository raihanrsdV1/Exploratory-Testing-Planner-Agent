# DataGhurhi — Requirements for the PUBLIC surface

Companion to `dataghurhi-spec.md`, which remains the complete reference. This
edition is ingested by the **`dataghurhi`** target profile, which runs with no
stored session and is therefore signed out throughout.

## Session the tests run under

- The browser is **not signed in** and must stay that way. Signing in would move
  the run onto the authenticated surface, which the `dataghurhi-auth` profile
  covers.
- Anything marked `[AUTH]` in the reference specification is out of scope here
  and is deliberately absent from this document.
- Creating an account is forbidden: it writes a real record on a live
  installation. Registration is specified below so that its **validation
  behaviour** can be exercised — fill the form and observe what it rejects —
  never so that an account is actually created.

---
## 1. Public survey response — `/v/:slug`

### FR-RESP-01 Rendering [PUBLIC]
DETAILED DESCRIPTION: A published survey opened at `/v/:slug` shall render its
title and its questions. The page is populated from
`GET /api/fetch-survey-user/:slug`. *Observed: `/v/e07-ba4` renders the heading
"Required Field Test Survey", three radio options and a free-text "other" field.*

### FR-RESP-02 Unknown or unpublished slug [PUBLIC]
DETAILED DESCRIPTION: A slug that does not resolve shall render an explicit
message and a route back. It shall not render a blank page, an endless spinner,
or a raw error. *Observed: `/v/999-nonexistent` renders "Error | This survey does
not exist." with a "Return Home" link.*

### FR-RESP-03 Required-question enforcement **[UNVERIFIED]**
*Could not be confirmed: submitting the response form produced no visible validation, because the reCAPTCHA gates submission before any field check runs. Testable only once the test environment uses reCAPTCHA test keys.*
DETAILED DESCRIPTION: A question marked required shall prevent submission until
answered, and the refusal shall name the offending question.

### FR-RESP-04 Answer types
DETAILED DESCRIPTION: The response form shall support single-choice, an "other"
free-text option alongside choices, multi-choice, free text, and the question
types the builder offers (Section 4). Each shall record what the respondent
selected and nothing else.
 *Observed on `/v/e07-ba4`: radio options plus an "other" free-text box. Only one survey was available to sample, so the full type list in FR-SURV-02 is confirmed from the builder, not from a rendered response form.* **[PARTIALLY VERIFIED]**

### FR-RESP-05 Human verification [PUBLIC]
DETAILED DESCRIPTION: The response page presents a reCAPTCHA challenge before
submission. *Observed: an `I'm not a robot` widget on `/v/e07-ba4`.*
**Testing consequence:** submission is unreachable to automation until the test
environment uses reCAPTCHA test keys. Anything before the challenge — rendering,
validation, retention — remains testable.

### FR-RESP-06 Submission and confirmation
DETAILED DESCRIPTION: A completed submission shall persist the response and
confirm it. *Observed: `/survey-success` renders "Survey Response Submitted
Successfully".*

### FR-RESP-07 Answer retention on a refused submission
DETAILED DESCRIPTION: When a submission is refused, previously entered answers
shall remain in the form.
 *Observed: a value typed into the "other" box was still present after the submission was refused.*

### FR-RESP-08 Draft retention across navigation
DETAILED DESCRIPTION: A partially completed response shall survive navigating
away and returning to the same slug. *Observed: a value typed into the "other"
box was still present after navigating to `/dashboard` and back.*

**Defect candidate.** The value returns but the radio that enables the box does
not: the field comes back holding its old answer while `disabled`. The stored
answer and the control state disagree, so a respondent cannot edit what the form
is still showing them.

---


## 2. Authentication and account

### FR-AUTH-01 Registration [PUBLIC]
DETAILED DESCRIPTION: `/signup` shall collect a title, name and email address and
begin verification by one-time code. *Observed: three text fields and a
"Send OTP" button; `POST /api/register`, `/api/register/check-email`,
`/api/send-otp`, `/api/verify-otp` exist in the client.*

### FR-AUTH-02 Email uniqueness [PUBLIC]
DETAILED DESCRIPTION: Registration shall refuse an address already registered,
via `/api/register/check-email`, and say so without revealing other account data.

### FR-AUTH-03 One-time code [PUBLIC] **[UNVERIFIED]**
DETAILED DESCRIPTION: Registration shall require a code sent to the address.
A wrong or expired code shall be refused. **Out of scope for automation:** the
code arrives out of band.

### FR-AUTH-04 Sign-in [PUBLIC]
DETAILED DESCRIPTION: `POST /api/login` shall accept valid credentials and refuse
invalid ones with a message that does not disclose whether the address is
registered.

### FR-AUTH-05 Password recovery [PUBLIC]
DETAILED DESCRIPTION: `/forgot-password` shall describe the reset path and
accept a recovery request (`/api/login/reset-password`) without revealing whether
an address is registered.

### FR-AUTH-06 Session and access control
DETAILED DESCRIPTION: An authenticated route requested without a session shall
refuse access rather than render its contents or another user's data.
 *Observed: `/dashboard` requested with no session renders the public landing page, not the dashboard; `GET /api/project` without a token returns 401.*

### FR-AUTH-09 Password policy
DETAILED DESCRIPTION: A password shall be at least 8 characters and include a
number and a special character. *Observed verbatim: "Password must be at least 8
characters long, include a number and a special character."*


## 9. Search, FAQ and internationalisation

### FR-FAQ-01 Help [PUBLIC]
DETAILED DESCRIPTION: `/faq` shall present browsable topics (`GET /api/faq`),
`/faq/:topic` a single topic, and `/faq/help-videos` video help.

### FR-I18N-01 Language [PUBLIC]
DETAILED DESCRIPTION: The interface offers a language switch. Switching shall not
lose entered data. *Observed: content in both English and Bangla — `/faq/help-videos`
renders "সাহায্য ভিডিও".*

### FR-I18N-02 Bangla text [PUBLIC]
DETAILED DESCRIPTION: Bangla text, including conjuncts, shall be accepted,
stored and redisplayed unchanged.
 *Observed: Bangla with conjuncts typed into Project Name read back byte-identical; CSV export preserves Bangla column headers.*

### FR-I18N-03 Translation service
DETAILED DESCRIPTION: The translation call shall succeed.
**Observed defect:** it returns **403 on every page, every run** —
`Translation error: AxiosError: Request failed with status code 403`.

---


## 11. Non-functional

### NFR-01 Client health
DETAILED DESCRIPTION: No ordinary journey shall produce an uncaught exception or
a 5xx. **Observed violations:** `GET /api/project/create-project` → 500;
`GET /api/sa/save-results/` → 500; `GET /api/project/:id/surveys` → 500 for some
ids. A GET on a create endpoint should be 404 or 405, never a server error.

### NFR-02 Accessibility
DETAILED DESCRIPTION: Interactive controls shall be reachable by keyboard and
expose a name and role. **Observed violation:** the `/security-settings`
accordion headers are `<div class="sec-card-header">` with `cursor: pointer`, no
`role`, no `tabindex` and no `aria-expanded` — not keyboard reachable.

### NFR-03 Responsive layout
DETAILED DESCRIPTION: At 375 CSS pixels the interface shall remain usable with no
horizontal page scrolling.
 *Observed at 375px on `/dashboard` and `/v/:slug`: scrollWidth equals clientWidth — no horizontal page scrolling.*
