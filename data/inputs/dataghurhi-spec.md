# DataGhurhi — Requirements Specification

**Provenance.** Written on 12 Sep 2026 from the running application at
`https://dataghurhi.cse.buet.ac.bd`, not from description. Three sources, in
order of authority:

1. **The application's own router and client bundle** — 31 declared routes and
   129 `/api/*` paths, read out of `/assets/index-*.js`.
2. **A read-only walk of every route** with an authenticated session, recording
   headings, form fields, controls and the API calls each page issued.
3. **Direct probes** of individual endpoints to separate "exists" from "named".

The previous edition of this document was written without seeing the site. It
claimed 19 API routes; several did not exist, one was misspelled, and whole
subsystems (admin, payments, vouchers, question bank, template sharing) were
missing. Anything below that is *not* directly observed is marked
**[UNVERIFIED]** so it can never again be mistaken for fact.

---

## 0. Surfaces

**Public** — reachable signed out: `/`, `/v/:slug`, `/signup`,
`/forgot-password`, `/about`, `/faq`, `/faq/:topic`, `/faq/help-videos`.

**Authenticated** — everything else: `/dashboard`, `/addproject`,
`/view-survey/:survey_id`, `/survey-responses/:survey_id`, `/my-templates`,
`/my-templates/:template_id/edit`, `/edit-profile`, `/security-settings`,
`/subscription`, `/saved-files`, `/search-results`, `/analysis`, `/preprocess`,
`/visualization`, `/report`, `/qualitative-analysis`,
`/qualitative-analysis/:survey_id`, `/group-preview`, `/user-response-view`,
`/preview`, `/survey-success`, `/home`.

**Observed oddity.** `/signup` and `/forgot-password` render their forms **while
a session is active**, rather than redirecting a signed-in user away. Worth a
decision: intended, or a missing guard?

**Rendered nothing.** `/about`, `/preview` and `/user-response-view` produced
zero interactive controls and no heading on a signed-in visit. Either they are
unfinished, or they require state this walk did not have.

### Verified API surface

`data/inputs/dataghurhi-api.md` holds the evidence table, and
`data/api/<project>.json` the machine-readable list the executor is given.
**No test may cite an endpoint absent from that list.** The route is
`/api/project` — singular. `/api/projects` does not exist.

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

### FR-AUTH-07 Secret question [AUTH]
DETAILED DESCRIPTION: `/security-settings` shall let a signed-in user set and
update a secret question and answer
(`/api/profile/get-secret-question`, `/api/profile/update-secret-question`).
The stored answer shall never be displayed again in clear.

### FR-AUTH-08 Password change [AUTH]
DETAILED DESCRIPTION: `/security-settings` shall offer a password change
requiring the current password (`/api/profile/match-password`,
`/api/profile/update-password`), and shall enforce the policy in FR-AUTH-09.
*Observed: the section is a collapsed card; expanding it reveals current, new and
confirm fields plus "Update Password".*
**Out of scope for automation:** changing it invalidates the stored session.

### FR-AUTH-09 Password policy
DETAILED DESCRIPTION: A password shall be at least 8 characters and include a
number and a special character. *Observed verbatim: "Password must be at least 8
characters long, include a number and a special character."*

### FR-AUTH-10 Profile [AUTH]
DETAILED DESCRIPTION: `/edit-profile` shall present basic info and contact
details and save them (`/api/profile/update-profile`,
`/api/profile/update-profile-image`). *Observed: sections "Basic Info" and
"Contact", with a "Save Profile" control.*

---

## 3. Projects

### FR-PROJ-01 Creation [AUTH]
DETAILED DESCRIPTION: `/addproject` shall create a project from a name, a
research field, an optional description and a visibility of Private or Public
(`POST /api/project/create-project`). *Observed: exactly those fields, with
Private/Public radios, "Create Project" and "Cancel".*

### FR-PROJ-02 Required fields [AUTH]
DETAILED DESCRIPTION: Project name and research field are required and enforced
by native browser validation. *Observed: submitting empty yields the browser's
own "Please fill out this field." on both.*

### FR-PROJ-03 Listing [AUTH]
DETAILED DESCRIPTION: `/dashboard?tab=projects` shall list the caller's projects
(`GET /api/project`) plus those shared with them
(`GET /api/collaborator/all-projects`), and no others.

### FR-PROJ-04 Detail and surveys [AUTH]
DETAILED DESCRIPTION: Opening a project shall show its surveys
(`GET /api/project/:id/surveys`) and the caller's access level
(`GET /api/project/:id/fetchaccess`).

### FR-PROJ-05 Update and deletion [AUTH]
DETAILED DESCRIPTION: A project may be renamed (`/api/project/:id/update-project`)
and deleted (`/api/project/:id/delete-project`).
**Out of scope for automation:** deletion is irreversible.

### FR-PROJ-06 Title is escaped, not executed [AUTH]
DETAILED DESCRIPTION: A project title containing markup shall be displayed as
text. *Observed: a project literally titled `<script>alert('XSS')</script>`
renders as a label and does not execute — correct behaviour, left by earlier
testing.*

---

## 4. Survey design

### FR-SURV-01 Builder [AUTH]
DETAILED DESCRIPTION: `/view-survey/:survey_id` shall present the survey editor
with its questions, loaded from `GET /api/surveytemplate/:id`.

### FR-SURV-02 Question types [AUTH]
DETAILED DESCRIPTION: The editor shall offer Checkbox, Radio, Text, Dropdown,
Rating, Likert Scale, Linear Scale, Date/Time and Tick Box Grid. *Observed
verbatim from the type selector.*

### FR-SURV-03 Saving [AUTH]
DETAILED DESCRIPTION: Edits shall persist via `/api/surveytemplate/save` and
confirm. *Observed: "Survey Saved successfully!".*

### FR-SURV-04 Settings [AUTH]
DETAILED DESCRIPTION: Per-survey settings shall be readable and writable at
`/api/surveytemplate/settings/:survey_id`.

### FR-SURV-05 Preview [AUTH] [UNVERIFIED]
DETAILED DESCRIPTION: The editor shows a "Preview" control. *Observed: clicking
it changed nothing on the page — no navigation, no panel, no dialog.* Whether it
is unimplemented, or requires an unsaved-changes state, is unestablished. **This
is a defect candidate.**

### FR-SURV-06 AI question generation [AUTH]
DETAILED DESCRIPTION: The builder may generate questions via
`/api/generate-question-with-llm/` and `/api/generate-multiple-questions-with-llm`,
and tags via `/api/generate-tags/`.

### FR-SURV-07 Quotas [AUTH]
DETAILED DESCRIPTION: Creating surveys, questions and tags decrements the
caller's allowance (`/api/reduce-survey-count`, `/api/reduce-question-count`,
`/api/reduce-tag-count`) and shall be refused once exhausted.

---

## 5. Templates and question bank

### FR-TMPL-01 My templates [AUTH]
DETAILED DESCRIPTION: `/my-templates` shall list the caller's saved templates
(`GET /api/get-saved-survey`, `/api/get-saved-survey/user-template`).

### FR-TMPL-02 Editing [AUTH]
DETAILED DESCRIPTION: `/my-templates/:template_id/edit` shall edit a template.
 *Observed: `GET /api/get-saved-survey` returns saved templates with an `id`, so the `:template_id` route has real values to address.*

### FR-TMPL-03 Sharing [AUTH]
DETAILED DESCRIPTION: A template may be shared, and invitations accepted,
declined, duplicated, left or revoked
(`/api/template-share/*`). `GET /api/template-share/shared-with-me` lists those
shared with the caller.

### FR-QB-01 Question bank [AUTH]
DETAILED DESCRIPTION: `/api/question-bank` shall store reusable questions, with
create, update, delete, share, revoke and `semantic-search`.

---

## 6. Collaboration

### FR-COLL-01 Project collaborators [AUTH]
DETAILED DESCRIPTION: A project owner may invite collaborators
(`/api/project/:id/invite-collaborator`) and list them
(`GET /api/project/:id/collaborators`).

### FR-COLL-02 Invitations [AUTH]
DETAILED DESCRIPTION: `GET /api/collaborator/all-invitations` lists pending
invitations; each may be accepted or declined.
**Out of scope for automation:** sending reaches a real mailbox.

### FR-COLL-03 Survey collaborators [AUTH]
DETAILED DESCRIPTION: Surveys carry their own collaborator set
(`/api/survey-collaborator/*`), including removal and a search over previously
used addresses.

---
 *Observed: `GET /api/survey-collaborator/get-survey-collaborators/602` returns `{"collaborators":[],"canManage":true}`.*
## 7. Responses, data and analysis

### FR-DATA-01 Responses [AUTH]
DETAILED DESCRIPTION: `/survey-responses/:survey_id` shall present a survey's
responses with Summary, Responses and Questions views. *Observed, along with a
Download control and `GET /api/generatecsv/:id`.*

### FR-DATA-02 Export [AUTH]
DETAILED DESCRIPTION: Responses shall be exportable as CSV.
 *Observed: `GET /api/generatecsv/:id` returns 200 with `content-type: text/csv` and quoted rows, including Bangla headers.*

### FR-DATA-03 Personal storage [AUTH]
DETAILED DESCRIPTION: `/saved-files` ("Personal Storage") shall list files the
caller has saved.
 *Observed: `/saved-files` renders the heading "Personal Storage" with 22 controls.*

### FR-ANLY-01 Statistical analysis [AUTH]
DETAILED DESCRIPTION: `/analysis` shall accept a data file by drag-and-drop,
browse, or import from saved files, and run analyses (`/api/sa/*`).

### FR-ANLY-02 Preprocessing [AUTH]
DETAILED DESCRIPTION: `/preprocess` shall present the data as an editable grid
with a formula bar and operations including Find and Replace and Categorize
Column, savable to a folder. *Observed verbatim.*

### FR-ANLY-03 Qualitative analysis [AUTH]
DETAILED DESCRIPTION: `/qualitative-analysis` and
`/qualitative-analysis/:survey_id` shall analyse free-text responses.
 *Observed: renders "Qualitative Analysis" and "Start a new analysis".*

### FR-ANLY-04 Visualization [AUTH]
DETAILED DESCRIPTION: `/visualization` shall render charts from selected data.
 *Observed: renders "Visualize the Data".*

### FR-ANLY-05 Reports [AUTH]
DETAILED DESCRIPTION: `/report` shall list saved analysis reports.
 *Observed: renders "Statistical Analysis Reports".*

### FR-ANLY-06 Grouped preview [AUTH]
DETAILED DESCRIPTION: `/group-preview` shall preview grouped data.

---
 *Observed: renders "Preview of Grouped Data".*
## 8. Subscription and payment

### FR-SUBS-01 Packages [AUTH]
DETAILED DESCRIPTION: `/subscription` shall show "Premium Packages" and
"Subscription History" (`GET /api/get-user-packages`).

### FR-SUBS-02 Purchase [AUTH]
DETAILED DESCRIPTION: A package may be purchased
(`/api/subscription/create`, `/api/payment/initiate`).
**Out of scope for automation:** it spends money.

### FR-SUBS-03 Vouchers [AUTH]
DETAILED DESCRIPTION: Vouchers may be validated and redeemed
(`/api/vouchers/validate`, `/api/vouchers/public`, `/api/voucher-used/create`).

---

## 9. Search, FAQ and internationalisation

### FR-SRCH-01 Global search [AUTH]
DETAILED DESCRIPTION: The header search shall query projects, surveys and
accounts, filtered by a scope selector of All / Project / Survey / Account
(`GET /api/search`), with results at `/search-results`.

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

## 10. Administration [UNVERIFIED]

The client declares 18 `/api/admin/*` routes — packages, validity periods, unit
price, coupons, revenue, user- and survey-growth statistics. **No admin UI route
was found among the 31 declared**, and the test account cannot reach them. Out of
scope until an admin surface and account are identified.

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