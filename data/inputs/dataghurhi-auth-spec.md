# DataGhurhi — Requirements for the SIGNED-IN surface

Companion to `dataghurhi-spec.md`, which remains the complete reference. This
edition is ingested by the **`dataghurhi-auth`** target profile, which runs with a
stored session and is therefore permanently signed in.

## Session the tests run under

- The browser starts **already signed in**, from a saved session. There is no
  sign-in step to perform and no sign-in form to reach.
- **Signing out is forbidden.** The batch shares one session; ending it would
  abort every remaining test. `Logout` is blocked in the dispatcher, not merely
  discouraged.
- Therefore **no test may have a precondition of "the user is not signed in."**
  Registration, sign-in, password recovery and the public password policy are
  covered by the `dataghurhi` (signed-out) profile and are deliberately absent
  from this document.
- **Changing the account password is forbidden** for the same reason: it
  invalidates the saved session and locks every later run out of the account.

## Writing a testable precondition

Prefer preconditions the agent can satisfy in two or three actions. A test that
spends its whole step budget building fixtures never reaches its own assertion.

Good: *"Open an existing project from the project list."*
Bad: *"Create a project, then a survey, then add a Likert question with
requireEachRowResponse set, then open preview."*

Where a fixture is genuinely needed, create the smallest possible one and assert
on it immediately.

---
## 1. Public survey response — `/v/:slug`

The primary anonymous surface. A published survey is opened by slug (for example
`/v/53d-edc`) and rendered from `GET /api/fetch-survey-user/:slug`.

### FR-RESP-01 Survey rendering [PUBLIC]
DETAILED DESCRIPTION: Opening a valid survey slug shall render the survey's
title, its description, and its questions grouped under their section headings,
in the order defined by the template. The page shall render without requiring
sign-in when the survey's `isLoggedInRequired` flag is false.

RATIONALE: A respondent arriving from a shared link is anonymous by default; a
sign-in wall on an open survey silently destroys the response rate.

### FR-RESP-02 Unknown or unpublished slug [PUBLIC]
DETAILED DESCRIPTION: A slug that does not exist, or that names a survey which is
not published, shall present an explicit "not found / unavailable" message. It
shall NOT present a blank page, an endless loading indicator, or a raw error
object.

RATIONALE: A blank page is indistinguishable from a failed load, and is the most
common way a broken link is misdiagnosed as a network problem.

### FR-RESP-03 Section presentation
DETAILED DESCRIPTION: Sections shall display their title when `showTitle` is set,
and their description when one is present. When `autoNumbering` is enabled for a
section, its questions shall be numbered contiguously and consistently; numbering
shall not restart, skip, or duplicate within a section.

### FR-RESP-04 Required-question enforcement
DETAILED DESCRIPTION: A question marked `required` shall be visibly marked as
required, and submission shall be refused while it is unanswered. The refusal
shall name or scroll to the specific unanswered question. Submission shall not be
silently discarded, and the page shall not navigate away.

RATIONALE: A refusal that does not identify *which* question is missing forces the
respondent to re-read the whole form; on a 43-question instrument that is an
abandonment.

### FR-RESP-05 Optional-question handling
DETAILED DESCRIPTION: A question not marked `required` shall be submittable while
left blank, and its absence shall not block submission or be recorded as an
empty-string answer that is indistinguishable from a deliberate blank.

### FR-RESP-06 Text answers
DETAILED DESCRIPTION: A `text` question shall accept free text, including:
- leading and trailing whitespace (trimmed before storage);
- Unicode beyond Latin-1, specifically Bangla script (for example `ঢাকা`);
- long answers of at least 2,000 characters without layout overflow or client error;
- characters with markup meaning (`<`, `>`, `&`, `"`, `'`) stored and re-displayed
  as literal text, never interpreted as markup.

RATIONALE: The platform is explicitly multilingual and research-facing; a text
field that mangles Bangla or executes markup is both a data-integrity and a
security defect.

### FR-RESP-07 Single-choice (radio) answers
DETAILED DESCRIPTION: A `radio` question shall permit exactly one selection from
its options. Selecting a second option shall replace the first, never add to it.

### FR-RESP-08 "Other" option
DETAILED DESCRIPTION: A question with `otherAsOption` shall present an "Other"
choice with an accompanying free-text input. Selecting "Other" shall enable that
input; the typed value shall be submitted in place of a fixed option. Selecting
"Other" and leaving the text empty on a *required* question shall be refused.

### FR-RESP-09 Multi-choice (checkbox) answers
DETAILED DESCRIPTION: A `checkbox` question shall permit zero or more selections.
Where `requireAtLeastOneSelection` is set, submission shall be refused until at
least one box is ticked, with a message stating that at least one selection is
required.

### FR-RESP-10 Likert matrix answers
DETAILED DESCRIPTION: A `likert` question shall render its statement rows against
its scale columns (for example: Strongly Agree · Agree · Neutral · Disagree ·
Strongly Disagree · Never Used). Exactly one column shall be selectable per row.
Where `requireEachRowResponse` is set, submission shall be refused until every row
carries a response, with a message stating that a response is required for each
row.

RATIONALE: A matrix that permits two answers in one row, or that reports "complete"
with rows unanswered, corrupts the analysis silently — the response is stored and
looks valid.

### FR-RESP-11 Other supported question types
DETAILED DESCRIPTION: Where a survey uses them, the platform shall render and
validate: `dropdown` (one selection), `linear` scale, `rating`, `date`, `time`,
tick-box grid, and file upload. Each shall enforce its own type constraint — a
date field shall reject a non-date, a rating shall not exceed its maximum.

### FR-RESP-12 File-upload answers
DETAILED DESCRIPTION: A file-upload question shall accept a file, display the
selected filename, permit removal before submission, and enforce any configured
size and type limits with an explicit message. An oversized or wrong-type file
shall be refused at selection time, not silently at submit time.

### FR-RESP-13 Submission and confirmation
DETAILED DESCRIPTION: A valid submission shall be recorded via
`POST /api/submit-survey/:id` and shall present an explicit confirmation stating
that the response was recorded. The confirmation shall be a distinct state — not
merely a cleared form, which is indistinguishable from a lost submission.

### FR-RESP-14 Double submission
DETAILED DESCRIPTION: Activating the submit control twice in rapid succession
shall record at most one response. The control shall become inert, or the second
attempt shall be rejected.

RATIONALE: Duplicate responses are indistinguishable from genuine ones after the
fact and silently bias every statistic computed from the dataset.

### FR-RESP-15 Answer retention on a refused submission
DETAILED DESCRIPTION: When submission is refused for a validation reason, every
answer already provided shall be retained. A validation failure shall never clear
the form.

RATIONALE: Discarding twenty minutes of answers over one missed required field is
the most expensive single defect this surface can have.

### FR-RESP-16 Closed and time-bounded surveys
DETAILED DESCRIPTION: A survey whose `ending_date` has passed, or whose
`collect_response` flag is false, shall refuse new responses and state why. It
shall not present a submittable form that silently discards the answer.

### FR-RESP-17 Sign-in-required surveys
DETAILED DESCRIPTION: A survey with `isLoggedInRequired` set shall require an
authenticated respondent, and shall say so on arrival rather than after the form
is filled in.

### FR-RESP-18 Preview mode
DETAILED DESCRIPTION: A survey opened in preview shall state that responses will
not be recorded, and shall not record them.

### FR-RESP-19 Question shuffling
DETAILED DESCRIPTION: When `shuffleQuestions` is enabled, question order shall
vary between renders while every question remains present exactly once. Shuffling
shall never drop, duplicate, or reorder a question across a section boundary.

---


## 2. Authentication and account

### FR-AUTH-09 Secret question [AUTH]
DETAILED DESCRIPTION: A user shall be able to set and update a secret question and
answer (`/api/profile/get-secret-question`, `/api/profile/update-secret-question`)
and the answer shall never be displayed in plain text after it is stored.


## 3. Projects

### FR-PROJ-01 Project creation [AUTH]
DETAILED DESCRIPTION: A signed-in user shall create a project
(`POST /api/project/create-project`) with a title. A project shall not be created
with an empty or whitespace-only title.

### FR-PROJ-02 Project listing [AUTH]
DETAILED DESCRIPTION: `GET /api/project` shall list only projects the requesting
user owns or has been invited to collaborate on. A project belonging to another
user shall never appear.

### FR-PROJ-03 Surveys within a project [AUTH]
DETAILED DESCRIPTION: A project shall list its surveys
(`GET /api/project/:id/surveys`) and permit creating a new one
(`POST /api/project/:id/create-survey`).

---


## 4. Survey design

### FR-SURV-01 Survey builder [AUTH]
DETAILED DESCRIPTION: The builder shall permit adding, editing, reordering, and
deleting sections and questions, and shall persist the result
(`POST /api/surveytemplate/save`). An unsaved change shall not be lost silently on
navigation — the user shall be warned or the change preserved.

### FR-SURV-02 Question configuration [AUTH]
DETAILED DESCRIPTION: Each question shall support setting its text, type,
required flag, options, and (where applicable) `otherAsOption`,
`requireAtLeastOneSelection`, and `requireEachRowResponse`. A choice-type question
shall refuse to be saved with zero options, with a message stating that at least
one option is required.

### FR-SURV-03 Survey settings [AUTH]
DETAILED DESCRIPTION: Settings (`/api/surveytemplate/settings/:survey_id`) shall
control at least: sign-in requirement, response collection on/off, question
shuffling, and ending date. A setting saved shall be the setting applied on the
public response page.

RATIONALE: A settings screen whose values do not take effect on the live survey is
worse than none — it reports a protection that is not in force.

### FR-SURV-04 Templates [AUTH]
DETAILED DESCRIPTION: A survey shall be savable as a template
(`/api/get-saved-survey/save-as-template`), listable (`/my-templates`), editable,
and copyable (`/api/surveytemplate/copy/:id`). Copying shall duplicate content
without linking the copy's later edits back to the original.

### FR-SURV-05 Question bank [AUTH]
DETAILED DESCRIPTION: Questions shall be creatable in
(`POST /api/question-bank/create`), searchable in — including semantic search
(`/api/question-bank/semantic-search`) — and reusable from a question bank. A
semantic search returning nothing shall say so explicitly rather than presenting
an empty region.

### FR-SURV-06 AI question generation [AUTH]
DETAILED DESCRIPTION: The AI generator
(`/api/generate-question-with-llm/`, `/api/generate-multiple-questions-with-llm`)
shall produce questions of a valid type with valid options, insert them into the
survey under construction, and remain editable afterwards. A generation failure or
timeout shall be reported to the user, never left as a silent no-op.

### FR-SURV-07 Quiz mode [AUTH]
DETAILED DESCRIPTION: A survey with `is_quiz` set shall support per-question point
values, a total, and a marks-release policy (immediately or later), and shall
present the score to the respondent only according to that policy.

---


## 5. Collaboration and sharing

### FR-COLL-01 Invitations [AUTH]
DETAILED DESCRIPTION: A survey owner shall invite a collaborator by email
(`/api/survey-collaborator/send-survey-collaboration-request`). The invitee shall
see the invitation (`/api/survey-collaborator/all-invitations`) and be able to
accept or decline it.

### FR-COLL-02 Collaborator permissions [AUTH]
DETAILED DESCRIPTION: A collaborator shall access only the surveys or projects
they were invited to, and only at the permission level granted. Removal
(`/remove-collaborator/:id`) shall revoke that access immediately.

RATIONALE: This is the platform's principal data-isolation boundary; a leak here
exposes another researcher's unpublished instrument and respondent data.

---


## 6. Responses and data collection

### FR-DATA-01 Response listing [AUTH]
DETAILED DESCRIPTION: `/survey-responses/:survey_id` shall list responses to a
survey the user owns or collaborates on, with the answer to each question
attributable to the response it came from.

### FR-DATA-02 Response export [AUTH]
DETAILED DESCRIPTION: Responses shall be exportable. An export shall contain every
recorded response and every question, with Bangla text preserved intact in the
exported file.

RATIONALE: Encoding loss at export is invisible in the application and fatal in
the analysis.

### FR-DATA-03 Empty state [AUTH]
DETAILED DESCRIPTION: A survey with no responses yet shall state that explicitly.
It shall not render an empty table indistinguishable from a failed load.

---


## 7. Preprocessing, analysis and visualization

### FR-ANLY-01 Column selection guards [AUTH]
DETAILED DESCRIPTION: An analysis requiring a minimum number or type of columns
shall refuse to run until they are chosen, and shall say what is missing — for
example that at least one numerical column, at least one categorical column, or at
least two columns for a network graph, is required.

### FR-ANLY-02 Type correctness [AUTH]
DETAILED DESCRIPTION: An operation shall refuse a column whose data type it cannot
process, with a message naming the mismatch, rather than producing a result
computed from coerced or dropped values.

RATIONALE: A statistic silently computed over coerced data is a wrong answer
presented with full confidence — the worst failure mode an analysis tool has.

### FR-ANLY-03 Preprocessing operations [AUTH]
DETAILED DESCRIPTION: Preprocessing (`/preprocess`) shall support at least
duplicate detection, column deletion, and cell edits, and shall reject a cell edit
whose value does not match the column's data type.

### FR-ANLY-04 Quantitative analysis [AUTH]
DETAILED DESCRIPTION: `/analysis` shall run the offered statistical procedures over
the selected columns and persist results (`/api/sa/save-results/`). A procedure
that cannot run on the given data shall explain why.

### FR-ANLY-05 Qualitative analysis [AUTH]
DETAILED DESCRIPTION: `/qualitative-analysis/:survey_id` shall support coding and
theming of free-text responses, and shall require at least one code and one theme
before rendering a connection view, saying so when they are absent.

### FR-ANLY-06 Visualization [AUTH]
DETAILED DESCRIPTION: `/visualization` shall render a chart from the selected data
with labelled axes and a legend where more than one series is present, and shall
remain readable when a label is long or non-Latin.

### FR-ANLY-07 Report and saved files [AUTH]
DETAILED DESCRIPTION: `/report` and `/saved-files` shall list artefacts belonging
to the user, permit download, and shall not expose another user's artefacts.

---


## 8. Subscription and packages

### FR-SUBS-01 Package presentation [AUTH]
DETAILED DESCRIPTION: `/subscription` shall present available packages with their
price, validity period, and included quotas as returned by the server, not as
hard-coded values.

### FR-SUBS-02 Quota enforcement [AUTH]
DETAILED DESCRIPTION: Actions that consume quota (survey creation, question
creation, tag creation) shall decrement the user's remaining allowance and shall
refuse the action, with an explicit message, once the allowance is exhausted.

RATIONALE: A quota that is displayed but not enforced, or enforced but not
displayed, both manifest as the user losing work at an unpredictable moment.

---


## 9. Internationalisation

### FR-I18N-01 Language switching [PUBLIC]
DETAILED DESCRIPTION: The interface shall offer a language switch. Switching shall
translate interface text without losing the user's place, without clearing entered
data, and without leaving a mixture of both languages in the same view.

### FR-I18N-02 Content translation [PUBLIC]
DETAILED DESCRIPTION: Where survey content is translated, the translation shall not
alter the meaning of scale labels, and an untranslatable string shall fall back to
the original rather than rendering blank.

### FR-I18N-03 Bangla input and rendering [PUBLIC]
DETAILED DESCRIPTION: Bangla text shall be accepted in every free-text field,
stored without corruption, re-displayed identically, and rendered with correct
conjunct glyph shaping.

---


## 10. Non-functional requirements

### NFR-PERF-01 Page responsiveness
DETAILED DESCRIPTION: A public survey page shall become interactive within 3
seconds on a normal connection. A materially slower load is a performance defect
and shall be reported with the measured duration.

### NFR-PERF-02 Large instruments
DETAILED DESCRIPTION: A survey of at least 50 questions across at least 10 sections
shall render, scroll, and submit without a materially degraded interaction.

### NFR-SEC-01 Transport
DETAILED DESCRIPTION: Every request shall be served over HTTPS. A page shall not
load an active mixed-content subresource.

### NFR-SEC-02 Input neutralisation
DETAILED DESCRIPTION: No user-supplied value shall be interpreted as markup or
script anywhere it is later displayed — in the builder, in the response view, in
the response listing, or in an export.

### NFR-SEC-03 Error disclosure
DETAILED DESCRIPTION: An error shown to a user shall not contain a stack trace, a
database message, an internal path, or a token.

### NFR-REL-01 Client stability
DETAILED DESCRIPTION: No user journey in Section 1 shall raise an uncaught
exception in the browser console, and no request in that journey shall return 5xx.

### NFR-REL-02 State preservation
DETAILED DESCRIPTION: Navigating away from a partially completed form and
returning shall either restore the entered answers or warn before discarding them.

### NFR-COMP-01 Viewport compatibility
DETAILED DESCRIPTION: Every page in Section 1 shall remain usable at a 360 px-wide
mobile viewport: no horizontal page scroll, no control pushed off-screen, no text
clipped by an overlapping element.

### NFR-A11Y-01 Accessible controls
DETAILED DESCRIPTION: Every interactive control shall have an accessible name.
Every input shall have an associated label. A required field shall be marked
programmatically, not by colour alone.

### NFR-A11Y-02 Keyboard operation
DETAILED DESCRIPTION: A survey shall be completable using the keyboard alone, with
a visible focus indicator at every step and no keyboard trap.

---


## 12. Guardrails — these override every requirement above

**This is a live research platform. Its database holds real researchers' projects
and real respondents' answers.**

### OOS-01 Do not submit to a live research survey
The survey at `/v/53d-edc` ("Software Quality Evaluation Questionnaire") is an
active instrument collecting real responses for research about DataGhurhi itself
(`collect_response: true`). **A test shall never submit a response to it, or to
any survey the tester does not own.** Every submission requirement in Section 1
(FR-RESP-13, FR-RESP-14) shall be exercised only against a survey created by the
tester for that purpose, or in preview mode (FR-RESP-18).

Validation, rendering, navigation, and refusal behaviour (FR-RESP-01 to FR-RESP-12,
FR-RESP-15) are all testable **without** submitting, by leaving a required field
blank so submission is refused. Prefer that route.

### OOS-02 Do not create accounts or send email
Registration (FR-AUTH-01) and password recovery (FR-AUTH-07) send real email to
real addresses and consume one-time codes. Test their **client-side validation
only** — invalid email format, weak password, mismatched confirmation — by
inspecting the refusal, never by completing the flow.

### OOS-03 Do not exercise payment
Subscription purchase, payment initiation, and coupon redemption shall not be
exercised. Package *display* (FR-SUBS-01) is in scope; buying is not.

### OOS-04 Do not destroy data
No test shall delete a project, survey, template, question, response, collaborator,
or account, and no test shall remove its own access to anything. Deletion
behaviour is out of scope.

### OOS-05 Do not act on other users' data
No test shall attempt to reach another user's project, survey, or responses in
order to confirm an isolation defect. Where FR-PROJ-02, FR-COLL-02 or FR-ANLY-07
describe an isolation boundary, verify it only from the tester's own account, by
confirming that only the tester's own items are listed.

### OOS-06 Do not sign out or change credentials
Sign-out, password change, and secret-question change end or invalidate the
session the run depends on. FR-AUTH-09 and FR-AUTH-11 are documented for
completeness and are out of scope for automated exploration.

### OOS-07 Stay on the application
No test shall follow a link off `dataghurhi.cse.buet.ac.bd`, including social
share links, embedded video, and external documentation.

### OOS-08 Load
No test shall issue repeated rapid requests against any endpoint. FR-RESP-14
(double submission) is exercised only against the tester's own survey, once.
