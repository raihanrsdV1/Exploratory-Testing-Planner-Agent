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

### FR-AUTH-01 Registration [PUBLIC]
DETAILED DESCRIPTION: `/signup` shall collect the fields required to create an
account and shall validate them before submission. Registration shall verify
email availability (`POST /api/register/check-email`) and refuse an address that
is already registered, with a message saying so.

### FR-AUTH-02 Password policy [PUBLIC]
DETAILED DESCRIPTION: A password shall be accepted only when it satisfies **all**
of: at least 8 characters, at least one uppercase letter, at least one lowercase
letter, at least one number, and at least one special character. The unmet
criteria shall be shown to the user as they type.

### FR-AUTH-03 Consistent password messaging [PUBLIC]
DETAILED DESCRIPTION: Every place the password rule is stated shall state the
**same** rule. A message describing a weaker rule than the one enforced (for
example, one that omits the uppercase and lowercase requirements) is a defect.

RATIONALE: The application contains at least two differently-worded statements of
this rule; a user who satisfies the weaker one and is refused has been misled by
the product itself.

### FR-AUTH-04 Password confirmation [PUBLIC]
DETAILED DESCRIPTION: Where a password is confirmed, a mismatch shall be refused
with a message stating that the passwords do not match, before any request is sent.

### FR-AUTH-05 Email validation [PUBLIC]
DETAILED DESCRIPTION: An email field shall refuse a syntactically invalid address
with an explicit message, and shall refuse an empty value with a message stating
the address is required. Validation shall accept legitimate forms including
subdomains, plus-addressing, and long TLDs.

### FR-AUTH-06 Sign-in [PUBLIC]
DETAILED DESCRIPTION: `POST /api/login` shall accept valid credentials and
establish a session. It shall refuse invalid credentials with a message that does
**not** disclose which of the two was wrong.

RATIONALE: Distinguishing "no such user" from "wrong password" is an account
enumeration vector on a platform holding named researcher accounts.

### FR-AUTH-07 Password recovery [PUBLIC]
DETAILED DESCRIPTION: `/forgot-password` shall send a one-time code to the
registered address (`POST /api/send-otp`), verify it (`POST /api/verify-otp`), and
only then permit a new password subject to FR-AUTH-02. An incorrect, expired, or
reused code shall be refused with an explicit message.

### FR-AUTH-08 Recovery does not disclose registration [PUBLIC]
DETAILED DESCRIPTION: Requesting a code for an address that is not registered
shall respond identically to one that is, without revealing whether the account
exists.

### FR-AUTH-10 Session and access control
DETAILED DESCRIPTION: Requesting an authenticated route (Section 0, Authenticated
surface) while signed out shall refuse access and route the visitor to sign-in. It
shall not render the page's contents, and shall not expose data belonging to
another user in the process.


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
