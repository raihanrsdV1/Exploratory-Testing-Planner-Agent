# DataGhurhi — verified API surface

Compiled 12 Sep 2026 by driving the running application at
`https://dataghurhi.cse.buet.ac.bd` with an authenticated session and recording
every request it issued, then probing each route named in the specification.

**Why this document exists.** The planner wrote a test objective citing
`GET /api/projects`. That route does not exist — the real one is `/api/project`,
singular. The executor chased the invented path six times and the test died in a
livelock. Endpoint names in a specification are worth nothing until something has
called them.

**How to read the evidence column.** This is an Express application, and Express
answers **404** for a path/method pair it has no handler for. So `GET → 404` means
*"there is no GET handler here"*, which is **not** proof the route is absent for
`POST`. Only `WORKS` and `exists` rows are conclusive.

---

## 1. Confirmed working (observed during ordinary use)

Every one of these was issued by the application itself while browsing, and
returned success. This is the list the agent is given.

| Route | Method | Purpose |
|---|---|---|
| `/api/project` | GET 200 | the caller's project list — **not** `/api/projects` |
| `/api/project/:id` | GET 200 | one project |
| `/api/project/:id/surveys` | GET 200 | surveys within a project |
| `/api/project/:id/collaborators` | GET 200 | collaborators on a project |
| `/api/project/:id/fetchaccess` | GET 200 | the caller's access level |
| `/api/profile` | GET 200 | the signed-in user's profile |
| `/api/profile/get-secret-question` | GET 200 | security question |
| `/api/get-user-packages` | GET 200 | subscription packages |
| `/api/get-saved-survey` | GET 200 | saved surveys |
| `/api/surveytemplate/:id` | GET 200 | one survey template |
| `/api/surveytemplate/stream/:id` | GET 200 | template stream |
| `/api/fetch-survey-user/:slug` | GET 200 | public survey by slug |
| `/api/collaborator/all-invitations` | GET 200 | invitations |
| `/api/collaborator/all-projects` | GET 200 | projects shared with the caller |
| `/api/survey-collaborator/all-invitations` | GET 200 | invitations (older path, still live) |
| `/api/template-share/shared-with-me` | GET 200 | templates shared with the caller |

## 2. Exists, but not usable by GET

| Route | Evidence | Reading |
|---|---|---|
| `/api/surveytemplate/save` | GET 403 | present; refuses a GET |
| `/api/project/create-project` | GET 500 | a GET handler exists and **crashes** |
| `/api/sa/save-results/` | GET 500 | same |

## 3. Named in the specification, no GET handler found

Unverified rather than disproven — most are plausibly POST-only. **Do not cite
these in a test objective without checking first.**

`/api/generate-question-with-llm/` · `/api/generate-multiple-questions-with-llm` ·
`/api/get-saved-survey/save-as-template` · `/api/profile/update-secret-question` ·
`/api/project/:id/create-survey` · `/api/question-bank/create` ·
`/api/question-bank/semantic-search` · `/api/submit-survey/:id` ·
`/api/survey-collaborator/send-survey-collaboration-request` ·
`/api/surveytemplate/copy/:id` · `/api/surveytemplate/settings/:survey_id`

## 4. Confirmed NOT to exist

Invented by the planner, or plausible-but-wrong pluralisations:

`/api/projects` · `/api/surveys` · `/api/users`

---

## 5. Defects this survey found

Two are worth reporting to the DataGhurhi team regardless of any test run:

1. **`GET /api/project/create-project` returns 500.** A GET on a creation
   endpoint should be a 404 or a 405, never an unhandled server error. The same
   applies to `GET /api/sa/save-results/`.
2. **`GET /api/project/:id/surveys` returned both 200 and 500** during the walk,
   depending on the id — a missing or inaccessible project id should give 404 or
   403, not a server error.

## 6. Reachable pages

`/` · `/dashboard` · `/dashboard?tab=projects` · `/dashboard?tab=surveys` ·
`/dashboard?tab=projectdetails&projectId=:id` · `/security-settings` ·
`/subscription` · `/view-survey/:id` · `/v/:slug`

`/profile`, `/question-bank` and `/templates` rendered no interactive controls —
either they are not routes, or they render empty for this account. Worth a manual
check before any test targets them.

---

## Keeping this current

`web_player/api_registry.py` holds the machine-readable version at
`data/api/<project>.json`. It is seeded from this survey and **grows on every
run**: the oracle already sees every response, so any route the agent reaches is
recorded with the status it returned. The agent is shown the working list each
turn, so it selects from real routes instead of inventing one.
