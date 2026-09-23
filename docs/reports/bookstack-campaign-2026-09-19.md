# BookStack (WebTestPilot benchmark) — Exploratory Testing Campaign Report

**Date:** 19 September 2026
**Target:** BookStack v25.02.1, `http://localhost:8081`, Docker (`linux/amd64`, emulated)
**Source:** [code-philia/WebTestPilot](https://github.com/code-philia/WebTestPilot) — `webapps/bookstack`
**Tester:** Exploratory Testing Planner Agent (planner `deepseek-v4-flash`, executor `qwen3.5-flash`)
**Run:** 30 test cases, 2728 s (45.5 min), project `bookstack`

---

## 1. Headline

Thirty test cases were generated and executed against a clean BookStack instance.
**No defect was found in the application, and no defect was falsely reported.**
Fourteen tests reached a verdict and passed; the remaining sixteen ended without
an assertion, and every one of those was attributable to the tester or to the
environment rather than to BookStack.

The single most useful result is a **specificity measurement**: on an application
seeded to a known-good state, the system produced **zero false defect reports in
30 attempts**. Read together with the DataGhurhi campaign — where the same system
surfaced one defect that was independently reproduced by hand — this is evidence
that the pipeline does not manufacture findings to justify itself.

Two minor deviations from the specification were observed and **confirmed
manually**; three tests raised an alarm that manual checking proved **false**.

## 2. Setup

The benchmark app was deployed with its own `docker-compose.yaml` and seeded from
`seed.sql` to a deterministic state: **3 books, 4 chapters, 6 pages, 1 shelf, 28
activity records**. The agent ran as an administrator via a captured Playwright
storage state. `webapps/start_app.sh` was not used directly because it relies on
GNU `date -d`, which is unavailable on macOS; compose was driven directly with a
portable date instead.

A 120-line SRS (`data/inputs/bookstack-spec.md`) was written from BookStack's
documented feature set and ingested as testing knowledge. **The benchmark's
injected-bug definitions were deliberately not shown to the planner.**

## 3. Results

| Outcome | N | Share | Meaning |
|---|---|---|---|
| Passed | 14 | 47% | assertion reached, application behaved correctly |
| **Site failures** | **0** | **0%** | **no defect attributable to BookStack** |
| Blocked (`PRECONDITION_NOT_MET`) | 11 | 37% | tester could not arrange or reach what the test needed |
| Agent errors | 5 | 17% | 3 livelock, 1 stale element, 1 navigation failure |

Coverage recorded: 30 test cases, 20 features, 27 validation rules, 48 entities.

Passing tests covered real workflows end to end: creating and renaming pages and
chapters, deleting books and shelves with confirmation, un-favouriting, shelf
creation, settings persistence, content counts, and search result snippets.

**Whitespace-name validation passed on three separate object types** (book,
chapter, page, shelf). This is worth calling out because the equivalent test
produced a false defect during the DataGhurhi campaign: the agent now reads the
browser's own validation message instead of concluding the click did nothing.

## 4. Findings, after manual verification

Every candidate below was re-checked by driving the UI directly, independently of
the agent. That step changed the conclusion in three of six cases.

### 4.1 CONFIRMED — over-long names are silently truncated (minor)
Submitting a 203-character book name stores a **191-character** name, with no
validation message and no indication that anything was discarded. 191 is the
MySQL `utf8mb4` index limit, so the field simply overflows.
Expected per **FR-UI-02**: refusal naming the field and its limit.
*Note:* the agent reported the cut-off as 128 characters. That figure is wrong;
the verified value is 191. The phenomenon is real, the detail was not.

### 4.2 CONFIRMED — an empty search returns everything (minor, arguable)
`/search?term=` reports "28 total results found" and lists content, rather than
stating that no term was given. Expected per **FR-SRCH-01**: say no matches were
found, or refuse the search. Reasonable people may call this intended behaviour;
it is reported as an observation, not asserted as a bug.

### 4.3 FALSE ALARM — "Recently Viewed never populates" (3 tests)
TC-001, TC-018 and TC-026 each concluded that the dashboard's *My Recently
Viewed* list does not track opened items. **Manual checking disproves this:**
opening a book placed it at the head of the list immediately. The feature works.
Three of thirty runs were spent on an effect that does not exist — an agent
observation failure, not an application defect.

## 5. What the campaign revealed about the tester

### 5.1 The agent cannot see inside iframes (the dominant limitation)
Four blocked tests — TC-007 (templates), TC-011 (comments), TC-012 (book
description), TC-014 (page content) — all failed the same way: the agent reported
formatting controls (Bold, Italic) but "no editable area". BookStack's editors are
**TinyMCE instances inside `<iframe>` elements**, and the DOM snapshot observes
only the top-level document.

This is decisive rather than incidental: WebTestPilot's own ground truth for the
same workflow uses
`page.locator('iframe[title="Rich Text Area"]').content_frame`. Roughly **a third
of the blocked runs are this one gap**, and page-body editing is central to a
wiki. Extending the snapshot to walk same-origin frames is the highest-value fix
available to this system.

### 5.2 Livelock remains the second failure mode
Three runs ended in the repeat guard. This matches the DataGhurhi campaign, where
it dominated, and is unchanged by the mitigations attempted so far.

### 5.3 Blocked is over-used as a verdict
Several `PRECONDITION_NOT_MET` results (4.1, 4.2) were genuine observations about
the application filed as "could not test". The honest-exit wording succeeded in
stopping the agent from thrashing, but it now absorbs outcomes that deserve to be
reported as findings.

## 6. Three blockers fixed to make the run possible

The first two attempts produced 0 usable runs. All three causes were in our own
stack, introduced by the merge of `origin/jonayed`:

1. **Executor returned no valid action, ever.** A newly added `strict: true`
   JSON-schema response format drove the executor model into degenerate numeric
   output (`-1.0000000000000002e+308`) on every step. Proven by running the same
   prompt with and without the schema. Now gated behind `WEB_LLM_JSON_SCHEMA`
   (default off).
2. **Reviewer rejected every verdict.** Same degeneration, plus `review.parse`
   requiring strict `json.loads` of a reply the model wraps in reasoning prose.
   The schema is gated and the parser now reuses the tolerant extractor.
3. **A failed review discarded finished tests.** An unparseable reviewer reply
   returned `Verdict unverified`, throwing away everything the run had observed.
   It now retries once, then keeps the agent's verdict marked unreviewed.

All 12 test modules (560+ checks) pass with these changes.

## 7. Honest summary

The system ran a clean 30-case campaign against an unfamiliar application with no
human help beyond deployment, and reported nothing false about it. That is the
result worth reporting.

It is also true that **only 14 of 30 runs produced evidence about BookStack**, and
that the agent raised a three-times-repeated alarm that turned out to be wrong.
The two confirmed findings are minor. This campaign demonstrates reliability, not
bug-finding power — the app was clean, so there was little to find.

## 8. Recommended next step: measure detection rate

The benchmark ships **27 injected bugs** for BookStack
(`benchmark/bookstack/bugs/*.js`) — DOM mutations that delete list entries or
corrupt titles and descriptions. They are applied as a Playwright init script,
which this system could replicate in `BrowserSession` with a few lines.

Running the campaign against a seeded bug would convert this report's qualitative
result into a quantified one: **detection rate against known ground truth**, which
is what the benchmark exists to measure. Fixing the iframe gap (5.1) first would
matter, since several injected bugs live on editor-bearing pages.

---

### Artefacts
- `docs/reports/bookstack-run-2026-09-19.pdf` — generated report (9 pages)
- `logs/bookstack_run4.log` — full campaign log
- `data/inputs/bookstack-spec.md` — SRS used as testing knowledge
- `targets/profiles/bookstack.json` — target profile
- `logs/web_shots/` — per-test screenshots

---

## 9. Bug accuracy: overlap with the benchmark's canonical bugs

The benchmark defines **27 canonical bugs** for BookStack. None were injected
during this run, so nothing here is a detection rate. What can be measured is
**overlap**: how many canonical bugs our 30 generated tests would have been in a
position to catch had they been active.

A bug counts as **covered** when one of our test cases targets the same feature
*and* the surface the bug mutates, so the mutation would fall inside that test's
assertion. **Effective** additionally requires that the test reached a verdict —
a blocked or livelocked test could not have detected anything.

| Metric | Value |
|---|---|
| Canonical bugs | 27 |
| **Covered** (a generated test targets the bug's surface) | **12 / 27 = 44.4%** |
| **Effective** (covered *and* the test reached a verdict) | **7 / 27 = 25.9%** |

### Covered (12)

| Bug | Our test(s) | Reached a verdict |
|---|---|---|
| recently_viewed_page | TC-001, TC-018 | no |
| recently_viewed_chapter | TC-018 | no |
| recently_viewed_book | TC-026 | no |
| favourite_page | TC-002 | no |
| favourite_book | TC-008 | yes |
| count_recently_created_pages | TC-006, TC-028 | yes |
| recent_activity_all | TC-005 | yes |
| search | TC-020, TC-021, TC-025 | yes |
| create_page | TC-003, TC-006, TC-014 | yes |
| delete_book | TC-005 | yes |
| update_book | TC-012, TC-013 | yes |
| comment | TC-011 | no |

### Not covered (15), and why

The misses are not random — they fall into **entity-type variants of families we
did cover**:

- *Recently Viewed* (5 bugs): covered page, chapter, book; missed **shelf**, **page template**
- *Favourites* (5 bugs): covered page, book; missed **chapter**, **page template**, **unfavourite-shelf**
- *Recently Created counts* (4 bugs): covered pages; missed **books**, **chapters**, **shelves**
- *Recent Activity* (5 bugs): covered the aggregate feed; missed the four per-entity variants
- *Settings/sorting* (2 bugs): `/settings/sorting` was never visited — TC-016 sorted at book level, TC-017 used `/settings/features` and `/settings/customization`
- *create_book*: books were created repeatedly, but no test asserted a new book's **card and description in the `/books` listing**, which is what the bug corrupts

### What the two numbers mean

**The 44% → 26% drop is our tester's failure modes, not planning.** Five covered
bugs sit behind tests that never reached a verdict: three Recently Viewed tests
(the false-alarm family, §4.3), one favourites test (livelock), and the comment
test (iframe, §5.1). Fixing the iframe gap and the livelock guard converts
already-planned coverage into real detection opportunity without generating a
single extra test.

**The 44% ceiling is a planning-strategy result.** The planner diversifies across
*features* — it wrote one Recently Viewed test, one favourites test, one search
test — while the benchmark enumerates the same feature across *every entity type*
(book, chapter, page, page template, shelf). That is sound exploratory behaviour
and poor benchmark coverage at the same time. Prompting the planner to sweep
entity types within a covered feature would raise overlap substantially, and the
SRS already names the hierarchy (FR-HIER-01) it would need to do so.

**Caveat.** The bug-to-test mapping above is a documented judgement, made by
reading each bug's trigger path and mutated selector against the routes our tests
actually visited. It is reproducible from `benchmark/bookstack/bugs/*.js` and the
recorded execution routes, but another reader could reasonably score one or two
borderline cases (notably `create_book` and `create_sort_rule`) differently.
