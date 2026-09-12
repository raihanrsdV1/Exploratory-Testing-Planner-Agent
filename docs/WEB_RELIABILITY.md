# Web execution reliability

The September 12 DataGhurhi campaign mixed several different outcomes under
"failed": a forbidden password control, a navigation loop, two copies of an
API-method test, and an import requiring tools the browser did not have.
The HTTP 500 responses may be real application defects; a healthy test runner
must preserve those failures rather than make every test pass.

## Changes

- The browser sends its target's guardrails, available actions, budget, and
  attempted titles with each planner request. The first plan also receives the
  actual entry-page observation. Constraints are request-scoped because the
  gateway may serve multiple targets.
- Both planner modes reject malformed or duplicate proposals. A rejected retry
  cannot be forcibly accepted. Earlier feature-keyword restrictions and the
  simple-first-test preference have been removed; see [passive exploration](WEB_EXPLORATION.md).
- File upload, drag-and-drop, and download inspection remain unsupported tools,
  but mentioning them or an API path no longer rejects a proposal. Existing
  target safety guards still govern actual actions.
- The browser requests JSON schema responses and validates finish booleans.
  Model calls run off the browser event loop, so dialogs and network events can
  be handled while the model responds. Per-test deadlines include initial
  navigation, model calls, and recovery. Screenshot/logging overhead is separate.
- History includes observed outcomes and an initial baseline. Cycle detection
  distinguishes actual action arguments and ignores irrelevant schema fields
  and changing element ref numbers. Unchanged DOM alone no longer permanently
  blacklists a control.
- An evidence reviewer checks proposed passes AND failures. It can request a
  specific missing check, at most twice, or mark a missing prerequisite blocked.
  This catches cases such as opening a language menu without selecting a language
  and claiming the switch failed. Review is an additional LLM check, not a proof
  that every assertion is correct.
- Cancelled requests are no longer HTTP 5xx failures. Transport errors have a
  separate category; real same-origin 5xx responses still count. Query secrets
  are redacted in diagnostic URL summaries.
- Summaries separate site failures, agent errors, and environment blocks. The
  existing graph verdict values remain compatible; `error_type` provides the
  distinction. Execution records use the same test ID key as test records, and
  verdict logging preserves the planner's test type.

## Verification commands

Verified on September 12, 2026:

- All 11 test modules passed, including 21 focused web regression checks.
- Both isolated Chromium smoke cases passed with the configured MiniMax web
  model and GLM evidence reviewer.
- A planner-generated, read-only DataGhurhi Question Bank navigation test
  (TC-008) passed in 4 steps / 30.7 seconds. The destination screenshot shows
  the Question Bank heading and loaded questions; its execution links to the
  correct test record and retains `test_type=positive`.
- Two earlier diagnostic executions exposed premature verdicts. Their stored
  records were corrected to `PRECONDITION_NOT_MET` and `VERDICT_UNVERIFIED`;
  neither is counted as a demonstrated site defect or successful assertion.

This verifies the repaired web path and shared regression suite. It does not
establish that every DataGhurhi feature passes or that a new Android campaign
has been exercised.

From the activated Python environment in the repository root:

```powershell
py tests/run_all.py
py -m scripts.web_smoke
py -m scripts.web_smoke --live-model
```

The smoke script uses Chromium against a locally fulfilled HTML fixture. Its
default mode uses deterministic actions. `--live-model` exercises the configured
browser model and evidence reviewer and incurs normal model API charges. Both
versions independently assert the resulting DOM and do not use the target
account or knowledge graph.

Reload/restart the gateway and RAG API after code changes if they are not running
with `--reload`, then use the normal command:

```powershell
py -m targets.run dataghurhi-auth --rounds 5
```

Use `--ingest` when requirements changed; it resets/re-ingests the project slice.
The target's existing `run.clean_slate` setting still controls campaign history
reset, independently of `--ingest`.

`WEB_VERIFY_VERDICTS=true` is the default. With OpenRouter the reviewer uses
`EVALUATOR_MODEL`, falling back to the web model. With direct Gemini it uses the
web model. Review adds a model call to each proposed verdict, within the same
execution timeout. The browser model must support the requested structured
response format.
