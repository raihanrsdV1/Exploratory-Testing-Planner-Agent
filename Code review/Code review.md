# Repository Code Review

**Repository:** Exploratory Testing Planner Agent

**Reviewed branch:** `Niloy_44-working`

**Reviewed revision:** `5105760`

**Review date:** 2026-09-02

**Decision:** **Request changes - not ready for deployment on a shared network or for use with non-disposable credentials/data.**

## Executive Assessment

The repository has a coherent experimental architecture: source ingestion feeds a Neo4j knowledge graph, a LangGraph planner creates tests, Droidrun executes them, and observations and verdicts feed subsequent planning. The code also contains useful safeguards such as parameterized Cypher values, graph constraints, explicit failure attribution, project-scoped identifiers, degradation reporting, and defensive Android command construction.

The current implementation is not production-ready. Two P0 security chains allow unauthenticated network access to sensitive operations and expose test-account secrets through execution artifacts. Several P1 correctness defects then undermine the system's central claim that execution history improves later planning: execution records are usually disconnected from test cases, non-defect outcomes are recorded as defects, and replacement ingestion can leave the knowledge graph partially erased. These are release blockers because they affect both security and the validity of reported results.

This review uses a large-engineering-organization style rubric: findings are ordered by user impact and exploitability, each has evidence, impact, a required fix, and a verification expectation. P0/P1 findings are release-blocking.

## Severity Definitions

| Priority | Meaning | Release treatment |
|---|---|---|
| **P0 - Critical** | Direct secret/data compromise, destructive unauthenticated operation, or system-wide integrity failure | Stop-ship; fix immediately |
| **P1 - High** | Major security boundary failure or core behavior produces materially incorrect results | Must fix before release |
| **P2 - Medium** | Reliability, operability, maintainability, or defense-in-depth gap with a credible failure mode | Fix in the next milestone |
| **P3 - Low** | Localized quality or documentation debt | Track and schedule |

## Findings

### P0-01 - Externally bound APIs permit unauthenticated file exfiltration and destructive operations

**Evidence**

- Both services bind to every network interface in [start.sh](../start.sh#L167), while blank API keys disable authentication in [planner/config.py](../planner/config.py#L31), [rag_api/main.py](../rag_api/main.py#L31), and the shipped [.env.example](../.env.example#L6).
- Gateway path validation rejects only a literal `..` component, so absolute paths remain valid for SRS, Figma, and defect ingestion in [planner/pipeline.py](../planner/pipeline.py#L46). The RAG API directly reads SRS and Figma paths without a containment check in [rag_api/main.py](../rag_api/main.py#L694) and accepts absolute defect paths in [rag_api/main.py](../rag_api/main.py#L2203).
- Ingested SRS text is persisted and returned through retrieval/graph endpoints. An attacker who can reach port 9010 can therefore ingest a readable local file such as `.env` and retrieve its contents.
- The same unauthenticated RAG surface includes project reset and unbounded state observation. A separate path escape in [gateway/main.py](../gateway/main.py#L325) accepts an absolute or parent-relative trajectory directory and reads its `trajectory.json`.

**Impact**

Any host able to reach the service can read supported local files indirectly, overwrite a project's knowledge slice, delete project data, submit large payloads, and inspect graph/test data. Binding to `0.0.0.0` turns a default intended for local development into a network attack surface.

**Required change**

Default both services to loopback. Refuse non-loopback startup unless a mandatory API key or stronger identity-aware proxy is configured. Enforce authentication through one application-wide dependency/middleware rather than per-handler calls. Replace server-side caller-supplied file paths with uploads/inline content, or resolve paths against a configured ingest root and verify `resolved_path.is_relative_to(resolved_root)`. Apply authorization separately to read, ingest, reset, and diagnostic operations.

**Verification**

Add integration tests proving that unauthenticated ingest/reset/retrieval fails, absolute paths and symlink escapes fail, allowed-root files succeed, and a non-loopback bind without authentication is rejected at startup.

### P0-02 - Test-account credentials are persisted in plaintext execution artifacts

**Evidence**

- `APP_LOGIN_IDENTIFIER`, `APP_LOGIN_SECRET`, and free-form login hints are converted into a plaintext prompt fragment in [settings.py](../settings.py#L197) and [settings.py](../settings.py#L271).
- That fragment is included in the Droidrun goal in [clients/executor_runner.py](../clients/executor_runner.py#L322). The full goal is printed and emitted to the optional cloud logger in [clients/executor_runner.py](../clients/executor_runner.py#L903); `start.sh --with-executor` redirects stdout to a persistent file in [start.sh](../start.sh#L206).
- Droidrun is configured to save every trajectory in [clients/executor_runner.py](../clients/executor_runner.py#L953). The gateway returns agent thoughts and unredacted tool arguments, including typed `text`, from an unauthenticated endpoint in [gateway/main.py](../gateway/main.py#L325). The React view explicitly renders `text` tool arguments in [dashboard-react/src/RunSteps.jsx](../dashboard-react/src/RunSteps.jsx#L6).

**Impact**

A password, fixed OTP, PIN, phone number, or other login hint can remain in local logs, third-party logging, saved trajectories, and dashboard responses after a run. Anyone with dashboard/network, log, backup, or observability access may recover the test account. This also expands the blast radius of prompt injection.

**Required change**

Treat login values as secrets end to end. Do not print or remotely log complete goals. Redact secret values and text-entry tool arguments before trajectory persistence and dashboard serialization. Authenticate diagnostic endpoints, add strict retention and filesystem permissions, and keep secrets in a dedicated secret provider rather than general settings objects. Audit existing logs/trajectories/cloud records and rotate any credential that may already have been recorded.

**Verification**

Run a canary login with recognizable fake values and assert that none appear in stdout, local logs, JSONL, cloud-handler payloads, trajectory JSON, dashboard APIs, screenshots, or backups.

### P1-01 - The documented authenticated mode is incomplete and breaks first-party clients

**Evidence**

- The gateway documentation promises authentication on every request in [gateway/main.py](../gateway/main.py#L49), but metrics, dashboard data, logs, planner traces, run steps, screenshots, and health handlers do not call the auth check; examples start at [gateway/main.py](../gateway/main.py#L76) and [gateway/main.py](../gateway/main.py#L194).
- The executor calls protected Gateway and RAG endpoints without either key in [clients/executor_runner.py](../clients/executor_runner.py#L236), [clients/executor_runner.py](../clients/executor_runner.py#L289), and [clients/executor_runner.py](../clients/executor_runner.py#L842).
- The primary ingestion script also omits authorization headers, including on destructive reset, in [scripts/ingest_all.py](../scripts/ingest_all.py#L14). The simulator and verification scripts have the same pattern.
- The dashboard screenshot proxy omits the configured RAG authorization header in [gateway/main.py](../gateway/main.py#L393). Dashboard coverage calls the authenticated coverage function with `None` and silently returns an empty section when the Gateway key is enabled in [planner/pipeline.py](../planner/pipeline.py#L381).

**Impact**

Leaving keys blank exposes the services; setting keys causes the executor, ingestion workflow, screenshot proxy, and dashboard coverage to fail partially. Health checks remain green, so the stack can report healthy while its primary workflow is unusable.

**Required change**

Centralize service authentication and centralize HTTP clients that always attach the appropriate key. Define an explicit public allowlist, normally only liveness, and keep readiness diagnostics authenticated or redacted. Add a secured-stack smoke test covering ingest, plan, execute-log, screenshot, dashboard, and reset authorization.

### P1-02 - Execution records use a different test identity and disconnect the learning loop

**Evidence**

- Planned/verdict records build the internal TestCase ID from the external test-case ID in [rag_api/main.py](../rag_api/main.py#L1486).
- Execution records instead build the TestCase ID from the test title before attempting `FOR_TEST` linkage in [rag_api/main.py](../rag_api/main.py#L2118).
- Effectiveness metrics rely on the missing `ExecutionLog-[:FOR_TEST]->TestCase` relationship in [rag_api/metrics.py](../rag_api/metrics.py#L45), and strategy learning queries the same title-derived ID in [rag_api/main.py](../rag_api/main.py#L2159).
- The checked-in recent backup corroborates the defect: it contains 53 `ExecutionLog` nodes but only 5 `FOR_TEST` relationships in [neo4j_backup_20260830_183050.json](../data/backups/neo4j_backup_20260830_183050.json).

**Impact**

Most executions do not contribute to per-test effectiveness. Strategy memory falls back to `unspecified`, so the agent cannot reliably learn which test strategies work. The dashboard can show plausible records while the central feedback relationship is absent.

**Required change**

Create one canonical TestCase ID function and use it in planning, verdict logging, execution logging, retrieval, and migration scripts. Prefer an immutable campaign-scoped UUID or `(project, campaign_id, external_id)` over title slugs. Backfill existing relationships by project plus external ID, then verify counts and reject orphan execution logs.

**Verification**

An end-to-end test should plan one test, log one execution and verdict, then assert exactly one TestCase, one linked ExecutionLog, the expected TestRuns, a non-`unspecified` strategy update, and changed effectiveness metrics.

### P1-03 - Replacement ingestion is not atomic and can destroy the last good graph

**Evidence**

- SRS ingestion deletes/replaces the document slice in one auto-committed `session.run`, computes embeddings afterward, then writes chunks and the entity graph through many additional calls in [rag_api/main.py](../rag_api/main.py#L694) and [rag_api/main.py](../rag_api/main.py#L787).
- Entity extraction deletes the previous Requirement and Entity slices before writing replacements in [rag_api/main.py](../rag_api/main.py#L814).
- Figma ingestion deletes all existing project UI data, then rebuilds screens and elements through separate calls in [rag_api/main.py](../rag_api/main.py#L1027).

**Impact**

An embedding outage, malformed extracted item, Neo4j error, timeout, or process crash after deletion leaves a partially populated or empty knowledge base. Subsequent test generation can silently use incomplete requirements/UI data.

**Required change**

Perform external parsing and embedding before mutation. Write a versioned staging subgraph, validate expected counts and relationships, and switch the project's active version in one managed Neo4j transaction. Delete the previous version only after the new version commits. At minimum, wrap each replacement write set in `session.execute_write` with a single transaction object.

**Verification**

Inject failures after deletion, mid-chunk, mid-requirement, and mid-screen writes; the previously active graph must remain fully queryable in every case.

### P1-04 - `blocked` and `skipped` outcomes are persisted as application failures

**Evidence**

- The public schema distinguishes pass, failed, blocked, and skipped but documents that the last two become failed in [planner/schemas.py](../planner/schemas.py#L54).
- The conversion occurs before persistence in [planner/pipeline.py](../planner/pipeline.py#L244).
- Coverage hot spots treat a failed verdict without an execution error type as application evidence in [planner/coverage.py](../planner/coverage.py#L25), while regression risk counts `last_verdict='failed'` directly in [rag_api/risk.py](../rag_api/risk.py#L40).

**Impact**

Environment blocks and intentional skips inflate bug counts and regression risk, create false hot spots, pollute failed-title context, and steer later tests toward areas where the app did not actually fail.

**Required change**

Persist the original outcome as a closed enum. Store failure attribution (`app`, `agent`, `environment`, `not_executed`) separately and make every metric query use that attribution. Migrate existing records by parsing the legacy note prefix where possible.

**Verification**

Table-driven tests must cover all verdict/error-type combinations and prove blocked/skipped/environment failures do not change defect counts, hot spots, risk, strategy effectiveness, or app pass rate.

### P1-05 - The served dashboard fallback contains a stored-XSS path

**Evidence**

- No React `dist` is present, so the gateway serves the vanilla dashboard fallback selected in [gateway/main.py](../gateway/main.py#L81).
- The fallback places an escaped UI-state label in an SVG data attribute, later reads the decoded attribute value, and inserts it into `innerHTML` without re-escaping in [dashboard/index.html](../dashboard/index.html#L410).
- UI-state labels can originate from accessibility `content_description` or text in caller-supplied observation data in [ingestion/app_state.py](../ingestion/app_state.py#L72). The observation endpoint is exposed by the default configuration.

**Impact**

A crafted state label can execute script in the operator's Gateway origin when a graph node is hovered. That origin can call unauthenticated diagnostic and mutation endpoints and read operational data.

**Required change**

Build the tooltip with DOM nodes and `textContent`; never move data from an attribute back into `innerHTML`. Eliminate dynamic HTML assembly for untrusted graph/test data, add a restrictive Content Security Policy, and apply authentication independently of browser protections.

**Verification**

Add a browser test with labels containing tags, quotes, event handlers, SVG payloads, and encoded variants. Rendering and hover must display literal text and must not create executable DOM nodes.

### P1-06 - Untrusted knowledge can become privileged device instructions

**Evidence**

- Requirements, SRS, Figma, defects, navigation history, and prior errors are inserted into the planner prompt as raw text without an untrusted-data boundary in [planner/prompts.py](../planner/prompts.py#L200).
- Model output is parsed as any JSON object; only the presence of `title` is required before an ID is assigned and the test is auto-logged in [planner/langgraph_agent.py](../planner/langgraph_agent.py#L363) and [planner/langgraph_agent.py](../planner/langgraph_agent.py#L499).
- Generated steps are copied verbatim into a Droidrun goal that also contains credentials in [clients/executor_runner.py](../clients/executor_runner.py#L322).
- The strict application boundary is disabled by default in [settings.py](../settings.py#L335), allowing the device agent to leave the target app unless separately configured.

**Impact**

A poisoned SRS, design label, defect report, observed UI text, or prior error can instruct the planner to create device actions that disclose credentials, leave the app, alter settings, or perform unrelated state-changing operations. This is indirect prompt injection across a high-privilege agent boundary.

**Required change**

Treat all retrieved content as data, label its provenance, and explicitly prohibit following instructions contained in it. Validate planner output against a strict schema and a deterministic action policy before execution. Default to the target package, allow only reviewed companion packages/actions, block install/uninstall/settings/account changes, redact secrets, and require approval for sensitive excursions. Maintain an adversarial prompt-injection evaluation set.

**Verification**

Run poisoned-source tests across every ingestion source and live UI. The final action plan must remain inside policy, must not reproduce secrets, and must reject instructions to open unrelated apps, change credentials, install/uninstall software, or access system settings.

### P1-07 - Full knowledge-graph backups are committed to source control

**Evidence**

- Two tracked JSON backups total approximately 7.5 MB: [backup 1](../data/backups/neo4j_backup_20260830_182735.json) and [backup 2](../data/backups/neo4j_backup_20260830_183050.json).
- The backup script exports every node and relationship property without filtering in [scripts/backup_neo4j.py](../scripts/backup_neo4j.py#L20).
- The backups include full SRS and chunk text, defect descriptions, execution errors, test notes, UI labels, paths, and embeddings. The current [.gitignore](../.gitignore#L1) does not exclude `data/backups/`.

**Impact**

Repository readers and every fork/clone receive operational test data and potentially proprietary requirements, defect history, identifiers, or sensitive error text. Deleting a file in a later commit does not remove it from Git history.

**Required change**

Classify the backup contents and run secret/PII scanning. Store backups in access-controlled, encrypted artifact storage with retention controls; export only required properties and redact sensitive text. Ignore generated backup paths. If any content is confidential, purge it from repository history using the organization's incident process and rotate exposed credentials.

### P1-08 - Configuration drift can select the wrong backend, credential, or application data

**Evidence**

- [settings.py](../settings.py#L1) declares itself the single configuration source and defaults the model backend to OpenRouter and the Neo4j password to empty.
- [planner/config.py](../planner/config.py#L12) reloads the environment independently and defaults the backend to `ngrok`; [rag_api/main.py](../rag_api/main.py#L31) independently defaults the Neo4j password to the hard-coded value `hihi`.
- Although settings and tests intentionally remove application-specific path defaults, [start.sh](../start.sh#L34) restores Contacts project/SRS/Figma defaults. `start.sh --ingest` then resets and ingests that project in [start.sh](../start.sh#L192).

**Impact**

Components can start with different effective configuration. A missing environment value can silently target the wrong backend or authenticate with an unintended password. An unconfigured destructive ingest can wipe or populate the Contacts project despite the repository's stated fail-fast policy.

**Required change**

Load one immutable typed settings object from the repository root and inject it into every component. Remove all fallback credentials and app-specific shell defaults. Validate required values at startup and log a redacted configuration fingerprint. Make destructive commands require an explicit project and input paths plus confirmation/non-interactive force flag.

**Verification**

Contract tests should launch each component from different working directories with missing, blank, and conflicting values and assert identical results or an explicit startup failure. Extend the existing app-default guard to shell scripts.

### P2-01 - Verdict-and-next is neither atomic nor idempotent

**Evidence**

- The endpoint claims it “atomically” logs and generates in [gateway/main.py](../gateway/main.py#L583), but implementation performs a durable RAG write followed by an independent LLM workflow in [planner/pipeline.py](../planner/pipeline.py#L275).
- Each verdict call creates a timestamp-derived TestRun ID in [rag_api/main.py](../rag_api/main.py#L1486), so retrying after a timeout creates another run.

**Impact**

If generation fails after the verdict commits, a client sees an error despite a partial success. Retrying duplicates run history and changes metrics; not retrying loses the next test. The API contract therefore cannot support reliable client recovery.

**Required change**

Remove the atomicity claim and expose the two-step workflow explicitly, or implement a persisted operation state machine. Require an idempotency key/run ID for verdict writes and return the existing result on retry. Test timeouts and failures at every boundary.

### P2-02 - Payload and log growth are insufficiently bounded

**Evidence**

- Figma JSON, defect input, normalized UI trees, and base64 screenshots have no meaningful request-size limits in [planner/schemas.py](../planner/schemas.py#L100) and [rag_api/schemas.py](../rag_api/schemas.py#L80). Screenshots are decoded and written directly in [rag_api/main.py](../rag_api/main.py#L1964).
- Logging uses an unbounded `FileHandler` in [observability/logger.py](../observability/logger.py#L30), and dashboard polling reads and splits entire log files before returning a tail in [gateway/main.py](../gateway/main.py#L194) and [gateway/main.py](../gateway/main.py#L226).
- The cross-process degradation sink is also append-only, read in full, and deleted without an inter-process lock in [observability/degradations.py](../observability/degradations.py#L35).

**Impact**

Large or repeated requests can exhaust memory/disk, while long campaigns make every dashboard poll progressively more expensive. Concurrent append/read/reset operations can lose or corrupt degradation evidence.

**Required change**

Set server and schema limits for bodies, collection sizes, decoded images, dimensions, and text fields. Verify image formats before storage. Rotate/compress logs with retention limits, tail files without full reads, cap trajectories/screenshots per campaign, and use a real cross-process event store or locked atomic files.

### P2-03 - LLM output and prompt budgets are not enforced as contracts

**Evidence**

- JSON parsing falls back to `{"raw": ...}` and otherwise accepts any dictionary in [planner/textutil.py](../planner/textutil.py#L34).
- The planner assigns an ID when `title` merely exists; it does not validate required steps, types, enum values, requirement IDs, or expected result before auto-logging in [planner/langgraph_agent.py](../planner/langgraph_agent.py#L422).
- Prompt budget priority-zero blocks are explicitly included even when they exceed the configured cap in [planner/budget.py](../planner/budget.py#L76). Requirements, SRS, and Figma are all priority zero in [planner/prompts.py](../planner/prompts.py#L226).

**Impact**

Malformed model output can create phantom planned tests, strings can be treated as step iterables, and invalid cases reach executor-side late checks. Large mandatory context can exceed model context/cost expectations despite a configured budget.

**Required change**

Use a strict Pydantic model or provider-supported structured output for planner responses, validate before deduplication/persistence, and retry only on typed validation errors. Make the prompt ceiling hard: summarize/chunk mandatory sources, reserve output tokens, and fail with an explicit budget diagnostic if the minimum safe prompt cannot fit.

### P2-04 - Restore interpolates untrusted schema identifiers into Cypher

**Evidence**

- Backup labels and relationship types are read from JSON and concatenated directly into Cypher in [scripts/restore_neo4j.py](../scripts/restore_neo4j.py#L52) and [scripts/restore_neo4j.py](../scripts/restore_neo4j.py#L73).

**Impact**

A modified or third-party backup can alter the generated query rather than merely restore data. This is especially risky because restore can run after wiping the target database.

**Required change**

Treat backups as untrusted input. Validate every label/type against a strict identifier allowlist, reject unknown schema types, verify a signed checksum/manifest, and restore into a disposable database before promotion. Add malicious-label/type tests.

### P2-05 - Automated quality gates are incomplete and the documented suite is stale

**Evidence**

- Python dependencies are entirely unpinned in [requirements.txt](../requirements.txt); there is no Python lockfile, lint/type/test configuration, or CI workflow in the reviewed tree.
- Tests are top-level script checks run by [tests/run_all.py](../tests/run_all.py#L1), not a standard discoverable suite. The runner captures stderr but does not print it for failed modules, so failures appear only as `no summary`.
- [README.md](../README.md#L161) documents five modules including `test_livelock.py`, but that file is absent and the runner discovers four.
- In this review environment, `tests/run_all.py` reported 2/4 modules passing. The two failures occurred during import because `structlog` was not installed; the aggregate runner hid that cause. A read-only AST parse succeeded for all 64 Python files.

**Impact**

Dependency updates can change behavior without review, regressions are not blocked on pull requests, and the current tests do not cover FastAPI authorization, Neo4j transactions, persistence identity, dashboard security, model validation, or secured end-to-end operation.

**Required change**

Adopt a pinned/hashed Python lock, standard test runner, formatter/linter, type checking for core schemas, dependency/security scanning, and required CI checks. Preserve fast pure tests, then add service tests with Neo4j fixtures and HTTP clients plus contract tests for auth, ingestion rollback, identity linkage, verdict attribution, XSS, prompt injection, and idempotency. Print captured stdout/stderr on failure and keep the README manifest generated from the suite.

### P3-01 - Core service modules are too large for effective ownership and isolated testing

`rag_api/main.py`, `clients/executor_runner.py`, and `gateway/main.py` combine transport, configuration, persistence, domain policy, serialization, and operational UI concerns. Global driver/settings initialization and broad exception handling make unit isolation difficult and helped identity/auth rules diverge across call paths.

Split by domain boundary after the P0/P1 fixes: authenticated transport adapters, ingestion services, test/run repositories, learning/metrics services, executor policy, and dashboard diagnostics. Keep one canonical identity/outcome/configuration module and inject external clients. This should be an incremental extraction backed by characterization tests, not a wholesale rewrite.

## Positive Observations

- Most Cypher data values are parameterized, and the graph initializes useful uniqueness constraints.
- Android commands use argument arrays instead of shell interpolation.
- Reset schemas reject unknown destructive flags.
- The failure taxonomy separates app, agent, and environment faults conceptually.
- The degradation subsystem and comments document real operational failures instead of hiding all fallback behavior.
- The state abstraction has focused pure tests for identity stability and merging behavior.

## Required Remediation Order

1. Close P0-01 and P0-02: loopback/default-deny networking, centralized authentication, safe ingestion roots/uploads, secret redaction, diagnostic protection, and credential rotation/audit.
2. Fix P1-01 through P1-06: authenticated end-to-end operation, canonical TestCase identity, transactional versioned ingestion, correct outcome attribution, dashboard XSS, and an executor action policy.
3. Remove/classify tracked backups and consolidate fail-fast configuration.
4. Add idempotency, size/retention bounds, strict model schemas, prompt caps, and safe restore validation.
5. Establish required CI and integration/security regression tests before refactoring module boundaries.

## Release Gate

Do not expose ports 9010/9100 beyond loopback, use real test-account credentials, ingest confidential specifications, or rely on effectiveness/risk/strategy metrics until all P0 and P1 findings are fixed and their verification tests pass. P2 items should be accepted only with named owners and dated follow-up work.

## Review Method and Limitations

The review covered Python services and clients, planner/RAG/ingestion logic, shell scripts, schemas, observability, both dashboard implementations, tests, dependency manifests, documentation, and tracked runtime data. Findings were validated by call-path tracing, static searches, inspection of backup schema/relationship counts, a read-only AST parse, and the repository's test runner.

No live Neo4j instance, Android device, Droidrun session, external LLM, or cloud logging account was available, so runtime performance and model/device behavior were not exercised. The test failures described above are environment import failures, not proof that the two affected test assertions fail when dependencies are installed.
