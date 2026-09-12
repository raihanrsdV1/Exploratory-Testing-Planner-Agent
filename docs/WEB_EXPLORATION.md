# Passive web exploration graph

Normal web-player runs now record their existing observations and attempted actions.
There are no extra clicks, navigation, snapshots, model calls, or changes to verdicts.
This first stage does **not** feed the graph into the planner, replay paths, or crawl
unexplored controls. An observed control is not evidence it was tested.

## Run and inspect

```powershell
py -m targets.run dataghurhi-auth --rounds 5
py -m web_player.exploration --project dataghurhi-auth
py -m web_player.exploration --project dataghurhi-auth --json
```

By default the graph lives in `data/exploration/web.sqlite3` (gitignored), surviving
separate runs and test ingestion resets. SQLite is independent of Neo4j and requires
no extra service. Project, normalized base URL, and query policy partition the data.
The CLI opens the database read-only. Use `--db PATH` for a custom database.

Settings: `WEB_EXPLORATION_ENABLED=true`, optional `WEB_EXPLORATION_DB`, and
`WEB_EXPLORATION_QUERY_KEYS=tab,view,mode,section,step,lang,page,route`. Set enabled
to false to stop recording. Storage/normalization errors disable the recorder with
a warning, not the test. SQLite lock waits are bounded to 200 ms.

## Identity and evidence

Nodes combine normalized route, dialog identity, headings and deduplicated control
structure (including selected/expanded states). Numeric/UUID path IDs become `:id`.
Query values are retained only for configured navigation keys; other non-secret
parameters retain their name with a placeholder value. Tokens and tracking keys are
removed. Hash-router paths are supported. Transient element refs and typed values
are not state identity. Repeated cards with identical structure do not multiply nodes.

Edges hold source, optional destination, action/target hints, outcome, error category,
observation count and first/last timestamps. Target hints use role/name/href and an
ordinal, not transient `e<N>` refs. They are not guaranteed replay selectors.

Outcomes distinguish `state_changed`, `observation_changed`, `no_visible_change`,
`dispatch_failed`, and `destination_unobserved`. An action at the step/deadline limit
has no invented destination. Dispatch failures do not claim the page stayed unchanged.
No outcome means that an assertion passed, nor that an action was safe or side-effect-free.

## Boundaries and privacy

Modules are split into `state.py` (pure normalization), `store.py` (persistence),
and `recorder.py` (fail-open lifecycle). The player calls observe/attempted/close.
A future graph consumer or storage adapter can be added without changing model policy.

No full DOM, cookies, raw URLs, typed values, model reasoning, or raw error text are
persisted. Accessible names/headings and opaque path slugs can still contain project
or personal data; treat the database as sensitive local test evidence. Do not add
secret-bearing parameters to navigation keys. Secret-key filtering always takes priority.

Identity is heuristic: opaque slugs remain distinct; different record labels may
split otherwise equivalent screens; unknown query-driven screens may merge until
their query key is configured. Observation truncation can also affect identity.
This is a partial graph of what the player saw, not exhaustive site coverage.

## Planner restriction removal

The earlier simple-first-test preference and blanket keyword gates for API paths,
uploads, blocked-control mentions, empty accounts and console assertions are removed.
Structural validation, duplicate checks, truthful tool capabilities, evidence checks,
execution budgets and the target's existing action safety guards remain. Removing a
keyword gate does not add upload/drag tools or authorize destructive operations.
