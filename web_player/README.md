# Web Player — Playwright test executor

A second **player**, parallel to the Android one. The planner, the investigator,
the gateway and the Neo4j knowledge graph are shared and unchanged; only the
driving of the target differs.

```
                      ┌──────────────────────────────┐
                      │  Planner + RAG + Neo4j       │   ← unchanged
                      │  (gateway :9100, rag :9010)  │
                      └──────────┬───────────────────┘
                    test case    │    verdict
              ┌──────────────────┴──────────────────┐
              ▼                                     ▼
   clients/executor_runner.py              web_player/runner.py
   mobilerun + adb → Android device        Playwright → browser
```

## Run it

```bash
pip install -r requirements.txt
playwright install chromium

py -m targets.run --list          # what can I test?
py -m targets.run wikipedia       # run a target profile
```

A **target profile** (`targets/profiles/*.json`) says which site to test, which
`project` scopes its knowledge graph, and what the agent may not touch — see
[targets/README.md](../targets/README.md). Prefer it over editing `.env`: the
variable that decides what the planner retrieves is `PROJECT`, and a half-edited
`.env` is why a run "keeps testing the old app".

To drive this player directly instead, set `WEB_BASE_URL` and `PROJECT` in `.env`:

```bash
py -m web_player.runner --rounds 5
```

Preflight fails loudly if `WEB_BASE_URL`, the gateway, the RAG API, the executor
model key, or Playwright is missing — before the first test case, not during it.

## How one test case runs

1. `gateway.next_testcase()` — asks the planner, tagged `platform="web"`
2. `goal.build_goal(tc)` — planner JSON → goal text (site identity, credentials,
   guardrails, preconditions, input guidance, verification discipline, steps)
3. `agent.WebAgent.run()` — observe → decide → act, one action per turn
4. `oracles` — the browser's own signals are folded in, and can override a
   "pass" the agent claimed
5. `failures.classify()` — attribute the failure: app, agent, or environment
6. self-heal — one adaptive retry for recoverable categories (WP7)
7. `gateway.log_verdict()` + `log_execution()` — which drives the next test case

## Watching a run

Four views, in increasing order of detail.

**1. The terminal / the trace log.** Every agent turn is printed as it happens and
appended to `logs/web_player.log`:

```
  [4/30] https://shop.example.com/cart
      think: The cart shows 0 items after adding one, so the count did not update.
      act:   {"action": "finish", "success": false, "reason": "Cart still shows 0 items"}
      page:  12 controls, messages: Item added to cart
      !!     FINISH success=False: Cart still shows 0 items after adding one.
```

`think` is the model's own reasoning for that step — it is the difference between
watching an exploration and watching a cursor move. From another terminal:

```bash
tail -f logs/web_player.log
```

**2. The browser itself — on by default.** The run opens a real Chromium window
and you watch it work. `WEB_SLOW_MO_MS` (default `300`) pauses before each action,
because Playwright otherwise drives faster than anyone can follow.

```bash
WEB_HEADLESS=true     # no window — for CI or a long unattended batch
WEB_SLOW_MO_MS=0      # full speed
```

**3. The dashboard.** The log panel has a **🌐 Web (Playwright)** stream next to
the device and planner ones (`/dashboard/logs?source=web`). Coverage, verdicts
and the execution timeline populate from Neo4j exactly as they do for Android,
because the player writes the same `/tests/log` and `/execution/log` records.

**4. Screenshots.** One per test case at its final state, plus one on a crash, in
`logs/web_shots/<test-id>-<verdict>.png`. The path is also written into the
verdict notes, so it is reachable from the graph.

## The observation

The agent never sees raw HTML. Each turn it gets a compact, ref-addressable view:

```
URL: https://shop.example.com/checkout
TITLE: Checkout
HEADINGS: Payment
MESSAGES ON PAGE: Card number is invalid
PAGE TEXT: Items: 0 | Total: $42.00
INTERACTIVE ELEMENTS (9):
  [e1] textbox "Card number" value="4111" required
  [e2] password "CVC" value="*******"
  [e3] select "Size" value="Large" options=[Small, Large]
  [e4] checkbox "Save card" checked
  [e5] button "Pay now"
  [e7] button "Continue" DISABLED
```

Three things this format gets right, each of which cost a real failed run to learn:

- **`PAGE TEXT` is not optional.** Most assertions are about ordinary text
  (`Items: 0`, a total, "No results"), which belongs to no control. Without it
  the agent can see every button on the page and none of its content, and
  thrashes looking for the value it was asked to check.
- **Refs are stamped (`data-etp-ref`), not selectors.** Hashed class names and
  nth-child paths break on the next render; a stamp does not. When the element
  *is* replaced, the locator misses and we report `STALE_ELEMENT` honestly
  instead of clicking whatever moved into its place.
- **A password value is masked at the source**, so a secret never enters the
  prompt or the logs.

## Guardrails

An exploratory agent on a website has no sandbox. Three limits are enforced in
the dispatcher — not merely requested in the prompt, because a prompt is a
request and a dispatcher is a rule:

| Setting | Default | Stops |
|---|---|---|
| `WEB_SAME_ORIGIN_ONLY` | `true` | wandering into OAuth screens and the open internet |
| `WEB_BLOCKED_TEXTS` | delete/close/deactivate account, log out, sign out | ending its own session or destroying the account |
| `WEB_BLOCKED_URL_PATTERNS` | `/logout`, `/signout`, `/sign-out` | the same, by URL |

A refusal is classified `BLOCKED_BY_GUARDRAIL` (an *environment* fault) and ends
the test case immediately — it is never counted as a defect, and never retried.

Override any of them when that control is the thing under test.

## Oracles — what web has that Android does not

The browser reports failures the app never says out loud, with no LLM judgment
involved: uncaught page exceptions, HTTP 4xx/5xx, console errors, failed
requests. These are **collected always** and folded into the verdict notes, so a
test that "passed" while the page threw is visible as such.

They can also fail a test on their own, but that is **off by default**:

```bash
WEB_FAIL_ON_PAGE_ERROR=true   # an uncaught exception fails the test
WEB_FAIL_ON_HTTP_5XX=true     # a 5xx from the app's own backend fails the test
```

Turn these on once you know the site's baseline noise. A site that logs errors
during normal operation would otherwise fail every test it has, and a run where
everything fails carries no information. A console error alone never fails a
test in any configuration.

## Authentication

Spending agent steps on a login form before every test is the largest avoidable
waste in a run, and some logins the agent cannot complete at all. Capture the
session once:

```bash
playwright codegen --save-storage=auth.json https://your-site/login
# then: WEB_STORAGE_STATE=auth.json
```

This is the web equivalent of "the device is already signed in". Failing that,
`WEB_LOGIN_USER` / `WEB_LOGIN_PASSWORD` are handed to the browser agent only —
never to the planner, whose prompts are far larger and far more widely logged.

## Deliberately not implemented

**No Live App Model / UI state graph.** Web state identity is a research problem
of its own: SPA routes that change nothing in the DOM, `/orders/1042` and
`/orders/9981` that are the same state, hashed class names, and content that
changes on every load. A signature tuned wrong either fragments one page into
fifty states or collapses fifty pages into one — and the resulting graph is
shared with the Android side, so bad web states would degrade Android retrieval
too.

The player therefore logs verdicts and execution records with `path: []` and puts
the route in `path_labels` (free text), so the trace is visible in reports
without inventing graph nodes nothing verified. `agent._signature()` does hash
the page, but only for in-run livelock detection — it is never persisted.

Also absent, and worth having later: responsive/viewport dimensions, cross-browser
runs, `sitemap.xml` / OpenAPI knowledge sources, and axe-core accessibility audits.

## Tests

```bash
py tests/test_web_player.py     # 73 checks, no browser needed
py tests/run_all.py             # full suite, Android included
```

The browser-facing half (snapshot JS, action dispatch, oracle capture) is
verified against real Chromium; see the smoke and loop scripts referenced in the
project notes.
