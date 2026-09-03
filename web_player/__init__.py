"""Web player — a Playwright-driven test executor, parallel to the Android one.

The planner, the gateway and the knowledge graph are shared and unchanged. Only
the *driving of the target* differs, so this package contains a browser and an
agent loop and nothing else:

    gateway.py   talk to the planner gateway / RAG (next test case, verdict, log)
    llm.py       minimal chat client for the executor model
    browser.py   Playwright lifecycle — browser, context, page, screenshots
    snapshot.py  observe the page as a compact, ref-addressable element list
    actions.py   the action vocabulary and how each one is performed
    oracles.py   passive browser signals (console, page errors, HTTP failures)
    failures.py  web failure taxonomy + self-healing recovery strategies
    goal.py      planner test case -> a goal an agent can execute
    agent.py     the observe -> decide -> act loop
    runner.py    CLI entry point: preflight, batch loop, summary

Deliberately NOT here (yet): the Live App Model / UI state graph. Web state
identity (SPA routes, hashed class names, volatile content) is a research
problem of its own; recording bad states would poison a graph the Android side
depends on. The executor logs verdicts and execution records without it.

Run:  py -m web_player.runner
"""
