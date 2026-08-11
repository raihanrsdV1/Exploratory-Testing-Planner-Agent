# Building the ScreenGraph — Detailed Implementation Guide

How to auto-build a navigation graph of the app from what MobileRun already observes while it
executes, and feed it to the planner as **runtime context** — so the planner stops planning blind
against design-time SRS/Figma and can reason about screens it has actually reached.

This is the practical realization of "auto-capture instead of hand-built Figma." Build the
**after-run** version first (simplest, deterministic), then optionally move to live capture.

> Grounded in the verified MobileRun/DroidRun **0.6.8** API. Field names and shapes below were read
> from the installed package, not assumed.

---

## 1. What we're building

- **Node = a screen** (a distinct UI state), identified by a stable *signature*.
- **Edge = the action** that moved the app from one screen to another, labelled with the action and
  its outcome.
- Accumulated across steps and runs → the real navigation graph: reachable screens, which actions
  connect them, dead ends, loops, and (by comparing against the SRS) screens the spec implies but
  you've never reached.

Output: a compact text block the planner reads in its prompt, e.g.

```
SCREENS REACHED:
  S1 [contacts]: buttons: Add, Search; list: contact_row (x4)          (visits=5)
  S2 [create]:   inputs: First name, Last name, Phone; buttons: Save   (visits=2)
TRANSITIONS:
  S1 --click "Add"--> S2        (ok)
  S2 --click "Save"--> S1       (ok)
  S2 --system_button back--> S1 (ok)
FRONTIER (untried actions): S2:"Phone"; spec screens never reached: settings, edit_contact
```

---

## 2. Data sources (verified, MobileRun 0.6.8)

You need, per step: the on-screen elements (a11y) and the action taken. Both are available.

| What | Where | Notes |
|---|---|---|
| Per-step a11y (the screens) | `agent.trajectory.ui_states: List[List[Dict]]` | One entry per step. **Requires** `config.logging.save_trajectory != "none"`. |
| Actions taken | `agent.shared_state.action_history: List[Dict]` | Each `{"action": <type>, <arg>: <val>, ...}` e.g. `{"action":"click","index":5}`. |
| Action outcomes | `agent.shared_state.action_outcomes: List[bool]` | Parallel to `action_history`. |
| Action errors | `agent.shared_state.error_descriptions: List[str]` | Parallel; `""` when ok. |
| Final activity | `agent.shared_state.current_activity_name: str` | **Only the final** screen's activity; not per-step (see §4). |
| Screenshots | trajectory folder `0000.png…` + `ScreenshotEvent` | For optional VLM captions (§9). |
| Live events | `RecordUIStateEvent(ui_state=list[Dict])`, `ExecutorResultEvent(action, outcome, error)` | For the live path (§10). |

**a11y element shape** (flat list — already flattened by `IndexedFormatter`, `children` is always `[]`):

```python
{
  "index": 7,                       # position in the flat list
  "resourceId": "com.app:id/save",  # STABLE identity — prefer this
  "className": "Button",            # short (last dotted segment)
  "text": "Save",                   # visible label / content-desc / fallback
  "bounds": "10,20,300,80",         # "left,top,right,bottom"
  "checkedState": "",               # "isChecked=True/False" when checkable
  "children": []                    # always empty (flattened)
}
```

---

## 3. The screen signature (the heart of it)

A node key must collapse *re-visits of the same screen* into one node, while separating genuinely
different screens. Activity name alone is too coarse (one activity = many visual states). Use:

> **signature = hash( activity + sorted set of stable element keys )**

where each element's key is its `resourceId` (preferred — stable across content changes), falling
back to `text` when there's no id. **Drop volatile text** (clocks, timestamps, battery, dynamic list
values) or the same screen hashes differently every visit.

```python
# planner/screen_graph.py
from __future__ import annotations
import hashlib
import re

_VOLATILE = re.compile(r"\b(\d{1,2}:\d{2}|\d+%|am|pm|today|yesterday|\d{4}-\d{2}-\d{2})\b", re.I)

def _looks_volatile(text: str) -> bool:
    return bool(_VOLATILE.search(text)) or text.strip().isdigit()

def screen_signature(a11y: list[dict], activity: str = "") -> str:
    tokens: list[str] = []
    for el in a11y or []:
        rid = (el.get("resourceId") or "").strip()
        txt = (el.get("text") or "").strip()
        if rid:
            tokens.append(rid)                       # stable structural key
        elif txt and not _looks_volatile(txt):
            tokens.append(f"~{txt}")                 # text fallback, volatile dropped
    basis = f"{activity}|" + "|".join(sorted(set(tokens)))
    return "S_" + hashlib.sha1(basis.encode("utf-8")).hexdigest()[:12]
```

**Tuning knob:** the token set defines granularity. Too strict (include `text` for everything) →
every list change is a "new" screen (graph explodes). Too loose (activity only) → distinct screens
merge. Start with resourceId-only + text-fallback as above; if the graph is too noisy, drop the text
fallback; if it over-merges, add `className` into the token. (This is the abstraction-granularity
problem APE/Q-testing study — see the research doc. Empirically tune on your first runs.)

---

## 4. A note on per-step activity

`shared_state` only keeps the **final** activity, and `trajectory.ui_states` entries are a11y lists
without an activity field. So in the pure after-run path you build signatures from **a11y only**
(the `activity` arg stays `""`). That's fine — the a11y token set alone is a strong discriminator,
and `resourceId`s already carry the package prefix.

If you want activity in the signature, get it in the **live path** (§10): at each
`RecordUIStateEvent` also read `agent.shared_state.current_activity_name` (it's current at that
moment). That's the main reason the live path yields slightly better nodes.

---

## 5. The ScreenGraph module

```python
# planner/screen_graph.py  (continued)
from dataclasses import dataclass, field

def _summarize(a11y: list[dict]) -> dict:
    """Compact, human/LLM-readable view of a screen's interactive elements."""
    out: dict[str, list[str]] = {}
    for el in a11y or []:
        label = (el.get("text") or el.get("resourceId") or "").strip()
        if not label:
            continue
        kind = (el.get("className") or "element").lower()
        out.setdefault(kind, [])
        if label not in out[kind]:
            out[kind].append(label)
    return {k: v[:8] for k, v in out.items()}          # cap for prompt size

@dataclass
class ScreenNode:
    sig: str
    activity: str = ""
    elements: dict = field(default_factory=dict)       # {kind: [labels]}
    visits: int = 0
    caption: str = ""                                  # optional VLM caption (§9)

@dataclass
class ScreenEdge:
    src: str
    dst: str
    action: str                                        # "click \"Save\"", "system_button back"
    outcome: bool | None = None
    count: int = 0

class ScreenGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, ScreenNode] = {}
        self.edges: dict[tuple, ScreenEdge] = {}       # (src, dst, action) -> edge

    def add_screen(self, a11y: list[dict], activity: str = "") -> str:
        sig = screen_signature(a11y, activity)
        node = self.nodes.get(sig)
        if node is None:
            node = ScreenNode(sig=sig, activity=activity, elements=_summarize(a11y))
            self.nodes[sig] = node
        node.visits += 1
        return sig

    def add_edge(self, src: str, dst: str, action: dict, outcome: bool | None) -> None:
        label = self._action_label(action)
        key = (src, dst, label)
        edge = self.edges.get(key)
        if edge is None:
            edge = ScreenEdge(src=src, dst=dst, action=label, outcome=outcome)
            self.edges[key] = edge
        edge.count += 1
        if outcome is not None:
            edge.outcome = outcome

    @staticmethod
    def _action_label(action: dict) -> str:
        a = (action or {}).get("action", "?")
        args = {k: v for k, v in (action or {}).items() if k != "action"}
        # Prefer a human tag; the executor's a11y indices aren't meaningful to the planner.
        if "index" in args:   return f'{a}(index={args["index"]})'
        if "button" in args:  return f'{a}({args["button"]})'
        if "package" in args: return f'{a}({args["package"]})'
        return a if not args else f"{a}({list(args.values())[0]})"
```

---

## 6. Build the graph after a run

`ui_states[i]` is the screen the agent saw **before** taking `action_history[i]`; `ui_states[i+1]`
is the screen **after**. So an edge is `sig(ui_states[i]) --action[i]--> sig(ui_states[i+1])`.
Lengths are usually `len(ui_states) == len(action_history) + 1`; zip defensively.

```python
# planner/screen_graph.py  (continued)
def build_from_run(agent, into: "ScreenGraph | None" = None) -> "ScreenGraph":
    sg = into or ScreenGraph()
    ui_states = list(getattr(getattr(agent, "trajectory", None), "ui_states", []) or [])
    actions   = list(agent.shared_state.action_history)
    outcomes  = list(agent.shared_state.action_outcomes)

    if not ui_states:
        return sg                                       # trajectory saving was off — see §7

    prev = sg.add_screen(ui_states[0])
    for i in range(1, len(ui_states)):
        cur = sg.add_screen(ui_states[i])
        act = actions[i - 1] if i - 1 < len(actions) else {}
        out = outcomes[i - 1] if i - 1 < len(outcomes) else None
        if cur != prev:                                 # only real transitions become edges
            sg.add_edge(prev, cur, act, out)
        prev = cur
    return sg
```

Pass a persistent `ScreenGraph` across rounds (`into=session_graph`) to accumulate the map over a
whole session, not just one test.

> **First-run check (do this before trusting it):** dump `len(ui_states)`, `len(actions)`, and the
> node/edge list after one run and eyeball that nodes match the screens you saw and the off-by-one
> alignment holds. This is roadmap task 3.1's verify step.

---

## 7. Wire it into the executor

In `clients/executor_runner.py :: execute_test_on_device`, (a) enable trajectory capture and
(b) build the graph after the run.

```python
# enable per-step a11y capture (default is "none")
config.logging.save_trajectory = "all"     # any non-"none" value turns it on

agent = MobileAgent(goal=goal, llms={"default": llm}, driver=driver,
                    timeout=EXECUTOR_TIMEOUT, config=config)
result = await agent.run()

# build/accumulate the screen graph (session_graph lives across rounds in main())
from planner.screen_graph import build_from_run
build_from_run(agent, into=session_graph)
```

If you'd rather **not** depend on the trajectory config, collect a11y yourself from the event stream
(this is the bridge to the live path — §10):

```python
handler = agent.run()
ui_states = []
async for ev in handler.stream_events():
    if type(ev).__name__ == "RecordUIStateEvent":
        ui_states.append(ev.ui_state)
result = await handler
# then feed ui_states + agent.shared_state.action_history into ScreenGraph
```

---

## 8. Serialize for the planner prompt

Keep it compact — this goes into every planning prompt.

```python
# planner/screen_graph.py  (continued)
def to_prompt_text(self, max_screens: int = 20, max_edges: int = 30) -> str:
    lines = ["SCREENS REACHED:"]
    for n in sorted(self.nodes.values(), key=lambda x: -x.visits)[:max_screens]:
        el = "; ".join(f"{k}: {', '.join(v)}" for k, v in n.elements.items())
        cap = f" — {n.caption}" if n.caption else ""
        lines.append(f"  {n.sig} [{n.activity or '?'}]: {el} (visits={n.visits}){cap}")
    lines.append("TRANSITIONS:")
    for e in list(self.edges.values())[:max_edges]:
        ok = "ok" if e.outcome else ("fail" if e.outcome is False else "?")
        lines.append(f"  {e.src} --{e.action}--> {e.dst} ({ok})")
    frontier = self.frontier()
    if frontier:
        lines.append("FRONTIER (screens with likely-untried actions): " + ", ".join(frontier[:10]))
    return "\n".join(lines)

def frontier(self) -> list[str]:
    """Screens whose interactive element count exceeds their outgoing edge count —
    i.e. actions probably not yet tried. Cheap heuristic; refine later."""
    out_counts: dict[str, int] = {}
    for (src, _, _) in self.edges:
        out_counts[src] = out_counts.get(src, 0) + 1
    result = []
    for sig, n in self.nodes.items():
        interactive = sum(len(v) for v in n.elements.values())
        if interactive > out_counts.get(sig, 0):
            result.append(sig)
    return result

def to_json(self) -> dict:
    return {
        "nodes": [vars(n) for n in self.nodes.values()],
        "edges": [vars(e) for e in self.edges.values()],
    }
```

---

## 9. Feed it to the planner

The planner runs in the gateway; the executor holds the graph. Two ways to bridge:

**A. Pass it in the request (simplest — start here).**
- Add `screen_graph_text: str = ""` to `NextTestCaseRequest` in `planner/schemas.py`.
- Executor sends `session_graph.to_prompt_text()` in the `/agent/next-testcase` payload.
- In `planner/prompts.py`, inject it as a new context block in `build_testcase_prompt`
  (alongside `figma_context`), labelled e.g. *"OBSERVED RUNTIME UI (ground truth — prefer these exact
  element labels; note screens the SRS implies but that are absent here)"*.
- Thread it through `langgraph_agent`'s `AgentState` like the other context blocks.

No schema/graph migration; you see value immediately.

**B. Persist to Neo4j (cleaner, later).**
- New node/edge types: `(:RunScreen {sig, activity, elements})` and
  `(:RunScreen)-[:TRANSITION {action, outcome, count}]->(:RunScreen)`, project-scoped.
- Cross-link to the spec side: `(:RunScreen)-[:MATCHES]->(:FigmaScreen)` (by element overlap) and to
  requirements, so coverage ("spec'd but never reached") lives in one graph and the planner reads it
  in `bootstrap_context`. More work; do it once the after-run text version proves useful.

---

## 10. Optional — screenshots & image context

MobileRun already saves screenshots (`0000.png…` in the trajectory folder; `ScreenshotEvent` live).
When a **new** signature first appears, run a VLM once to produce a short caption and store it on the
node (`ScreenNode.caption`). The planner then gets visual context (layout, empty/error states) the
a11y tree misses — cheaply, as text, and model-agnostic. Keep the PNG path on the node so you can go
fully multimodal later for the final generation step only. Reuse the pattern you already have for
LLM screen-purpose classification of Figma.

---

## 11. Optional — go live (later)

Replace after-run building with streaming, and let the planner replan when something interesting
happens rather than every step (LLM rounds are 30–60 s):

```python
handler = agent.run()
async for ev in handler.stream_events():
    name = type(ev).__name__
    if name == "RecordUIStateEvent":
        sig = session_graph.add_screen(ev.ui_state, agent.shared_state.current_activity_name)
        if is_new_screen or last_action_failed:        # replan triggers
            new_objective = call_planner(session_graph.to_prompt_text())
            agent.shared_state.queue_user_message(new_objective)   # inject mid-run (0.6.8 supports this)
    elif name == "ExecutorResultEvent":
        session_graph.add_edge(prev_sig, cur_sig, ev.action, ev.outcome)
result = await handler
```

Note `MobileAgentState.queue_user_message()` exists in 0.6.8 for mid-run injection — that's the hook
to steer MobileRun from your planner without forking it. Keep MobileRun as the low-level actor and
your planner as the high-level intent-setter so they don't fight.

---

## 12. Build order & verification

| Step | Do | Verify |
|---|---|---|
| 1 | `planner/screen_graph.py` (signature + `ScreenGraph` + `build_from_run`) | unit-feed two hand-made a11y lists + one action → 2 nodes, 1 edge |
| 2 | Enable `save_trajectory`, call `build_from_run` after a real run | dump graph after one session → nodes match screens seen, edges match taps |
| 3 | `to_prompt_text()` + pass via request field (§9A) | planner prompt contains the runtime UI block; generated steps use real observed labels |
| 4 | (opt) VLM captions, then Neo4j persistence, then live | each adds one verifiable capability |

## 13. Pitfalls
- **Volatile text** inflates node count → keep the `_looks_volatile` filter tight for your app.
- **Off-by-one** between `ui_states` and `action_history` → verify alignment on run 1 (§6).
- **Trajectory off** → `ui_states` is empty and `build_from_run` no-ops; make sure `save_trajectory`
  is set, or use the event-stream collector (§7).
- **Graph explosion** across many runs → cap `to_prompt_text` (already does) and consider persisting
  full graph to Neo4j while sending only the top-N relevant screens to the prompt.
- **Signature drift** for the *same* screen across app versions → expected; the graph is per-app-build.
```
