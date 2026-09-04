# Target profiles

One JSON file per thing under test. It decides **what** is tested and **which
player** runs — Playwright for a website, Droidrun for an Android app.

```bash
py -m targets.run --list                # what can I test?
py -m targets.run wikipedia             # run it (opens a visible browser)
py -m targets.run wikipedia --dry-run   # resolve the config, run nothing
py -m targets.run wikipedia --ingest    # load its documents first
py -m targets.run contacts-app          # same command, Android target
py -m targets.run ./my-site.json        # any path works too
```

## Why this exists

`.env` is one global configuration for one machine. Switching from the contacts
app to Wikipedia means changing a dozen variables and remembering which ones —
and the one that matters most is easy to miss:

> **`PROJECT` scopes the entire knowledge graph.** Leave it on `contacts-app` and
> the planner keeps retrieving the contacts SRS, screens and test history, and
> keeps writing tests about contacts, no matter what the other variables say.

A profile sets all of them together, so a target is switched atomically or not at
all. `--dry-run` prints exactly what will be used before anything runs.

## Precedence

```
target profile   →   .env   →   settings.py defaults
   (this run)     (machine)        (fallback)
```

The profile wins. `.env` supplies what is machine-level rather than
target-level: API keys, service URLs, Neo4j credentials. `targets/env.py` is the
only module that knows `settings.py` variable names.

This depends on one ordering rule: `settings.py` reads the environment when it is
first imported, so `env.apply()` must run first. `targets/run.py` imports
`settings` and both players **inside functions** for that reason, asserts the
ordering at runtime, and `tests/test_targets.py` fails if a top-level import is
ever added.

## Profile shape

Only `name`, `kind`, `project` and the target block are required; everything else
has a default.

```json
{
  "name": "wikipedia",
  "kind": "web",
  "project": "wikipedia",
  "display_name": "Wikipedia",
  "description": "Public encyclopedia — read-only exploration.",

  "web": {
    "base_url": "https://en.wikipedia.org",
    "browser": "chromium",
    "headless": false,
    "slow_mo_ms": 300,
    "viewport": "1280x800",
    "same_origin_only": true,
    "blocked_texts": ["edit", "undo", "log out"],
    "blocked_url_patterns": ["action=edit"],
    "storage_state": "",
    "fail_on_page_error": false,
    "fail_on_http_5xx": true,
    "console_ignore": ["favicon"],
    "login": { "url": "", "user": "", "password": "", "hint": "", "role": "" }
  },

  "android": {
    "package": "com.example.app",
    "activity": "",
    "labels": ["My App"],
    "target_app_only": true,
    "device_reset": "pm_clear",
    "login": { "user": "", "password": "", "role": "" }
  },

  "knowledge": { "srs_path": "", "figma_path": "", "defects_path": "" },
  "run": { "rounds": 5, "max_steps": 30, "timeout": 420,
           "clean_slate": true, "self_heal": true },
  "model": { "provider": "", "model": "" }
}
```

Use the `web` block for `kind: "web"` and the `android` block for
`kind: "android"` — the other is ignored, and the settings it would write are
never applied.

`knowledge` may be empty. That is **zero-doc exploration**: with no SRS and no
Figma the planner works from the live UI and heuristics, which is the right setup
for a site you do not own (like Wikipedia).

## Validation

`validate()` **returns** its problems rather than exiting, so the same code
serves the CLI and a future UI:

```
targets/profiles/broken.json is not a usable target profile
  - project: required — it scopes the Neo4j knowledge graph. Reusing another
    target's project makes the planner generate tests from that target's
    documents and history.
  - web.base_url: must start with http:// or https:// (got 'demo.example.com')
  - headles: unknown setting — check the spelling; it is being ignored
```

Unknown keys are reported, not ignored — a typo in a hand-written profile would
otherwise be invisible.

## Adding a target

Copy a profile, change `name`, `project` and the target block, and run it. Give
every target its **own `project`**; sharing one merges two apps' knowledge and
history into a single graph slice.

Think about `blocked_texts` before the first run. The agent has no sandbox on the
web, and the defaults only cover account and session destruction — the Wikipedia
profile adds `edit`, `undo`, `rollback` and friends because on that site those
are the destructive controls.

## Porting to a UI

The layering is already the one a UI needs:

| Module | Role | What a UI does with it |
|---|---|---|
| `schema.py` | shape + `validate()` | generate the form from `dataclasses.fields`; show returned errors inline |
| `loader.py` | find / read / write | swap for a database or an API; `save()` is already there |
| `env.py` | profile → settings | untouched — the UI never learns a variable name |
| `run.py` | `run_profile(profile)` | the single call to start a run |

`env.redacted()` masks passwords for anything displayed. Nothing in the layer
prints, exits, or reads `sys.argv` except `run.py`'s `main()`.
