#!/usr/bin/env python3
"""Inject one WebTestPilot canonical bug into the running BookStack container.

Why server-side rather than Playwright's add_init_script (which is how
WebTestPilot's own harness does it): injecting from the tester would mean the
tester knows where the bug is. Writing it into the application's own
"custom head" setting makes the fault a property of the deployment, so the
exploratory agent meets it exactly as a user would - blind.

The bug lives in the `app-custom-head` setting, rendered into every page by
layouts/base.blade.php. It is stored in the database, so it survives restarts
and is removed again with `clear`.

    python3 scripts/bookstack_bug.py list
    python3 scripts/bookstack_bug.py inject create_book
    python3 scripts/bookstack_bug.py status
    python3 scripts/bookstack_bug.py clear
"""
from __future__ import annotations

import base64
import pathlib
import re
import subprocess
import sys

BUGS = pathlib.Path("/Users/khalidhasantuhin/Documents/Terms/4-1/Capstone/WebTestPilot/benchmark/bookstack/bugs")
TEMPLATE = pathlib.Path("/Users/khalidhasantuhin/Documents/Terms/4-1/Capstone/WebTestPilot/baselines/bug_injector.js")
DB, APP = "bookstack-db-1", "bookstack-app-1"


def sql(statement: str) -> str:
    out = subprocess.run(
        ["docker", "exec", "-i", DB, "mysql", "-u", "admin", "-padmin", "bookstack", "-e", statement],
        capture_output=True, text=True)
    return "\n".join(l for l in (out.stdout + out.stderr).splitlines() if "Using a password" not in l)


def build(name: str, persistent: bool = True) -> str:
    """Merge one bug's two function bodies into the harness template.

    ``persistent`` removes the template's one-shot sessionStorage sentinel. The
    benchmark fires each bug at most once per browser session, which suits its
    scripted step sequences: the script is already standing where the fault will
    appear. An exploratory agent arrives in its own order, so the single trigger
    is routinely spent on a page where the target is not yet present - observed
    here, where a login redirect consumed the trigger and the fault then disarmed
    itself without changing anything. Re-arming on each page load makes the fault
    a standing property of the deployment, which is what an unscripted tester has
    to be able to find. Pass faithful=one-shot to reproduce benchmark semantics.
    """
    path = BUGS / f"{name}.js"
    if not path.exists():
        sys.exit(f"no such bug: {name}. Run 'list' to see the {len(list(BUGS.glob('*.js')))} available.")
    code = path.read_text()
    cond = re.findall(r"// BEGIN isConditionMet\s*(.*?)\s*// END isConditionMet", code, re.DOTALL)
    on = re.findall(r"// BEGIN onConditionMet\s*(.*?)\s*// END onConditionMet", code, re.DOTALL)
    if not cond or not on:
        sys.exit(f"{name}.js does not carry both function blocks")
    tpl = TEMPLATE.read_text()
    tpl = tpl.replace("const isConditionMet = () => {};", cond[-1])
    tpl = tpl.replace("const onConditionMet = () => {};", on[-1])
    if persistent:
        # window[NAMESPACE] still guards duplicate observers within one page load;
        # only the cross-page-load sentinel is dropped.
        tpl = tpl.replace('if (window[NAMESPACE] || sessionStorage.getItem(STORAGE_KEY)) return;',
                          'if (window[NAMESPACE]) return;')
        tpl = tpl.replace('sessionStorage.setItem(STORAGE_KEY, "true");', '')
    # Several bugs share these globals verbatim ("__prev_path__",
    # "__visit_count__", one NAMESPACE). Concatenated unchanged they would
    # overwrite each other's path history and visit counters, so each bug gets
    # its own copy of every piece of cross-call state.
    tag = re.sub(r"[^A-Za-z0-9_]", "_", name)
    for key in ("__BUG_INJECTOR__", "__BUG_INJECTOR_TRIGGERED__",
                "__prev_path__", "__visit_count__", "__triggered__"):
        tpl = tpl.replace(key, f"{key.rstrip('_')}_{tag}__")
    mode = "persistent" if persistent else "faithful-one-shot"
    return f"<!-- etp-bug: {name} ({mode}) -->\n<script>\n{tpl}\n</script>"


def write_head(html: str) -> None:
    payload = base64.b64encode(html.encode()).decode()
    sql(f"UPDATE settings SET value = FROM_BASE64('{payload}') WHERE setting_key = 'app-custom-head';")
    if "app-custom-head" not in sql("SELECT setting_key FROM settings WHERE setting_key='app-custom-head';"):
        sql(f"INSERT INTO settings (setting_key, value, created_at, updated_at) "
            f"VALUES ('app-custom-head', FROM_BASE64('{payload}'), NOW(), NOW());")
    # Settings are cached; without this the page keeps serving the old head.
    subprocess.run(["docker", "exec", APP, "php", "/var/www/bookstack/artisan", "cache:clear"],
                   capture_output=True, text=True)


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "list":
        for f in sorted(BUGS.glob("*.js")):
            print(" ", f.stem)
    elif cmd == "inject":
        if len(sys.argv) < 3:
            sys.exit("usage: inject <bug_name>")
        names = [a for a in sys.argv[2:] if not a.startswith("--")]
        persistent = "--faithful" not in sys.argv
        write_head("\n".join(build(n, persistent) for n in names))
        mode = "persistent" if persistent else "faithful one-shot"
        print(f"injected {len(names)} bug(s) ({mode}): {', '.join(names)}")
        print("verify with: curl -s http://localhost:8081/ | grep etp-bug")
    elif cmd == "clear":
        write_head("")
        print("cleared app-custom-head - the deployment is clean again")
    else:
        row = sql("SELECT LENGTH(value) len, LEFT(value, 60) head FROM settings WHERE setting_key='app-custom-head';")
        print(row or "(no app-custom-head row)")


if __name__ == "__main__":
    main()
