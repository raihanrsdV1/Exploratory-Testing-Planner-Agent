#!/usr/bin/env python3
"""Merge one WebTestPilot bug file into their injector template.

WebTestPilot ships each bug as two function bodies (isConditionMet /
onConditionMet) that only make sense spliced into their bug_injector.js
template. This produces that one ready-to-inject file, the same way
WEB_STORAGE_STATE is produced once by hand and then just pointed at.

Usage:
    py scripts/prepare_webtestpilot_bug.py bookstack create_book

Writes data/bugs/webtestpilot-bookstack-create_book.js
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLONE = ROOT / "webtestbenchmark" / "WebTestPilot"


def load_prepare_bug_script():
    spec = importlib.util.spec_from_file_location(
        "bug_injector", CLONE / "baselines" / "bug_injector.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.prepare_bug_script


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <app> <bug_name>", file=sys.stderr)
        sys.exit(1)
    app, bug_name = sys.argv[1], sys.argv[2]

    bug_path = CLONE / "benchmark" / app / "bugs" / f"{bug_name}.js"
    if not bug_path.exists():
        print(f"No such bug: {bug_path}", file=sys.stderr)
        sys.exit(1)

    script = load_prepare_bug_script()(bug_path)

    out_dir = ROOT / "data" / "bugs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"webtestpilot-{app}-{bug_name}.js"
    out_path.write_text(script, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
