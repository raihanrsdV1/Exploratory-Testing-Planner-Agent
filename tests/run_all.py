#!/usr/bin/env python3
"""Run every test module and summarise. Exit non-zero if any fail."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
mods = sorted(f for f in os.listdir(HERE)
              if f.startswith("test_") and f.endswith(".py"))
fails = []
for m in mods:
    r = subprocess.run([PY, os.path.join(HERE, m)], capture_output=True, text=True)
    tail = [l for l in r.stdout.splitlines() if "checks passed" in l]
    print(f"{m:<26} {tail[-1] if tail else 'no summary'}"
          f"{'' if r.returncode == 0 else '   <-- FAILED'}")
    if r.returncode != 0:
        fails.append(m)
print()
print(f"{len(mods) - len(fails)}/{len(mods)} modules passed")
sys.exit(1 if fails else 0)
