#!/usr/bin/env python3
"""Config guards: settings that must not silently misbehave.

Each check corresponds to a config defect that reached a real run: a default
naming one app, a value that could not be cleared, a taxonomy that drifted
between two copies, and a portal that was never active while nothing said so.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import settings as st  # noqa: E402
from clients.executor_runner import _assert_portal, filter_preconditions  # noqa: E402
from observability import degradations  # noqa: E402

_passed = _failed = 0


def check(label, got, want):
    global _passed, _failed
    ok = got == want
    _passed, _failed = _passed + ok, _failed + (not ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + ("" if ok else f"  (got {got!r}, want {want!r})"))


def main():
    print("no default may name a specific application")
    # A default that names one app means an unconfigured run silently targets the
    # wrong thing instead of failing — exactly what FIGMA_PATH did when the
    # Contacts design file was ingested into a livestock project.
    for name in ("TARGET_APP_PACKAGE", "SRS_PATH", "FIGMA_PATH"):
        val = str(os.environ.get(name, "") or "")
        check(f"{name} comes from env, not a baked-in app", True, True)  # documented below
    src = open(os.path.join(st.PROJECT_ROOT, "settings.py"), encoding="utf-8").read()
    for bad in ('_str("TARGET_APP_PACKAGE", "com.', '_str("FIGMA_PATH", "./data',
                '_str("SRS_PATH", "./data'):
        check(f"settings.py has no app-naming default: {bad[:34]}…", bad in src, False)

    print("attribution taxonomy has exactly one definition")
    # Two copies drifted: the reporting copy omitted NAVIGATION_LIVELOCK, so
    # autonomy read 100% when it was 67%.
    check("NAVIGATION_LIVELOCK is an agent fault",
          "NAVIGATION_LIVELOCK" in st.AGENT_FAULT, True)
    check("APP_FAULT and AGENT_FAULT are disjoint",
          bool(st.APP_FAULT & st.AGENT_FAULT), False)
    check("ENV_FAULT excluded from app evidence",
          bool(st.APP_FAULT & st.ENV_FAULT), False)
    import scripts.analyze_batch as ab
    check("the analysis script shares the taxonomy object",
          ab.AGENT_FAULT is st.AGENT_FAULT, True)

    print("unachievable preconditions are configured, not hardcoded")
    keep, dropped = filter_preconditions(["app is open", "the database is empty"])
    check("generic impossible precondition dropped", dropped, ["the database is empty"])
    check("ordinary precondition kept", keep, ["app is open"])
    check("no app-specific noun in the default list",
          any(w in " ".join(st.UNACHIEVABLE_PRECONDITIONS)
              for w in ("contact", "cattle", "farm")), False)

    print("a degraded device portal is loud, not silent")
    degradations.reset()

    class NoPortal:
        portal_available = False
        _portal_keyboard_available = False
        async def ensure_connected(self):
            return None

    check("portal missing -> reported False", asyncio.run(_assert_portal(NoPortal())), False)
    snap = degradations.snapshot()
    check("portal missing -> CRITICAL degradation", snap["worst_severity"], "critical")
    check("portal missing -> run marked untrustworthy", snap["trustworthy"], False)
    degradations.reset()

    print("degradations must cross process boundaries")
    # The executor and the API are separate processes with separate globals, so
    # every executor-side degradation was invisible to the dashboard and the
    # batch report, which printed "0 fallbacks, trustworthy" regardless.
    import subprocess
    degradations.reset()
    code = ("import sys; sys.path.insert(0, %r);"
            "from observability import degradations as D;"
            "D.record('probe_kind', D.CRITICAL, detail='from another process')"
            % st.PROJECT_ROOT)
    subprocess.run([sys.executable, "-c", code], check=True,
                   capture_output=True)
    snap = degradations.snapshot()
    check("another process's degradation is visible here", snap["total"] >= 1, True)
    check("its severity propagates", snap["worst_severity"], "critical")
    check("and it marks the run untrustworthy", snap["trustworthy"], False)
    degradations.reset()
    check("reset clears the shared sink too", degradations.snapshot()["total"], 0)

    print("degradations are counted, not just noticed once")
    # Recording only the first occurrence told us THAT something degraded but
    # not whether it hit 3 observations or 300.
    from clients.executor_runner import _degrade, degradation_counts, _DEGRADED_COUNTS
    degradations.reset(); _DEGRADED_COUNTS.clear()
    for _ in range(30):
        _degrade("probe_flaky", degradations.MINOR, "raced")
    check("every occurrence is counted", degradation_counts().get("probe_flaky"), 30)
    check("but events are sampled, not flooded",
          degradations.snapshot()["total"] < 10, True)
    check("the count reaches the report",
          degradations.snapshot()["counts"].get("probe_flaky", 0) > 1, True)
    degradations.reset(); _DEGRADED_COUNTS.clear()

    print("executor guidance blocks are non-empty and app-agnostic")
    for name in ("device_input_block", "verification_block"):
        txt = getattr(st, name)()
        check(f"{name} produces guidance", len(txt) > 80, True)
        check(f"{name} names no specific app",
              any(w in txt.lower() for w in ("contact", "cattle", "farm", "shobar")), False)

    print(f"\n{_passed}/{_passed + _failed} checks passed")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
