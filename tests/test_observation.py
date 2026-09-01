#!/usr/bin/env python3
"""Observation pipeline: driver output -> usable app-model state.

Every check here is a regression guard for a defect that reached a real run.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clients.executor_runner import adapt_driver_tree, _observe_device  # noqa: E402
from mobilerun.macro.state import normalize_ui_state  # noqa: E402

_passed = _failed = 0


def check(label, got, want):
    global _passed, _failed
    ok = got == want
    _passed, _failed = _passed + ok, _failed + (not ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + ("" if ok else f"  (got {got!r}, want {want!r})"))


# Shape the AndroidDriver actually returns: a11y_tree is a single ROOT NODE with
# nested children, and phone_state uses packageName. normalize_ui_state expects a
# LIST and a key called `package`, so raw output yields 0 nodes and package=None —
# which is why every recorded state was labelled "android.view.View #2", had no
# package, and could not be told apart from the launcher.
DRIVER_TREE = {
    "a11y_tree": {
        "className": "android.widget.FrameLayout", "resourceId": "android:id/content",
        "contentDescription": "", "clickable": False, "text": "",
        "packageName": "com.example.app",
        "children": [
            {"className": "android.view.View", "resourceId": "", "text": "",
             "contentDescription": "Browse Cattle", "clickable": True,
             "packageName": "com.example.app", "children": []},
            {"className": "android.widget.Button", "resourceId": "", "text": "",
             "contentDescription": "Save", "clickable": True,
             "packageName": "com.example.app", "children": []},
        ],
    },
    "phone_state": {"packageName": "com.example.app",
                    "activityName": "com.example.app/.MainActivity"},
    "device_context": {},
}


def main():
    print("adapt_driver_tree — unlocks the data normalize_ui_state could not see")
    raw = normalize_ui_state(DRIVER_TREE)
    check("raw driver output yields no nodes (the bug)", len(raw.get("nodes") or []), 0)
    check("raw driver output yields no package (the bug)",
          raw["phone_state"]["package"], None)

    fixed = normalize_ui_state(adapt_driver_tree(DRIVER_TREE))
    nodes = fixed.get("nodes") or []
    check("adapted: nodes are found", len(nodes) > 0, True)
    check("adapted: package resolved from packageName",
          fixed["phone_state"]["package"], "com.example.app")
    check("adapted: activity resolved (restores the identity bucket)",
          fixed["phone_state"]["activity"], "com.example.app/.MainActivity")
    check("adapted: content-descriptions survive",
          sorted(n["content_description"] for n in nodes if n.get("content_description")),
          ["Browse Cattle", "Save"])
    check("adapted: clickable survives",
          sum(1 for n in nodes if n.get("clickable")), 2)

    print("adapt_driver_tree — leaves non-driver payloads alone")
    already = {"elements": [], "phone_state": {"package": "x", "activity": "y"}}
    check("passes through an already-normalised payload", adapt_driver_tree(already), already)
    check("passes through a bare list", adapt_driver_tree([1, 2]), [1, 2])

    print("_observe_device — falls back rather than losing the observation")

    class Boom:
        async def get_ui_tree(self):
            raise RuntimeError("uiautomator output did not contain XML")

    sentinel = {"elements": [], "phone_state": {}}
    got = asyncio.run(_observe_device(Boom(), sentinel))
    check("driver failure falls back to the event payload", got, sentinel)
    check("no driver at all falls back too", asyncio.run(_observe_device(None, sentinel)), sentinel)

    print(f"\n{_passed}/{_passed + _failed} checks passed")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
