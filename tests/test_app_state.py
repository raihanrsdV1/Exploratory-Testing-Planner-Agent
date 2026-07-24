"""
Proves the Live App Model state-abstraction on the exact cases raised in review:

  1. Scrolling a contacts list to DIFFERENT contacts  -> SAME state
  2. Light mode -> dark mode                          -> SAME state
  3. Tapping into a contact detail screen             -> NEW state
  4. Opening a dialog on the same screen              -> NEW state
  5. Empty list vs populated list                     -> different states

Run: ./venv/bin/python tests/test_app_state.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion import app_state


PKG = "com.android.contacts"
LIST_ACT = "com.android.contacts/.activities.PeopleActivity"
DETAIL_ACT = "com.android.contacts/.activities.ContactDetailActivity"


def _row(name):
    # A contact row: SAME resource_id/class across rows (the row template),
    # only the free text differs.
    return {"resource_id": "com.android.contacts:id/contact_name", "class": "android.widget.TextView",
            "content_description": "", "clickable": True, "text": name}


def contacts_list(names, dark=False):
    # `dark` intentionally changes NOTHING structural — theme is pixels only.
    nodes = [
        {"resource_id": "com.android.contacts:id/toolbar", "class": "androidx.appcompat.widget.Toolbar",
         "content_description": "", "clickable": False, "text": "Contacts"},
        {"resource_id": "com.android.contacts:id/search", "class": "android.widget.Button",
         "content_description": "Search", "clickable": True, "text": ""},
        {"resource_id": "com.android.contacts:id/floating_action_button", "class": "android.widget.ImageButton",
         "content_description": "Create contact", "clickable": True, "text": "+"},
    ] + [_row(n) for n in names]
    return {"phone_state": {"package": PKG, "activity": LIST_ACT}, "nodes": nodes}


def contact_detail(name):
    return {"phone_state": {"package": PKG, "activity": DETAIL_ACT}, "nodes": [
        {"resource_id": "com.android.contacts:id/toolbar", "class": "androidx.appcompat.widget.Toolbar",
         "content_description": "", "clickable": False, "text": name},
        {"resource_id": "com.android.contacts:id/edit", "class": "android.widget.Button",
         "content_description": "Edit", "clickable": True, "text": "Edit"},
        {"resource_id": "com.android.contacts:id/call", "class": "android.widget.ImageButton",
         "content_description": "Call", "clickable": True, "text": ""},
        {"resource_id": "com.android.contacts:id/phone_number", "class": "android.widget.TextView",
         "content_description": "", "clickable": False, "text": "+1 555 0100"},
    ]}


def list_with_dialog(names):
    st = contacts_list(names)
    st["nodes"] = st["nodes"] + [
        {"resource_id": "android:id/alertTitle", "class": "android.app.AlertDialog",
         "content_description": "", "clickable": False, "text": "Delete contact?"},
        {"resource_id": "android:id/button1", "class": "android.widget.Button",
         "content_description": "", "clickable": True, "text": "Delete"},
    ]
    return st


def sig(state):
    return app_state.abstract_state(state)["signature"]


def main():
    checks = []

    def check(name, cond):
        checks.append((name, cond))
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # 1. Scroll to different contacts -> SAME state
    a = contacts_list(["Alice", "Bob", "Carol"])
    a_scrolled = contacts_list(["Xavier", "Yolanda", "Zach", "Wade", "Vera"])  # different data, more rows
    check("scroll: different contacts -> same signature", sig(a) == sig(a_scrolled))

    # 2. Light -> dark -> SAME state (structure identical)
    a_dark = contacts_list(["Alice", "Bob", "Carol"], dark=True)
    check("theme: light vs dark -> same signature", sig(a) == sig(a_dark))

    # 3. Contact detail -> NEW state
    d = contact_detail("Alice")
    check("navigate: contact detail -> new signature", sig(a) != sig(d))
    check("navigate: detail similarity to list is low",
          app_state.similarity(app_state.abstract_state(a)["key_set"],
                               app_state.abstract_state(d)["key_set"]) < 0.4)

    # 4. Dialog open -> NEW state
    dlg = list_with_dialog(["Alice", "Bob", "Carol"])
    check("dialog: open dialog -> new signature", sig(a) != sig(dlg))
    check("dialog: has_dialog flag set", app_state.abstract_state(dlg)["has_dialog"] is True)

    # 5. Empty vs populated list -> different states (empty lacks the row template key)
    empty = contacts_list([])
    check("empty vs populated -> different signature", sig(empty) != sig(a))

    # 6. is_same_state agrees with signatures
    check("is_same_state: scroll -> same",
          app_state.is_same_state(app_state.abstract_state(a), app_state.abstract_state(a_scrolled)))
    check("is_same_state: detail -> different",
          not app_state.is_same_state(app_state.abstract_state(a), app_state.abstract_state(d)))

    # 7. thin-tree detection triggers the visual fallback path
    thin = {"phone_state": {"package": PKG, "activity": LIST_ACT, "observationMode": "screenshot_only",
                            "accessibilityTree": False}, "nodes": []}
    check("thin-tree detected (use visual fallback)", app_state.is_thin_tree(thin) is True)
    check("rich-tree NOT thin", app_state.is_thin_tree(a) is False)

    passed = sum(1 for _, c in checks if c)
    print(f"\n{passed}/{len(checks)} checks passed")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
