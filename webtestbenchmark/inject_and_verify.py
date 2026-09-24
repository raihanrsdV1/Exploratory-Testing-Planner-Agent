"""Inject one WebTestPilot bug into a running benchmark app and prove it fires.

Two arms, same Playwright steps (taken from benchmark/<app>/test_cases/<case>.yaml):

    control  - no injection, the ground-truth assertions must pass
    bug      - benchmark/<app>/bugs/<case>.js injected via context.add_init_script,
               the bug must trigger and the ground truth must now be violated

Only when BOTH hold is the injection proven: the bug is present, and it is the
bug (not our own steps) that breaks the page.

    venv/Scripts/python.exe webtestbenchmark/inject_and_verify.py            # verify
    venv/Scripts/python.exe webtestbenchmark/inject_and_verify.py --live     # headed browser,
                                                                              # bug injected, left open
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from playwright.sync_api import Page, expect, sync_playwright

HERE = Path(__file__).parent
REPO = HERE / "WebTestPilot"
BASE_URL = "http://localhost:8081"
USERNAME, PASSWORD = "admin@admin.com", "password"   # baselines/test_setup_functions.py


def load_prepare_bug_script():
    """baselines/ is not a package; import bug_injector.py by path."""
    spec = importlib.util.spec_from_file_location("bug_injector", REPO / "baselines" / "bug_injector.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.prepare_bug_script


def login(page: Page) -> None:
    page.goto(f"{BASE_URL}/login")
    page.get_by_role("textbox", name="Email").fill(USERNAME)
    page.get_by_role("textbox", name="Password").fill(PASSWORD)
    page.get_by_role("button", name="Log In").click()
    page.wait_for_load_state("domcontentloaded")


def create_book_steps(page: Page, name: str) -> None:
    """The action column of benchmark/bookstack/test_cases/create_book.yaml."""
    page.get_by_role("link", name="Books", exact=True).first.click()          # /books  visit 1
    expect(page.get_by_role("heading", name="Books", exact=True)).to_be_visible()
    page.get_by_role("link", name="Create New Book").click()
    expect(page.get_by_role("heading", name="Create New Book")).to_be_visible()
    page.get_by_role("textbox", name="Name").fill(name)
    page.frame_locator('iframe[title="Rich Text Area"]').locator("body").fill("New Book Description")
    page.get_by_role("button", name="Save Book").click()
    expect(page.get_by_role("heading", name=name)).to_be_visible()
    page.get_by_role("link", name="Books", exact=True).first.click()          # /books  visit 2 -> bug fires
    expect(page.get_by_role("heading", name="Books", exact=True)).to_be_visible()


def bug_triggered(page: Page) -> bool:
    # Set by bug_injector.js cleanup() the moment onConditionMet() has run.
    return page.evaluate("sessionStorage.getItem('__BUG_INJECTOR_TRIGGERED__')") == "true"


def run_arm(pw, bug_script: str | None, book_name: str, headless: bool) -> dict:
    browser = pw.chromium.launch(headless=headless)
    context = browser.new_context(viewport={"width": 1280, "height": 800})
    if bug_script:
        # Context-level, not page-level: survives every navigation and new tab.
        context.add_init_script(bug_script)
    page = context.new_page()
    try:
        login(page)
        create_book_steps(page, book_name)

        # create_book.yaml final ground truth: the new book is listed under #recents
        recents_link = page.locator("#recents").get_by_role("link", name=book_name)
        listed = recents_link.count() > 0 and recents_link.first.is_visible()

        # What create_book.js does when it fires: either rewrites the card's description
        # to 'Bad Description' or removes the entry from the 'New Books' section.
        desc_corrupted = page.get_by_text("Bad Description", exact=True).count() > 0
        return {"triggered": bug_triggered(page), "listed_in_recents": listed,
                "desc_corrupted": desc_corrupted}
    finally:
        page.screenshot(path=str(HERE / f"screenshot_{'bug' if bug_script else 'control'}.png"), full_page=True)
        context.close()
        browser.close()


def verify(headless: bool) -> int:
    bug_script = load_prepare_bug_script()(REPO / "benchmark" / "bookstack" / "bugs" / "create_book.js")
    with sync_playwright() as pw:
        control = run_arm(pw, None, "Control Book", headless)
        print(f"control : {control}")
        bug = run_arm(pw, bug_script, "New Book", headless)   # bug matches on title 'New Book'
        print(f"bug     : {bug}")

    ok_control = control["listed_in_recents"] and not control["triggered"] and not control["desc_corrupted"]
    ok_bug = bug["triggered"] and (bug["desc_corrupted"] or not bug["listed_in_recents"])
    print()
    print("control arm:", "PASS - ground truth holds without injection" if ok_control else "FAIL")
    print("bug arm    :", "PASS - bug fired and ground truth is violated" if ok_bug else "FAIL")
    return 0 if (ok_control and ok_bug) else 1


def live() -> None:
    """Headed browser with the bug injected, logged in, left open until the window closes."""
    bug_script = load_prepare_bug_script()(REPO / "benchmark" / "bookstack" / "bugs" / "create_book.js")
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False)
        context = browser.new_context(viewport={"width": 1280, "height": 800})
        context.add_init_script(bug_script)
        page = context.new_page()
        login(page)
        print(f"LIVE: {BASE_URL} with bookstack/create_book.js injected. Trigger: create a book named "
              f"'New Book', then open Books a second time. Close the window to stop.", flush=True)
        page.wait_for_event("close", timeout=0)
        browser.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="headed browser with the bug injected, kept open")
    ap.add_argument("--headed", action="store_true", help="run the verification with a visible browser")
    args = ap.parse_args()
    if args.live:
        live()
        sys.exit(0)
    sys.exit(verify(headless=not args.headed))
