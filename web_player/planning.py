"""The browser's execution contract, shared with both planner paths.

Keep this module pure: the gateway may serve several targets and must use the
request's constraints, never its own process-wide browser settings.
"""
import json
import re


def contract(cfg):
    from .actions import ACTION_FIELDS
    return {
        "blocked_controls": list(cfg.WEB_BLOCKED_TEXTS),
        "blocked_urls": list(cfg.WEB_BLOCKED_URL_PATTERNS),
        "base_url": cfg.WEB_BASE_URL,
        "max_steps": cfg.WEB_MAX_STEPS,
        "actions": list(ACTION_FIELDS),
        "unsupported": ["file upload", "drag-and-drop", "download inspection", "external email", "CAPTCHA solving"],
    }


def guidance(constraints):
    if not constraints:
        return ""
    return (
        "\nBROWSER EXECUTION CONTRACT (takes precedence over coverage pressure):\n"
        + json.dumps(constraints, ensure_ascii=False)
        + "\nChoose one bounded, observable UI behavior. Include setup in the budget. "
        "Do not require blocked controls, unsupported actions, external fixtures, or a CAPTCHA. "
        "A file import requires an upload/drag tool and a fixture; neither is available. "
        "Do not substitute typing a filename for uploading a file. "
        "Use the normal UI; do not navigate directly to API endpoints to invent method-mismatch tests. "
        "Include positive tests as well as negative tests. On a new session choose a simple "
        "navigation or visible-state test first to establish that execution works. "
        "Do not infer an application defect from a previous agent livelock or timeout. "
        "Previously blocked or failed attempts should guide you to a different feasible behavior."
        " Never assume an empty account or no existing records; the stored session contains user data. "
        "Do not require global absence of console or HTTP errors for an unrelated UI test; "
        "assert the specific visible behavior under test. Browser diagnostics are reported separately."
    )


def rejection_errors(tc, constraints, excluded_titles=()):
    """Conservative admission gate for behavior the current browser cannot execute."""
    from planner.textutil import is_similar_to_existing

    if not isinstance(tc, dict):
        return ["A test case must be an object."]
    errors = []
    for key in ("title", "objective", "expected_result"):
        if not str(tc.get(key) or "").strip():
            errors.append(f"Missing {key}.")
    title = str(tc.get("title") or "")
    if title and is_similar_to_existing(title, list(excluded_titles), threshold=0.60):
        errors.append("This duplicates an already attempted or rejected test. Choose a different behavior.")
    if not constraints:
        return errors
    # Do not match the expected result: 'logout must not occur' does not require logout.
    text = " ".join(str(tc.get(k) or "") for k in
                    ("title", "objective", "steps", "preconditions")).lower()
    for label in constraints.get("blocked_controls", []):
        if label and re.search(rf"(?<!\w){re.escape(label.lower())}(?!\w)", text):
            errors.append(f"Requires blocked control '{label}'. Choose an unrelated allowed behavior.")
    for path in constraints.get("blocked_urls", []):
        if path and path.lower() in text:
            errors.append(f"Requires blocked URL '{path}'.")
    actions = set(constraints.get("actions", []))
    preconditions = " ".join(str(p) for p in (tc.get("preconditions") or [])).lower()
    if re.search(r"no (?:existing |existing\s+\w+\s+)?(?:projects|surveys|records|data)|"
                 r"empty (?:account|database)|database is empty|fresh install", preconditions):
        errors.append("Cannot assume an empty account/database in a reused session. Choose a test using existing data or a search with no matches.")
    if not {"upload", "drag"} & actions and re.search(
        r"drag[\s-]+and[\s-]+drop|upload|(?:import|load)\w*\s+(?:a\s+|an\s+|data\s+|csv\s+|excel\s+|local\s+)*(?:file|spreadsheet)|file\s+import", text
    ):
        errors.append("File import/upload/drag-and-drop is unsupported; choose a behavior using available actions.")
    if re.search(r"/api/", text):
        errors.append("Direct API probes are outside this UI executor contract; test the behavior through the UI.")
    expected = str(tc.get("expected_result") or "").lower()
    if re.search(r"(?:no|zero|without)(?:\s+\w+){0,3}\s+console errors|no.{0,35}http\s+4xx", expected):
        if not re.search(r"console|http|network|status code", str(tc.get("objective") or "").lower()):
            errors.append("Do not append a global clean-console/HTTP assertion to an unrelated UI objective. Assert the visible behavior only.")
    return errors
