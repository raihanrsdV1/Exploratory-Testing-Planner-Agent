"""Factual executor context and structural validation, not coverage policy."""
import json


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
        "\nBROWSER EXECUTION CONTRACT (capabilities and target safety rules):\n"
        + json.dumps(constraints, ensure_ascii=False)
        + "\nUse these facts when describing setup and execution. Test complexity and "
        "coverage are not restricted to simple navigation or visible-state checks. "
        "A missing capability is a coverage gap, not evidence that the website is defective."
    )


def rejection_errors(tc, constraints, excluded_titles=()):
    """Reject malformed or duplicate proposals, never feature-name keywords."""
    from planner.textutil import is_similar_to_existing
    if not isinstance(tc, dict):
        return ["A test case must be an object."]
    errors = [f"Missing {key}." for key in ("title", "objective", "expected_result")
              if not str(tc.get(key) or "").strip()]
    title = str(tc.get("title") or "")
    if title and is_similar_to_existing(title, list(excluded_titles), threshold=0.60):
        errors.append("This duplicates an already attempted or rejected test. Choose a different behavior.")
    return errors
