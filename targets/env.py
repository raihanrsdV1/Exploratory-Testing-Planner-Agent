"""Profile -> environment overrides.

The one module that knows `settings.py` variable names. Everything else deals in
profile fields, so replacing or renaming a setting is a change here and nowhere
else — and a UI never has to learn that "which site" is spelled ``WEB_BASE_URL``.

**Ordering matters.** ``settings.py`` reads the environment at import time, so
``apply()`` must run before anything imports it. ``load_dotenv`` does not override
variables that are already set, so these overrides win over `.env` — which is the
whole point: the profile is authoritative for the run, `.env` supplies the
machine-level rest (API keys, service URLs, Neo4j credentials).
"""

from __future__ import annotations

import os

from .schema import TargetProfile


def build(profile: TargetProfile) -> dict[str, str]:
    """The environment this profile implies. Pure — nothing is written."""
    env: dict[str, str] = {
        "PROJECT": profile.project,
        "PROJECT_NAME": profile.project,
        "APP_NAME": profile.label(),
    }

    know = profile.knowledge
    env["SRS_PATH"] = know.srs_path
    env["FIGMA_PATH"] = know.figma_path

    run = profile.run
    env["CLEAN_SLATE"] = _bool(run.clean_slate)
    env["SELF_HEAL"] = _bool(run.self_heal)

    if profile.kind == "web":
        env.update(_web(profile))
    else:
        env.update(_android(profile))
    return env


def _web(profile: TargetProfile) -> dict[str, str]:
    web, run, model = profile.web, profile.run, profile.model
    env = {
        "WEB_BASE_URL": web.base_url,
        "WEB_SITE_NAME": profile.label(),
        "WEB_BROWSER": web.browser,
        "WEB_HEADLESS": _bool(web.headless),
        "WEB_SLOW_MO_MS": str(max(0, int(web.slow_mo_ms))),
        "WEB_VIEWPORT": web.viewport,
        "WEB_SAME_ORIGIN_ONLY": _bool(web.same_origin_only),
        "WEB_STORAGE_STATE": web.storage_state,
        "WEB_FAIL_ON_PAGE_ERROR": _bool(web.fail_on_page_error),
        "WEB_FAIL_ON_HTTP_5XX": _bool(web.fail_on_http_5xx),
        "WEB_ROUNDS": str(run.rounds),
        "WEB_MAX_STEPS": str(run.max_steps),
        "WEB_TIMEOUT": str(run.timeout),
        "WEB_LOGIN_URL": web.login.url,
        "WEB_LOGIN_USER": web.login.user,
        "WEB_LOGIN_PASSWORD": web.login.password,
        "WEB_LOGIN_HINT": web.login.hint,
    }
    # A blank list must NOT be written: settings falls back to its default on an
    # empty value, so "" and "unset" mean the same thing — and writing "" for a
    # guardrail list would read as "no guardrails" while actually restoring the
    # defaults. Omit it instead, and let the default apply explicitly.
    _set_list(env, "WEB_BLOCKED_TEXTS", web.blocked_texts)
    _set_list(env, "WEB_BLOCKED_URL_PATTERNS", web.blocked_url_patterns)
    _set_list(env, "WEB_CONSOLE_IGNORE", web.console_ignore)
    if model.provider:
        env["WEB_LLM_PROVIDER"] = model.provider
    if model.model:
        env["WEB_LLM_MODEL"] = model.model
    return env


def _android(profile: TargetProfile) -> dict[str, str]:
    android, run, model = profile.android, profile.run, profile.model
    env = {
        "TARGET_APP_PACKAGE": android.package,
        "TARGET_APP_ACTIVITY": android.activity,
        "TARGET_APP_ONLY": _bool(android.target_app_only),
        "DEVICE_RESET": android.device_reset,
        "EXECUTOR_ROUNDS": str(run.rounds),
        "EXECUTOR_MAX_STEPS": str(run.max_steps),
        "EXECUTOR_TIMEOUT": str(run.timeout),
        "APP_LOGIN_ROLE": android.login.role,
        "APP_LOGIN_IDENTIFIER": android.login.user,
        "APP_LOGIN_SECRET": android.login.password,
        "APP_LOGIN_HINT": android.login.hint,
    }
    _set_list(env, "TARGET_APP_LABELS", android.labels)
    if model.provider:
        env["EXECUTOR_LLM_PROVIDER"] = model.provider
    if model.model:
        env["EXECUTOR_LLM_MODEL"] = model.model
    return env


def apply(profile: TargetProfile) -> dict[str, str]:
    """Write the profile's environment. Call BEFORE importing ``settings``."""
    env = build(profile)
    for key, value in env.items():
        os.environ[key] = value
    return env


_SECRET_HINTS = ("PASSWORD", "SECRET", "TOKEN", "API_KEY")


def redacted(env: dict[str, str]) -> dict[str, str]:
    """The same mapping with secrets masked — safe for logs, a UI, or --dry-run."""
    return {
        k: ("*" * 8 if v and any(h in k for h in _SECRET_HINTS) else v)
        for k, v in env.items()
    }


def _bool(value) -> str:
    return "true" if value else "false"


def _set_list(env: dict[str, str], key: str, values: list[str]) -> None:
    cleaned = [str(v).strip() for v in (values or []) if str(v).strip()]
    if cleaned:
        env[key] = ",".join(cleaned)
