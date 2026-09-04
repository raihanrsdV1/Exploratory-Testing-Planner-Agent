"""Business logic behind the target-configuration UI endpoints.

Kept out of gateway/main.py to match its "thin router" design (see its module
docstring), and out of targets/ itself since that package is deliberately
UI-agnostic (see targets/README.md's "Porting to a UI" table) — this is the
UI-serving layer that wraps loader.py/schema.py, not a change to them.

Running a profile is deliberately a SEPARATE PROCESS, never in-process here:
targets/run.py's own docstring warns that settings.py is read once at first
import and is process-global, and the gateway has ALREADY imported settings at
its own startup — calling targets.run.run_profile() directly would silently
apply one profile's config to the gateway's own process (and therefore every
other request), not just to that one run.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from fastapi import HTTPException

from planner import config
from targets import loader
from targets.schema import TargetProfile

_REPO_ROOT = Path(__file__).resolve().parent.parent
_LOG_DIR = _REPO_ROOT / "logs"

# name -> subprocess.Popen. In-memory only, resets on gateway restart — the same
# tradeoff the dashboard's other "live" state already makes.
_RUNS: dict[str, subprocess.Popen] = {}


def _redact(data: dict) -> dict:
    """Blank credential fields before a profile is sent to the browser."""
    out = json.loads(json.dumps(data))  # cheap deep copy, data is plain JSON
    for section in ("web", "android"):
        login = (out.get(section) or {}).get("login")
        if isinstance(login, dict) and login.get("password"):
            login["password"] = "•" * 8
    return out


def list_targets(authorization: str | None) -> dict:
    """Every profile on disk, valid or not — an invalid one still needs to be
    editable in a UI, unlike targets.loader.list_profiles() which skips it."""
    config.check_gateway_auth(authorization)
    out = []
    profile_dir = loader.PROFILE_DIR
    if os.path.isdir(profile_dir):
        for filename in sorted(os.listdir(profile_dir)):
            if not filename.endswith(".json"):
                continue
            name = filename[:-5]
            path = os.path.join(profile_dir, filename)
            try:
                profile = loader.load(path)
                out.append({
                    "name": profile.name, "kind": profile.kind, "project": profile.project,
                    "display_name": profile.display_name, "description": profile.description,
                    "where": profile.web.base_url if profile.kind == "web" else profile.android.package,
                    "valid": True, "errors": [],
                })
            except loader.ProfileError as exc:
                out.append({
                    "name": name, "kind": "", "project": "", "display_name": "", "description": "",
                    "where": "", "valid": False, "errors": exc.problems or [str(exc)],
                })
    return {"profiles": out, "profile_dir": str(profile_dir)}


def get_target(name: str, authorization: str | None) -> dict:
    """Raw profile + validation errors — read directly rather than via loader.load()
    so an INVALID profile can still be loaded into an edit form to fix it."""
    config.check_gateway_auth(authorization)
    try:
        path = loader.resolve_path(name)
    except loader.ProfileError as exc:
        raise HTTPException(status_code=404, detail=exc.report())
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    raw.setdefault("name", os.path.splitext(os.path.basename(path))[0])
    profile = TargetProfile.from_dict(raw)
    errors = profile.validate(raw)
    return {"profile": _redact(raw), "errors": errors, "valid": not errors, "path": path}


def validate_target(body: dict, authorization: str | None) -> dict:
    """Validate without saving — for live inline form feedback."""
    config.check_gateway_auth(authorization)
    body = dict(body or {})
    profile = TargetProfile.from_dict(body)
    errors = profile.validate(body)
    return {"errors": errors, "valid": not errors}


def save_target(name: str, body: dict, authorization: str | None) -> dict:
    """Create or overwrite a profile. Refuses to save an invalid one."""
    config.check_gateway_auth(authorization)
    body = dict(body or {})
    body.setdefault("name", name)
    if str(body.get("name", "")) != name:
        raise HTTPException(status_code=400, detail="Profile 'name' in the body must match the URL")
    profile = TargetProfile.from_dict(body)
    errors = profile.validate(body)
    if errors:
        raise HTTPException(status_code=422, detail={"errors": errors})
    path = loader.save(profile)
    return {"saved": True, "path": path, "profile": _redact(profile.to_dict())}


def run_target(name: str, rounds: int | None, authorization: str | None) -> dict:
    """Launch `py -m targets.run <name>` as a detached subprocess."""
    config.check_gateway_auth(authorization)
    try:
        loader.load(name)  # exists and is usable before spawning anything
    except loader.ProfileError as exc:
        raise HTTPException(status_code=422, detail=exc.report())

    existing = _RUNS.get(name)
    if existing and existing.poll() is None:
        raise HTTPException(
            status_code=409,
            detail=f"A run for '{name}' is already in progress (pid {existing.pid}).",
        )

    _LOG_DIR.mkdir(exist_ok=True)
    log_path = _LOG_DIR / f"targets_run_{name}.log"
    cmd = [sys.executable, "-m", "targets.run", name]
    if rounds:
        cmd += ["--rounds", str(int(rounds))]
    with open(log_path, "w", encoding="utf-8") as log_fh:
        proc = subprocess.Popen(cmd, cwd=str(_REPO_ROOT), stdout=log_fh, stderr=subprocess.STDOUT)
    _RUNS[name] = proc
    return {"started": True, "pid": proc.pid, "log_path": str(log_path)}


def run_status(name: str, authorization: str | None) -> dict:
    config.check_gateway_auth(authorization)
    proc = _RUNS.get(name)
    if not proc:
        return {"running": False}
    code = proc.poll()
    return {"running": code is None, "pid": proc.pid, "exit_code": code}
