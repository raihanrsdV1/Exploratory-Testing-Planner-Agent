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
import re
import subprocess
import sys
from pathlib import Path

from fastapi import HTTPException

from planner import config
from targets import loader
from targets.schema import TargetProfile

_REPO_ROOT = Path(__file__).resolve().parent.parent
_LOG_DIR = _REPO_ROOT / "logs"

# Uploads land in their OWN directory, never straight into data/inputs/. That
# directory holds curated, version-controlled specs (wikipedia-spec.md,
# ShobarKhamar-SRS.txt, ...); an upload that happened to share a filename would
# otherwise overwrite one of them from the browser, with no way back.
_UPLOAD_DIR = _REPO_ROOT / "data" / "inputs" / "uploads"

# Per-kind extension allowlists. 'srs' mirrors ingestion/document_loader.py's
# supported set; the other two are read as structured data, not prose.
_DOC_EXTS = {".txt", ".md", ".markdown", ".text", ".rst",
             ".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
             ".html", ".htm", ".rtf", ".odt", ".epub", ".csv"}
_ALLOWED_EXTS = {"srs": _DOC_EXTS, "figma": {".json"}, "defects": {".json", ".csv"}}

_MAX_UPLOAD_BYTES = 25 * 1024 * 1024
_CHUNK = 1024 * 1024


def _safe_upload_name(filename: str, kind: str) -> str:
    """Reduce a browser-supplied filename to a safe basename.

    ``Path(...).name`` drops any directory component (including '../' and a
    Windows 'C:\\...' prefix), then everything outside a conservative character
    set is replaced — a filename is attacker-controlled input and is about to
    become a path on disk.
    """
    base = Path(str(filename or "")).name
    stem, ext = os.path.splitext(base)
    ext = ext.lower()
    allowed = _ALLOWED_EXTS.get(kind, _DOC_EXTS)
    if ext not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"'{ext or base}' is not a supported {kind} file. Allowed: "
                   f"{', '.join(sorted(allowed))}",
        )
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-.") or "upload"
    return f"{stem[:80]}{ext}"

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


def upload_knowledge(kind: str, upload, authorization: str | None) -> dict:
    """Store an uploaded SRS/Figma/defects file and return the path to put in the
    profile, so a tester can supply a document from their own machine instead of
    typing a server-side path.

    The file is streamed in bounded chunks rather than read whole: an UploadFile
    is only spooled to disk past a threshold, so reading it in one call lets a
    large upload decide how much gateway memory to consume.
    """
    config.check_gateway_auth(authorization)
    kind = (kind or "srs").lower()
    if kind not in _ALLOWED_EXTS:
        raise HTTPException(status_code=400, detail=f"Unknown knowledge kind '{kind}'")
    if upload is None or not getattr(upload, "filename", ""):
        raise HTTPException(status_code=400, detail="No file was uploaded")

    safe_name = _safe_upload_name(upload.filename, kind)
    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = (_UPLOAD_DIR / safe_name).resolve()
    # Defence in depth: even with the sanitising above, never write outside the
    # upload directory.
    if not str(dest).startswith(str(_UPLOAD_DIR.resolve()) + os.sep):
        raise HTTPException(status_code=400, detail="Refusing to write outside the upload directory")

    written = 0
    try:
        with open(dest, "wb") as out:
            while True:
                chunk = upload.file.read(_CHUNK)
                if not chunk:
                    break
                written += len(chunk)
                if written > _MAX_UPLOAD_BYTES:
                    out.close()
                    dest.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File is larger than the {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
                    )
                out.write(chunk)
    except HTTPException:
        raise
    except OSError as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Could not store the upload: {exc}")
    finally:
        try:
            upload.file.close()
        except Exception:
            pass

    if written == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="The uploaded file is empty")

    # Relative, so a saved profile stays portable across machines — the same
    # form every hand-written profile in targets/profiles/ already uses.
    rel = "./" + dest.relative_to(_REPO_ROOT).as_posix()
    return {"stored": True, "path": rel, "filename": safe_name, "bytes": written, "kind": kind}


def ingest_target(name: str, authorization: str | None) -> dict:
    """Load this profile's configured documents into its own project slice.

    DESTRUCTIVE: targets.run's ingest resets the project slice first (tests, SRS
    and Figma are deleted before the documents are re-read), so the caller is
    responsible for confirming intent. Run as a subprocess for the same reason
    run_target is — see the module docstring.
    """
    config.check_gateway_auth(authorization)
    try:
        profile = loader.load(name)
    except loader.ProfileError as exc:
        raise HTTPException(status_code=422, detail=exc.report())

    know = profile.knowledge
    if not (know.srs_path or know.figma_path or know.defects_path):
        raise HTTPException(
            status_code=422,
            detail=f"'{name}' has no documents configured — set an SRS, Figma or defects "
                   f"path first (an empty knowledge section is valid, but there is "
                   f"nothing to ingest).",
        )
    for label, path in (("srs_path", know.srs_path), ("figma_path", know.figma_path),
                        ("defects_path", know.defects_path)):
        if path and not (_REPO_ROOT / path).exists() and not Path(path).exists():
            raise HTTPException(status_code=422, detail=f"knowledge.{label}: file not found: {path}")

    key = f"{name}::ingest"
    existing = _RUNS.get(key)
    if existing and existing.poll() is None:
        raise HTTPException(status_code=409, detail=f"An ingest for '{name}' is already running.")

    _LOG_DIR.mkdir(exist_ok=True)
    log_path = _LOG_DIR / f"targets_ingest_{name}.log"
    cmd = [sys.executable, "-m", "targets.run", name, "--ingest-only"]
    with open(log_path, "w", encoding="utf-8") as log_fh:
        proc = subprocess.Popen(cmd, cwd=str(_REPO_ROOT), stdout=log_fh, stderr=subprocess.STDOUT)
    _RUNS[key] = proc
    return {"started": True, "pid": proc.pid, "log_path": str(log_path), "project": profile.project}


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
    """State of this profile's run and of its ingest, which are separate processes."""
    config.check_gateway_auth(authorization)

    def _state(proc):
        if not proc:
            return {"running": False}
        code = proc.poll()
        return {"running": code is None, "pid": proc.pid, "exit_code": code}

    out = _state(_RUNS.get(name))
    out["ingest"] = _state(_RUNS.get(f"{name}::ingest"))
    return out
