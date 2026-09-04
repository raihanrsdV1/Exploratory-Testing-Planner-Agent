"""Find, read and write target profiles.

Kept apart from ``schema`` and ``env`` so a UI can swap the storage (a database,
an API) without touching validation or the settings mapping.
"""

from __future__ import annotations

import json
import os

from .schema import TargetProfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILE_DIR = os.path.join(_ROOT, "targets", "profiles")


class ProfileError(Exception):
    """A profile could not be loaded or is invalid. Carries every problem found."""

    def __init__(self, message: str, problems: list[str] | None = None):
        super().__init__(message)
        self.problems = problems or []

    def report(self) -> str:
        lines = [str(self)]
        lines += [f"  - {p}" for p in self.problems]
        return "\n".join(lines)


def list_profiles() -> list[TargetProfile]:
    """Every valid profile on disk, name-sorted. Invalid ones are skipped."""
    out = []
    if not os.path.isdir(PROFILE_DIR):
        return out
    for filename in sorted(os.listdir(PROFILE_DIR)):
        if not filename.endswith(".json"):
            continue
        try:
            out.append(load(os.path.join(PROFILE_DIR, filename)))
        except ProfileError:
            continue
    return out


def resolve_path(name_or_path: str) -> str:
    """Accept a profile name, a bare filename, or a path to a JSON file."""
    candidates = [
        name_or_path,
        os.path.join(PROFILE_DIR, name_or_path),
        os.path.join(PROFILE_DIR, f"{name_or_path}.json"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    available = ", ".join(p.name for p in list_profiles()) or "(none found)"
    raise ProfileError(
        f"No target profile '{name_or_path}'. Available: {available}",
        [f"looked in: {PROFILE_DIR}"],
    )


def load(name_or_path: str) -> TargetProfile:
    """Load and validate a profile. Raises ProfileError listing every problem."""
    path = resolve_path(name_or_path)
    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ProfileError(f"{path} is not valid JSON", [f"line {exc.lineno}: {exc.msg}"]) from exc
    except OSError as exc:
        raise ProfileError(f"Could not read {path}", [str(exc)]) from exc

    if not isinstance(raw, dict):
        raise ProfileError(f"{path} must contain a JSON object")

    # A profile with no name defaults to its filename, so the two cannot drift.
    raw.setdefault("name", os.path.splitext(os.path.basename(path))[0])

    profile = TargetProfile.from_dict(raw)
    problems = profile.validate(raw)
    if problems:
        raise ProfileError(f"{path} is not a usable target profile", problems)
    return profile


def save(profile: TargetProfile, path: str | None = None) -> str:
    """Write a profile back to disk. Present for the UI; unused by the CLI."""
    path = path or os.path.join(PROFILE_DIR, f"{profile.name}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(profile.to_dict(), fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return path
