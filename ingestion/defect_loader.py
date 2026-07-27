"""
Format-agnostic defect-history loader (ETA-REQ-301.1).

Turns any structured defect export into a list of canonical defect dicts the RAG
API can graph. Accepts, transparently:

  * a JSON array of objects
  * a JSON object with a top-level ``defects`` / ``issues`` / ``items`` list
  * a single JSON object (one defect)
  * CSV (header row → field names)

It is deliberately app-agnostic: no field is required, common field aliases from
different issue trackers (Jira/GitHub/Bugzilla/CSV exports) are mapped onto the
canonical schema, and anything unmapped is preserved under ``extra``.

Canonical defect schema:
    id, title, description, severity, status, area, root_cause_category,
    frequency, first_seen, last_seen, resolution, affected_screen,
    requirement_ids (list), extra (dict)
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

# Aliases seen across common trackers → canonical field name. Matching is
# case-insensitive and ignores spaces / underscores / hyphens.
_ALIASES = {
    "id": ["id", "defectid", "bugid", "issueid", "key", "number", "ticket"],
    "title": ["title", "summary", "name", "headline", "shortdescription"],
    "description": ["description", "desc", "details", "body", "detail", "notes"],
    "severity": ["severity", "priority", "impact", "criticality"],
    "status": ["status", "state", "resolutionstatus"],
    "area": ["area", "component", "feature", "module", "featurearea", "category", "affectedarea"],
    "root_cause_category": ["rootcause", "rootcausecategory", "cause", "category2", "defecttype", "type"],
    "frequency": ["frequency", "count", "occurrences", "hits", "seen"],
    "first_seen": ["firstseen", "created", "createdat", "opened", "reporteddate", "datecreated"],
    "last_seen": ["lastseen", "updated", "updatedat", "modified", "lastoccurred", "dateupdated"],
    "resolution": ["resolution", "fix", "resolutionnotes", "howfixed"],
    "affected_screen": ["affectedscreen", "screen", "page", "view", "location"],
    "requirement_ids": ["requirementids", "requirements", "requirement", "reqids", "coveredrequirements", "relatedrequirements"],
}

_LIST_KEYS = ("defects", "issues", "items", "bugs", "records", "data")


def _norm_key(k: str) -> str:
    return "".join(ch for ch in str(k).lower() if ch.isalnum())


def _canonical_field(raw_key: str) -> str | None:
    nk = _norm_key(raw_key)
    for canonical, aliases in _ALIASES.items():
        if nk in aliases:
            return canonical
    return None


def _coerce_requirement_ids(value) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    # split "FR-5; FR-7" / "FR-5, FR-7" / "FR-5 FR-7"
    parts = [p.strip() for p in str(value).replace(";", ",").replace("|", ",").split(",")]
    out: list[str] = []
    for p in parts:
        for tok in p.split():
            tok = tok.strip()
            if tok:
                out.append(tok)
    return out


def normalize_defect(raw: dict, index: int = 0) -> dict:
    """Map one raw record onto the canonical schema (app-agnostic)."""
    canon: dict = {
        "id": "",
        "title": "",
        "description": "",
        "severity": "",
        "status": "",
        "area": "",
        "root_cause_category": "",
        "frequency": 1,
        "first_seen": "",
        "last_seen": "",
        "resolution": "",
        "affected_screen": "",
        "requirement_ids": [],
        "extra": {},
    }
    for raw_key, value in (raw or {}).items():
        field = _canonical_field(raw_key)
        if field is None:
            canon["extra"][str(raw_key)] = value
            continue
        if field == "requirement_ids":
            canon[field] = _coerce_requirement_ids(value)
        elif field == "frequency":
            try:
                canon[field] = int(float(value))
            except (TypeError, ValueError):
                canon[field] = 1
        else:
            canon[field] = str(value).strip() if value is not None else ""

    if not canon["id"]:
        canon["id"] = f"D{index + 1}"
    if not canon["title"]:
        canon["title"] = (canon["description"][:80] or canon["id"])
    if not canon["area"]:
        canon["area"] = "general"
    if canon["frequency"] < 1:
        canon["frequency"] = 1
    return canon


def _load_raw(source_path: str | None, raw_text: str | None) -> list[dict]:
    text = raw_text
    fmt = "inline"
    if not text:
        src = Path(source_path or "")
        if not src.exists():
            raise FileNotFoundError(f"Defect file not found: {source_path}")
        text = src.read_text(encoding="utf-8", errors="ignore")
        fmt = src.suffix.lower().lstrip(".") or "txt"

    text = text.strip()
    # Try JSON first (covers .json and JSON pasted into .txt).
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
        if isinstance(data, dict):
            for key in _LIST_KEYS:
                if isinstance(data.get(key), list):
                    return [d for d in data[key] if isinstance(d, dict)]
            return [data]  # single defect object
    except json.JSONDecodeError:
        pass

    # Fall back to CSV.
    reader = csv.DictReader(io.StringIO(text))
    rows = [dict(r) for r in reader]
    if rows:
        return rows
    raise ValueError("Could not parse defect data as JSON or CSV")


def load_defects(source_path: str | None = None, raw_text: str | None = None) -> list[dict]:
    """Load + normalize defects from a file path or inline text. Returns canonical dicts."""
    raw_records = _load_raw(source_path, raw_text)
    return [normalize_defect(r, i) for i, r in enumerate(raw_records)]
