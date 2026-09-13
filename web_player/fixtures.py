"""Files the agent is allowed to upload, addressed by key rather than by path.

Two tests in a row died reporting that DataGhurhi's `/analysis` upload "cannot be
tested in this environment". That was never true of Playwright — `set_input_files`
writes straight to the `<input type=file>` and the native picker never opens. The
harness simply had no upload action, no way to see a hidden file input, and a goal
prompt that told the agent uploads were impossible. So the planner kept selecting
the untested upload requirements, and every test that reached one was spent.

The agent names a fixture (`"csv"`), never a filesystem path. That keeps an
arbitrary-file-read off the table: a model that can pick its own path can upload
`.env` to a live website.

Text fixtures are written on first use so the repository carries no generated
files; binary ones point at what is already in `data/fixtures/media/`.
"""

from __future__ import annotations

import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORE_DIR = os.path.join(_ROOT, "data", "fixtures", "web")
MEDIA_DIR = os.path.join(_ROOT, "data", "fixtures", "media")

# A small, genuinely analysable dataset: numeric columns for summary statistics,
# a categorical column to group by, and enough rows to chart.
_CSV = """\
respondent_id,age,region,satisfaction,monthly_spend
1,24,Dhaka,4,1200
2,31,Chattogram,5,2400
3,45,Dhaka,2,900
4,38,Khulna,3,1750
5,29,Dhaka,5,3100
6,52,Rajshahi,1,650
7,41,Chattogram,4,2200
8,27,Khulna,3,1400
9,36,Dhaka,5,2800
10,48,Rajshahi,2,1050
"""

_CSV_MALFORMED = """\
respondent_id,age,region,satisfaction
1,24,Dhaka,4
2,31,Chattogram
3,45,Dhaka,2,extra,columns,here
4,,Khulna,3
"""

_CATALOGUE: dict[str, dict] = {
    "csv": {
        "filename": "survey-responses.csv",
        "content": _CSV,
        "describe": "a valid 10-row CSV with numeric and categorical columns",
    },
    "csv_empty": {
        "filename": "empty.csv",
        "content": "",
        "describe": "a zero-byte CSV (edge case)",
    },
    "csv_headers_only": {
        "filename": "headers-only.csv",
        "content": "respondent_id,age,region,satisfaction\n",
        "describe": "a CSV with headers but no data rows",
    },
    "csv_malformed": {
        "filename": "malformed.csv",
        "content": _CSV_MALFORMED,
        "describe": "a CSV with ragged rows and a missing value",
    },
    "xlsx": {
        "filename": "survey-responses.xlsx",
        "build": "xlsx",
        "describe": "the same 10-row dataset as a real .xlsx workbook",
    },
    "txt": {
        "filename": "notes.txt",
        "content": "This is a plain text file, not a dataset.\n",
        "describe": "a plain .txt file — use to test rejection of unsupported types",
    },
    "image": {
        "media": "feed.jpg",
        "describe": "a JPEG image — use to test rejection of a non-data file",
    },
}


def _xlsx_rows() -> list[list]:
    """The CSV fixture's data, as rows, for the spreadsheet build."""
    rows = [line.split(",") for line in _CSV.strip().split("\n")]
    out: list[list] = [rows[0]]
    for row in rows[1:]:
        out.append([c if not c.lstrip("-").isdigit() else int(c) for c in row])
    return out


def _write_xlsx(path: str, rows: list[list]) -> None:
    """Write the dataset as a real .xlsx workbook.

    DataGhurhi's analysis upload declares accept=".xls,.xlsx" — a CSV is refused
    at the picker, so testing that feature at all needs a genuine spreadsheet.
    """
    from openpyxl import Workbook

    wb = Workbook()
    sheet = wb.active
    sheet.title = "Responses"
    for row in rows:
        sheet.append(row)
    wb.save(path)


def keys() -> list[str]:
    return list(_CATALOGUE)


def resolve(key: str) -> str | None:
    """Absolute path for a fixture key, materialising it on first use.

    Returns None for an unknown key — the caller turns that into an actionable
    error listing what is available.
    """
    entry = _CATALOGUE.get((key or "").strip().lower())
    if entry is None:
        return None

    if "media" in entry:
        path = os.path.join(MEDIA_DIR, entry["media"])
        return path if os.path.exists(path) else None

    os.makedirs(STORE_DIR, exist_ok=True)
    path = os.path.join(STORE_DIR, entry["filename"])
    if not os.path.exists(path):
        if entry.get("build") == "xlsx":
            _write_xlsx(path, _xlsx_rows())
        else:
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(entry["content"])
    return path


def describe(key: str) -> str:
    entry = _CATALOGUE.get((key or "").strip().lower()) or {}
    return entry.get("describe", "")


def prompt_block() -> str:
    """The catalogue as the agent sees it."""
    lines = ["UPLOADABLE FILES (name one with the 'file' field — paths are not accepted):"]
    for key, entry in _CATALOGUE.items():
        lines.append(f"  {key:<18} {entry['describe']}")
    return "\n".join(lines)
