"""Target profile — what is under test, as data.

One JSON file describes one thing to test: an Android app or a website, which
knowledge to load, how long to run, and what the agent may not touch. The runner
reads it and dispatches to the right player.

Why this exists as a layer above ``settings.py`` rather than inside it: `.env` is
one global configuration for one machine, so switching from the contacts app to
Wikipedia means editing a dozen variables and remembering which ones. Nearly
every "the planner is testing the wrong app" symptom traces to a half-switched
`.env` — most often ``PROJECT``, which scopes the whole knowledge graph, so a
stale value keeps serving another app's SRS, screens and test history no matter
what the other variables say.

Designed to be driven by a UI later, so:

* every field is a plain JSON type — the whole profile round-trips through
  ``to_dict``/``from_dict`` with no custom encoding
* the field lists are introspectable (``dataclasses.fields``), so a form can be
  generated from them rather than hand-written twice
* ``validate()`` RETURNS its errors instead of exiting, so a UI can show them
  next to the offending input, and unknown keys are reported rather than
  silently ignored — a typo in a hand-written profile is otherwise invisible
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field, fields

KINDS = ("web", "android")


@dataclass
class Login:
    """Credentials handed to the PLAYER only, never to the planner."""
    url: str = ""
    user: str = ""
    password: str = ""
    hint: str = ""
    role: str = ""          # planner sees only this ("admin", "guest")


@dataclass
class WebTarget:
    base_url: str = ""
    browser: str = "chromium"          # chromium | firefox | webkit
    headless: bool = False       # watch it by default; true for CI / long batches
    slow_mo_ms: int = 300        # pause before each action so a headed run is followable
    viewport: str = "1280x800"
    same_origin_only: bool = True
    blocked_texts: list[str] = field(default_factory=list)
    blocked_url_patterns: list[str] = field(default_factory=list)
    storage_state: str = ""            # Playwright auth state file
    fail_on_page_error: bool = False
    fail_on_http_5xx: bool = False
    console_ignore: list[str] = field(default_factory=list)
    login: Login = field(default_factory=Login)


@dataclass
class AndroidTarget:
    package: str = ""
    activity: str = ""
    labels: list[str] = field(default_factory=list)
    target_app_only: bool = False
    device_reset: str = "pm_clear"     # pm_clear | force_stop | none
    login: Login = field(default_factory=Login)


@dataclass
class Knowledge:
    """Optional documents. All absent is a valid, supported ('zero-doc') setup."""
    srs_path: str = ""
    figma_path: str = ""
    defects_path: str = ""


@dataclass
class RunBudget:
    rounds: int = 5           # test cases per batch
    max_steps: int = 30       # player actions per test case
    timeout: int = 420        # wall-clock seconds per test case
    clean_slate: bool = True  # wipe this project's test history before the batch
    self_heal: bool = True


@dataclass
class Model:
    """Executor model override. Blank means "use the .env default"."""
    provider: str = ""
    model: str = ""


@dataclass
class TargetProfile:
    name: str = ""
    kind: str = "web"
    project: str = ""          # scopes the Neo4j knowledge graph
    display_name: str = ""     # what the planner calls it in prompts
    description: str = ""
    web: WebTarget = field(default_factory=WebTarget)
    android: AndroidTarget = field(default_factory=AndroidTarget)
    knowledge: Knowledge = field(default_factory=Knowledge)
    run: RunBudget = field(default_factory=RunBudget)
    model: Model = field(default_factory=Model)

    # ── serialisation ────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TargetProfile":
        """Build a profile, ignoring unknown keys (``validate`` reports those)."""
        data = dict(data or {})
        return cls(
            name=str(data.get("name", "")),
            kind=str(data.get("kind", "web")).lower(),
            project=str(data.get("project", "")),
            display_name=str(data.get("display_name", "")),
            description=str(data.get("description", "")),
            web=_build(WebTarget, data.get("web")),
            android=_build(AndroidTarget, data.get("android")),
            knowledge=_build(Knowledge, data.get("knowledge")),
            run=_build(RunBudget, data.get("run")),
            model=_build(Model, data.get("model")),
        )

    # ── validation ───────────────────────────────────────────────────────────

    def validate(self, raw: dict | None = None) -> list[str]:
        """Return every problem found. Empty list means the profile is usable."""
        errors: list[str] = []

        if not self.name.strip():
            errors.append("name: required (used to select the profile on the CLI)")
        if self.kind not in KINDS:
            errors.append(f"kind: must be one of {', '.join(KINDS)} (got {self.kind!r})")
        if not self.project.strip():
            errors.append(
                "project: required — it scopes the Neo4j knowledge graph. Reusing "
                "another target's project makes the planner generate tests from "
                "that target's documents and history."
            )

        if self.kind == "web":
            errors += self._validate_web()
        elif self.kind == "android":
            if not self.android.package.strip():
                errors.append("android.package: required for an android target")
            if self.android.device_reset not in ("pm_clear", "force_stop", "none"):
                errors.append("android.device_reset: must be pm_clear, force_stop or none")

        for label, path in (("srs_path", self.knowledge.srs_path),
                            ("figma_path", self.knowledge.figma_path),
                            ("defects_path", self.knowledge.defects_path)):
            if path and not os.path.exists(path):
                errors.append(f"knowledge.{label}: file not found: {path}")

        for label, value in (("rounds", self.run.rounds),
                             ("max_steps", self.run.max_steps),
                             ("timeout", self.run.timeout)):
            if not isinstance(value, int) or value < 1:
                errors.append(f"run.{label}: must be a positive integer (got {value!r})")

        if raw is not None:
            errors += _unknown_keys(raw)
        return errors

    def _validate_web(self) -> list[str]:
        errors = []
        url = self.web.base_url.strip()
        if not url:
            errors.append("web.base_url: required for a web target")
        elif not re.match(r"^https?://", url):
            errors.append(f"web.base_url: must start with http:// or https:// (got {url!r})")
        if self.web.browser not in ("chromium", "firefox", "webkit"):
            errors.append(f"web.browser: must be chromium, firefox or webkit (got {self.web.browser!r})")
        if not re.match(r"^\d+x\d+$", str(self.web.viewport)):
            errors.append(f"web.viewport: must look like 1280x800 (got {self.web.viewport!r})")
        return errors

    # ── convenience ──────────────────────────────────────────────────────────

    def label(self) -> str:
        """Human name for prompts and logs."""
        return self.display_name.strip() or self.name

    def summary(self) -> str:
        """One line, for `--list` and for a UI's profile picker."""
        where = self.web.base_url if self.kind == "web" else self.android.package
        return f"{self.name:<18} {self.kind:<8} project={self.project:<16} {where}"


def _build(cls, data):
    """Instantiate a section dataclass from a dict, keeping only known fields."""
    if not isinstance(data, dict):
        return cls()
    known = {f.name for f in fields(cls)}
    kwargs = {k: v for k, v in data.items() if k in known}
    if "login" in known and isinstance(kwargs.get("login"), dict):
        kwargs["login"] = _build(Login, kwargs["login"])
    return cls(**kwargs)


def _unknown_keys(raw: dict, prefix: str = "") -> list[str]:
    """Report keys the schema does not define — i.e. typos that would be ignored."""
    section_types = {
        "": TargetProfile, "web": WebTarget, "android": AndroidTarget,
        "knowledge": Knowledge, "run": RunBudget, "model": Model,
        "web.login": Login, "android.login": Login,
    }
    cls = section_types.get(prefix)
    if cls is None or not isinstance(raw, dict):
        return []
    known = {f.name for f in fields(cls)}
    out = []
    for key, value in raw.items():
        path = f"{prefix}.{key}" if prefix else key
        if key not in known:
            out.append(f"{path}: unknown setting — check the spelling; it is being ignored")
        elif isinstance(value, dict) and path in section_types:
            out += _unknown_keys(value, path)
    return out
