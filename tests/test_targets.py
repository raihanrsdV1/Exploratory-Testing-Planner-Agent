#!/usr/bin/env python3
"""Target profiles: the layer that decides WHAT is under test.

The failure this guards against is the one that actually happened: `.env` still
said `PROJECT=contacts-app` while the intent was to test Wikipedia, so the
planner kept retrieving the contacts SRS, screens and history and kept writing
tests about contacts. Every check below is about that class of silent mismatch —
a profile that does not override, a document path that leaks between targets, a
typo that is ignored rather than reported.
"""
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from targets import env as env_mod  # noqa: E402
from targets import loader  # noqa: E402
from targets.schema import TargetProfile  # noqa: E402

_passed = _failed = 0


def check(label, got, want):
    global _passed, _failed
    ok = got == want
    _passed, _failed = _passed + ok, _failed + (not ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + ("" if ok else f"  (got {got!r}, want {want!r})"))


WEB = {"name": "demo", "kind": "web", "project": "demo-web",
       "display_name": "Demo Shop", "web": {"base_url": "https://demo.example.com"}}
ANDROID = {"name": "app", "kind": "android", "project": "demo-app",
           "android": {"package": "com.example.app"}}


def main():
    print("a profile round-trips through JSON without losing anything")
    p = TargetProfile.from_dict(WEB)
    check("to_dict/from_dict is stable", TargetProfile.from_dict(p.to_dict()), p)
    check("the whole profile is JSON-serialisable", bool(json.dumps(p.to_dict())), True)

    print("validation catches what would silently produce a wrong run")
    check("a good web profile validates", p.validate(), [])
    check("a good android profile validates", TargetProfile.from_dict(ANDROID).validate(), [])

    def errs(data):
        prof = TargetProfile.from_dict(data)
        return " | ".join(prof.validate(data))

    check("a missing project is refused",
          "project: required" in errs({**WEB, "project": ""}), True)
    check("a web target with no URL is refused",
          "web.base_url: required" in errs({**WEB, "web": {}}), True)
    check("a URL without a scheme is refused",
          "must start with http" in errs({**WEB, "web": {"base_url": "demo.example.com"}}), True)
    check("an android target with no package is refused",
          "android.package: required" in errs({**ANDROID, "android": {}}), True)
    check("an unknown kind is refused", "kind: must be one of" in errs({**WEB, "kind": "ios"}), True)
    check("a bad viewport is refused",
          "web.viewport" in errs({**WEB, "web": {"base_url": "https://a.b", "viewport": "big"}}), True)
    check("a non-positive budget is refused",
          "run.rounds" in errs({**WEB, "run": {"rounds": 0}}), True)
    check("a missing document is refused, not ignored",
          "file not found" in errs({**WEB, "knowledge": {"srs_path": "./nope/missing.txt"}}), True)
    check("a typo'd key is reported rather than silently dropped",
          "unknown setting" in errs({**WEB, "headles": True}), True)
    check("a typo inside a section is reported too",
          "web.base_urls" in errs({**WEB, "web": {"base_url": "https://a.b", "base_urls": "x"}}), True)
    check("validation returns errors instead of exiting (a UI needs this)",
          isinstance(TargetProfile.from_dict({}).validate(), list), True)

    print("the profile — not .env — decides what is under test")
    env = env_mod.build(TargetProfile.from_dict(WEB))
    check("project is set from the profile", env["PROJECT"], "demo-web")
    check("the planner's app name follows display_name", env["APP_NAME"], "Demo Shop")
    check("the site name matches", env["WEB_SITE_NAME"], "Demo Shop")
    check("the base URL is mapped", env["WEB_BASE_URL"], "https://demo.example.com")
    # The reported bug: another target's SRS still being ingested/retrieved.
    check("an absent SRS is cleared, not inherited from .env", env["SRS_PATH"], "")
    check("an absent Figma is cleared too", env["FIGMA_PATH"], "")

    print("the browser is watchable by default")
    check("headless defaults to False", env["WEB_HEADLESS"], "false")
    check("slow motion is on by default", env["WEB_SLOW_MO_MS"], "300")
    fast = env_mod.build(TargetProfile.from_dict(
        {**WEB, "web": {"base_url": "https://a.b", "headless": True, "slow_mo_ms": 0}}))
    check("both are overridable per profile",
          (fast["WEB_HEADLESS"], fast["WEB_SLOW_MO_MS"]), ("true", "0"))

    print("each kind maps to its own player's settings, and only those")
    android_env = env_mod.build(TargetProfile.from_dict(ANDROID))
    check("android sets the package", android_env["TARGET_APP_PACKAGE"], "com.example.app")
    check("android sets EXECUTOR_ROUNDS", "EXECUTOR_ROUNDS" in android_env, True)
    check("android writes no WEB_* settings",
          [k for k in android_env if k.startswith("WEB_")], [])
    check("web writes no TARGET_APP_* settings",
          [k for k in env if k.startswith("TARGET_APP_")], [])

    print("an empty guardrail list must not read as 'no guardrails'")
    # settings falls back to its default on an empty value, so writing "" would
    # quietly restore the defaults while looking like it disabled them.
    check("an empty blocked list is omitted, not blanked",
          "WEB_BLOCKED_TEXTS" in env, False)
    filled = env_mod.build(TargetProfile.from_dict(
        {**WEB, "web": {"base_url": "https://a.b", "blocked_texts": ["Delete", " ", "Log out"]}}))
    check("a set list is joined and stripped", filled["WEB_BLOCKED_TEXTS"], "Delete,Log out")

    print("secrets never reach a log, a UI, or --dry-run")
    secret = env_mod.build(TargetProfile.from_dict(
        {**WEB, "web": {"base_url": "https://a.b", "login": {"user": "qa", "password": "hunter2"}}}))
    check("the password IS applied to the environment", secret["WEB_LOGIN_PASSWORD"], "hunter2")
    shown = env_mod.redacted(secret)
    check("but is masked when displayed", shown["WEB_LOGIN_PASSWORD"], "*" * 8)
    check("the username is not masked (it is not a secret)", shown["WEB_LOGIN_USER"], "qa")
    check("no secret survives redaction",
          [v for k, v in shown.items() if "PASSWORD" in k and v == "hunter2"], [])

    print("the shipped profiles load and are valid")
    names = {p.name for p in loader.list_profiles()}
    check("wikipedia profile is present and valid", "wikipedia" in names, True)
    check("contacts-app profile is present and valid", "contacts-app" in names, True)
    wiki = loader.load("wikipedia")
    check("wikipedia is a web target", wiki.kind, "web")
    check("wikipedia has its own project (not contacts-app)", wiki.project, "wikipedia")
    check("wikipedia blocks destructive wiki controls",
          all(t in wiki.web.blocked_texts for t in ("edit", "undo", "rollback")), True)
    check("wikipedia is watchable by default", wiki.web.headless, False)

    print("every shipped profile is valid — none is silently skipped")
    # list_profiles() drops what it cannot load, so a broken shipped profile would
    # quietly disappear from `--list` rather than announce itself.
    shipped = [f for f in sorted(os.listdir(loader.PROFILE_DIR)) if f.endswith(".json")]
    check("there are shipped profiles to check", len(shipped) >= 3, True)
    for filename in shipped:
        try:
            prof = loader.load(os.path.join(loader.PROFILE_DIR, filename))
            check(f"{filename} loads and validates", prof.name, os.path.splitext(filename)[0])
        except loader.ProfileError as exc:
            check(f"{filename} loads and validates", exc.report(), "valid")
    check("no two profiles share a project (that would merge their graphs)",
          len({p.project for p in loader.list_profiles()}), len(loader.list_profiles()))

    print("loader failures are actionable, never a stack trace")
    with tempfile.TemporaryDirectory() as tmp:
        bad = os.path.join(tmp, "broken.json")
        open(bad, "w", encoding="utf-8").write("{not json")
        try:
            loader.load(bad)
            check("malformed JSON raises ProfileError", "no error", "ProfileError")
        except loader.ProfileError as exc:
            check("malformed JSON raises ProfileError", "not valid JSON" in str(exc), True)
            check("the report names the line", "line" in exc.report(), True)

        empty = os.path.join(tmp, "empty.json")
        open(empty, "w", encoding="utf-8").write("{}")
        try:
            loader.load(empty)
            check("an incomplete profile is rejected", "no error", "ProfileError")
        except loader.ProfileError as exc:
            check("an incomplete profile is rejected", len(exc.problems) >= 2, True)

        # A profile with no "name" adopts its filename, so the two cannot drift.
        named = os.path.join(tmp, "my-site.json")
        json.dump({k: v for k, v in WEB.items() if k != "name"}, open(named, "w", encoding="utf-8"))
        check("a nameless profile takes its filename", loader.load(named).name, "my-site")

    try:
        loader.load("does-not-exist")
        check("an unknown profile name is refused", "no error", "ProfileError")
    except loader.ProfileError as exc:
        check("an unknown profile name lists what IS available",
              "Available:" in str(exc), True)

    print("the CLI enforces its own import ordering")
    # settings reads the environment at import time, so a stray top-level import
    # in targets/run.py would make every profile a silent no-op.
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "targets", "run.py"), encoding="utf-8").read()
    header = src.split("def cmd_list")[0]
    check("run.py does not import settings at module level",
          "\nimport settings" in header or "\nfrom settings" in header, False)
    check("run.py does not import a player at module level",
          "\nfrom web_player" in header or "\nfrom clients" in header, False)

    print("end to end: the profile beats .env in a real interpreter")
    # The actual reported symptom, reproduced as a subprocess so the import
    # ordering is real rather than simulated.
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = (
        "from targets import loader, env; env.apply(loader.load('wikipedia'));"
        "import settings as s; print(s.PROJECT, s.APP_NAME, repr(s.SRS_PATH))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=root, env={**os.environ, "PROJECT": "contacts-app",
                                        "SRS_PATH": "./data/inputs/Sample-Contacts-App-SRS.txt"})
    check("the profile wins over an inherited PROJECT",
          out.stdout.split()[0] if out.stdout.split() else out.stderr[-120:], "wikipedia")
    # The profile supplies its own spec; the inherited contacts SRS must be gone.
    # Replacement, not emptiness — a profile with no SRS clears it instead.
    check("another target's SRS does not leak in",
          "Sample-Contacts-App-SRS" not in out.stdout, True)
    check("the profile's own SRS is the one used", "wikipedia-spec" in out.stdout, True)

    nodoc = subprocess.run(
        [sys.executable, "-c",
         "from targets.schema import TargetProfile; from targets import env;"
         "print(repr(env.build(TargetProfile.from_dict("
         "{'name':'x','kind':'web','project':'x','web':{'base_url':'https://a.b'}}))['SRS_PATH']))"],
        capture_output=True, text=True, cwd=root)
    check("a profile with no SRS clears it rather than inheriting",
          nodoc.stdout.strip(), "''")

    print(f"\n{_passed}/{_passed + _failed} checks passed")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
