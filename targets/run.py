#!/usr/bin/env python3
"""One entry point for both players, selected by a target profile.

    py -m targets.run --list                 # what can I test?
    py -m targets.run wikipedia              # run the profile
    py -m targets.run wikipedia --dry-run    # resolve config, run nothing
    py -m targets.run wikipedia --ingest     # load its documents first
    py -m targets.run contacts-app           # same command, Android target

The profile decides which player runs, so nothing here (or in a future UI) needs
to know that a website means Playwright and an app means Droidrun.

**Import order is load-bearing.** ``settings`` reads the environment when it is
first imported, so this module must not import it — directly or through a player
— until ``env.apply()`` has run. Every such import is therefore inside a
function, and the ordering is asserted at runtime rather than left to
convention.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from targets import env as env_mod  # noqa: E402
from targets import loader  # noqa: E402
from targets.schema import TargetProfile  # noqa: E402


def cmd_list() -> int:
    profiles = loader.list_profiles()
    if not profiles:
        print(f"No target profiles in {loader.PROFILE_DIR}")
        return 1
    print(f"Target profiles in {loader.PROFILE_DIR}:\n")
    for profile in profiles:
        print("  " + profile.summary())
        if profile.description:
            print(f"  {'':<18} {profile.description}")
    print("\nRun one with:  py -m targets.run <name>")
    return 0


def show(profile: TargetProfile, env: dict[str, str]) -> None:
    print("=" * 72)
    print(f"TARGET: {profile.label()}  ({profile.kind})")
    print("=" * 72)
    print(f"  project (scopes the knowledge graph): {profile.project}")
    if profile.kind == "web":
        print(f"  site:      {profile.web.base_url}")
        print(f"  browser:   {profile.web.browser} (headless={profile.web.headless}, "
              f"{profile.web.viewport})")
    else:
        print(f"  package:   {profile.android.package}")
    know = profile.knowledge
    docs = [f"{k}={v}" for k, v in
            (("srs", know.srs_path), ("figma", know.figma_path), ("defects", know.defects_path))
            if v]
    print(f"  knowledge: {', '.join(docs) if docs else 'none (zero-doc exploration)'}")
    print(f"  budget:    {profile.run.rounds} test cases x {profile.run.max_steps} steps, "
          f"{profile.run.timeout}s each")
    print(f"  applied {len(env)} settings from the profile")


def cmd_ingest(profile: TargetProfile) -> int:
    """Load this profile's documents into its own project slice."""
    import requests  # noqa: E402  (after env.apply)
    import settings as cfg  # noqa: E402

    rag, gw = cfg.RAG_URL.rstrip("/"), cfg.GATEWAY_URL.rstrip("/")
    print(f"\n[ingest] project '{profile.project}'")

    def post(url, payload, label):
        try:
            resp = requests.post(url, json=payload, timeout=5400)
            resp.raise_for_status()
            print(f"  ✅ {label}: {str(resp.json())[:160]}")
            return True
        except Exception as exc:
            print(f"  ❌ {label} failed: {exc}")
            return False

    post(f"{rag}/project/reset",
         {"project": profile.project, "delete_tests": True,
          "delete_srs": True, "delete_figma": True},
         "reset project slice")

    know = profile.knowledge
    if know.srs_path:
        post(f"{gw}/srs/ingest", {"project": profile.project, "source_path": know.srs_path}, "SRS")
    else:
        print("  – SRS: none configured (fine — the planner explores without it)")
    if know.figma_path:
        post(f"{gw}/figma/ingest", {"project": profile.project, "source_path": know.figma_path}, "Figma")
    if know.defects_path:
        post(f"{rag}/ingest/defects", {"project": profile.project, "source_path": know.defects_path}, "defects")
    return 0


def run_profile(profile: TargetProfile, rounds: int | None = None) -> int:
    """Dispatch to the player this profile targets.

    The single function a UI needs: hand it a validated profile and it runs.
    """
    import asyncio

    if profile.kind == "web":
        from web_player import runner as player
        asyncio.run(player.main(rounds=rounds or profile.run.rounds))
    else:
        from clients import executor_runner as player
        asyncio.run(player.main(rounds=rounds or profile.run.rounds))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the exploratory tester against a configured target.")
    parser.add_argument("profile", nargs="?", help="profile name, or a path to a JSON file")
    parser.add_argument("--list", action="store_true", help="list available profiles and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve and print the configuration without running")
    parser.add_argument("--ingest", action="store_true",
                        help="ingest the profile's documents before running")
    parser.add_argument("--ingest-only", action="store_true", help="ingest and stop")
    parser.add_argument("--rounds", type=int, help="override the profile's round count")
    args = parser.parse_args(argv)

    if args.list or not args.profile:
        return cmd_list()

    try:
        profile = loader.load(args.profile)
    except loader.ProfileError as exc:
        print(exc.report(), file=sys.stderr)
        return 2

    # Guard the ordering this module depends on rather than trusting it: an
    # import added above would silently make the profile a no-op, and the symptom
    # ("it is still testing the old app") looks like anything but an import.
    if "settings" in sys.modules:
        print("BUG: settings was imported before the profile was applied; "
              "the profile would be ignored.", file=sys.stderr)
        return 3

    env = env_mod.apply(profile)
    show(profile, env)

    if args.dry_run:
        print("\n--dry-run: resolved settings (secrets masked)\n")
        for key, value in sorted(env_mod.redacted(env).items()):
            print(f"  {key:<26} {value!r}")
        print("\nNothing was run.")
        return 0

    if args.ingest or args.ingest_only:
        cmd_ingest(profile)
        if args.ingest_only:
            return 0

    return run_profile(profile, rounds=args.rounds)


if __name__ == "__main__":
    raise SystemExit(main())
