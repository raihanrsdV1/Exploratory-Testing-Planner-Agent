"""Target profiles — what to test, as data.

One JSON file per thing under test. ``py -m targets.run <name>`` loads it,
applies it over `.env`, and dispatches to the right player (Playwright for a
website, Droidrun for an Android app).

    schema.py   the profile shape + validation (returns errors, never exits)
    loader.py   find / read / write profiles      (swap this for a DB in a UI)
    env.py      profile -> settings.py variables  (the only name-aware module)
    run.py      CLI + ``run_profile()``, the single call a UI would make
"""
