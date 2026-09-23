"""Exercise the real browser loop against an isolated fixture.

    py -m scripts.web_smoke                 # deterministic, no model or services
    py -m scripts.web_smoke --live-model    # also verifies configured WEB_LLM_MODEL

All browser requests are fulfilled locally. No target account or graph is changed.
"""
import argparse
import asyncio
import json
import os
import re
import tempfile
from types import SimpleNamespace

os.environ.setdefault("WEB_TRACE_FILE", os.path.join(tempfile.gettempdir(), "qa_web_smoke.log"))
import settings
from web_player.agent import WebAgent
from web_player.browser import BrowserSession
from web_player.llm import ChatClient
from web_player.oracles import Collector

HTML = """<!doctype html><html lang="en"><title>QA fixture</title><body>
<h1>Projects</h1>
<button onclick="document.getElementById('panel').hidden=false">Open details</button>
<section id="panel" hidden><h2>Project details</h2><p>Project: Alpha</p></section>
<form onsubmit="event.preventDefault(); document.getElementById('result').textContent=
document.getElementById('name').value.trim() ? 'Project created' : 'Name is required'">
<label for="name">Project name</label><input id="name" autocomplete="off">
<button type="submit">Create project</button></form>
<p role="status" id="result"></p></body></html>"""


class ScriptedClient:
    def __init__(self, label, evidence):
        self.label, self.evidence = label, evidence

    def chat(self, messages):
        observation = messages[-1]["content"].split("## Current page\n")[-1]
        if self.evidence in observation:
            return json.dumps({"action": "finish", "success": True, "reason": self.evidence})
        line = next(line for line in observation.splitlines() if self.label in line and "[e" in line)
        ref = re.search(r"\[(e\d+)\]", line).group(1)
        return json.dumps({"action": "click", "ref": ref})


async def main(live=False):
    cfg = SimpleNamespace(**{k: getattr(settings, k) for k in dir(settings) if not k.startswith("_")})
    cfg.WEB_BASE_URL = "https://qa-fixture.test"
    cfg.WEB_STORAGE_STATE = ""
    cfg.WEB_HEADLESS = True
    cfg.WEB_SLOW_MO_MS = 0
    cfg.WEB_FAIL_ON_HTTP_5XX = True
    cfg.WEB_FAIL_ON_PAGE_ERROR = True
    cases = [
        ("navigation", "Click Open details and verify Project details and Project: Alpha are visible. Stop once verified.",
         "Open details", "Project: Alpha"),
        ("empty-name validation", "Leave Project name empty, click Create project, and verify Name is required. "
         "No project should be created. Stop once the validation message is visible.", "Create project", "Name is required"),
    ]
    passed = 0
    async with BrowserSession(cfg) as session:
        async def serve(route):
            await route.fulfill(status=200, content_type="text/html", body=HTML)
        await session.context.route("**/*", serve)
        collector = Collector(session.page, cfg)
        collector.attach()
        for name, goal, label, evidence in cases:
            await session.reset_to_base()
            collector.reset()
            client = ChatClient(cfg) if live else ScriptedClient(label, evidence)
            a = WebAgent(session.page, cfg, client)
            a.collector = collector
            result = await a.run(goal, 12, 90)
            # Independently check the actual DOM, not just the agent's verdict.
            visible = await session.page.get_by_text(evidence, exact=True).is_visible()
            ok = result.success and visible and not collector.findings.verdict_override(cfg)
            passed += ok
            print(f"{'PASS' if ok else 'FAIL'} {name}: {result.steps} steps; {result.reason}")
    print(f"{passed}/{len(cases)} browser smoke tests passed ({'configured model' if live else 'scripted client'})")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-model", action="store_true")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.live_model)))
