"""Regression checks for the failed DataGhurhi campaign. No live site or LLM."""
import asyncio
import contextlib
import io
import json
import os
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch, AsyncMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("WEB_TRACE_FILE", os.path.join(tempfile.gettempdir(), "web_regressions.log"))
from web_player import agent, gateway, llm, planning, runner, failures
from web_player.oracles import Collector, Findings, _short_url
from web_player.actions import _goto_settled


def cfg():
    return NS(WEB_BASE_URL="https://test.example", WEB_MAX_STEPS=30,
              WEB_BLOCKED_TEXTS=["change password", "delete account"],
              WEB_BLOCKED_URL_PATTERNS=["/logout"], WEB_SAME_ORIGIN_ONLY=True,
              WEB_SNAPSHOT_MAX_ELEMENTS=60, WEB_HEADLESS=True,
              WEB_FAIL_ON_PAGE_ERROR=False, WEB_FAIL_ON_HTTP_5XX=True,
              WEB_STALL_STEPS=6)


class AdmissionTests(unittest.TestCase):
    def test_feasible_navigation_allowed(self):
        self.assertEqual(planning.rejection_errors({"title": "Project navigation", "objective": "Open Projects",
                          "expected_result": "The project list appears"}, planning.contract(cfg())), [])

    def test_unexecutable_tests_rejected(self):
        for objective in ("Click Change Password", "Verify drag-and-drop data file import",
                          "Upload a CSV", "GET /api/project/create-project"):
            with self.subTest(objective=objective):
                self.assertTrue(planning.rejection_errors({"title": objective, "objective": objective,
                                "expected_result": "works"}, planning.contract(cfg())))

    def test_empty_account_assumption_rejected(self):
        self.assertTrue(planning.rejection_errors({"title": "Empty state", "objective": "Open Projects",
                        "expected_result": "Empty", "preconditions": ["User has no existing projects"]},
                        planning.contract(cfg())))

    def test_duplicate_rejected_even_if_graph_has_not_logged_it(self):
        tc = {"title": "Verify project search clears results", "objective": "Search", "expected_result": "Results"}
        self.assertTrue(planning.rejection_errors(tc, planning.contract(cfg()), [tc["title"]]))

    def test_gateway_repairs_rejected_proposals_before_execution(self):
        old = {"title": "Verify project search clears results", "objective": "Search", "expected_result": "Results"}
        good = {"title": "Account navigation displays profile", "objective": "Open profile", "expected_result": "Name appears"}
        responses = [NS(status_code=200, json=lambda: {"next_testcase": old}, raise_for_status=lambda: None),
                     NS(status_code=200, json=lambda: {"next_testcase": good}, raise_for_status=lambda: None)]
        with patch.object(gateway.requests, "post", side_effect=responses) as post:
            result = gateway.next_testcase(excluded_titles=[old["title"]])
        self.assertEqual(result["next_testcase"], good)
        self.assertEqual(post.call_count, 2)
        self.assertIn("executor_constraints", post.call_args.kwargs["json"])


class EvidenceTests(unittest.TestCase):
    def test_cancelled_requests_do_not_become_http_500(self):
        col = Collector(None, cfg())
        col._on_request_failed(NS(url="https://test.example/stream", method="GET", failure="net::ERR_ABORTED"))
        self.assertTrue(col.findings.is_empty())
        col._on_request_failed(NS(url="https://test.example/stream", method="GET", failure="net::ERR_CONNECTION_RESET"))
        self.assertEqual(len(col.findings.request_failures), 1)
        self.assertEqual(col.findings.http_failures, [])
        self.assertIsNone(col.findings.verdict_override(cfg()))

    def test_real_server_error_still_fails(self):
        col = Collector(None, cfg())
        col._on_response(NS(status=500, url="https://test.example/api/items", request=NS(method="GET")))
        self.assertEqual(col.findings.verdict_override(cfg())[0], "HTTP_ERROR")

    def test_third_party_500_is_not_site_failure(self):
        col = Collector(None, cfg())
        col._on_response(NS(status=500, url="https://translation.example/api", request=NS(method="GET")))
        self.assertIsNone(col.findings.verdict_override(cfg()))

    def test_query_tokens_redacted(self):
        self.assertNotIn("secret-value", _short_url("https://test.example/?token=secret-value"))

    def test_signature_distinguishes_navigation_destinations_and_keys(self):
        snap = {"url": "https://test.example", "elements": []}
        self.assertNotEqual(agent._signature(snap, {"action": "goto", "url": "/one"}),
                            agent._signature(snap, {"action": "goto", "url": "/two"}))
        self.assertNotEqual(agent._signature(snap, {"action": "press", "key": "Enter"}),
                            agent._signature(snap, {"action": "press", "key": "Tab"}))

    def test_signature_survives_ref_renumbering(self):
        a = {"url": "https://test.example", "elements": [{"ref": "e1", "name": "Edit", "role": "button"}]}
        b = {"url": "https://test.example", "elements": [{"ref": "e99", "name": "Edit", "role": "button"}]}
        self.assertEqual(agent._signature(a, {"action": "click", "ref": "e1"}),
                         agent._signature(b, {"action": "click", "ref": "e99"}))

    def test_string_false_cannot_pass(self):
        self.assertEqual(llm.parse_action('{"action":"finish","success":"false","reason":"broken"}')["action"], "_error")

    def test_unrelated_schema_fields_cannot_evade_livelock(self):
        snap = {"url": "https://test.example", "elements": []}
        self.assertEqual(agent._signature(snap, {"action": "scroll", "direction": "down", "text": "first"}),
                         agent._signature(snap, {"action": "scroll", "direction": "down", "text": "second"}))

    def test_summary_separates_agent_and_environment(self):
        records = [{"round": i, "test_case_id": str(i), "title": "t", "area": "a", "notes": "n",
                    "duration_seconds": 1, "verdict": "failed", "error_type": kind}
                   for i, kind in enumerate(["BLOCKED_BY_GUARDRAIL", "NAVIGATION_LIVELOCK", "HTTP_ERROR"], 1)]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            runner._summarize(records)
        self.assertIn("BLOCKED", output.getvalue())
        self.assertIn("AGENT ERROR", output.getvalue())
        self.assertIn("Completed assertions: 1/3", output.getvalue())


class PlannerGateTests(unittest.TestCase):
    def test_pipeline_does_not_accept_duplicate_from_retry(self):
        from planner import langgraph_agent as p
        tc = {"title": "Verify search clears results", "objective": "Search", "expected_result": "Results"}
        state = {
            "project": "regression-fixture", "next_testcase": tc,
            "done_titles": [tc["title"]], "failed_titles": [], "debug_trace": False,
            "selected_screens": [], "figma_screens": [], "srs_context_blocks": [],
            "figma_overview": [], "flow_context_blocks": [], "coverage_map": {},
            "app_name": "fixture", "objective": "test search", "recent_tests": [],
            "max_new_tokens": 8000, "enable_thinking": False, "llm_call_log": [],
        }
        with patch.object(p.rag_client, "semantic_dedup_check", return_value={"is_duplicate": False}), \
             patch.object(p.prompts, "build_testcase_prompt", return_value="make a different test"), \
             patch.object(p.model_client, "call_model", return_value={"answer": json.dumps(tc)}):
            result = p.duplicate_check(state)
        self.assertEqual(result["next_testcase"], {})
        self.assertEqual(result["finalization_mode"], "proposal_rejected")
        self.assertTrue(result["planning_errors"])

    def test_tools_never_force_a_rejected_proposal_into_browser(self):
        from planner import agent_loop as p
        tc = {"title": "Verify file upload", "objective": "Upload a file", "expected_result": "Imported"}
        message = {"role": "assistant", "content": None, "tool_calls": [
            {"id": "proposal", "type": "function", "function": {
                "name": "propose_test_case", "arguments": json.dumps(tc)}}]}
        with patch.object(p.rag_client, "get_brief_context", return_value={}), \
             patch.object(p.rag_client, "rag_post") as write, \
             patch.object(p.model_client, "chat_tools", return_value=message), \
             patch.object(p.proposal_mod, "validate", return_value=(True, [])), \
             patch.object(p.degradations, "record"):
            result = p.run_agent_tools({"project": "regression-fixture",
                                       "executor_constraints": planning.contract(cfg())})
        self.assertEqual(result["next_testcase"], {})
        self.assertTrue(result["planning_errors"])
        write.assert_not_called()


class LoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_reviewer_corrects_open_menu_mistaken_for_broken_language_switch(self):
        actions = iter([
            {"action": "click", "ref": "e1"},
            {"action": "finish", "success": False, "reason": "Language did not change"},
            {"action": "click", "ref": "e2"},
            {"action": "finish", "success": True, "reason": "Bangla label is visible"},
        ])
        class Client:
            def chat(self, messages): return json.dumps(next(actions))
        class Reviewer:
            def __init__(self): self.calls = 0
            def chat(self, messages, timeout_s):
                self.calls += 1
                return json.dumps({"decision": "continue" if self.calls == 1 else "accept",
                                   "reason": "Select Bangla from the open menu" if self.calls == 1 else "Label changed"})
        stage = [0]
        async def observe(*args):
            return {"url": "https://test.example", "elements": [
                {"ref": "e1", "name": "Language", "role": "button"},
                {"ref": "e2", "name": "Bangla", "role": "button"}],
                "texts": ["Bangla label" if stage[0] == 2 else "English label"]}
        async def perform(action, snap):
            stage[0] += 1
            return "opened language menu" if stage[0] == 1 else "selected Bangla"
        a = agent.WebAgent(None, cfg(), Client())
        a.reviewer = Reviewer()
        a.dispatcher.perform = perform
        with patch.object(agent.snapshot, "observe", observe):
            result = await a.run("Switch to Bangla and verify the label", 8, 5)
        self.assertTrue(result.success)
        self.assertEqual(stage[0], 2)
        self.assertEqual(a.reviewer.calls, 2)
        self.assertTrue(any("EVIDENCE REVIEW" in h for h in result.history))

    async def test_streaming_navigation_does_not_reload_or_wait_full_timeout(self):
        page = NS(goto=AsyncMock(), wait_for_load_state=AsyncMock(side_effect=TimeoutError("Timeout")))
        await _goto_settled(page, "https://test.example", NS(WEB_NAV_TIMEOUT_MS=30000))
        page.goto.assert_awaited_once()
        self.assertEqual(page.goto.call_args.kwargs["wait_until"], "domcontentloaded")
        self.assertEqual(page.wait_for_load_state.call_args.kwargs["timeout"], 1500)

    async def test_partial_verification_is_not_a_pass(self):
        fake_agent = NS(run=AsyncMock(return_value=agent.AgentResult(
            True, "Navigation worked, but the precondition of empty state was not met.", 3)))
        session = NS(reset_to_base=AsyncMock(), screenshot=AsyncMock(return_value=""), headless=True, page=None)
        collector = NS(reset=lambda: None, findings=Findings())
        with patch.object(runner, "WebAgent", return_value=fake_agent), \
             patch.object(runner.gateway, "log_execution") as logged, \
             patch.object(runner.goal_mod, "build_goal", return_value="verify empty state"):
            outcome = await runner.execute_test_case(session, collector, None,
                       {"test_case_id": "fixture", "title": "Empty state"})
        self.assertEqual(outcome["verdict"], "failed")
        self.assertEqual(outcome["error_type"], "PRECONDITION_NOT_MET")
        self.assertEqual(outcome["attribution"], "environment")
        self.assertEqual(logged.call_args.kwargs["error_type"], "PRECONDITION_NOT_MET")

    async def test_model_wait_does_not_block_browser_event_loop(self):
        beats = []
        class Slow:
            def chat(self, messages):
                time.sleep(.2)
                return '{"action":"finish","success":true,"reason":"ok"}'
        async def observe(*args):
            return {"url": "https://test.example", "elements": []}
        async def heartbeat():
            await asyncio.sleep(.01)
            beats.append(True)
        with patch.object(agent.snapshot, "observe", observe):
            task = asyncio.create_task(heartbeat())
            start = time.monotonic()
            result = await agent.WebAgent(None, cfg(), Slow()).run("goal", 10, .05)
            self.assertLess(time.monotonic() - start, .15)
            self.assertFalse(result.success)
            self.assertIn("Timed out", result.reason)
            await task
        self.assertTrue(beats)

    async def test_prior_field_value_reaches_next_model_turn(self):
        snapshots = iter([
            {"url": "https://test.example", "elements": [{"ref": "e1", "name": "Name", "value": "Alice"}]},
            {"url": "https://test.example", "elements": [{"ref": "e1", "name": "Name", "value": ""}]},
        ])
        prompts = []
        class Client:
            def chat(self, messages):
                prompts.append(messages[-1]["content"])
                return ('{"action":"fill","ref":"e1","text":""}' if len(prompts) == 1 else
                        '{"action":"finish","success":true,"reason":"Name is empty"}')
        async def observe(*args): return next(snapshots)
        async def perform(*args): return "cleared Name"
        a = agent.WebAgent(None, cfg(), Client())
        a.dispatcher.perform = perform
        with patch.object(agent.snapshot, "observe", observe):
            result = await a.run("Clear the name and verify it", 5, 2)
        self.assertTrue(result.success)
        self.assertIn("Observed after the previous action", prompts[-1])
        self.assertIn("Name=''", prompts[-1])


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    print(f"{result.testsRun - len(result.failures) - len(result.errors)}/{result.testsRun} checks passed")
    raise SystemExit(not result.wasSuccessful())
