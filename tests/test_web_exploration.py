"""Offline passive-graph tests: identity, persistence, privacy and player hooks."""
import asyncio
import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from web_player.exploration.state import state, route
from web_player.exploration.state import observation_signature, action_descriptor
from web_player.exploration.store import GraphStore
from web_player.exploration.recorder import Recorder, create_recorder
from web_player import agent
from web_player.actions import ActionError


def snap(url="https://site.test/?tab=projects", **extra):
    return dict(url=url, elements=[{"ref": "e1", "role": "button", "name": "New Project"}], **extra)


class IdentityTests(unittest.TestCase):
    def test_query_navigation_and_record_ids(self):
        self.assertNotEqual(route("https://s/?tab=projects"), route("https://s/?tab=projectdetails"))
        self.assertEqual(route("https://s/?tab=projectdetails&projectId=2"),
                         route("https://s/?projectId=77&tab=projectdetails"))
        self.assertEqual(route("https://s/view-survey/600"), route("https://s/view-survey/601"))

    def test_secrets_tracking_and_userinfo_removed(self):
        result = route("https://alice:secret@s/42?token=secret&projectId=99&utm_source=secret&tab=projects")
        self.assertNotIn("secret", result)
        self.assertNotIn("alice", result)
        self.assertNotIn("99", result)
        self.assertIn("projects", result)

    def test_modal_selected_tabs_and_ref_stability(self):
        a = snap()
        b = copy.deepcopy(a)
        b["elements"][0]["ref"] = "e99"
        self.assertEqual(state(a), state(b))
        b["dialog_open"] = True
        b["dialog_name"] = "New Project"
        self.assertNotEqual(state(a)[0], state(b)[0])
        b = copy.deepcopy(a)
        b["elements"][0]["selected"] = True
        self.assertNotEqual(state(a)[0], state(b)[0])

    def test_hash_routes_and_configurable_query_keys(self):
        self.assertNotEqual(route("https://s/#/projects"), route("https://s/#/surveys"))
        self.assertNotEqual(route("https://s/?screen=one", ("screen",)),
                            route("https://s/?screen=two", ("screen",)))

    def test_values_and_counts_not_state_identity(self):
        a = snap(headings=["Projects 23"])
        b = snap(headings=["Projects 24"])
        b["elements"][0]["value"] = "private input"
        self.assertEqual(state(a), state(b))

    def test_ref_change_does_not_claim_an_observed_change(self):
        a, b = snap(), snap()
        b["elements"][0]["ref"] = "e99"
        self.assertEqual(observation_signature(a), observation_signature(b))

    def test_unlabelled_input_value_not_persisted_as_name(self):
        s = snap()
        s["elements"] = [{"role": "textbox", "name": "PRIVATE", "value": "PRIVATE"}]
        self.assertNotIn("PRIVATE", json.dumps(state(s)))

    def test_navigation_keys_still_cannot_persist_token(self):
        self.assertNotIn("private", route("https://s/?token=private", ("token",)))

    def test_safe_action_arguments_and_disambiguation(self):
        s = snap()
        s["elements"].append({"ref": "e2", "role": "button", "name": "New Project"})
        a = action_descriptor({"action": "click", "ref": "e2", "text": "PRIVATE"}, s)
        self.assertEqual(a["ordinal"], 1)
        self.assertNotIn("PRIVATE", json.dumps(a))
        self.assertEqual(action_descriptor({"action": "press", "key": "Enter"}, s)["key"], "Enter")
        self.assertNotIn("key", action_descriptor({"action": "press", "key": "PRIVATE"}, s))


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "graph.sqlite3"
        self.store = GraphStore(self.path, "project-a")
        self.recorder = Recorder(self.store)

    def tearDown(self):
        self.recorder.close()
        self.temp.cleanup()

    def test_success_noop_changed_observation_and_failure(self):
        a, b = snap(), snap(dialog_open=True, dialog_name="New Project")
        r = self.recorder
        r.observe(a)
        r.attempted({"action": "click", "ref": "e1"}, a)
        r.observe(b)
        r.attempted({"action": "click", "ref": "e1"}, b)
        r.observe(b)
        r.attempted({"action": "fill", "ref": "e1", "text": "PRIVATE"}, b)
        c = copy.deepcopy(b)
        c["elements"][0]["value"] = "PRIVATE"
        r.observe(c)
        r.attempted({"action": "click", "ref": "e1"}, c, "BLOCKED_BY_GUARDRAIL")
        result = self.store.export()
        self.assertEqual(len(result["nodes"]), 2)
        self.assertEqual({e["outcome"] for e in result["edges"]},
                         {"state_changed", "no_visible_change", "observation_changed", "dispatch_failed"})
        self.assertNotIn("PRIVATE", json.dumps(result))
        self.assertNotIn('"ref"', json.dumps(result))
        failure = next(e for e in result["edges"] if e["outcome"] == "dispatch_failed")
        self.assertIsNone(failure["destination"])

    def test_reopen_counts_and_project_isolation(self):
        self.recorder.observe(snap())
        self.recorder.close()
        self.store = GraphStore(self.path, "project-a")
        self.recorder = Recorder(self.store)
        self.recorder.observe(snap())
        self.assertEqual(self.store.export()["nodes"][0]["observations"], 2)
        other = GraphStore(self.path, "project-b")
        self.assertEqual(other.export()["nodes"], [])
        other.close()

    def test_unobserved_last_action_not_assumed_success(self):
        self.recorder.observe(snap())
        self.recorder.attempted({"action": "click", "ref": "e1"}, snap())
        self.recorder.close()
        self.store = GraphStore(self.path, "project-a")
        self.recorder = Recorder(self.store)
        edge = self.store.export()["edges"][0]
        self.assertEqual(edge["outcome"], "destination_unobserved")
        self.assertIsNone(edge["destination"])

    def test_failure_is_fail_open_and_can_disable(self):
        with patch.object(self.store, "node", side_effect=OSError("private")):
            with self.assertLogs("web_player.exploration.recorder", level="WARNING") as logs:
                self.recorder.observe(snap())
        self.assertTrue(self.recorder.disabled)
        self.assertNotIn("private", str(logs.output))
        self.assertIsNone(create_recorder(NS(WEB_EXPLORATION_ENABLED=False)))


class IntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_player_uses_only_existing_observations_and_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "graph.sqlite3")
            cfg = NS(PROJECT="fixture", WEB_BASE_URL="https://site.test", WEB_EXPLORATION_ENABLED=True,
                     WEB_EXPLORATION_DB=path, WEB_SNAPSHOT_MAX_ELEMENTS=60,
                     WEB_STALL_STEPS=6, WEB_BLOCKED_TEXTS=[], WEB_BLOCKED_URL_PATTERNS=[])
            actions = iter([{"action": "click", "ref": "e1"},
                            {"action": "finish", "success": True, "reason": "Dialog appeared"}])
            client = NS(chat=lambda messages: json.dumps(next(actions)))
            player = agent.WebAgent(None, cfg, client)
            with patch.object(agent.snapshot, "observe", AsyncMock(side_effect=[snap(), snap(dialog_open=True)])) as observe, \
                 patch.object(player.dispatcher, "perform", AsyncMock(return_value="clicked")) as perform:
                result = await player.run("Open New Project", 3, 5)
            self.assertTrue(result.success)
            self.assertEqual(observe.await_count, 2)
            self.assertEqual(perform.await_count, 1)
            store = GraphStore(path, "fixture|https://site.test/|tab,view,mode,section,step,lang,page,route")
            self.assertEqual(len(store.export()["edges"]), 1)
            store.close()

    async def test_guardrail_failure_is_recorded_without_success_edge(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = GraphStore(Path(tmp) / "graph.sqlite3", "fixture")
            recorder = Recorder(store)
            cfg = NS(WEB_BASE_URL="https://site.test", WEB_SNAPSHOT_MAX_ELEMENTS=60,
                     WEB_STALL_STEPS=6, WEB_BLOCKED_TEXTS=[], WEB_BLOCKED_URL_PATTERNS=[])
            player = agent.WebAgent(None, cfg, NS(chat=lambda messages: '{"action":"click","ref":"e1"}'))
            with patch.object(agent, "create_recorder", return_value=recorder), \
                 patch.object(agent.snapshot, "observe", AsyncMock(return_value=snap())), \
                 patch.object(player.dispatcher, "perform", AsyncMock(side_effect=ActionError("Forbidden", "BLOCKED_BY_GUARDRAIL"))):
                result = await player.run("Attempt click", 3, 5)
            self.assertFalse(result.success)
            reader = GraphStore(Path(tmp) / "graph.sqlite3", "fixture", read_only=True)
            edges = reader.export()["edges"]
            reader.close()
            self.assertEqual(len(edges), 1)
            self.assertEqual(edges[0]["outcome"], "dispatch_failed")
            self.assertEqual(edges[0]["error_category"], "BLOCKED_BY_GUARDRAIL")

    async def test_last_step_records_unknown_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "graph.sqlite3"
            recorder = Recorder(GraphStore(path, "fixture"))
            cfg = NS(WEB_BASE_URL="https://site.test", WEB_SNAPSHOT_MAX_ELEMENTS=60,
                     WEB_STALL_STEPS=6, WEB_BLOCKED_TEXTS=[], WEB_BLOCKED_URL_PATTERNS=[])
            player = agent.WebAgent(None, cfg, NS(chat=lambda messages: '{"action":"click","ref":"e1"}'))
            with patch.object(agent, "create_recorder", return_value=recorder), \
                 patch.object(agent.snapshot, "observe", AsyncMock(return_value=snap())) as observe, \
                 patch.object(player.dispatcher, "perform", AsyncMock(return_value="clicked")):
                result = await player.run("Attempt click", 1, 5)
            self.assertFalse(result.success)
            self.assertEqual(observe.await_count, 1)
            reader = GraphStore(path, "fixture", read_only=True)
            edges = reader.export()["edges"]
            reader.close()
            self.assertEqual(edges[0]["outcome"], "destination_unobserved")


if __name__ == "__main__":
    unittest.main()
