"""Tool candidates share schema validation, dispatch and read-only boundaries."""
from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_agent.maps.home_store import HomeMapStore
from embodied_agent.maps.store import MapStore
from embodied_agent.models.contracts import PlannerError
from embodied_agent.tools.capabilities import build_capabilities, validate_actions
from embodied_agent.tools.query import QueryTools
from embodied_agent.tools.registry import QueryToolCatalog, QueryToolDefinition


class QueryToolRegistryTests(unittest.TestCase):
    def setUp(self):
        self.world = HomeMapStore().load()
        self.snapshot = self.world.snapshot()

    def assert_error(self, code, operation):
        with self.assertRaises(PlannerError) as error:
            operation()
        self.assertEqual(error.exception.code, code)

    def test_defaults_keep_queries_frozen_and_do_not_save_or_execute(self):
        before = self.world.to_dict()
        query = QueryTools(self.world, self.snapshot)
        self.assertEqual([schema["function"]["name"] for schema in query.schemas],
                         ["observe", "query_world", "inspect", "get_capabilities"])
        self.snapshot["objects"]["remote"]["position_m"][0] = 99
        result = query.call("query_world", {"object_id": "remote"})
        self.assertEqual(result["object"]["position_m"], [-1, .43, .625])
        result["object"]["position_m"][0] = 88
        observed = query.call("observe", {})
        observed["snapshot"]["objects"]["remote"]["position_m"][0] = 77
        self.assertEqual(query.call("inspect", {"object_id": "remote"})["object"]["position_m"][0], -1)
        self.assertEqual(query.call("query_world", {"object_id": "tea_table"})["object_id"], "tea_table")
        self.assertEqual(query.call("query_world", {"object_id": "floor"})["object"]["kind"], "floor")
        self.assertEqual(self.world.to_dict(), before)
        for name in ("save", "step", "navigate", "set_position"):
            self.assert_error("UNKNOWN_TOOL", lambda name=name: query.call(name, {}))

    def test_selected_tools_match_schema_dispatch_and_capability_advertisement(self):
        config = {"components": {"tools": {"query_names": ["query_world", "get_capabilities"]}}}
        catalog = QueryToolCatalog(config)
        query = QueryTools(self.world, self.snapshot, catalog=catalog)
        self.assertEqual([schema["function"]["name"] for schema in query.schemas], list(catalog.names))
        self.assertEqual(query.call("get_capabilities", {})["query_tools"], list(catalog.names))
        self.assert_error("UNKNOWN_TOOL", lambda: query.call("observe", {}))
        self.assert_error("UNKNOWN_TOOL", lambda: query.call("inspect", {"object_id": "remote"}))
        disabled = QueryTools(self.world, self.snapshot,
            catalog=QueryToolCatalog({"components": {"tools": {"query_names": []}}}))
        self.assertEqual(disabled.schemas, [])
        self.assertEqual(disabled.capabilities()["query_tools"], [])
        self.assert_error("UNKNOWN_TOOL", lambda: disabled.call("query_world", {"object_id": "remote"}))

    def test_candidate_descriptions_are_detached_and_independently_fingerprinted(self):
        config = {"components": {"tools": {"descriptions": {"observe": "Read a frozen observation."}}}}
        catalog = QueryToolCatalog(config)
        baseline = QueryToolCatalog()
        config["components"]["tools"]["descriptions"]["observe"] = "changed after construction"
        schema = catalog.schemas
        self.assertEqual(schema[0]["function"]["description"], "Read a frozen observation.")
        schema[0]["function"]["description"] = "changed returned schema"
        self.assertEqual(catalog.schemas[0]["function"]["description"], "Read a frozen observation.")
        self.assertNotEqual(catalog.fingerprint, baseline.fingerprint)
        self.assertEqual(baseline.fingerprint, QueryToolCatalog().fingerprint)

    def test_replacement_schema_controls_validation_and_replacement_handler(self):
        parameters = {"type": "object", "properties": {"object_id": {
            "type": "string", "enum": ["remote"], "description": "Only the remote fixture."}},
            "required": ["object_id"], "additionalProperties": False}
        calls = []

        def fixture_query(context, arguments):
            calls.append(copy.deepcopy(arguments))
            result = context._query_entity(arguments["object_id"], inspect=False)
            arguments["object_id"] = "detached handler argument"
            return {**result, "handler_version": "fixture-v2"}

        definition = QueryToolDefinition("query_world", "Read one fixture.", parameters,
                                         fixture_query, version="fixture-v2")
        catalog = QueryToolCatalog(definitions={"query_world": definition})
        parameters["properties"]["object_id"]["enum"].append("book")
        query = QueryTools(self.world, self.snapshot, catalog=catalog)
        arguments = {"object_id": "remote"}
        self.assertEqual(query.call("query_world", arguments)["handler_version"], "fixture-v2")
        self.assertEqual(arguments, {"object_id": "remote"})
        self.assertEqual(calls, [arguments])
        self.assert_error("INVALID_TOOL_ARGUMENTS", lambda: query.call("query_world", {"object_id": "book"}))
        schema = next(schema for schema in catalog.schemas if schema["function"]["name"] == "query_world")
        self.assertEqual(schema["function"]["parameters"]["properties"]["object_id"]["enum"], ["remote"])

    def test_invalid_configuration_cannot_expand_model_permissions(self):
        for settings in ({"query_names": ["save"]}, {"query_names": ["observe", "observe"]},
                         {"query_names": "observe"}, {"descriptions": {"step": "Execute"}},
                         {"descriptions": {"observe": False}}, {"unexpected": True}):
            with self.subTest(settings=settings):
                self.assert_error("INVALID_TOOL_CONFIG",
                    lambda: QueryToolCatalog({"components": {"tools": settings}}))
        self.assert_error("INVALID_TOOL_CONFIG", lambda: QueryToolCatalog({"components": None}))
        definition = QueryToolDefinition("observe", "Read", {"type": "object", "properties": {},
            "required": [], "additionalProperties": False}, lambda context, arguments: {})
        for candidate in (replace(definition, name="save"),
                          replace(definition, parameters={**definition.parameters, "additionalProperties": True}),
                          replace(definition, parameters={"type": "object", "properties": {"save": {"type": "string"}},
                              "required": ["save"], "additionalProperties": False})):
            self.assert_error("INVALID_TOOL_CONFIG", lambda candidate=candidate:
                QueryToolCatalog(definitions={candidate.name: candidate}))

    def test_invalid_arguments_and_inspection_permissions_keep_original_errors(self):
        query = QueryTools(self.world, self.snapshot)
        for name, arguments in (("observe", {"object_id": "remote"}), ("query_world", {}),
                                ("inspect", {"object_id": ""}), ("query_world", {"object_id": True}),
                                ("query_world", {"object_id": "remote", "save": True}), ("observe", [])):
            with self.subTest(name=name, arguments=arguments):
                self.assert_error("INVALID_TOOL_ARGUMENTS", lambda: query.call(name, arguments))
        self.assert_error("UNKNOWN_OBJECT", lambda: query.call("query_world", {"object_id": "missing"}))
        self.assert_error("UNKNOWN_OBJECT", lambda: query.call("inspect", {"object_id": "tea_table"}))
        restricted = copy.deepcopy(self.snapshot)
        restricted["objects"]["remote"]["operations"] = []
        self.assert_error("UNSUPPORTED_OPERATION", lambda:
            QueryTools(self.world, restricted).call("inspect", {"object_id": "remote"}))

    def test_default_desktop_and_environment_edit_capabilities_are_preserved(self):
        world = MapStore().load("classic")
        query = QueryTools(world)
        self.assertEqual(query.call("query_world", {"object_id": "danger_zone"})["object"],
                         world.to_dict()["danger_zone"])
        self.assertEqual(query.call("get_capabilities", {}), build_capabilities(world, world.to_dict()))
        desktop_edit = QueryTools(world, edit_mode=True).call("get_capabilities", {})
        self.assertEqual(desktop_edit["operations"], ["set_position", "set_half_size", "scale", "shift_axis"])
        home_edit = QueryTools(self.world, edit_mode=True).call("get_capabilities", {})
        self.assertEqual(home_edit["operations"], ["set_position", "set_name", "set_description"])
        self.assertNotIn("save", home_edit["operations"])
        self.assert_error("QUERY_TOOL_REQUIRED", lambda:
            validate_actions([{"skill": "observe"}], build_capabilities(world, world.to_dict())))

    def test_component_import_does_not_load_simulator_model_sdk_or_old_tools(self):
        program = """import sys
sys.path.insert(0, 'src')
from embodied_agent.tools import QueryTools, QueryToolCatalog, QueryToolDefinition
from embodied_agent.tools.capabilities import build_capabilities
assert not any(name in sys.modules for name in ('mujoco', 'numpy', 'tkinter', 'torch', 'openai'))
assert not any(name in sys.modules for name in ('embodied_agent.agents.query_tools', 'embodied_agent.agents.capabilities'))
"""
        result = subprocess.run([sys.executable, "-c", program], cwd=ROOT,
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
