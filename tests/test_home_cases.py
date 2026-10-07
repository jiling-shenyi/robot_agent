"""Current robot case catalog and trusted fixture validation."""
import sys
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from embodied_agent.evaluation.cases import load_cases, validate_case

class UnifiedHomeCaseTests(unittest.TestCase):
    def test_catalog_keeps_legacy_cases_and_registers_nine_original_demo_home_cases(self):
        cases=load_cases()
        self.assertEqual(len(cases),34)
        self.assertEqual(len([c for c in cases if "legacy_scenario_id" in c]),20)
        home=[c for c in cases if c.get("map_id")=="home_living_room"]
        self.assertEqual(len(home),9)
        self.assertTrue(all(c["agent"]=="robot" for c in home))
        self.assertEqual({c["case_id"] for c in home if c.get("test_fault_protocol")},{"home_navigation_collision"})

    def test_trusted_case_plan_fault_and_transport_assertions_are_strict(self):
        base={"case_id":"home_test","agent":"robot","instruction":"观察房间",
              "robot_plan":{"schema_version":1,"actions":[{"skill":"observe"}]}}
        for fields in ({"robot_plan":{"schema_version":1,"actions":[{"skill":"toggle","object_id":"kettle"}]}},
                       {"test_fault_protocol":"arbitrary_state_edit"},
                       {"test_fault_protocol":"navigation_waypoint_into_tea_table_v1"},
                       {"expected":{"status":"SUCCESS","transport_success":1}}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                validate_case({**base,**fields},{"robot","environment"})
