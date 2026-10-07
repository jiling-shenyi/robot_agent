from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from embodied_agent.execution.home_contracts import HomeExecutionError, validate_home_plan


class HomeContractTests(unittest.TestCase):
    def test_mutation_and_half_plan_cannot_be_executed(self):
        for actions in ([{"skill": "set_state", "object_id": "kettle", "hot": False}],
                        [{"skill": "stop"}, {"skill": "exec", "code": "arbitrary"}],
                        [{"skill": "pick", "object_id": "remote", "override_policy": True}],
                        [{"skill": "place", "support_id": "dining_table", "target_xy": [float("nan"), 1]}],
                        [{"skill": "wait", "seconds": True}], []):
            with self.subTest(actions=actions), self.assertRaises(HomeExecutionError):
                validate_home_plan({"schema_version": 1, "actions": actions})

    def test_full_contract_keeps_parameters_and_copies_plan(self):
        plan = {"schema_version": 1, "actions": [
            {"skill": "observe"}, {"skill": "navigate", "target": "tea_table_dock"},
            {"skill": "inspect", "object_id": "remote"}, {"skill": "pick", "object_id": "remote"},
            {"skill": "carry", "target": "dining_table_dock"}, {"skill": "stop"},
            {"skill": "wait", "seconds": 0.2},
            {"skill": "place", "support_id": "dining_table", "target_xy": [1.0, 2.0]},
        ]}
        result = validate_home_plan(plan)
        self.assertEqual(result, plan["actions"])
        plan["actions"][-1]["target_xy"][0] = 999
        self.assertEqual(result[-1]["target_xy"], [1, 2])


if __name__ == "__main__":
    unittest.main()
