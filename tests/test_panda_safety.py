"""Physical safety remains independent of language planning."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from embodied_agent.contracts import ContractError
from embodied_agent.safety.path_precheck import check_skill_path
from embodied_agent.simulation.episode import Episode
from embodied_agent.simulation.scenarios import load_json
from embodied_agent.simulation.errors import M2Failure

class PandaSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = load_json(ROOT / 'configs' / 'agent_runtime.json')
        cls.thresholds = load_json(ROOT / 'configs' / 'm2_thresholds.json')
        cls.scenario = load_json(ROOT / 'configs' / 'm2_scenarios.json')['scenarios'][0]

    def test_danger_zone_path_precheck_rejects_a_segment_through_box(self) -> None:
        episode = SimpleNamespace(
            mocap_id=0,
            cube_body_id=0,
            danger_zone_geom_id=0,
            target_center=np.array([0.8, 0.8, 0.0]),
            table_top_z=0.0,
            cube_half_size=np.array([0.025, 0.025, 0.025]),
            thresholds={"pregrasp_clearance_m": 0.1, "lift_height_m": 0.1},
            data=SimpleNamespace(
                mocap_pos=np.array([[0.0, 0.0, 0.0]]),
                xpos=np.array([[0.8, 0.0, 0.0]]),
                geom_xpos=np.array([[0.4, 0.0, 0.0]]),
            ),
            model=SimpleNamespace(geom_size=np.array([[0.1, 0.1, 0.1]])),
        )
        with self.assertRaises(ContractError) as raised:
            check_skill_path(episode, "pick", self.runtime)
        self.assertEqual(raised.exception.code, "PATH_REJECTED")

    def test_place_skill_requires_a_verified_grasp(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            episode = Episode(self.scenario, self.thresholds, Path(temporary))
            with self.assertRaises(M2Failure) as raised:
                episode.place_skill()
        self.assertEqual(raised.exception.code, "PRECONDITION_FAILED")
        self.assertEqual(episode.total_steps, 0)
