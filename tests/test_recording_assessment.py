"""Trusted specifications, independent outcomes and immutable assessment hashes."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_agent.recording.assessment import EvaluationStore, assess_task
from embodied_agent.recording.store import TaskRecordStore
from embodied_agent.evaluation.task_writer import TaskRunWriter

GOALS = [{"predicate": "held", "object_id": "cube", "requested_method": "pick"}]
SPEC = {"trusted": True, "source": "fixture", "kind": "robot_goal", "goals": GOALS}


class RecordingAssessmentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "records"
        self.store = TaskRecordStore(self.root)
        self.evaluations = EvaluationStore(self.root, task_store=self.store)

    def robot(self, *, spec=None, goals=None, verified=True, recording_errors=None, outcome_fields=None):
        task = self.store.begin(map_id="classic", map_definition={"revision": 1}, initial_state={"step": 0},
            natural_language="pick cube", metadata={"task_kind": "robot_task", "agent_role": "instruction",
            "trusted_task_spec": spec})
        task.append_attempt()
        task.append_event({"event": "goal.evaluated", "detail": {"passed": verified}, "source": "evaluator"})
        task.finish(final_state={"held": "cube"}, outcome={"status": "SUCCESS", "executor_verified": verified,
            "agent_interpreted_goals": copy.deepcopy(GOALS if goals is None else goals), "goal_check": {"passed": verified},
            "recording_errors": recording_errors or [], **(outcome_fields or {})})
        return task

    def test_free_task_success_does_not_become_intent_or_scalar_reward(self):
        task = self.robot()
        reference = assess_task(self.store, task.task_id, eval_run_id="free", scalar_reward=10)
        result = self.evaluations.load(reference)
        self.assertTrue(result["components"]["goal_verification"])
        self.assertIsNone(result["components"]["intent_correctness"])
        self.assertIsNone(result["scalar_reward"])
        self.assertFalse(result["validity"]["valid"])

    def test_executor_success_for_wrong_goal_is_not_intent_correct(self):
        task = self.robot(spec=SPEC, goals=[{"predicate": "held", "object_id": "book", "requested_method": "pick"}])
        result = self.evaluations.load(assess_task(self.store, task.task_id, eval_run_id="wrong"))
        self.assertFalse(result["components"]["intent_correctness"])
        self.assertTrue(result["components"]["execution_success"])
        self.assertIsNone(result["scalar_reward"])

    def test_trusted_success_keeps_only_explicit_reward_and_freezes_hashes(self):
        task = self.robot(spec=SPEC)
        reference = assess_task(self.store, task.task_id, eval_run_id="trusted", scalar_reward=2.5,
            evaluator_config={"reward_supplied_by": "test"})
        result = self.evaluations.load(reference)
        self.assertTrue(result["components"]["intent_correctness"])
        self.assertEqual(result["scalar_reward"], 2.5)
        self.assertEqual(len(result["evaluator"]["code_sha256"]), 64)
        self.assertEqual(len(result["source_task"]["seal_sha256"]), 64)
        self.assertTrue(result["evidence"])
        with self.assertRaises(FileExistsError):
            assess_task(self.store, task.task_id, eval_run_id="trusted")
        self.assertEqual(self.evaluations.load(reference), result)

    def test_ref_hash_and_evidence_ids_are_enforced(self):
        task = self.robot(spec=SPEC)
        reference = assess_task(self.store, task.task_id, eval_run_id="hash")
        modified = {**reference, "sha256": "0" * 64}
        with self.assertRaises(ValueError):
            self.evaluations.load(modified)
        with self.assertRaises(ValueError):
            self.evaluations.save(task.task_id, eval_run_id="wrong-event", evaluator_name="test",
                evaluator_version="1", components={}, evidence_event_ids=["missing"])

    def test_map_commit_and_reload_failure_are_independent(self):
        operations = [{"op": "shift_axis", "object_id": "cube", "axis": "y", "delta_m": .01}]
        before = {"revision": 1, "name": "map", "cube_position_m": [.4, -.29, .425]}
        after = {**before, "revision": 2, "cube_position_m": [.4, -.28, .425]}
        task = self.store.begin(map_id="classic", map_definition=before, initial_state=before,
            natural_language="raise y .01", metadata={"task_kind": "map_edit", "agent_role": "environment",
            "trusted_task_spec": {"trusted": True, "source": "fixture", "kind": "map_edit", "expected_operations": operations}})
        task.append_attempt()
        task.append_event({"event": "proposal.created", "detail": {"proposal": {"operations": operations}}})
        task.append_event({"event": "commit.finished", "detail": {"map_before": before, "map_after": after}})
        task.append_event({"event": "scene.reload.failed", "detail": {"error_code": "REFRESH_FAILED"}})
        task.finish(final_state=None, outcome={"status": "FAILED", "error_code": "REFRESH_FAILED",
            "commit_status": "success", "reload_status": "failed", "persisted": True})
        result = self.evaluations.load(assess_task(self.store, task.task_id, eval_run_id="map"))
        self.assertTrue(result["components"]["intent_correctness"])
        self.assertTrue(result["components"]["operation_effect"])
        self.assertTrue(result["components"]["execution_success"])
        self.assertEqual(result["components"]["reload_status"], "failed")
        self.assertIsNone(result["scalar_reward"])

    def test_correct_refusal_can_be_a_valid_result_while_status_failed(self):
        spec = {"trusted": True, "source": "fixture", "kind": "rejection", "expected_error_code": "CAPABILITY_GAP"}
        task = self.store.begin(map_id="classic", map_definition={}, initial_state={}, natural_language="throw",
            metadata={"trusted_task_spec": spec})
        task.append_attempt()
        task.finish(final_state={}, outcome={"status": "FAILED", "error_code": "CAPABILITY_GAP"})
        result = self.evaluations.load(assess_task(self.store, task.task_id, eval_run_id="refusal"))
        self.assertTrue(result["validity"]["valid"])
        self.assertTrue(result["components"]["intent_correctness"])
        self.assertTrue(result["components"]["execution_success"])

    def test_retrospective_spec_does_not_modify_sealed_facts(self):
        task = self.robot()
        before = task.journal_path.read_bytes()
        reference = assess_task(self.store, task.task_id, eval_run_id="retrospective", trusted_task_spec=SPEC)
        self.assertEqual(task.journal_path.read_bytes(), before)
        self.assertIsNone(self.store.load(task.task_id)["trusted_task_spec"])
        self.assertEqual(self.evaluations.load(reference)["trusted_task_spec"], SPEC)

    def test_success_without_physical_goal_evidence_keeps_reward_unknown(self):
        task = self.store.begin(map_id="classic", map_definition={}, initial_state={}, natural_language="pick",
            metadata={"trusted_task_spec": SPEC})
        task.append_attempt()
        task.finish(final_state={}, outcome={"status": "SUCCESS", "executor_verified": True,
            "agent_interpreted_goals": GOALS})
        result = self.evaluations.load(assess_task(self.store, task.task_id, eval_run_id="missing-goal", scalar_reward=1))
        self.assertIsNone(result["components"]["goal_verification"])
        self.assertIsNone(result["components"]["execution_success"])
        self.assertIsNone(result["scalar_reward"])
        self.assertFalse(result["validity"]["valid"])

    def test_generic_query_kind_does_not_invent_observe_intent(self):
        task = self.store.begin(map_id="classic", map_definition={}, initial_state={}, natural_language="query",
            metadata={"trusted_task_spec": {"trusted": True, "source": "fixture", "kind": "query"}})
        task.append_attempt()
        task.append_event({"event": "tool.finished", "detail": {"name": "observe", "result": {"state": 1}}})
        task.finish(final_state={}, outcome={"status": "SUCCESS"})
        result = self.evaluations.load(assess_task(self.store, task.task_id, eval_run_id="generic-query", scalar_reward=1))
        self.assertIsNone(result["components"]["intent_correctness"])
        self.assertIsNone(result["scalar_reward"])

    def test_real_writer_preserves_map_commit_despite_reload_failure(self):
        before = {"revision": 1, "name": "map", "cube_position_m": [.4, -.29, .425]}
        after = {**before, "revision": 2, "cube_position_m": [.4, -.28, .425]}
        operations = [{"op": "shift_axis", "object_id": "cube", "axis": "y", "delta_m": .01}]
        writer = TaskRunWriter(self.root.parent / "writer-reports", "writer-assessment", records_dir=self.root)
        self.addCleanup(writer.close)
        record = writer.begin_task(instruction="shift cube y .01", map_id="classic", map_definition=before,
            initial_state=before, metadata={"agent": "environment", "trusted_task_spec": {
                "trusted": True, "source": "fixture", "kind": "map_edit", "expected_operations": operations}})
        writer.record_event("environment_validated", {"plan": {"operations": operations},
            "stored_map_before": before, "candidate_map_after": after})
        writer.record_event("environment_commit_finished", {"operations": operations,
            "stored_map_before": before, "stored_map_after": after, "commit_status": "success"})
        writer.record_event("environment_reload_failed", {"reload_status": "failed", "error_code": "REFRESH_FAILED"})
        writer.finish_task({"status": "FAILED", "error_code": "REFRESH_FAILED", "persisted": True,
            "commit_status": "success", "reload_status": "failed"}, final_state=after)
        result = self.evaluations.load(assess_task(self.store, record.task_id, eval_run_id="writer-map"))
        self.assertTrue(result["components"]["intent_correctness"])
        self.assertTrue(result["components"]["operation_effect"])
        self.assertTrue(result["components"]["execution_success"])
        self.assertEqual(result["components"]["reload_status"], "failed")
        self.assertIsNone(result["components"]["goal_verification"])

    def test_claimed_commit_without_commit_event_keeps_effect_unknown(self):
        operations = [{"op": "shift_axis", "object_id": "cube", "axis": "y", "delta_m": .01}]
        task = self.store.begin(map_id="classic", map_definition={"cube_position_m": [.4, -.29, .425]},
            initial_state={}, natural_language="shift", metadata={"trusted_task_spec": {
                "trusted": True, "source": "fixture", "kind": "map_edit", "expected_operations": operations}})
        task.append_attempt()
        task.append_event({"event": "proposal.validated", "detail": {"plan": {"operations": operations}}})
        task.finish(final_state={}, outcome={"status": "SUCCESS", "commit_status": "success", "persisted": True})
        result = self.evaluations.load(assess_task(self.store, task.task_id, eval_run_id="missing-commit", scalar_reward=1))
        self.assertIsNone(result["components"]["operation_effect"])
        self.assertIsNone(result["components"]["execution_success"])
        self.assertIsNone(result["scalar_reward"])

    def test_recording_gap_preserves_known_components_but_invalidates_reward(self):
        variants = [({"recording_errors": [{"event": "proposal_rejected", "message": "disk failed"}]}, False),
                    ({"error_code": "RECORDING_FAILED"}, False),
                    ({"record_integrity": "incomplete"}, False),
                    ({"component_errors": [{"component": "memory_backend", "message": "lookup failed"}]}, True)]
        for index, (fields, complete) in enumerate(variants):
            with self.subTest(fields=fields):
                task = self.robot(spec=SPEC, outcome_fields=fields)
                reference = assess_task(self.store, task.task_id, eval_run_id=f"recording-gap-{index}", scalar_reward=1)
                result = self.evaluations.load(reference)
                self.assertTrue(self.store.verify(task.task_id)["valid"])
                self.assertTrue(result["components"]["goal_verification"])
                self.assertTrue(result["components"]["intent_correctness"])
                self.assertTrue(result["components"]["execution_success"])
                self.assertEqual(result["components"]["recording_evidence_complete"], complete)
                self.assertEqual(result["validity"]["valid"], complete)
                if complete:
                    self.assertEqual(result["scalar_reward"], 1)
                else:
                    self.assertIn("recording_evidence_incomplete", result["validity"]["reasons"])
                    self.assertIsNone(result["scalar_reward"])


if __name__ == "__main__":
    unittest.main()
