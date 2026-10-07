"""Frozen data views, raw assistant targets and connected provenance splits."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_agent.recording.assessment import EvaluationStore, assess_task
from embodied_agent.recording.export import export_dataset
from embodied_agent.recording.store import TaskRecordStore

GOALS = [{"predicate": "held", "object_id": "cube", "requested_method": "pick"}]
SPEC = {"trusted": True, "source": "fixture", "kind": "robot_goal", "goals": GOALS}
RAW = '{"schema_version":1,"goals":[{"predicate":"held","object_id":"cube","requested_method":"pick"}],"decision":"plan","actions":[{"skill":"pick","object_id":"cube"}]}'
BUDGET_CONFIG = {"budgets": {"planner_timeout_s": 30, "max_episode_steps": 30000},
    "agent": {"max_rounds": 8, "max_tool_calls": 24, "max_output_tokens": 4096},
    "task_budget": {"max_requests": 24, "max_tokens": 131072, "max_actions": 128, "max_replans": 4, "timeout_s": 300},
    "execution": {"max_decision_rounds": 8}}


class RecordingExportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = TaskRecordStore(self.root / "records")

    def task(self, *, metadata=None, raw=RAW, status="SUCCESS", source="llm", spec=SPEC, request=None, initial=None,
             consumed=True, delivered=False, before_events=(), runtime_config=BUDGET_CONFIG, recording_errors=None):
        info = {"task_kind": "robot_task", "agent_role": "instruction", "trusted_task_spec": copy.deepcopy(spec),
                "reset_spec": {"map_id": "classic", "seed": 0}, "reset_verified": True, **(metadata or {})}
        task = self.store.begin(map_id="classic", map_definition={"revision": 1}, initial_state=initial or {"step": 0},
            natural_language="pick cube", versions={"model": "fixture", "runtime_config": copy.deepcopy(runtime_config)}, metadata=info)
        task.append_attempt()
        identity = {"model_call_id": "call-1", "decision_id": "decision-1"}
        task.append_event({"event": "decision.started", "decision_id": "decision-1", "observation": {"step": 0}})
        for event in before_events:
            task.append_event(event)
        task.append_event({"event": "model.requested", **identity, "source": "runtime",
            "detail": {"request": request or {"model": "fixture", "messages": [{"role": "user", "content": "pick cube"}], "tools": [], "max_tokens": 4096}}})
        task.append_event({"event": "model.responded", **identity, "source": source,
            "detail": {"response_text": raw, "tool_calls": None, "finish_reason": "stop", "consumed": consumed,
                       "behavior_logprobs": None, "completion_token_ids": None}})
        if delivered:
            task.append_event({"event": "feedback.delivered", **identity,
                "detail": {"recipient": "dialogue_response_parser"}})
        task.append_event({"event": "proposal.validated", **identity,
            "detail": {"plan": {"schema_version": 1, "actions": [{"skill": "pick", "object_id": "cube"}]}}})
        task.append_event({"event": "action.finished", "decision_id": "decision-1", "action_id": "action-1",
            "detail": {"action": {"skill": "pick", "object_id": "cube"}, "status": status}, "observation": {"step": 5, "held": "cube"}})
        task.finish(final_state={"held": "cube"}, outcome={"status": status, "executor_verified": status == "SUCCESS",
            "agent_interpreted_goals": GOALS, "goal_check": {"passed": status == "SUCCESS"}, "terminated": True, "truncated": False,
            "recording_errors": recording_errors or []})
        return task

    def files(self, directory):
        return (json.loads((directory / "samples.json").read_text(encoding="utf-8")),
                json.loads((directory / "qualification.json").read_text(encoding="utf-8")))

    def test_sft_uses_raw_assistant_output_and_freezes_sources(self):
        task = self.task()
        evaluation = assess_task(self.store, task.task_id, eval_run_id="sft")
        output = self.root / "sft"
        manifest = export_dataset(self.store, output, kind="sft", evaluation_refs=[evaluation])
        data, report = self.files(output)
        self.assertEqual(manifest["sample_count"], 1)
        self.assertEqual(data["samples"][0]["assistant"]["content"], RAW)
        self.assertIn('"goals"', data["samples"][0]["assistant"]["content"])
        self.assertTrue(report["tasks"][0]["eligible"])
        self.assertFalse(report["tasks"][0]["rl_update"]["eligible"])
        self.assertIn("behavior_logprobs_unavailable", report["tasks"][0]["rl_update"]["exclusion_reasons"])
        self.assertEqual(manifest["file_hashes"]["samples.json"], hashlib.sha256((output / "samples.json").read_bytes()).hexdigest())
        self.assertTrue(manifest["source_tasks"][0]["hashes"]["seal_sha256"])
        with self.assertRaises(FileExistsError):
            export_dataset(self.store, output, kind="sft")

    def test_sft_excludes_success_without_trusted_independent_evaluation(self):
        self.task(spec=None)
        output = self.root / "free"
        manifest = export_dataset(self.store, output, kind="sft")
        self.assertEqual(manifest["sample_count"], 0)
        report = self.files(output)[1]["tasks"][0]
        self.assertIn("trusted_task_spec_unavailable", report["exclusion_reasons"])
        self.assertIn("trusted_evaluation_unavailable", report["exclusion_reasons"])

    def test_stub_proposals_do_not_become_assistant_demonstrations(self):
        task = self.task(source="stub")
        evaluation = assess_task(self.store, task.task_id, eval_run_id="stub")
        output = self.root / "stub"
        result = export_dataset(self.store, output, kind="sft", evaluation_refs=[evaluation])
        self.assertEqual(result["sample_count"], 0)
        self.assertIn("trusted_raw_assistant_output_unavailable", self.files(output)[1]["tasks"][0]["exclusion_reasons"])

    def test_transitive_session_family_and_memory_groups_never_cross_split(self):
        first = self.task(initial={"step": 1}, metadata={"run_id": "run-a", "task_family_id": "family-a"})
        second = self.task(initial={"step": 2}, metadata={"run_id": "run-b", "task_family_id": "family-a", "memory_connection_ids": ["memory-x"]})
        third = self.task(initial={"step": 3}, metadata={"run_id": "run-c", "memory_connection_ids": ["memory-x"]})
        output = self.root / "groups"
        result = export_dataset(self.store, output, kind="trajectory")
        groups = [row["split"] for row in result["source_tasks"]]
        self.assertEqual(len(groups), 3)
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(groups[1], groups[2])
        self.assertEqual({row["task_id"] for row in result["source_tasks"]}, {first.task_id, second.task_id, third.task_id})

    def test_decision_view_marks_observation_and_keeps_raw_model_calls(self):
        task = self.task()
        output = self.root / "decision"
        export_dataset(self.store, output, kind="decision")
        sample = self.files(output)[0]["samples"][0]
        self.assertEqual(sample["observation_before"], {"step": 0})
        self.assertEqual(sample["next_observation"]["step"], 5)
        self.assertEqual(sample["model_calls"][0]["raw_assistant"]["content"], RAW)
        self.assertTrue(sample["terminated"])

    def test_task_pool_requires_verified_reset_and_does_not_promote_snapshot(self):
        self.task(metadata={"reset_spec": None, "reset_verified": False,
                            "reproducibility": {"level": "trace_only", "restorable": False}})
        output = self.root / "pool"
        result = export_dataset(self.store, output, kind="task_pool")
        self.assertEqual(result["sample_count"], 0)
        self.assertIn("verified_reset_or_checkpoint_unavailable", self.files(output)[1]["tasks"][0]["exclusion_reasons"])

    def test_preference_requires_same_input_start_and_explicit_evaluations(self):
        good = self.task()
        bad = self.task(raw='{"schema_version":1,"goals":[],"actions":[]}', status="FAILED")
        references = [assess_task(self.store, good.task_id, eval_run_id="pair", scalar_reward=1),
                      assess_task(self.store, bad.task_id, eval_run_id="pair", scalar_reward=0)]
        output = self.root / "pairs"
        result = export_dataset(self.store, output, kind="preference", evaluation_refs=references)
        self.assertEqual(result["sample_count"], 1)
        sample = self.files(output)[0]["samples"][0]
        self.assertEqual(sample["task_ids"], [good.task_id, bad.task_id])
        self.assertEqual(sample["chosen"]["content"], RAW)

    def test_preference_does_not_pair_different_prompts(self):
        first = self.task()
        second = self.task(request={"model": "fixture", "messages": [{"role": "user", "content": "different input"}], "tools": [], "max_tokens": 4096})
        references = [assess_task(self.store, first.task_id, eval_run_id="inputs", scalar_reward=1),
                      assess_task(self.store, second.task_id, eval_run_id="inputs", scalar_reward=0)]
        output = self.root / "no-pairs"
        self.assertEqual(export_dataset(self.store, output, kind="preference", evaluation_refs=references)["sample_count"], 0)

    def test_preference_does_not_compare_different_reward_configurations(self):
        first, second = self.task(), self.task(raw='{"schema_version":1,"goals":[],"actions":[]}', status="FAILED")
        references = [assess_task(self.store, first.task_id, eval_run_id="reward-config", scalar_reward=1,
                                  evaluator_config={"reward_version": "a"}),
                      assess_task(self.store, second.task_id, eval_run_id="reward-config", scalar_reward=0,
                                  evaluator_config={"reward_version": "b"})]
        result = export_dataset(self.store, self.root / "incomparable-rewards", kind="preference", evaluation_refs=references)
        self.assertEqual(result["sample_count"], 0)

    def test_sft_requires_actual_response_delivery_or_explicit_consumption(self):
        undelivered = self.task(consumed=None)
        delivered = self.task(consumed=None, delivered=True)
        references = [assess_task(self.store, task.task_id, eval_run_id="delivery")
                      for task in (undelivered, delivered)]
        output = self.root / "delivery"
        result = export_dataset(self.store, output, kind="sft", evaluation_refs=references)
        self.assertEqual(result["sample_count"], 1)
        self.assertEqual(self.files(output)[0]["samples"][0]["task_id"], delivered.task_id)

    def test_nested_experiment_rollout_group_connects_distinct_sessions_and_starts(self):
        experiments = [
            {"candidate_id": "candidate-a", "rollout_group_id": "nested-shared-rollout",
             "optimization_target": "instruction.plan", "branch_id": "branch-a", "study": "retained"},
            {"candidate_id": "candidate-b", "rollout_group_id": "nested-shared-rollout",
             "optimization_target": "environment.desktop", "branch_id": "branch-b", "study": "retained"}]
        first = self.task(initial={"step": 101}, metadata={"run_id": "nested-run-a", "session_id": "nested-session-a",
                                                          "experiment": experiments[0]})
        second = self.task(initial={"step": 202}, metadata={"run_id": "nested-run-b", "session_id": "nested-session-b",
                                                           "experiment": experiments[1]})
        output = self.root / "nested-experiment"
        manifest = export_dataset(self.store, output, kind="configuration")
        self.assertEqual(manifest["source_tasks"][0]["split"], manifest["source_tasks"][1]["split"])
        samples = {sample["task_id"]: sample for sample in self.files(output)[0]["samples"]}
        for task, experiment in ((first, experiments[0]), (second, experiments[1])):
            self.assertEqual(samples[task.task_id]["experiment"], experiment)
            for field in ("candidate_id", "rollout_group_id", "optimization_target", "branch_id"):
                self.assertEqual(samples[task.task_id][field], experiment[field])

    def test_conflicting_top_level_experiment_identity_rejects_export(self):
        self.task(metadata={"candidate_id": "top-candidate", "experiment": {"candidate_id": "nested-candidate"}})
        output = self.root / "conflict"
        with self.assertRaisesRegex(ValueError, "Conflicting candidate_id"):
            export_dataset(self.store, output, kind="configuration")
        self.assertFalse(output.exists())

    def test_shared_optimization_target_and_candidate_do_not_merge_unrelated_tasks(self):
        shared = {"candidate_id": "shared-candidate", "optimization_target": "prompts"}
        first = self.task(initial={"step": 303}, metadata={"run_id": "control-run-a", "session_id": "control-session-a",
            "task_family_id": "control-family-a", "case_id": "control-case-a", "experiment": {
                **shared, "rollout_group_id": "control-rollout-a", "branch_id": "control-branch-a"}})
        second = self.task(initial={"step": 404}, metadata={"run_id": "control-run-b", "session_id": "control-session-b",
            "task_family_id": "control-family-b", "case_id": "control-case-b", "experiment": {
                **shared, "rollout_group_id": "control-rollout-b", "branch_id": "control-branch-b"}})
        result = export_dataset(self.store, self.root / "independent-controls", kind="configuration")
        groups = {row["task_id"]: row["split"]["group_id"] for row in result["source_tasks"]}
        self.assertNotEqual(groups[first.task_id], groups[second.task_id])
        self.assertEqual(result["experiment_metadata_policy"]["grouping_fields"], ["rollout_group_id", "branch_id"])

    def tool_turn(self, *, name="observe", facts=False, failed=False, rejected=False, offered_names=("observe",)):
        identity = {"model_call_id": "preceding-tool-turn", "decision_id": "decision-1"}
        call = {"id": "native-tool-turn", "type": "function", "function": {"name": name, "arguments": "{}"}}
        tools = [{"type": "function", "function": {"name": offered_name, "parameters": {
            "type": "object", "properties": {}, "additionalProperties": False}}} for offered_name in offered_names]
        rows = [
            {"event": "model.requested", **identity, "detail": {"request": {
                "messages": [{"role": "user", "content": "pick cube"}], "tools": tools, "max_tokens": 4096}}},
            {"event": "model.responded", **identity, "source": "llm", "detail": {
                "response_text": None, "tool_calls": [call], "finish_reason": "tool_calls", "consumed": True}}]
        if facts:
            detail = {"tool_call_id": "local-tool-turn", "native_tool_call_id": "native-tool-turn",
                      "name": name, "arguments": {}}
            rows.append({"event": "tool.requested", **identity, "tool_call_id": "local-tool-turn", "detail": detail})
            rows.append({"event": "tool.finished", **identity, "tool_call_id": "local-tool-turn", "detail": {
                **detail, "result": {"ok": False, "error": {"code": "QUERY_FAILED"}} if failed else {"state": 1}}})
        if rejected:
            rows.append({"event": "proposal.rejected", **identity, "detail": {"error_code": "UNKNOWN_TOOL"}})
        return rows

    def test_recovered_success_excludes_initial_illegal_tool_turn(self):
        task = self.task(before_events=self.tool_turn(name="unregistered_tool", rejected=True))
        reference = assess_task(self.store, task.task_id, eval_run_id="recovered-illegal")
        output = self.root / "recovered-illegal"
        result = export_dataset(self.store, output, kind="sft", evaluation_refs=[reference])
        samples, qualification = self.files(output)
        self.assertEqual(result["sample_count"], 1)
        self.assertEqual(samples["samples"][0]["model_call_id"], "call-1")
        bad = next(row for row in qualification["tasks"][0]["model_calls"] if row["model_call_id"] == "preceding-tool-turn")
        self.assertFalse(bad["sft_eligible"])
        self.assertIn("proposal_rejected", bad["exclusion_reasons"])

    def test_unproven_tool_request_cannot_be_positive_sft_without_rejection_marker(self):
        task = self.task(before_events=self.tool_turn())
        reference = assess_task(self.store, task.task_id, eval_run_id="unproven-tool")
        output = self.root / "unproven-tool"
        result = export_dataset(self.store, output, kind="sft", evaluation_refs=[reference])
        self.assertEqual(result["sample_count"], 1)
        bad = self.files(output)[1]["tasks"][0]["model_calls"][0]
        self.assertIn("validated_tool_request_unavailable", bad["exclusion_reasons"])

    def test_actual_tool_facts_allow_only_successful_tool_turn_as_positive_sft(self):
        for failed in (False, True):
            with self.subTest(failed=failed):
                task = self.task(before_events=self.tool_turn(facts=True, failed=failed))
                reference = assess_task(self.store, task.task_id, eval_run_id=f"tool-{failed}")
                output = self.root / f"tool-{failed}"
                result = export_dataset(self.store, output, kind="sft", task_ids=[task.task_id], evaluation_refs=[reference])
                self.assertEqual(result["sample_count"], 1 if failed else 2)
                report = self.files(output)[1]["tasks"][0]["model_calls"][0]
                self.assertTrue(report["tool_evidence"]["requests_validated"])
                self.assertTrue(report["tool_evidence"]["completed"])
                self.assertEqual(report["tool_evidence"]["successful"], not failed)
                if failed:
                    self.assertIn("tool_execution_success_unconfirmed", report["exclusion_reasons"])

    def test_preference_excludes_missing_budget_evidence(self):
        first = self.task(runtime_config=None)
        second = self.task(raw='{"schema_version":1,"goals":[],"actions":[]}', status="FAILED")
        references = [assess_task(self.store, first.task_id, eval_run_id="missing-budget", scalar_reward=1),
                      assess_task(self.store, second.task_id, eval_run_id="missing-budget", scalar_reward=0)]
        output = self.root / "missing-budget"
        result = export_dataset(self.store, output, kind="preference", evaluation_refs=references)
        self.assertEqual(result["sample_count"], 0)
        bad = next(row for row in self.files(output)[1]["tasks"] if row["task_id"] == first.task_id)
        self.assertIn("comparable_budget_unavailable", bad["exclusion_reasons"])

    def test_same_input_with_different_runtime_or_request_budget_is_not_preference(self):
        for changed in ("runtime", "request"):
            with self.subTest(changed=changed):
                first = self.task()
                runtime = copy.deepcopy(BUDGET_CONFIG)
                runtime["task_budget"]["max_replans"] = 1 if changed == "runtime" else 4
                request = {"model": "fixture", "messages": [{"role": "user", "content": "pick cube"}],
                           "tools": [], "max_tokens": 1024 if changed == "request" else 4096}
                second = self.task(runtime_config=runtime, request=request,
                                   raw='{"schema_version":1,"goals":[],"actions":[]}', status="FAILED")
                references = [assess_task(self.store, first.task_id, eval_run_id=f"budget-{changed}", scalar_reward=1),
                              assess_task(self.store, second.task_id, eval_run_id=f"budget-{changed}", scalar_reward=0)]
                output = self.root / f"budget-{changed}"
                result = export_dataset(self.store, output, kind="preference", task_ids=[first.task_id, second.task_id],
                                        evaluation_refs=references)
                self.assertEqual(result["sample_count"], 0)

    def single_tool_task(self, *, name="observe", failed=False, rejected=False):
        task = self.store.begin(map_id="classic", map_definition={"revision": 1}, initial_state={"step": 0},
            natural_language="pick cube", versions={"runtime_config": copy.deepcopy(BUDGET_CONFIG)}, metadata={
                "trusted_task_spec": SPEC, "reset_spec": {"map_id": "classic", "seed": 0}, "reset_verified": True})
        task.append_attempt()
        for row in self.tool_turn(name=name, facts=not rejected, failed=failed, rejected=rejected,
                                  offered_names=("observe", "inspect")):
            task.append_event(row)
        success = not failed and not rejected
        task.finish(final_state={"step": 5}, outcome={"status": "SUCCESS" if success else "FAILED",
            "executor_verified": success, "agent_interpreted_goals": GOALS, "goal_check": {"passed": success}})
        return task

    def test_preference_allows_proven_tool_failure_only_as_rejected_and_excludes_illegal_turn(self):
        for illegal in (False, True):
            with self.subTest(illegal=illegal):
                good = self.single_tool_task()
                bad = self.single_tool_task(name="unregistered" if illegal else "inspect", failed=not illegal, rejected=illegal)
                references = [assess_task(self.store, good.task_id, eval_run_id=f"tool-pref-{illegal}", scalar_reward=1),
                              assess_task(self.store, bad.task_id, eval_run_id=f"tool-pref-{illegal}", scalar_reward=0)]
                output = self.root / f"tool-pref-{illegal}"
                result = export_dataset(self.store, output, kind="preference", task_ids=[good.task_id, bad.task_id],
                                        evaluation_refs=references)
                self.assertEqual(result["sample_count"], 0 if illegal else 1)
                if not illegal:
                    self.assertEqual(self.files(output)[0]["samples"][0]["task_ids"], [good.task_id, bad.task_id])

    def test_identical_assistant_output_with_different_rewards_is_not_preference(self):
        first, second = self.task(), self.task()
        references = [assess_task(self.store, first.task_id, eval_run_id="identical-output", scalar_reward=1),
                      assess_task(self.store, second.task_id, eval_run_id="identical-output", scalar_reward=0)]
        result = export_dataset(self.store, self.root / "identical-output", kind="preference", evaluation_refs=references)
        self.assertEqual(result["sample_count"], 0)

    def test_recovered_success_with_missing_rejection_record_is_analysis_only_even_with_positive_evaluation(self):
        task = self.task(before_events=self.tool_turn(name="unregistered"), recording_errors=[{
            "event": "proposal_rejected", "message": "rejection recording failed before recovery"}])
        evaluator = EvaluationStore(self.store.root, task_store=self.store)
        reference = evaluator.save(task.task_id, eval_run_id="caller-positive", evaluator_name="caller",
            evaluator_version="1", components={"intent_correctness": True, "execution_success": True},
            scalar_reward=1, validity={"valid": True, "reasons": []})
        self.assertTrue(evaluator.load(reference)["validity"]["valid"])
        for kind in ("sft", "preference", "task_pool", "trajectory", "configuration", "decision"):
            with self.subTest(kind=kind):
                output = self.root / f"gap-{kind}"
                manifest = export_dataset(self.store, output, kind=kind, evaluation_refs=[reference])
                samples, report = self.files(output)
                qualification = report["tasks"][0]
                self.assertFalse(qualification["recording_evidence_complete"])
                self.assertIn("recording_evidence_incomplete", qualification["rl_update"]["exclusion_reasons"])
                if kind in {"sft", "preference", "task_pool"}:
                    self.assertEqual(manifest["sample_count"], 0)
                    self.assertFalse(qualification["eligible"])
                    self.assertIn("recording_evidence_incomplete", qualification["exclusion_reasons"])
                else:
                    self.assertEqual(manifest["sample_count"], 1)
                    self.assertTrue(qualification["analysis_only"])
                    self.assertFalse(samples["samples"][0]["recording_evidence_complete"])
                if kind == "decision":
                    self.assertFalse(samples["samples"][0]["availability"]["recording_evidence_complete"])
                    self.assertEqual(samples["samples"][0]["transition_status"], "incomplete_evidence_analysis_only")


if __name__ == "__main__":
    unittest.main()
