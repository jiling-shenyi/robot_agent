"""Shared, GUI-independent demo sessions and reproducible case execution.

Agent adapters receive the live session. Only selecting/resetting a map or running
a registered test case resets physics; free-form agent instructions continue from
the currently displayed state. Batch sessions use private map copies.
"""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import mujoco

from embodied_agent.contracts import ContractError
from embodied_agent.evaluation.evidence import jsonable
from embodied_agent.evaluation.task_writer import TaskRunWriter, recordable
from embodied_agent.models.tracing import capture_model_requests, trace_scope, record_model_event, new_trace_id
from embodied_agent.models.budget import TaskBudget, budget_context
from embodied_agent.context import AgentContext
from embodied_agent.prompts import PromptCatalog
from embodied_agent.tools import QueryToolCatalog
from embodied_agent.agents.map_environment import UnifiedEnvironmentAgent
from embodied_agent.maps.unified_store import UnifiedMapStore, DemoWorld, parse_demo_world
from embodied_agent.simulation.live_episode import DemoEpisode, FrameCallback, StatusCallback, create_demo_episode
from embodied_agent.evaluation.cases import check_expected, validate_case
from embodied_agent.evaluation.batch import evaluate_session
from embodied_agent.evaluation.provenance import source_hashes
from embodied_agent.paths import PROJECT_ROOT as ROOT

AgentAdapter = Callable[["DemoSession", str], dict[str, Any]]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def _robot_provenance(root: Path) -> dict[str, Any]:
    directories = {"panda": root / "assets" / "third_party" / "franka_emika_panda",
                   "stretch": root / "assets" / "third_party" / "hello_robot_stretch"}
    result = {}
    for name, directory in directories.items():
        files = sorted(path for path in directory.rglob("*") if path.is_file())
        if name == "panda":
            files += sorted(path for path in (root / "assets" / "scene").rglob("*") if path.is_file())
        hashes = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
        provenance = directory / "MODEL_PROVENANCE.json"
        result[name] = {"asset_directory": directory.relative_to(root).as_posix(),
            "asset_hashes": hashes, "license": (directory / "LICENSE").relative_to(root).as_posix(),
            "upstream": _read_json(provenance) if provenance.exists() else {"source": "https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda"}}
    return result


class _SessionWriter(TaskRunWriter):
    def __init__(self, session: "DemoSession"):
        super().__init__(session.output_dir, session.run_id,
                         records_dir=session.records_dir, versions=session.record_versions(), source_root=session.root)
        self.session = session


class DemoSession:
    """One selected map, live simulator and pluggable natural-language agents."""

    def __init__(self, *, root: Path = ROOT, map_dir: Path | None = None,
                 output_dir: Path | None = None, records_dir: Path | None = None, planner_kind: str = "llm",
                 environment_mode: str = "llm", on_frame: FrameCallback | None = None,
                 on_status: StatusCallback | None = None, _allow_batch_maps: bool = False,
                 runtime_config: dict | None = None, prompt_catalog=None, tool_catalog=None,
                 memory_store=None, knowledge_base=None):
        self.root = Path(root).resolve()
        self.run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.output_dir = Path(output_dir or self.root / "results" / "demo" / self.run_id).resolve()
        self.records_dir = Path(records_dir or self.root / "records").resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        existing = list(self.output_dir.iterdir())
        if existing and not (_allow_batch_maps and len(existing) == 1 and
                             existing[0].name == "maps" and existing[0].is_dir()):
            raise ValueError(f"Output directory already contains data: {self.output_dir}")
        self.store = UnifiedMapStore(Path(map_dir or self.root / "configs" / "maps"))
        map_root = Path(self.store.root).resolve()
        if self.records_dir.is_relative_to(map_root) or map_root.is_relative_to(self.records_dir):
            raise ValueError("Task record and map directories must not overlap")
        self.runtime = (copy.deepcopy(runtime_config) if runtime_config is not None else
                        _read_json(self.root / "configs" / "agent_runtime.json"))
        self.prompt_catalog = prompt_catalog if prompt_catalog is not None else PromptCatalog(
            self.runtime, project_root=self.root)
        self.tool_catalog = tool_catalog if tool_catalog is not None else QueryToolCatalog(self.runtime)
        self.contexts = {role: AgentContext(self.runtime, role=role,
            namespace=f"{self.run_id}:{role}", memory_store=memory_store,
            knowledge_base=knowledge_base, project_root=self.root)
            for role in ("instruction", "environment")}
        self.thresholds = _read_json(self.root / "configs" / "m2_thresholds.json")
        self.on_frame, self.on_status = on_frame, on_status
        self.planner_kind = planner_kind
        from embodied_agent.agents.instruction import InstructionAgent
        self.instruction_agent = InstructionAgent(self.runtime, kind=planner_kind,
            prompt_catalog=self.prompt_catalog, tool_catalog=self.tool_catalog,
            context_provider=self.contexts["instruction"])
        self.active_task_budget = None
        self._active_instruction_adapter = None
        self.environment_agent = UnifiedEnvironmentAgent(self.store, mode=environment_mode, config=self.runtime,
            prompt_catalog=self.prompt_catalog, tool_catalog=self.tool_catalog,
            context_provider=self.contexts["environment"], recording_enabled=False)
        self.episode: Any | None = None
        self.world: DemoWorld | None = None
        self._episode_serial = 0
        # Case initial_overrides affect only the displayed scene. Keep the
        # storage snapshot they were based on so a later environment edit can
        # distinguish that temporary scene from an external map change.
        self._case_override_base: DemoWorld | None = None
        self.results: list[dict[str, Any]] = []
        self.action_count = 0
        self.agents: dict[str, AgentAdapter] = {}
        self.register_agent("robot", lambda session, instruction: session._run_instruction(instruction))
        self.register_agent("environment", lambda session, instruction: session._edit_environment(instruction))
        self.batch_isolated = False
        self._batch_baselines: dict[str, dict[str, Any]] = {}
        self.source_map_dir: str | None = None
        self._source_hashes = source_hashes(self.root)
        self._robot_models = _robot_provenance(self.root)
        self._capability_scope = {"panda": "Shared instruction Agent over measured simulator state; top-grasp cube manipulation and dynamically named tabletop regions, monitored execution and independent support verification.",
            "stretch": "Shared instruction Agent over measured simulator state; geometry-generated manipulation candidates, known-map navigation with dynamic path review, supported horizontal surfaces and finite robot grasp/reach/payload limits. Physical contact, release and stable support are verified. No visual recognition, throwing, door/appliance or fluid/material simulation."}
        self.writer = _SessionWriter(self)
        _write_json(self.output_dir / "manifest.json", {
            "schema_version": 2, "run_id": self.run_id, "phase": "demo",
            "planner_kind": planner_kind, "environment_mode": environment_mode,
            "mujoco_version": mujoco.__version__, "maps_artifact_ref": self.writer.store.put_artifact([w.to_dict() for w in self.store.list_maps()]),
            "source_hashes": self._source_hashes, "robot_models": self._robot_models,
            "agent_components_ref": self.writer.store.put_artifact(self.record_versions()["agent_components"]),
            "robot_capability_scope": self._capability_scope,
            "task_record_schema_version": 2, "task_records_dir": str(self.records_dir),
            "run_record_uri": f"runs/{self.run_id}/manifest.json", "session_id": self.writer.session_id,
        })

    def record_versions(self) -> dict[str, Any]:
        prompts = {name: self.prompt_catalog.get(name).to_dict() for name in
                   ("instruction.intent", "instruction.plan", "environment.desktop", "environment.home")}
        source_ref = (getattr(self, "writer", None).versions.get("source_snapshot_ref")
                      if getattr(self, "writer", None) is not None else None)
        return {"source_hashes": self._source_hashes, "source_snapshot_ref": source_ref, "mujoco": mujoco.__version__,
                "planner_kind": self.planner_kind,
                "instruction_prompt_version": prompts["instruction.plan"]["version"],
                "instruction_runtime_version": "universal-instruction-v1",
                "agent_components": {"prompts": prompts,
                    "tools": {"fingerprint": self.tool_catalog.fingerprint,
                              "schemas": self.tool_catalog.schemas},
                    "context": {role: context.describe() for role, context in self.contexts.items()}},
                "robot_models": self._robot_models,
                "runtime_config": recordable(self.runtime), "threshold_config": recordable(self.thresholds),
                "runtime_config_sha256": hashlib.sha256(
                    json.dumps(self.runtime, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest(),
                "threshold_config_sha256": hashlib.sha256(
                    (self.root / "configs" / "m2_thresholds.json").read_bytes()).hexdigest()}

    def register_agent(self, name: str, handler: AgentAdapter) -> None:
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError("Agent name must be a lower-case identifier")
        if not callable(handler):
            raise ValueError("Agent handler must be callable")
        self.agents[name] = handler

    def map_evidence(self) -> dict[str, Any]:
        evidence = {"map_id": self.world.map_id if self.world else None,
                "map_revision": self.world.revision if self.world else None,
                "map_schema_kind": getattr(self.world, "schema_kind", "desktop") if self.world else None,
                "map_snapshot": self.world.to_dict() if self.world else None}
        if self.episode is not None and getattr(self.episode, "robot_kind", "panda") == "stretch":
            evidence["world_snapshot"] = self.episode.snapshot()
        return evidence

    def _show_world(self, world: DemoWorld) -> Any:
        self.writer.run.append("environment.reload.started", {"map_id": world.map_id,
            "map_ref": self.writer.store.put_artifact(world.to_dict()), "previous_instance_id": f"robot-{self._episode_serial:04d}"})
        old_episode = self.episode
        episode = create_demo_episode(
            world, self.thresholds, self.output_dir, root=self.root,
            on_frame=lambda ep: self.on_frame(ep) if self.on_frame is not None else None,
            on_status=lambda title, detail: self.on_status(title, detail) if self.on_status is not None else None,
        )
        # Showing an edited unsafe map must be possible. Action initialization and
        # every physics step still execute the original safety guard.
        if old_episode is not None:
            old_episode.close()
        episode.prepare(check_safety=False)
        self.world, self.episode = world, episode
        self._episode_serial += 1
        self._last_reload_recording_errors = []
        try:
            self.writer.run.append("environment.reload.finished", {"map_id": world.map_id,
                "environment_instance_id": f"{self.run_id}:robot-{self._episode_serial:04d}",
                "observation_ref": self.writer.store.put_artifact(recordable(episode.snapshot())),
                "checkpoint_availability": "unavailable", "restorable": False})
        except Exception as error:
            # The new episode is already active. Preserve that effect even if
            # publishing its audit event fails.
            self._last_reload_recording_errors.append({"event": "environment.reload.finished", "message": str(error)})
        if self.on_frame is not None:
            self.on_frame(episode)
        return episode

    def select_map(self, map_id: str) -> Any:
        self.writer.run.append("environment.map.selected", {"map_id": map_id})
        episode = self._show_world(self.store.load(map_id))
        self._case_override_base = None
        return episode

    def reset(self) -> Any:
        if self.world is None:
            raise ValueError("Select a map before resetting the environment")
        self.writer.run.append("environment.reset.requested", {"map_id": self.world.map_id,
            "environment_instance_id": f"{self.run_id}:robot-{self._episode_serial:04d}"})
        return self.select_map(self.world.map_id)

    def _require_episode(self) -> Any:
        if self.episode is None or self.world is None:
            raise ValueError("Select a map before running an agent")
        return self.episode


    def _run_instruction(self, instruction: str) -> dict[str, Any]:
        from embodied_agent.execution.supervisor import EmbodiedTaskRunner
        from embodied_agent.execution.panda_adapter import PandaInstructionAdapter
        episode = self._require_episode()
        adapter = episode if episode.robot_kind == "stretch" else PandaInstructionAdapter(episode, self.runtime)
        self._active_instruction_adapter = adapter
        runner = EmbodiedTaskRunner(self.instruction_agent, self.active_task_budget,
            record_event=self.writer.record_event, on_attempt=self.writer.begin_recovery_attempt,
            max_rounds=int(self.runtime.get("execution", {}).get("max_decision_rounds", 8)),
            max_recoveries=int(self.runtime.get("task_budget", {}).get("max_replans", 4)))
        result = runner.run(instruction, adapter, self.world, task_context={
            "episode_id": f"robot-{self._episode_serial:04d}", "map_revision": self.world.revision})
        budget = self.active_task_budget.snapshot() if self.active_task_budget is not None else {}
        dialogue = getattr(self.instruction_agent, "last_dialogue", {}) or {}
        for key in ("dialogue_messages", "tool_results", "raw_model_response"):
            if key in dialogue:
                result[key] = dialogue[key]
        record = self.writer.active_record
        query_calls = sum(event.get("event") == "agent_tool_call"
                          for attempt in record.data["attempts"] for event in attempt["events"]) if record else 0
        result.update(llm_request_count=budget.get("requests", 0), query_tool_count=query_calls,
                      tool_call_count=query_calls,
                      token_usage={key: budget.get(key, 0) for key in ("prompt_tokens", "completion_tokens", "total_tokens")})
        result.update(robot_kind=episode.robot_kind, planner_kind=self.planner_kind,
                      instruction_agent="universal-instruction-v1")
        targets = {g["target_id"] for g in (result.get("goals") or [])
                   if g.get("target_id") and g.get("predicate") in {"inside", "supported_on"}}
        if len(targets) == 1:
            result["target_id"] = next(iter(targets))
        return result

    def cancel_task(self, reason: str = "User cancelled the task", *, code: str = "TASK_CANCELLED") -> None:
        if self.active_task_budget is not None:
            self.active_task_budget.cancel(reason, code=code)

    def pump_planning_hold(self, seconds: float = .02) -> None:
        """Physics-thread heartbeat while a language worker waits for a reply."""
        if self._active_instruction_adapter is None:
            return
        if self.active_task_budget is not None:
            self.active_task_budget.check()
        episode = self.episode
        callback = getattr(episode, "on_frame", None)
        episode.on_frame = None
        try:
            self._active_instruction_adapter.hold(seconds=seconds)
        finally:
            episode.on_frame = callback

    def _run_robot_plan(self, plan: dict, *, case_id: str, test_fault_protocol: str | None = None) -> dict:
        episode = self._require_episode()
        if getattr(episode, "robot_kind", "panda") != "stretch":
            raise ContractError("INCOMPATIBLE_PLAN", "Registered robot plans require the mobile robot map")
        result = episode.run_plan(plan, case_id=case_id, fault_protocol=test_fault_protocol,
                                  record_event=self.writer.record_event)
        result["robot_kind"] = "stretch"
        result["robot_episode_id"] = f"robot-{self._episode_serial:04d}"
        return result

    def _physical_boundary(self) -> dict[str, Any]:
        episode = self.episode
        return {"episode": episode, "step": episode.total_steps if episode else 0,
                "events": len(episode.events) if episode else 0,
                "events_ref": episode.events if episode else None,
                "physics_events": len(episode.sim.events) if episode and hasattr(episode, "sim") else 0,
                "samples": len(episode.samples) if episode else 0,
                "samples_ref": episode.samples if episode else None,
                "episode_serial": self._episode_serial}

    def _task_execution(self, before: dict[str, Any]) -> dict[str, Any]:
        episode = self.episode
        if episode is None:
            return {"episode_events": [], "physics_events": [], "trajectory": [], "range": None}
        same = episode is before["episode"]
        # A new map episode starts new arrays; free tasks retain cumulative
        # telemetry. Capture only events/samples within the current task.
        cumulative = same and episode.events is before["events_ref"]
        event_start = before["events"] if cumulative else 0
        physics_start = before["physics_events"] if same else 0
        sample_start = before["samples"] if same and episode.samples is before["samples_ref"] else 0
        sim_events = episode.sim.events if hasattr(episode, "sim") else []
        return recordable({"episode_events": episode.events[event_start:],
            "physics_events": sim_events[physics_start:], "trajectory": episode.samples[sample_start:],
            "range": {"episode_id": f"robot-{self._episode_serial:04d}",
                      "episode_changed": not same, "start_step": before["step"] if cumulative else 0,
                      "end_step": episode.total_steps,
                      "events_start": event_start, "events_end": len(episode.events),
                      "physics_events_start": physics_start, "physics_events_end": len(sim_events),
                      "samples_start": sample_start, "samples_end": len(episode.samples)}})

    def _edit_environment(self, instruction: str) -> dict[str, Any]:
        self._require_episode()
        stored = self.store.load(self.world.map_id)
        case_base = self._case_override_base
        expected = case_base if case_base is not None else self.world
        if stored.to_dict() != expected.to_dict():
            raise ContractError("REVISION_CONFLICT", "The stored map changed since this scene was loaded. Reset the environment before editing.")
        # A case override is temporary: edits always start from the selected
        # persisted map. The store's revision check also guards a concurrent
        # change between this comparison and the save.
        result = self.environment_agent.apply(self.world.map_id, instruction, expected_revision=expected.revision)
        committed = result.to_dict()
        committed.update(getattr(self.environment_agent, "last_dialogue", {}) or {})
        if "tool_call_count" in committed:
            committed["query_tool_count"] = committed["tool_call_count"]
        if case_base is not None:
            committed["auto_reset_from_case"] = True
        try:
            record_model_event("environment_reload_started", {"map_id": self.world.map_id,
                "stored_map_after": committed["after"], "previous_environment_instance_id": f"{self.run_id}:robot-{self._episode_serial:04d}"})
        except Exception as error:
            committed["reload_status"] = "not_started_due_to_recording_failure"
            committed.setdefault("recording_errors", []).append({"event": "scene.reload.started", "message": str(error)})
            return committed
        try:
            self.reset()
            committed["reload_status"] = "success"
            if getattr(self, "_last_reload_recording_errors", None):
                committed.setdefault("recording_errors", []).extend(self._last_reload_recording_errors)
            if case_base is not None and self.on_status is not None:
                self.on_status("环境编辑", "已从用例临时布局恢复所选地图并保存修改")
        except Exception as error:
            closed = type(error).__name__ in {"ViewerClosed"}
            committed.update(status="ABORTED" if closed else "FAILED",
                             error_code="VIEWER_CLOSED" if closed else "REFRESH_FAILED",
                             error_message="The initial map was saved, but refreshing its display failed: " + str(error),
                             refresh_error={"type": type(error).__name__, "message": str(error)})
            committed["reload_status"] = "failed"
            try:
                record_model_event("environment_reload_failed", {"reload_status": "failed", "persisted": True,
                    "stored_map_after": committed["after"], "error_code": committed["error_code"], "message": str(error)})
            except Exception as recording_error:
                committed.setdefault("recording_errors", []).append({"event": "scene.reload.failed", "message": str(recording_error)})
        if committed["reload_status"] == "success":
            try:
                record_model_event("environment_reload_finished", {"map_id": self.world.map_id,
                    "environment_instance_id": f"{self.run_id}:robot-{self._episode_serial:04d}",
                    "displayed_state_after": recordable(self.episode.snapshot()), "reload_status": "success"})
            except Exception as recording_error:
                committed.setdefault("recording_errors", []).append({"event": "scene.reload.finished", "message": str(recording_error)})
        return committed

    def _restore_batch_baseline(self, map_id: str) -> None:
        """Restore only private batch copies; all steps within a case share edits."""
        baseline = self._batch_baselines.get(map_id)
        if not self.batch_isolated or baseline is None:
            return
        path = Path(self.store.root) / f"{map_id}.json"
        if self.store.load(map_id).to_dict() == baseline:
            return
        # These files belong exclusively to this batch. Preserve the baseline's
        # revision as well as geometry so reruns are independent of case order.
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".baseline-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            _write_json(temporary, baseline)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def run_agent(self, instruction: str, agent: str = "robot", *, record: bool = True,
                  trusted_task_spec: dict | None = None, task_family_id: str | None = None,
                  experiment: dict | None = None) -> dict[str, Any]:
        return self._execute_action(instruction, agent, record=record,
            trusted_task_spec=trusted_task_spec, task_family_id=task_family_id, experiment=experiment)

    def _execute_action(self, instruction: str, agent: str, *, record: bool = True,
                        robot_plan: dict | None = None, test_fault_protocol: str | None = None,
                        case_id: str | None = None, trusted_task_spec: dict | None = None,
                        task_family_id: str | None = None, experiment: dict | None = None) -> dict[str, Any]:
        try:
            return self._execute_action_impl(instruction, agent, record=record,
                robot_plan=robot_plan, test_fault_protocol=test_fault_protocol, case_id=case_id,
                trusted_task_spec=trusted_task_spec, task_family_id=task_family_id, experiment=experiment)
        finally:
            if self.active_task_budget is not None:
                self.active_task_budget.cancel("Task lifecycle ended; ignore late worker responses")
            self.active_task_budget = self._active_instruction_adapter = None

    def _execute_action_impl(self, instruction: str, agent: str, *, record: bool = True,
                        robot_plan: dict | None = None, test_fault_protocol: str | None = None,
                        case_id: str | None = None, trusted_task_spec: dict | None = None,
                        task_family_id: str | None = None, experiment: dict | None = None) -> dict[str, Any]:
        self.action_count += 1
        start = self.map_evidence()
        started = time.perf_counter()
        boundary = self._physical_boundary()
        initial_observation_error = None
        try:
            initial_state = self.episode.snapshot() if self.episode is not None else None
        except Exception as error:
            initial_observation_error = error
            initial_state = {"unavailable": "initial_observation_failed", "message": str(error)}
        # File-backed prompt candidates may change between tasks. The task
        # keeps a fresh component snapshot; events identify each actual call.
        self.writer.versions = self.record_versions()
        self.writer.begin_task(instruction=instruction, map_id=start["map_id"] or "unselected",
            map_definition=start["map_snapshot"], initial_state=initial_state,
            metadata={"agent": agent, "case_id": case_id, "action_index": self.action_count,
                      "robot_kind": getattr(self.episode, "robot_kind", None),
                      "capability_scope": self._capability_scope.get(getattr(self.episode, "robot_kind", "")),
                      "rule_version": getattr(self.world, "rule_version", None),
                      "task_kind": "map_edit" if agent == "environment" else "robot_task",
                      "planner_source": "registered_case_plan" if robot_plan is not None else
                          self.environment_agent.mode if agent == "environment" else self.planner_kind,
                      "environment_instance_id": f"{self.run_id}:robot-{self._episode_serial:04d}",
                      "trusted_task_spec": trusted_task_spec, "task_family_id": task_family_id,
                      "experiment": experiment, "memory_namespaces": {role: context.namespace for role, context in self.contexts.items()
                          if context.memory_enabled},
                      "displayed_state_before": initial_state, "case_override": self._case_override_base is not None,
                      "batch_isolated": self.batch_isolated})
        task_budget = TaskBudget(self.runtime)
        self.active_task_budget = task_budget
        try:
            if initial_observation_error is not None:
                raise initial_observation_error
            self._require_episode()
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError("Instruction must be a nonempty string")
            if agent not in self.agents:
                raise ValueError(f"Unknown agent: {agent}")
            if robot_plan is not None:
                if agent not in {"robot"}:
                    raise ContractError("INVALID_CASE_PLAN", "Robot plans belong to the existing robot agent")
                with capture_model_requests(self.writer.model_recorder()), trace_scope(**self.writer.trace_identity()):
                    result = self._run_robot_plan(robot_plan, case_id=case_id or f"{self.run_id}-{self.action_count:04d}", test_fault_protocol=test_fault_protocol)
                result["planner_kind"] = "registered_case_plan"
            else:
                if agent == "environment":
                    self.environment_agent.last_dialogue = {}
                with budget_context(task_budget), capture_model_requests(self.writer.model_recorder()), trace_scope(**self.writer.trace_identity()):
                    result = self.agents[agent](self, instruction)
            if not isinstance(result, dict) or result.get("status") not in {"SUCCESS", "FAILED", "ABORTED"}:
                raise ValueError("Agent adapter must return a result with SUCCESS, FAILED or ABORTED status")
        except Exception as error:
            closed = type(error).__name__ in {"ViewerClosed"} or getattr(error, "code", None) == "TASK_CANCELLED"
            result = {"status": "ABORTED" if closed else "FAILED",
                      "error_code": "VIEWER_CLOSED" if closed else getattr(error, "code", "DEMO_ERROR"),
                      "error_message": str(error), "error_details": getattr(error, "details", {})}
            if self._active_instruction_adapter is not None:
                try:
                    self._active_instruction_adapter.safe_stop()
                except Exception as stop_error:
                    result.setdefault("cleanup_errors", []).append({"stage": "safe_stop", "message": str(stop_error)})
            if agent == "environment":
                result.update(getattr(self.environment_agent, "last_dialogue", {}) or {})
                if "tool_call_count" in result:
                    result["query_tool_count"] = result["tool_call_count"]
            details = getattr(error, "details", {})
            if getattr(error, "recording_errors", None):
                result.setdefault("recording_errors", []).extend(error.recording_errors)
            if hasattr(error, "commit_status"):
                result["commit_status"] = error.commit_status
            if hasattr(error, "observed_persisted_map"):
                result["stored_map_observed_after_error"] = error.observed_persisted_map
            for key in ("llm_request_count", "request_count", "query_tool_count", "tool_call_count",
                        "dialogue_messages", "messages", "tool_results", "raw_model_response"):
                if key in details:
                    result[key] = details[key]
        if (self.on_frame is not None and self.episode is not None
                and result["status"] != "ABORTED" and "refresh_error" not in result):
            try:
                self.on_frame(self.episode)
            except Exception as error:
                closed = type(error).__name__ in {"ViewerClosed"}
                result.update(status="ABORTED" if closed else "FAILED",
                              error_code="VIEWER_CLOSED" if closed else "DISPLAY_ERROR", error_message=str(error))
        result = jsonable({**result, "agent": agent, "instruction": instruction,
                           "map_before": start, **self.map_evidence()})
        result.setdefault("wall_time_s", time.perf_counter() - started)
        try:
            final_state = self.episode.snapshot() if self.episode is not None else None
        except Exception as error:
            final_state = {"unavailable": "final_observation_failed", "message": str(error)}
            result.setdefault("cleanup_errors", []).append({"stage": "final_observation", "message": str(error)})
        try:
            execution = self._task_execution(boundary)
        except Exception as error:
            execution = {"episode_events": [], "physics_events": [], "trajectory": [], "range": None}
            result.setdefault("cleanup_errors", []).append({"stage": "execution_evidence", "message": str(error)})
        result["task_budget"] = task_budget.snapshot()
        if getattr(self.episode, "robot_kind", None) == "stretch":
            result.setdefault("robot_kind", "stretch")
            result.setdefault("transport_success", False)
            result.setdefault("final_snapshot", final_state)
            result["robot_episode_id"] = f"robot-{self._episode_serial:04d}"
            result["physical_evidence"] = {"directory": str(self.records_dir),
                "file": self.writer.active_record.path.name, "task_id": self.writer.active_record.task_id,
                "event_count": len(execution["episode_events"]),
                "physics_event_count": len(execution["physics_events"]),
                "trajectory_samples": len(execution["trajectory"]), "episode_id": result["robot_episode_id"]}
        try:
            context = self.contexts.get("instruction" if agent == "robot" else agent)
            if context is not None:
                try:
                    with capture_model_requests(self.writer.model_recorder()), trace_scope(**self.writer.trace_identity()):
                        context.remember(instruction, result, task_id=self.writer.active_record.task_id,
                                         map_id=result.get("map_id"))
                except Exception as error:
                    result.setdefault("component_errors", []).append({"component": "memory_write",
                        "type": type(error).__name__, "message": str(error)})
            if agent == "environment" and result.get("operations") is not None:
                self.writer.record_event("environment_plan", {"plan": {
                    "schema_version": 1, "base_revision": result.get("before", {}).get("revision"),
                    "operations": result["operations"]}}, initial_state)
            self.writer.finish_task(result, final_state=final_state, actual_execution=execution)
        except (OSError, ValueError) as error:
            # Never retry physical work because persistence failed. The last
            # atomic record remains RUNNING and the caller gets the real result.
            task = self.writer.active_record
            result["recording_error"] = {"type": type(error).__name__, "message": str(error)}
            if task is not None:
                result.update(task_id=task.task_id, task_record_path=str(task.path))
            self.writer.active_record, self.writer.active_attempt = None, None
        if record:
            self.results.append(result)
        return result

    def edit_environment(self, instruction: str) -> dict[str, Any]:
        return self.run_agent(instruction, "environment")

    def run_case(self, case: dict[str, Any]) -> dict[str, Any]:
        """Run an explicitly reset case and separately report actual/expected status."""
        validate_case(case, set(self.agents))
        actions: list[dict[str, Any]] = []
        try:
            map_id = case.get("map_id") or (self.world.map_id if self.world else "classic")
            self._restore_batch_baseline(map_id)
            self.select_map(map_id)
            overrides = case.get("initial_overrides")
            if overrides:
                case_base = self.world
                world_data = self.world.to_dict()
                world_data.update(copy.deepcopy(overrides))
                self._show_world(parse_demo_world(world_data))
                self._case_override_base = case_base
            if "legacy_scenario_id" in case:
                self.episode.scenario.update(scenario_id=case["legacy_scenario_id"], seed=case["legacy_seed"])
            initial_map = self.map_evidence()
            for step in case.get("steps", [case]):
                action = self._execute_action(step["instruction"], step.get("agent", "robot"), record=False,
                    robot_plan=step.get("robot_plan"), test_fault_protocol=step.get("test_fault_protocol"), case_id=case["case_id"],
                    trusted_task_spec=step.get("trusted_task_spec", case.get("trusted_task_spec")),
                    task_family_id=case.get("task_family_id", case["case_id"]), experiment=case.get("experiment"))
                action["expected"] = step.get("expected", {"status": "SUCCESS"})
                action["expected_pass"], action["expectation_failures"] = check_expected(action, action["expected"])
                actions.append(action)
                if not action["expected_pass"] or action["status"] == "ABORTED":
                    break
            final = actions[-1]
            expected_pass = all(a["expected_pass"] for a in actions) and len(actions) == len(case.get("steps", [case]))
            result = {"case_id": case["case_id"], "name": case.get("name", case["case_id"]),
                      "status": final["status"], "error_code": final.get("error_code"),
                      "error_message": final.get("error_message"), "expected_pass": expected_pass, "passed": expected_pass,
                      "expectation_failures": [failure for a in actions for failure in a["expectation_failures"]],
                      "initial_map": initial_map, "actions": actions, **self.map_evidence()}
            if getattr(self.episode, "robot_kind", "panda") == "stretch":
                result.update(robot_kind="stretch", transport_success=result["status"] == "SUCCESS" and any(a.get("transport_success", False) for a in actions),
                    test_fault_protocol=next((a.get("test_fault_protocol") for a in actions if a.get("test_fault_protocol")), None),
                    physical_evidence=final.get("physical_evidence"), final_snapshot=final.get("final_snapshot"),
                    error_evidence=final.get("error_evidence", {}))
        except Exception as error:
            closed = type(error).__name__ in {"ViewerClosed"}
            result = {"case_id": case["case_id"], "status": "ABORTED" if closed else "FAILED",
                      "expected_pass": False, "passed": False,
                      "error_code": "VIEWER_CLOSED" if closed else getattr(error, "code", "CASE_ERROR"), "error_message": str(error),
                      "actions": actions, **self.map_evidence()}
        self.results.append(result)
        return result

    def mark_not_run(self, cases: Iterable[dict[str, Any]], reason: str = "RUN_INTERRUPTED") -> None:
        for case in cases:
            result = {"case_id": case["case_id"], "status": "NOT_RUN", "error_code": reason,
                      "error_message": "Not started because the batch was interrupted",
                      "expected_pass": False, "passed": False, "actions": []}
            self.results.append(result)

    def record_case(self, result: dict[str, Any]) -> None:
        self.results.append(result)

    def configure_batch(self, source: Path) -> None:
        self.batch_isolated = True
        self._batch_baselines = {world.map_id: world.to_dict() for world in self.store.list_maps()}
        self.source_map_dir = str(source)

    def finish(self) -> dict[str, Any]:
        passed = sum(bool(r.get("expected_pass", r["status"] == "SUCCESS")) for r in self.results)
        summary = {"schema_version": 1, "run_id": self.run_id, "phase": "demo",
                   "planner_kind": self.planner_kind, "batch_isolated": self.batch_isolated,
                   "source_map_dir": self.source_map_dir, "episode_count": len(self.results),
                   "case_count": len(self.results), "passed_count": passed,
                   "failed_count": len(self.results) - passed,
                   "success_count": sum(r["status"] == "SUCCESS" for r in self.results),
                   "aborted_count": sum(r["status"] == "ABORTED" for r in self.results),
                   "not_run_count": sum(r["status"] == "NOT_RUN" for r in self.results),
                   "transport_success_count": sum(bool(r.get("transport_success")) for r in self.results),
                   "expected_rejection_count": sum(bool(r.get("expected_pass")) and r["status"] == "FAILED" and not r.get("test_fault_protocol") for r in self.results),
                   "expected_fault_count": sum(bool(r.get("expected_pass")) and r["status"] == "FAILED" and bool(r.get("test_fault_protocol")) for r in self.results),
                   "all_pass": bool(self.results) and passed == len(self.results), "results": self.results}
        summary["task_records_dir"] = str(self.records_dir)
        # A free robot task's actions are skill results; a case's actions are
        # task results. Only the latter should be flattened for task refs.
        task_results = [action for row in self.results
            for action in ([row] if "agent" in row or "task_id" in row else row.get("actions", [row]))]
        summary["task_records"] = [action["task_record"] for action in task_results
            if "task_record" in action]
        summary["recording_failure_count"] = sum(bool(action.get("recording_error") or action.get("recording_errors"))
            for action in task_results)
        if summary["recording_failure_count"]:
            summary["all_pass"] = False
        report = {**summary, "schema_version": 2,
                  "results": [TaskRunWriter.report_result(row) for row in self.results]}
        _write_json(self.output_dir / "summary.json", report)
        if getattr(self, "writer", None) is not None:
            self.writer.run.append("run.summary.written", {"report": str(self.output_dir / "summary.json"),
                "task_records": summary["task_records"]})
        return summary

    def close(self) -> None:
        if self.episode is not None:
            self.episode.close()
        if getattr(self, "writer", None) is not None:
            self.writer.close()


def create_batch_session(*, root: Path = ROOT, map_dir: Path | None = None,
                         output_dir: Path | None = None, **kwargs: Any) -> DemoSession:
    root = Path(root).resolve()
    if output_dir is None:
        kwargs.setdefault("records_dir", root / "records")
    source = Path(map_dir or root / "configs" / "maps").resolve()
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(output_dir or root / "results" / "demo" / run_id).resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Batch output and source map directories must not overlap")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory already contains data: {output}")
    output.mkdir(parents=True, exist_ok=True)
    copies = output / "maps"
    shutil.copytree(source, copies)
    session = DemoSession(root=root, map_dir=copies, output_dir=output, _allow_batch_maps=True, **kwargs)
    session.configure_batch(source)
    return session


def run_batch(cases: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
    return evaluate_session(cases, create_batch_session(**kwargs))
