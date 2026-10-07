"""Collect direct environment operations, or reuse the caller's active task."""
from __future__ import annotations

from dataclasses import replace
from functools import wraps
from pathlib import Path
import json
import uuid

from embodied_agent.models.tracing import capture_model_requests, current_trace, new_trace_id, record_model_event, trace_scope
from embodied_agent.paths import PROJECT_ROOT


def record_environment_operation(operation: str):
    def decorate(method):
        @wraps(method)
        def call(self, map_id, instruction, **kwargs):
            def invoke():
                if operation != "preview" or current_trace().get("decision_id"):
                    return method(self, map_id, instruction, **kwargs)
                with trace_scope(decision_id=new_trace_id("decision")):
                    record_model_event("decision_started", {"operation": "map_preview", "instruction": instruction})
                    try:
                        preview = method(self, map_id, instruction, **kwargs)
                    except Exception as error:
                        record_rejection("environment_rejected", {"error_code": getattr(error, "code", "ENVIRONMENT_ERROR"),
                            "message": str(error), "proposal_valid": False}, error)
                        raise
                    record_model_event("decision_finished", {"proposal_valid": True, "operations": list(preview.operations)})
                    return preview
            if current_trace().get("task_id") or not self.recording_enabled:
                return invoke()
            from embodied_agent.evaluation.task_writer import TaskRunWriter, recordable
            from embodied_agent.evaluation.provenance import source_hashes
            if self.mode == "llm" and self._planner is None:
                if self.config is None:
                    self.config = json.loads((PROJECT_ROOT / "configs" / "agent_runtime.json").read_text(encoding="utf-8"))
                self._configure_components(self.config)
            root = Path(self.records_dir or PROJECT_ROOT / "records").resolve()
            run_id = "environment_" + uuid.uuid4().hex
            output = root.parent / "results" / "environment" / run_id
            components = {}
            if self.prompt_catalog is not None:
                components["prompts"] = {key: self.prompt_catalog.get(key).to_dict()
                    for key in ("environment.desktop", "environment.home")}
            if self.tool_catalog is not None:
                components["tools"] = {"fingerprint": self.tool_catalog.fingerprint,
                                       "schemas": self.tool_catalog.schemas}
            if self.context_provider is not None:
                components["context"] = self.context_provider.describe()
            writer = TaskRunWriter(output, run_id, records_dir=root,
                versions={"agent_components": components, "runtime_config": self.config,
                          "source_hashes": source_hashes(PROJECT_ROOT), "planner_source": self.mode})
            before = None
            try:
                before = self.store.load(map_id).to_dict()
            except Exception:
                pass  # The real method supplies its original error below.
            self.last_record_ref = None
            self.last_recording_error = None
            record = None
            result = None
            run_close_attempted = False
            def close_run():
                nonlocal run_close_attempted
                if run_close_attempted:
                    return None
                run_close_attempted = True
                try:
                    writer.close()
                except Exception as error:
                    self.last_recording_error = {"event": "run.finished", "type": type(error).__name__, "message": str(error)}
                    return self.last_recording_error
                return None
            def remember(feedback):
                if operation == "preview" or self.context_provider is None:
                    return
                try:
                    with capture_model_requests(writer.model_recorder()), trace_scope(**writer.trace_identity()):
                        self.context_provider.remember(instruction, feedback, task_id=record.task_id, map_id=str(map_id))
                except Exception as error:
                    feedback.setdefault("component_errors", []).append({"component": "memory_write", "message": str(error)})
            try:
                record = writer.begin_task(instruction=instruction if isinstance(instruction, str) else repr(instruction),
                    map_id=str(map_id), map_definition=before, initial_state=before,
                    metadata={"agent": "environment", "task_kind": "map_edit", "operation": operation,
                              "preview_only": operation == "preview", "planner_source": self.mode,
                              "memory_namespace": self.context_provider.namespace if self.context_provider is not None and
                                  self.context_provider.memory_enabled else None,
                              "recording_scope": "direct_environment_call", "original_input": recordable(instruction)})
                with capture_model_requests(writer.model_recorder()), trace_scope(**writer.trace_identity()):
                    result = invoke()
                feedback = result.to_dict()
                feedback.update(reload_status="not_requested")
                remember(feedback)
                writer.finish_task(feedback, final_state=result.after.to_dict() if result.persisted else before)
                self.last_record_ref = feedback["task_record"]
                run_error = close_run()
                return replace(result, task_record=feedback["task_record"],
                    component_errors=tuple(feedback.get("component_errors", [])),
                    recording_errors=(*result.recording_errors, *((run_error,) if run_error else ())))
            except Exception as error:
                if record is None:
                    raise
                if result is None:
                    feedback = {"status": "FAILED", "error_code": getattr(error, "code", "ENVIRONMENT_ERROR"),
                                "error_message": str(error), "error_details": getattr(error, "details", {}),
                                "recording_errors": getattr(error, "recording_errors", []),
                                "commit_status": getattr(error, "commit_status", "unknown" if getattr(error, "code", None) == "RECORDING_FAILED" else "not_started"),
                                "reload_status": "not_requested"}
                    try:
                        remember(feedback)
                        writer.finish_task(feedback, final_state=getattr(error, "observed_persisted_map", before))
                        self.last_record_ref = feedback["task_record"]
                    except Exception as recording_error:
                        self.last_recording_error = {"type": type(recording_error).__name__, "message": str(recording_error)}
                    raise
                # The operation already returned its real result. A failed
                # record must not trigger a second map commit.
                self.last_recording_error = {"type": type(error).__name__, "message": str(error)}
                self.last_record_ref = {"task_id": record.task_id, "path": str(record.path),
                    "uri": record.path.relative_to(root).as_posix(), "schema_version": 2}
                run_error = close_run()
                return replace(result, task_record=self.last_record_ref,
                    recording_errors=(*result.recording_errors, {"type": type(error).__name__, "message": str(error)},
                        *((run_error,) if run_error else ())))
            finally:
                close_run()
        return call
    return decorate


def record_rejection(event: str, detail: dict, original_error: Exception) -> None:
    """An audit failure cannot replace the validation or CAS failure it describes."""
    try:
        record_model_event(event, detail)
    except Exception as error:
        errors = list(getattr(original_error, "recording_errors", []))
        errors.append({"event": event, "type": type(error).__name__, "message": str(error)})
        original_error.recording_errors = errors


def map_diff(before: dict, after: dict) -> list[dict]:
    result = []
    def visit(left, right, path):
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right)):
                visit(left.get(key), right.get(key), path + "/" + str(key).replace("~", "~0").replace("/", "~1"))
        elif left != right:
            result.append({"path": path or "/", "before": left, "after": right})
    visit(before, after, "")
    return result
