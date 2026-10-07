"""Query, verify, rebuild, assess and export schema-v2 records without backends."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from embodied_agent.paths import PROJECT_ROOT
from embodied_agent.recording.assessment import assess_task
from embodied_agent.recording.export import EXPORT_KINDS, export_dataset
from embodied_agent.recording.store import TaskRecordStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-dir", type=Path, default=PROJECT_ROOT / "records",
                        help="Canonical records root; schema-v1 files are unsupported")
    parser.add_argument("--task", help="Task UUID, task directory or task.json path")
    parser.add_argument("--map", dest="map_id")
    parser.add_argument("--date", help="Asia/Shanghai start date, YYYYMMDD or YYYY-MM-DD")
    parser.add_argument("--status", choices=("SUCCESS", "FAILED", "ABORTED"))
    parser.add_argument("--record-status", choices=("RUNNING", "COMPLETED", "FAILED", "ABORTED"))
    parser.add_argument("--task-kind")
    parser.add_argument("--agent-role")
    parser.add_argument("--integrity", choices=("complete", "valid", "invalid", "unsealed"))
    parser.add_argument("--verified-only", action="store_true",
                        help="Filter executor-verified successes; this does not establish training eligibility")
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--verify", action="store_true")
    operation.add_argument("--rebuild", action="store_true", help="Rebuild derived task.json, never modify the journal")
    operation.add_argument("--rebuild-index", action="store_true")
    operation.add_argument("--events", action="store_true", help="Print resolved canonical events; requires --task")
    operation.add_argument("--export", choices=EXPORT_KINDS, dest="export_kind")
    operation.add_argument("--assess", action="store_true", help="Apply the explicit typed trusted task specification")
    parser.add_argument("--output", type=Path, help="New dataset directory for --export")
    parser.add_argument("--evaluation", action="append", default=[], help="Explicit immutable assessment path; repeat per task")
    parser.add_argument("--eval-run-id", help="New evaluation-run identity for --assess")
    parser.add_argument("--scalar-reward", type=float, help="Explicit caller-supplied reward; never inferred from status")
    parser.add_argument("--trusted-task-spec", type=Path, help="Explicit retrospective typed specification for --assess")
    args = parser.parse_args(argv)
    if not args.records_dir.is_dir():
        parser.error(f"Task record directory does not exist: {args.records_dir}")
    if args.export_kind and args.output is None:
        parser.error("--export requires --output pointing to a new dataset directory")
    if args.events and not args.task:
        parser.error("--events requires --task")
    if args.assess and (not args.task or not args.eval_run_id):
        parser.error("--assess requires --task and --eval-run-id")
    try:
        if not args.task:
            legacy = [path for folder in (args.records_dir, args.records_dir / "tasks")
                      if folder.is_dir() for path in folder.glob("*.json")
                      if re.fullmatch(r"\d{8}_.+_\d+\.json", path.name)]
            if legacy:
                raise ValueError(f"Schema-v1 task records are unsupported: {legacy[0]}")
        store = TaskRecordStore(args.records_dir)
        if args.rebuild_index:
            result = store.rebuild_index()
        elif args.verify and args.task:
            result = store.verify(args.task)
        else:
            records = [store.load(args.task)] if args.task else store.query(
                map_id=args.map_id, status=args.status, date=args.date, record_status=args.record_status)
            selected = []
            for row in records:
                metadata, identity, outcome = row.get("metadata", {}), row.get("identity", {}), row.get("outcome") or {}
                if args.task_kind and metadata.get("task_kind", identity.get("task_kind", row.get("task_kind"))) != args.task_kind:
                    continue
                if args.agent_role and identity.get("agent_role", metadata.get("agent_role", row.get("agent_role"))) != args.agent_role:
                    continue
                if args.verified_only and not (outcome.get("status") == "SUCCESS" and
                        outcome.get("executor_verified", outcome.get("verified")) is True):
                    continue
                if args.integrity:
                    check = store.verify(row["task_id"])
                    match = {"complete": check.get("valid") is True and check.get("sealed") is True,
                        "valid": check.get("valid") is True, "invalid": check.get("valid") is not True,
                        "unsealed": check.get("sealed") is not True}
                    if not match[args.integrity]:
                        continue
                selected.append(row)
            if args.verify:
                reports = [store.verify(row["task_id"]) for row in selected]
                result = reports[0] if args.task and len(reports) == 1 else {"count": len(reports), "tasks": reports}
            elif args.rebuild:
                rebuilt = [store.rebuild(row["task_id"]) for row in selected]
                result = rebuilt[0] if args.task and len(rebuilt) == 1 else {"count": len(rebuilt), "tasks": rebuilt}
            elif args.events:
                result = {"task_id": selected[0]["task_id"], "events": store.read_events(selected[0]["task_id"], resolve=True)} if selected else {"count": 0, "events": []}
            elif args.assess:
                spec = json.loads(args.trusted_task_spec.read_text(encoding="utf-8")) if args.trusted_task_spec else None
                result = assess_task(store, selected[0]["task_id"], eval_run_id=args.eval_run_id,
                    scalar_reward=args.scalar_reward, trusted_task_spec=spec,
                    evaluator_config={"entry_point": "task_records_cli"}) if selected else {"count": 0}
            elif args.export_kind:
                result = export_dataset(store, args.output, kind=args.export_kind,
                    task_ids=[row["task_id"] for row in selected], evaluation_refs=args.evaluation)
            elif args.task and len(selected) == 1:
                result = selected[0]
            else:
                result = {"count": len(selected), "tasks": [{"task_id": row["task_id"],
                    "date": row.get("date"), "sequence": row.get("sequence"), "map_id": row.get("map", {}).get("id"),
                    "natural_language": row.get("natural_language"), "record_status": row.get("record_status"),
                    "task_kind": row.get("metadata", {}).get("task_kind"),
                    "agent_role": row.get("identity", {}).get("agent_role", row.get("metadata", {}).get("agent_role")),
                    "integrity": row.get("integrity"), "outcome": row.get("outcome"),
                    "attempt_count": len(row.get("attempts", []))} for row in selected],
                    "errors": getattr(store, "errors", [])}
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2, default=str))
        if args.verify:
            reports = [result] if "valid" in result else result.get("tasks", [])
            return 1 if getattr(store, "errors", []) or any(report.get("valid") is not True for report in reports) else 0
        return 1 if getattr(store, "errors", []) else 0
    except (ValueError, OSError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
