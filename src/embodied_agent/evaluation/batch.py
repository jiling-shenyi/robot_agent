"""Batch driver using the public session API; no GUI or application imports."""
from __future__ import annotations
from typing import Any, Protocol


class EvaluationSession(Protocol):
    """Runtime operations needed by the batch evaluator and future data collectors."""

    on_status: Any

    def run_case(self, case: dict[str, Any]) -> dict[str, Any]: ...
    def map_evidence(self) -> dict[str, Any]: ...
    def record_case(self, result: dict[str, Any]) -> None: ...
    def mark_not_run(self, cases: list[dict[str, Any]]) -> None: ...
    def finish(self) -> dict[str, Any]: ...
    def close(self) -> None: ...

def evaluate_session(cases: list[dict[str, Any]], session: EvaluationSession) -> dict[str, Any]:
    try:
        for index, case in enumerate(cases):
            try:
                result = session.run_case(case)
            except KeyboardInterrupt:
                result = {"case_id": case["case_id"], "status": "ABORTED", "error_code": "INTERRUPTED",
                          "error_message": "Batch interrupted by user", "expected_pass": False,
                          "passed": False, "actions": [], **session.map_evidence()}
                session.record_case(result)
            if result["status"] == "ABORTED":
                session.mark_not_run(cases[index + 1:])
                break
            if session.on_status is not None:
                session.on_status(f"CASE {case['case_id']}",
                                  f"{result['status']}; expected_pass={result['expected_pass']}")
        return session.finish()
    finally:
        session.close()
