"""The shared visible robot demo and explicit headless acceptance batches."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from embodied_agent.paths import PROJECT_ROOT as ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("free", "batch"), default="free")
    parser.add_argument("--map", default="home_living_room", dest="map_id", help="initial map; classic/alternate retain the Panda scenes")
    parser.add_argument("--maps-dir", type=Path, default=ROOT / "configs" / "maps")
    parser.add_argument("--planner", choices=("llm", "stub"), default="llm", help="robot instruction planner; stub is offline regression only")
    parser.add_argument("--environment-planner", choices=("llm", "rules"), default="llm")
    parser.add_argument("--cases", type=Path, help="case catalog; default configs/demo_cases.json")
    parser.add_argument("--case", action="append", dest="case_ids", help="select a case ID; repeat for several cases")
    view = parser.add_mutually_exclusive_group()
    view.add_argument("--viewer", action="store_true", help="explicitly select the default visible mode")
    view.add_argument("--headless", action="store_true", help="explicitly request batch without graphics")
    parser.add_argument("--output", type=Path, help="new evidence directory; existing nonempty output is rejected")
    parser.add_argument("--records-dir", type=Path, help="v2 records root (runs/tasks/artifacts/evaluations); independent of --output")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mode == "free" and args.headless:
        parser.error("自由测试必须可视化；--headless 仅适用于 --mode batch")

    from dotenv import load_dotenv
    from embodied_agent.apps.demo.session import DemoSession, create_batch_session, run_batch
    from embodied_agent.evaluation.cases import load_cases

    load_dotenv(ROOT / ".env")
    run_id = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime("%Y%m%d-%H%M%S-%f")
    output = (args.output or ROOT / "results" / "demo" / run_id).resolve()
    try:
        cases = load_cases(args.cases, root=ROOT)
        if args.case_ids:
            by_id = {case["case_id"]: case for case in cases}
            unknown = set(args.case_ids) - set(by_id)
            if unknown:
                parser.error(f"未知测试用例: {', '.join(sorted(unknown))}")
            cases = [by_id[case_id] for case_id in dict.fromkeys(args.case_ids)]
        # A case can pin a map; otherwise the CLI's initial map is its default.
        cases = [dict(case, map_id=case.get("map_id", args.map_id)) for case in cases]
        options = dict(root=ROOT, map_dir=args.maps_dir.resolve(), output_dir=output,
                       planner_kind=args.planner, environment_mode=args.environment_planner,
                       records_dir=(args.records_dir.resolve() if args.records_dir else ROOT / "records"))
        if args.mode == "batch" and args.headless:
            def progress(title, detail):
                if title.startswith("CASE "):
                    print(f"{title}: {detail}", flush=True)

            summary = run_batch(cases, on_status=progress, **options)
            for result in summary.get("results", []):
                print(f"{result.get('case_id', '?')}: {result['status']} "
                      f"{result.get('error_code') or ''}; "
                      f"expected_pass={result.get('expected_pass', False)}")
            print(json.dumps({key: value for key, value in summary.items() if key != "results"}, ensure_ascii=False, indent=2))
            print(f"测试证据: {output}")
            return 0 if summary["all_pass"] else 1

        # Import Tk/OpenGL only for visible mode; explicit headless is independent.
        from embodied_agent.visualization.demo_ui import DemoApp

        factory = create_batch_session if args.mode == "batch" else DemoSession
        session = factory(**options)
        try:
            app = DemoApp(session, cases, initial_map=args.map_id, batch=args.mode == "batch")
            return app.run()
        finally:
            session.close()
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Demo 启动/执行失败: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
