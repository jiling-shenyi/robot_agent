"""Run the shared instruction Agent, or replay the registered Panda scene set."""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from embodied_agent.paths import PROJECT_ROOT as ROOT


def _run_single(args) -> int:
    from embodied_agent.apps.demo.session import DemoSession
    from embodied_agent.maps.schema import WorldMap
    from embodied_agent.simulation.scenarios import load_json, make_scenario

    run_id = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = (args.output or ROOT / 'results' / 'instruction' / run_id).resolve()
    records = args.records_dir or ROOT / 'records'
    session = DemoSession(output_dir=output, records_dir=records,
                          planner_kind=args.planner, environment_mode='rules')
    try:
        session.select_map('classic')
        if args.seed:
            scenarios = load_json(ROOT / 'configs' / 'm2_scenarios.json')['scenarios']
            scenario = make_scenario(args.seed, 'a', scenarios)
            definition = session.world.to_dict()
            definition['cube_position_m'] = list(scenario['cube_position_m'])
            session._show_world(WorldMap.from_dict(definition))
        if args.headless:
            result = session.run_agent(args.instruction)
            session.finish()
        else:
            from embodied_agent.visualization.demo_ui import DemoApp
            app = DemoApp(session, [], initial_map='classic')
            results = []

            def execute():
                result = session.run_agent(args.instruction)
                results.append(result)
                app._report_result(result)

            app.window.after(150, lambda: app._execute(execute))
            app.run()
            result = results[-1] if results else {'status': 'ABORTED', 'error_code': 'VIEWER_CLOSED'}
        print(f"{result['status']} {result.get('error_code') or ''}; steps={result.get('physics_steps', 0)}")
        if result.get('task_record_path'):
            print(f"任务记录：{result['task_record_path']}")
        return 0 if result['status'] == 'SUCCESS' else 1
    finally:
        session.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    select = parser.add_mutually_exclusive_group(required=True)
    select.add_argument('--instruction', help='one natural-language task on the live Panda scene')
    select.add_argument('--batch', action='store_true', help='replay twenty registered Panda scenarios through InstructionAgent')
    parser.add_argument('--planner', choices=('llm', 'stub'), default='llm')
    parser.add_argument('--seed', type=int, default=0, help='initial scene seed for one task')
    view = parser.add_mutually_exclusive_group()
    view.add_argument('--headless', action='store_true', help='explicitly run without graphics')
    view.add_argument('--viewer', action='store_true', help='select the default visible mode')
    parser.add_argument('--records-dir', type=Path)
    parser.add_argument('--output', type=Path, help='new evidence directory; nonempty directories are rejected')
    args = parser.parse_args(argv)
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
    if args.batch:
        from embodied_agent.apps.demo.cli import main as demo_main
        forwarded = ['--mode', 'batch', '--map', 'classic', '--planner', args.planner,
                     '--environment-planner', 'rules', '--cases', str(ROOT / 'configs' / 'm3_cases.json'),
                     '--headless' if args.headless else '--viewer']
        for flag, value in (('--records-dir', args.records_dir), ('--output', args.output)):
            if value is not None:
                forwarded.extend([flag, str(value)])
        return demo_main(forwarded)
    return _run_single(args)


if __name__ == '__main__':
    raise SystemExit(main())
