"""Language entry points use the current shared Agent."""
import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from embodied_agent.apps.demo import instruction

class InstructionEntryTests(unittest.TestCase):
    def test_single_task_defaults_visible_and_uses_shared_session(self):
        with patch.object(instruction, '_run_single', return_value=0) as run:
            self.assertEqual(instruction.main(['--instruction', 'place on a new registered destination']), 0)
        self.assertFalse(run.call_args.args[0].headless)
        self.assertEqual(run.call_args.args[0].planner, 'llm')

    def test_batch_uses_current_demo_and_preserves_explicit_view_choice(self):
        for choice in ('--headless', '--viewer'):
            with patch('embodied_agent.apps.demo.cli.main', return_value=0) as run:
                self.assertEqual(instruction.main(['--batch', '--planner', 'stub', choice]), 0)
                argv = run.call_args.args[0]
                self.assertIn(choice, argv)
                self.assertIn(str(ROOT / 'configs' / 'm3_cases.json'), argv)

    def test_removed_protocol_and_conflicting_view_choices_are_rejected(self):
        for options in (['--batch', '--legacy-m3'], ['--batch', '--viewer', '--headless']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                instruction.main(options)
            self.assertEqual(error.exception.code, 2)
