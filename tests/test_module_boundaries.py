"""Check A0 dependencies in fresh processes with optional backends unavailable."""
from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ModuleBoundaryTests(unittest.TestCase):
    def run_isolated(self, body: str, blocked: tuple[str, ...]) -> None:
        program = '''
import importlib.abc
import sys
class Unavailable(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in BLOCKED):
            raise ImportError('Optional backend unavailable: ' + fullname)
sys.meta_path.insert(0, Unavailable())
'''.replace("BLOCKED", repr(blocked)) + body
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONIOENCODING": "utf-8"}
        result = subprocess.run([sys.executable, "-c", program], cwd=ROOT.parent,
                                env=env, text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_pure_contracts_maps_and_scoring_without_simulator(self):
        self.run_isolated('''
from embodied_agent.contracts import ContractError, Observation
from embodied_agent.maps import MapStore, WorldMap
from embodied_agent.evaluation.cases import check_expected, load_cases
assert MapStore().load('classic').map_id == 'classic'
assert len(load_cases()) >= 20
assert check_expected({'status': 'FAILED', 'error_code': 'UNSAFE_TARGET'},
                      {'status': 'FAILED', 'error_code': 'UNSAFE_TARGET'}) == (True, [])
''', ("mujoco", "numpy", "tkinter", "torch", "transformers", "peft"))

    def test_agent_components_import_without_agent_clients_or_optional_backends(self):
        self.run_isolated('''
import sys
from embodied_agent.prompts import PromptCatalog
from embodied_agent.tools import QueryToolCatalog, QueryToolDefinition, QueryTools
from embodied_agent.tools.capabilities import build_capabilities, validate_actions
from embodied_agent.memory import MemoryStore
from embodied_agent.knowledge import KnowledgeBase
assert PromptCatalog().get('instruction.plan').text
assert len(QueryToolCatalog().schemas) == 4
assert not any(name in sys.modules for name in (
    'embodied_agent.agents.instruction', 'embodied_agent.agents.environment',
    'embodied_agent.agents.query_tools', 'embodied_agent.agents.capabilities',
    'embodied_agent.models.prompts', 'embodied_agent.models.deepseek',
    'mujoco', 'numpy', 'tkinter', 'torch', 'openai'))
''', ("mujoco", "numpy", "tkinter", "torch", "transformers", "peft", "openai"))

    def test_recording_core_uses_only_standard_library(self):
        self.run_isolated('''
import sys
import tempfile
from pathlib import Path
from embodied_agent.recording import TaskRecordStore
with tempfile.TemporaryDirectory() as folder:
    store = TaskRecordStore(Path(folder) / 'records')
    task = store.begin(map_id='fixture', map_definition={}, initial_state={}, natural_language='record')
    task.append_attempt()
    task.append_event({'event': 'observed', 'detail': {'measured': True}})
    task.finish(final_state={}, outcome={'status': 'SUCCESS'})
    assert store.verify(task.task_id)['sealed']
assert not any(name in sys.modules for name in ('mujoco', 'numpy', 'openai', 'torch'))
''', ("mujoco", "numpy", "tkinter", "torch", "transformers", "peft", "openai"))

    def test_headless_real_execution_without_gui_or_training(self):
        self.run_isolated('''
import tempfile
from pathlib import Path
from embodied_agent.apps.demo.session import run_batch
case = {'case_id': 'safe_rejection', 'map_id': 'classic',
        'instruction': '把方块扔到 target_a',
        'expected': {'status': 'FAILED', 'error_code': 'CAPABILITY_GAP'}}
with tempfile.TemporaryDirectory() as folder:
    result = run_batch([case], output_dir=Path(folder) / 'run',
                       records_dir=Path(folder) / 'records',
                       planner_kind='stub', environment_mode='rules')
    assert result['all_pass'] and result['success_count'] == 0
    assert result['passed_count'] == 1
''', ("tkinter", "mujoco.viewer", "torch", "transformers", "peft", "openai"))

    def test_core_has_no_application_or_script_imports(self):
        forbidden = ("embodied_agent.apps", "embodied_agent.training", "scripts", "m2_pick_place")
        for directory in ("execution", "simulation", "skills", "safety", "evaluation", "recording"):
            for path in (ROOT / "src" / "embodied_agent" / directory).rglob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else (
                        [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
                    if isinstance(node, ast.ImportFrom) and node.level:
                        package = ".".join(path.relative_to(ROOT / "src").with_suffix("").parts[:-1])
                        names = [importlib.util.resolve_name("." * node.level + (node.module or ""), package)]
                    for name in names:
                        self.assertFalse(any(name == prefix or name.startswith(prefix + ".")
                                             for prefix in forbidden), f"{path}: {name}")

    def test_removed_legacy_modules_are_unimportable(self):
        self.run_isolated('''
import importlib
import importlib.util
from pathlib import Path
from embodied_agent.paths import PROJECT_ROOT
for name in ('demo', 'demo_ui', 'world_view', 'worlds', 'planner',
             'language_agent', 'environment_agent', 'runtime',
             'agents.home_language', 'agents.language', 'agents.planning',
             'agents.capabilities', 'agents.query_tools', 'models.prompts',
             'execution.runtime', 'apps.demo.m3', 'evaluation.task_records'):
    module = 'embodied_agent.' + name
    assert not (PROJECT_ROOT / 'src' / 'embodied_agent' / (name.replace('.', '/') + '.py')).exists()
    assert importlib.util.find_spec(module) is None, module
    try:
        importlib.import_module(module)
    except ModuleNotFoundError as error:
        assert error.name == module, error
    else:
        raise AssertionError('Removed module remains importable: ' + module)
''', ("tkinter", "mujoco.viewer", "torch", "transformers", "peft"))

    def test_sources_do_not_import_the_removed_agent_pipeline(self):
        removed = {"embodied_agent.agents.home_language", "embodied_agent.agents.language",
                   "embodied_agent.agents.planning", "embodied_agent.execution.runtime",
                   "embodied_agent.apps.demo.m3", "embodied_agent.agents.capabilities",
                   "embodied_agent.agents.query_tools", "embodied_agent.models.prompts",
                   "embodied_agent.evaluation.task_records"}
        for directory in (ROOT / "src", ROOT / "scripts"):
            for path in directory.rglob("*.py"):
                for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                    if isinstance(node, ast.Import):
                        names = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        module = node.module or ""
                        if node.level:
                            package = ".".join(path.relative_to(ROOT / "src").with_suffix("").parts[:-1])
                            module = importlib.util.resolve_name("." * node.level + module, package)
                        names = [module, *(module + "." + alias.name for alias in node.names)]
                    else:
                        continue
                    self.assertTrue(removed.isdisjoint(names), f"{path}: {names}")


if __name__ == "__main__":
    unittest.main()
