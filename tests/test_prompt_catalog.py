"""Exact default prompt migration and explicit replacement configuration."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_agent.prompts import PromptCatalog, PromptSpec


class PromptCatalogTests(unittest.TestCase):
    def config(self, prompt_id, override):
        return {"agent": {"max_rounds": 8},
                "components": {"prompts": {"overrides": {prompt_id: override}}}}

    def test_defaults_match_exact_pre_migration_request_text(self):
        # SHA-256 values measured from original AST literals. Environment values
        # include the original _llm_plan suffix, including its leading newline.
        expected = {
            "instruction.intent": ("instruction-agent-v2", 2393,
                "3fa6405bc0f2db7afb93552226e92e6a382a2e0cb938c0da869d1d5e85dce46c"),
            "instruction.plan": ("instruction-agent-v2", 2912,
                "59d017137747ace72c833726d9bb62bc1d4514e53e9f84d8a4d4e25ba91fd9fa"),
            "environment.desktop": ("environment-agent-desktop-v1", 2254,
                "1b985fa2c94ca9e9274ae0c28809c009e4d4a099bb612d6c9dcb4f6ba839dcbe"),
            "environment.home": ("environment-agent-home-v1", 1098,
                "bb896e5e238a707aff9fabbc5fa9ade2979b92bf011b87c3fd36af5943ee36ee"),
        }
        catalog = PromptCatalog()
        for prompt_id, (version, length, digest) in expected.items():
            with self.subTest(prompt_id=prompt_id):
                spec = catalog.get(prompt_id)
                self.assertEqual((spec.id, spec.version, len(spec.text), spec.sha256),
                                 (prompt_id, version, length, digest))

    def test_spec_is_immutable_and_serializes_exact_text(self):
        spec = PromptSpec("instruction.plan", "custom-v1", "Custom\r\n提示词\n")
        expected = {"id": spec.id, "version": spec.version, "text": spec.text,
                    "sha256": hashlib.sha256(spec.text.encode("utf-8")).hexdigest()}
        self.assertEqual(spec.to_dict(), expected)
        data = spec.to_dict()
        data["text"] = "mutated"
        self.assertEqual(spec.text, expected["text"])
        with self.assertRaises(FrozenInstanceError):
            spec.text = "mutated"

    def test_inline_override_is_complete_and_other_roles_keep_defaults(self):
        catalog = PromptCatalog(self.config("environment.desktop",
            {"text": "Replacement without the old suffix.", "version": "trial-v1"}))
        spec = catalog.get("environment.desktop")
        self.assertEqual(spec.text, "Replacement without the old suffix.")
        self.assertEqual(spec.version, "trial-v1")
        self.assertEqual(catalog.get("environment.home"), PromptCatalog().get("environment.home"))

    def test_config_mutation_requires_a_new_catalog(self):
        config = self.config("instruction.plan", {"text": "original", "version": "trial-v1"})
        catalog = PromptCatalog(config)
        config["components"]["prompts"]["overrides"]["instruction.plan"]["text"] = "changed"
        self.assertEqual(catalog.get("instruction.plan").text, "original")
        self.assertEqual(PromptCatalog(config).get("instruction.plan").text, "changed")

    def test_relative_file_override_preserves_bytes_and_sees_replacement(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "prompt.txt"
            path.write_bytes("文件提示词\r\n\n".encode("utf-8"))
            catalog = PromptCatalog(self.config("instruction.intent",
                {"path": "prompt.txt", "version": "file-v1"}), project_root=folder)
            first = catalog.get("instruction.intent")
            self.assertEqual(first.text, "文件提示词\r\n\n")
            path.write_bytes(b"replacement\n")
            second = catalog.get("instruction.intent")
            self.assertEqual(second.text, "replacement\n")
            self.assertNotEqual(first.sha256, second.sha256)

    def test_absolute_file_override(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "prompt.txt"
            path.write_bytes(b"absolute file text")
            catalog = PromptCatalog(self.config("environment.home",
                {"path": str(path), "version": "file-v1"}), project_root=ROOT)
            self.assertEqual(catalog.get("environment.home").text, "absolute file text")

    def test_missing_or_invalid_file_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as folder:
            catalog = PromptCatalog(self.config("instruction.plan",
                {"path": "missing.txt", "version": "file-v1"}), project_root=folder)
            with self.assertRaises(FileNotFoundError):
                catalog.get("instruction.plan")
            path = Path(folder) / "missing.txt"
            path.write_bytes(b"\xff")
            with self.assertRaises(UnicodeDecodeError):
                catalog.get("instruction.plan")
            path.write_bytes(b"   ")
            with self.assertRaises(ValueError):
                catalog.get("instruction.plan")

    def test_invalid_overrides_fail_before_model_work(self):
        invalid = [None, "text", {}, {"text": "x"},
            {"text": "x", "version": ""}, {"text": "x", "version": 1},
            {"text": " ", "version": "v1"}, {"path": "", "version": "v1"},
            {"text": "x", "path": "p", "version": "v1"},
            {"text": "x", "version": "v1", "typo": True}]
        for override in invalid:
            with self.subTest(override=override), self.assertRaises(ValueError):
                PromptCatalog(self.config("instruction.plan", override))

    def test_invalid_config_sections_and_unknown_ids(self):
        for config in [[], {"components": None}, {"components": {"prompts": []}},
                       {"components": {"prompts": {"overrides": None}}},
                       self.config("unknown", {"text": "x", "version": "v1"})]:
            with self.subTest(config=config), self.assertRaises(ValueError):
                PromptCatalog(config)
        with self.assertRaises(KeyError):
            PromptCatalog().get("instruction.typo")

    def test_ordinary_runtime_config_and_empty_overrides_use_defaults(self):
        expected = PromptCatalog().get("instruction.plan")
        for config in ({"agent": {"max_rounds": 8}}, {"components": {}},
                       {"components": {"prompts": {"overrides": {}}}}):
            self.assertEqual(PromptCatalog(config).get("instruction.plan"), expected)

    def test_import_and_defaults_without_agents_or_optional_backends(self):
        program = '''
import importlib.abc
import sys
class Unavailable(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = ('embodied_agent.agents', 'embodied_agent.models', 'mujoco',
                   'numpy', 'openai', 'torch', 'transformers', 'tkinter')
        if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
            raise ImportError('Unavailable: ' + fullname)
sys.meta_path.insert(0, Unavailable())
from embodied_agent.prompts import PromptCatalog
catalog = PromptCatalog()
for prompt_id in ('instruction.intent', 'instruction.plan', 'environment.desktop', 'environment.home'):
    assert catalog.get(prompt_id).text
'''
        result = subprocess.run([sys.executable, "-c", program], cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONIOENCODING": "utf-8"},
            capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
