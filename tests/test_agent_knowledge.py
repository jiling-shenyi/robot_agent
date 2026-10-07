"""Deterministic lexical retrieval and evidence isolation without optional dependencies."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from embodied_agent.knowledge import KnowledgeBase


class AgentKnowledgeTests(unittest.TestCase):
    def test_relevance_in_both_languages_and_stable_ties(self):
        knowledge = KnowledgeBase([
            {"id": "remote", "text": "遥控器放置后需要检查桌面支撑", "metadata": {"source": "guide"}},
            {"id": "b", "text": "book placement needs stable support"},
            {"id": "a", "text": "book placement needs stable support"},
        ], version="fixture-v2")
        result = knowledge.retrieve("遥控器")
        self.assertEqual(result["version"], "fixture-v2")
        self.assertEqual([row["id"] for row in result["documents"]], ["remote"])
        self.assertEqual(result["documents"][0]["metadata"], {"source": "guide"})
        self.assertEqual(knowledge.retrieve("遥")["documents"][0]["id"], "remote")
        self.assertEqual([row["id"] for row in knowledge.retrieve("BOOK")["documents"]], ["a", "b"])
        self.assertEqual(knowledge.retrieve("astronomy")["documents"], [])
        self.assertEqual(knowledge.retrieve(" ")["documents"], [])

    def test_budget_marks_actual_excerpt_and_retains_full_snapshot(self):
        text = "遥控器" + "支撑检查" * 400
        knowledge = KnowledgeBase([{"id": "remote", "text": text, "metadata": {"source": "manual-1"}}])
        result = knowledge.retrieve("遥控器", max_chars=220)
        self.assertLessEqual(result["returned_chars"], 220)
        row = result["documents"][0]
        self.assertTrue(row["truncated"])
        self.assertEqual(row["original_chars"], len(text))
        self.assertTrue(row["text"].endswith("[truncated]"))
        self.assertEqual(row["metadata"], {"source": "manual-1"})
        self.assertEqual(result["returned_chars"], len(json.dumps(row, ensure_ascii=False,
            separators=(",", ":"))))
        self.assertEqual(knowledge.snapshot()["documents"][0]["text"], text)
        self.assertEqual(knowledge.retrieve("遥控器", max_chars=1)["documents"], [])
        self.assertEqual(knowledge.retrieve("遥控器", max_chars=0)["documents"], [])
        self.assertEqual(knowledge.retrieve("遥控器", top_k=0)["documents"], [])

    def test_input_and_result_mutations_do_not_change_evidence(self):
        documents = [{"id": "manual", "text": "book support", "metadata": {"source": ["original"]}}]
        knowledge = KnowledgeBase(documents)
        documents[0]["metadata"]["source"].clear()
        result = knowledge.retrieve("book")
        result["documents"][0]["metadata"]["source"].clear()
        snapshot = knowledge.snapshot()
        snapshot["documents"].clear()
        self.assertEqual(knowledge.retrieve("book")["documents"][0]["metadata"]["source"], ["original"])
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: knowledge.retrieve("book"), range(32)))
        self.assertTrue(all(row == knowledge.retrieve("book") for row in results))

    def test_empty_catalog_loading_and_invalid_limits(self):
        self.assertEqual(KnowledgeBase().retrieve("book")["documents"], [])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "knowledge.json"
            path.write_text(json.dumps({"version": "file-v2", "documents": [
                {"id": "manual", "text": "book support"}]}), encoding="utf-8")
            self.assertEqual(KnowledgeBase.from_file(path).retrieve("book")["version"], "file-v2")
        for invalid in (-1, True, 1.5):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                KnowledgeBase().retrieve("book", top_k=invalid)
        for documents in ([{"id": "a", "text": "book"}, {"id": "a", "text": "duplicate"}],
                          [{"id": "a", "text": ""}], [{"id": "a", "text": "book", "metadata": {"password": "secret"}}],
                          [{"id": "a", "text": "-----BEGIN PRIVATE KEY-----data"}]):
            with self.subTest(documents=documents), self.assertRaises(ValueError):
                KnowledgeBase(documents)


if __name__ == "__main__":
    unittest.main()
