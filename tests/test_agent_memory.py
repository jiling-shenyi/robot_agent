"""Memory isolation, concurrency and bounded visible data without runtime imports."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from embodied_agent.memory import MemoryStore


class AgentMemoryTests(unittest.TestCase):
    def test_namespace_revision_and_copies(self):
        store, value = MemoryStore(), {"status": "FAILED", "errors": ["blocked"]}
        first = store.write("instruction", "last-task", value, metadata={"task_id": "task-1"})
        value["errors"].clear()
        first["metadata"]["task_id"] = "changed"
        self.assertEqual(store.read("instruction", "last-task")["value"]["errors"], ["blocked"])
        self.assertEqual(store.read("instruction", "last-task")["metadata"]["task_id"], "task-1")
        self.assertIsNone(store.read("environment", "last-task"))
        self.assertEqual(store.write("instruction", "last-task", {"status": "SUCCESS"})["revision"], 2)
        snapshot = store.snapshot("instruction")
        snapshot["entries"].clear()
        self.assertEqual(len(store.snapshot("instruction")["entries"]), 1)
        self.assertEqual(store.clear("instruction")["revision"], 3)
        self.assertEqual(store.snapshot("instruction")["entries"], [])

    def test_global_capacity_and_evicted_namespace_revision(self):
        store = MemoryStore(max_entries=2)
        store.write("a", "first", "first")
        store.write("b", "second", "second")
        store.write("b", "third", "third")
        self.assertIsNone(store.read("a", "first"))
        self.assertEqual(store.snapshot("a")["revision"], 2)
        self.assertEqual(len(store.snapshot("b")["entries"]), 2)

    def test_concurrent_writes_preserve_unique_revisions(self):
        store = MemoryStore(max_entries=64)
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda index: store.write("shared", str(index), {"index": index}), range(32)))
        self.assertEqual({row["revision"] for row in results}, set(range(1, 33)))
        self.assertEqual(len(store.snapshot("shared")["entries"]), 32)

    def test_chinese_english_relevance_and_content_budget(self):
        store = MemoryStore()
        store.write("agent", "remote", "遥控器必须放在桌面。" + "确认支撑。" * 200)
        store.write("agent", "book", {"experience": "book placement needs stable support"})
        self.assertEqual(store.search("agent", "遥控器")["entries"][0]["key"], "remote")
        self.assertEqual(store.search("agent", "遥")["entries"][0]["key"], "remote")
        self.assertEqual(store.search("agent", "BOOK")["entries"][0]["key"], "book")
        self.assertEqual(store.search("agent", "astronomy")["entries"], [])
        clipped = store.search("agent", "遥控器", max_chars=240)
        self.assertLessEqual(clipped["returned_chars"], 240)
        self.assertTrue(clipped["entries"][0]["truncated"])
        self.assertTrue(clipped["entries"][0]["value"].endswith("[truncated]"))
        self.assertEqual(clipped["returned_chars"], sum(len(json.dumps(row, ensure_ascii=False,
            separators=(",", ":"))) for row in clipped["entries"]))
        self.assertEqual(store.search("agent", "book", max_chars=1)["entries"], [])
        self.assertEqual(store.search("agent", "book", max_chars=0)["entries"], [])
        self.assertEqual(store.read("agent", "remote")["value"].count("确认支撑。"), 200)

    def test_invalid_data_and_explicit_credentials_are_rejected(self):
        store = MemoryStore()
        for value in ({"api_key": "secret-value"}, {"nested": {"password": "secret"}},
                      "-----BEGIN PRIVATE KEY-----data", {"value": float("nan")}, {1: "invalid key"}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                store.write("agent", "entry", value)
        self.assertEqual(store.snapshot("agent")["revision"], 0)
        store.write("agent", "usage", {"token_usage": 120, "max_tokens": 800})
        for invalid in (-1, True, 1.5):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                store.search("agent", "usage", max_chars=invalid)
        with self.assertRaises(ValueError):
            MemoryStore(max_entries=0)
        with self.assertRaises(ValueError):
            store.write("", "key", "text")


if __name__ == "__main__":
    unittest.main()
