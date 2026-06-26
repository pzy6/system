"""实时画面流水线防卡顿回归测试。"""
import ast
import sys
import unittest
from pathlib import Path
from queue import Queue

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from utils.common import put_latest


def _get_method(class_node, method_name):
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return node
    raise AssertionError(f"method not found: {method_name}")


def _first_call_line(func_node, attr_or_name):
    for node in ast.walk(func_node):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if isinstance(target, ast.Attribute) and target.attr == attr_or_name:
            return node.lineno
        if isinstance(target, ast.Name) and target.id == attr_or_name:
            return node.lineno
    raise AssertionError(f"call not found: {attr_or_name}")


def _system_class():
    tree = ast.parse((SRC / "main.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "SilverGuardianSystem":
            return node
    raise AssertionError("SilverGuardianSystem not found")


class TestLiveFramePipeline(unittest.TestCase):
    def test_put_latest_drops_stale_frames_when_full(self):
        q = Queue(maxsize=2)
        put_latest(q, "old-1")
        put_latest(q, "old-2")

        dropped = put_latest(q, "newest")

        self.assertEqual(dropped, 2)
        self.assertEqual(q.qsize(), 1)
        self.assertEqual(q.get_nowait(), "newest")

    def test_main_enqueues_dashboard_frame_before_adaptive_skip(self):
        process_frame = _get_method(_system_class(), "process_frame")

        raw_enqueue_line = _first_call_line(process_frame, "_enqueue_dashboard_frame")
        adaptive_skip_line = _first_call_line(process_frame, "should_process")

        self.assertLess(raw_enqueue_line, adaptive_skip_line)

    def test_dashboard_enqueue_uses_latest_frame_queueing(self):
        enqueue_method = _get_method(_system_class(), "_enqueue_dashboard_frame")

        self.assertIsInstance(_first_call_line(enqueue_method, "put_latest"), int)


if __name__ == "__main__":
    unittest.main()
