"""测试 YOLO 检测器"""
import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from algorithms.yolo_detector import YOLODetector


class TestYOLODetector:
    def setup_method(self):
        self.detector = YOLODetector({
            'model_path': os.path.join(
                os.path.dirname(__file__), '..', 'models', 'yolo_face_smoke'
            ),
            'confidence_threshold': 0.5,
            'frame_skip': 1,
        })

    def test_model_loaded(self):
        """模型应加载成功"""
        assert self.detector.face_loaded or self.detector.smoke_loaded

    def test_detect_returns_correct_structure(self):
        """检测返回格式正确"""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        result = self.detector.detect(frame)
        assert 'face_bboxes' in result
        assert 'smoke_detected' in result
        assert 'smoke_bboxes' in result
        assert 'smoke_confidence' in result
        assert isinstance(result['face_bboxes'], list)
        assert isinstance(result['smoke_detected'], bool)

    def test_detect_empty_frame_no_crash(self):
        """黑帧不崩溃"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.detector.detect(frame)
        assert result['smoke_detected'] is False

    def test_frame_skip_caches_result(self):
        """帧跳应缓存结果"""
        detector = YOLODetector({
            'model_path': os.path.join(
                os.path.dirname(__file__), '..', 'models', 'yolo_face_smoke'
            ),
            'frame_skip': 3,
        })
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        r1 = detector.detect(frame)
        r2 = detector.detect(frame)
        # 第二帧应复用缓存
        assert r1 == r2

    def test_get_status(self):
        """状态查询正常"""
        s = self.detector.get_status()
        assert 'loaded' in s
        assert 'message' in s

    def test_fallback_without_model(self):
        """无模型目录时应使用 Haar Cascade"""
        detector = YOLODetector({
            'model_path': './nonexistent_dir/',
        })
        assert detector.using_fallback
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        result = detector.detect(frame)
        assert 'face_bboxes' in result
