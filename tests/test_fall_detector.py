"""测试摔倒检测器"""
import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from algorithms.fall_detector import FallDetector


class TestFallDetector:
    def setup_method(self):
        self.detector = FallDetector(threshold=0.75)

    def _make_kp(self, parts):
        """创建关键点列表"""
        return [{'part': p, 'x': x, 'y': y, 'confidence': c}
                for p, x, y, c in parts]

    def test_standing_person_not_fall(self):
        """站立人不应检测为摔倒"""
        kp = self._make_kp([
            ('Nose', 320, 100, 0.9), ('Neck', 320, 150, 0.9),
            ('RShoulder', 280, 160, 0.9), ('LShoulder', 360, 160, 0.9),
            ('RHip', 290, 280, 0.9), ('LHip', 350, 280, 0.9),
            ('RKnee', 290, 380, 0.8), ('LKnee', 350, 380, 0.8),
            ('RAnkle', 285, 470, 0.8), ('LAnkle', 355, 470, 0.8),
        ])
        is_fall, conf = self.detector.detect(kp, 100)
        assert not is_fall
        assert conf < 0.75

    def test_lying_down_detected_as_fall(self):
        """平躺应检测为摔倒（宽高比大 + 角度小）"""
        kp = self._make_kp([
            ('Nose', 320, 460, 0.9), ('Neck', 320, 440, 0.9),
            ('RShoulder', 400, 440, 0.9), ('LShoulder', 240, 440, 0.9),
            ('RHip', 400, 460, 0.9), ('LHip', 240, 460, 0.9),
            ('RKnee', 400, 475, 0.8), ('LKnee', 240, 475, 0.8),
            ('RAnkle', 400, 480, 0.8), ('LAnkle', 240, 480, 0.8),
        ])
        for t in range(101, 115):  # 需要多帧确认
            is_fall, conf = self.detector.detect(kp, t)
        assert conf > 0.5  # 高置信度

    def test_ankle_above_hip_is_inverted(self):
        """踝高于髋应检测为倒置"""
        kp = self._make_kp([
            ('Nose', 320, 200, 0.9), ('Neck', 320, 250, 0.9),
            ('RHip', 290, 300, 0.9), ('LHip', 350, 300, 0.9),
            ('RKnee', 290, 200, 0.8), ('LKnee', 350, 200, 0.8),
            ('RAnkle', 285, 150, 0.8), ('LAnkle', 355, 150, 0.8),
        ])
        inv = self.detector._calculate_ankle_hip_inversion(kp)
        assert inv == 1.0  # 倒置

    def test_empty_keypoints_no_fall(self):
        """空关键点不应崩溃"""
        is_fall, conf = self.detector.detect([], 100)
        assert not is_fall
        assert conf == 0.0

    def test_get_status(self):
        """状态查询正常"""
        s = self.detector.get_status()
        assert s['loaded'] is True
        assert s['fallback'] is False
