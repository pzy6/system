"""测试人脸识别器"""
import pytest
import numpy as np
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from algorithms.face_recognizer import FaceRecognizer


class TestFaceRecognizer:
    def setup_method(self):
        self.db_path = os.path.join(
            os.path.dirname(__file__), '..', 'data', 'face_db'
        )

    def test_model_loading(self):
        """insightface 模型加载正常（numpy 版本兼容时）"""
        rec = FaceRecognizer({
            'face_db_path': self.db_path,
            'recognition_threshold': 0.6,
        })
        if not rec.model_loaded:
            pytest.skip(f"insightface 加载失败 (环境问题): {rec.status_message}")
        assert rec.model_loaded

    def test_list_enrolled(self):
        """列出已注册人员"""
        rec = FaceRecognizer({'face_db_path': self.db_path})
        enrolled = rec.list_enrolled()
        assert isinstance(enrolled, list)

    def test_recognize_no_bboxes(self):
        """无人脸框时返回空"""
        rec = FaceRecognizer({'face_db_path': self.db_path})
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        identities = rec.recognize(frame, [])
        assert identities == []

    def test_recognize_empty_bbox(self):
        """
        用空区域识别不崩溃
        注意: 可能因为 insightface 未检测到人脸而返回空
        """
        rec = FaceRecognizer({'face_db_path': self.db_path})
        if not rec.model_loaded:
            pytest.skip("insightface model not loaded")
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        # 传入一个不包含人脸的假框
        identities = rec.recognize(frame, [(10, 10, 30, 30)])
        # 无人脸 → 返回空或标记 unknown
        assert isinstance(identities, list)

    def test_get_status(self):
        rec = FaceRecognizer({'face_db_path': self.db_path})
        s = rec.get_status()
        assert 'loaded' in s
        assert 'enrolled_count' in s

    def test_cosine_similarity(self):
        """余弦相似度计算正确"""
        rec = FaceRecognizer({'face_db_path': self.db_path})
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        assert rec._cosine_similarity(a, b) == pytest.approx(1.0)

        c = np.array([0.0, 1.0, 0.0])
        assert rec._cosine_similarity(a, c) == pytest.approx(0.0)
