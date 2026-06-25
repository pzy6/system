"""测试 YOLO 姿态检测器"""
import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from algorithms.yolo_pose_detector import YOLOPoseDetector


class TestYOLOPoseDetector:
    def setup_method(self):
        self.model_path = os.path.join(
            os.path.dirname(__file__), '..', 'models', 'yolo_pose'
        )

    def test_model_loading(self):
        """模型文件存在时加载成功"""
        detector = YOLOPoseDetector({
            'model_path': self.model_path,
            'confidence_threshold': 0.5,
        })
        assert detector.model_loaded
        assert not detector.using_fallback

    def test_coco_to_posenet_mapping(self):
        """COCO 17关键点→PoseNet 18关键点映射正确"""
        detector = YOLOPoseDetector({'model_path': './nonexistent/'})
        kp_np = np.zeros((17, 3), dtype=np.float32)
        kp_np[0] = [320, 100, 0.9]   # nose
        kp_np[5] = [280, 160, 0.9]   # left_shoulder
        kp_np[6] = [360, 160, 0.9]   # right_shoulder
        kp_np[11] = [290, 280, 0.9]  # left_hip
        kp_np[12] = [350, 280, 0.9]  # right_hip

        result = detector._coco_to_posenet_keypoints(kp_np, 640, 480)

        # 应包含 18 个关键点
        assert len(result) == 18

        # Neck 应由双肩中点合成
        neck = next(k for k in result if k['part'] == 'Neck')
        assert neck['x'] == 320  # (280+360)//2
        assert neck['y'] == 160

        # Nose 直接映射
        nose = next(k for k in result if k['part'] == 'Nose')
        assert nose['x'] == 320
        assert nose['y'] == 100

    def test_hog_fallback_when_no_model(self):
        """无模型时回退到 HOG 检测"""
        detector = YOLOPoseDetector({'model_path': './nonexistent/'})
        assert detector.using_fallback
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        skeletons = detector.detect(frame)
        assert isinstance(skeletons, list)

    def test_draw_skeleton_no_crash(self):
        """骨架绘制不崩溃"""
        detector = YOLOPoseDetector({'model_path': './nonexistent/'})
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        persons = [{
            'keypoints': [
                {'part': 'Neck', 'x': 320, 'y': 150, 'confidence': 0.9},
                {'part': 'Nose', 'x': 320, 'y': 100, 'confidence': 0.9},
                {'part': 'RShoulder', 'x': 360, 'y': 160, 'confidence': 0.9},
                {'part': 'LShoulder', 'x': 280, 'y': 160, 'confidence': 0.9},
            ],
            'confidence': 0.8,
        }]
        result = detector.draw_skeleton(frame, persons)
        assert result.shape == frame.shape

    def test_get_status(self):
        detector = YOLOPoseDetector({'model_path': self.model_path})
        s = detector.get_status()
        assert 'loaded' in s
        assert 'message' in s
