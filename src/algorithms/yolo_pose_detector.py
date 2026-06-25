"""
YOLOv11n-pose 姿态检测器
使用 YOLO11n-pose 模型检测人体 17 个 COCO 关键点，
映射为 PoseNet 18 关键点格式以保持向后兼容。
无模型时回退到 HOG 人体检测。
"""

import cv2
import numpy as np
import logging
import os
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# COCO 17 关键点索引 → 名称
COCO_KEYPOINTS = {
    0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear",
    5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow", 8: "right_elbow",
    9: "left_wrist", 10: "right_wrist", 11: "left_hip", 12: "right_hip",
    13: "left_knee", 14: "right_knee", 15: "left_ankle", 16: "right_ankle"
}

# COCO → PoseNet 关键点映射（用于向后兼容 FallDetector / BehaviorAnalyzer）
# PoseNet 18 部件: Nose, Neck, RShoulder, RElbow, RWrist, LShoulder, LElbow,
#   LWrist, RHip, RKnee, RAnkle, LHip, LKnee, LAnkle, REye, LEye, REar, LEar, Background
COCO_TO_POSENET_PART = {
    0:  "Nose",
    5:  "LShoulder",
    6:  "RShoulder",
    7:  "LElbow",
    8:  "RElbow",
    9:  "LWrist",
    10: "RWrist",
    11: "LHip",
    12: "RHip",
    13: "LKnee",
    14: "RKnee",
    15: "LAnkle",
    16: "RAnkle",
    1:  "LEye",
    2:  "REye",
    3:  "LEar",
    4:  "REar",
}


class YOLOPoseDetector:
    """YOLOv11n-pose 姿态检测器，输出与 SkeletonDetector 兼容的关键点格式"""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.model_path = self.config.get("model_path", "./models/yolo_pose/")
        self.confidence_threshold = self.config.get("confidence_threshold", 0.5)
        self.iou_threshold = self.config.get("iou_threshold", 0.5)
        self.device = self.config.get("device", 0)  # 默认 GPU
        self.frame_skip = self.config.get("frame_skip", 5)  # 内部跳帧

        self.model_loaded = False
        self.using_fallback = False
        self.status_message = "YOLO-pose 模型未加载，使用 HOG 回退检测"
        self.model = None
        self._pose_frame_count = 0
        self._cached_result: List[Dict] = []

        # 姿态绘制连接对（PoseNet 风格）
        self.POSE_PAIRS = [
            ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
            ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
            ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"],
            ["Neck", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"],
            ["Neck", "Nose"], ["Nose", "REye"], ["REye", "REar"],
            ["Nose", "LEye"], ["LEye", "LEar"]
        ]

        self._load_model()

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        """加载 YOLOv11n-pose 模型（支持 .pt / .onnx）"""
        model_file = self._find_model_file()
        if model_file is None:
            self.using_fallback = True
            self.status_message = (
                f"YOLO-pose 模型文件未找到于 {self.model_path}，使用 HOG 回退"
            )
            logger.warning(self.status_message)
            return

        try:
            from ultralytics import YOLO
        except Exception as e:
            self.using_fallback = True
            self.status_message = f"ultralytics 导入失败: {str(e)[:80]}，使用 HOG 回退"
            logger.warning(self.status_message)
            return

        try:
            is_onnx = model_file.endswith('.onnx')
            self.model = YOLO(model_file, task='pose') if is_onnx else YOLO(model_file)
            self.model_loaded = True
            self.using_fallback = False
            self.status_message = f"YOLO-pose 模型已加载: {model_file}"
            logger.info(self.status_message)
            # GPU 预热
            import numpy as _np
            dummy = _np.zeros((640, 640, 3), dtype=_np.uint8)
            try:
                self.model(dummy, device=self.device, imgsz=416, half=True, verbose=False)
            except Exception:
                pass
        except Exception as e:
            self.using_fallback = True
            self.status_message = f"YOLO-pose 模型加载失败: {str(e)}，使用 HOG 回退"
            logger.warning(self.status_message)

    def _find_model_file(self) -> Optional[str]:
        """在 model_path 目录中查找模型文件（OpenVINO > YOLO26 > .pt > .onnx）"""
        if not os.path.isdir(self.model_path):
            return None
        # 优先 OpenVINO IR 目录
        for name in os.listdir(self.model_path):
            if 'openvino_model' in name.lower() and os.path.isdir(
                os.path.join(self.model_path, name)
            ):
                return os.path.join(self.model_path, name)
        # YOLO26n-pose .pt
        for name in os.listdir(self.model_path):
            if 'yolo26' in name.lower() and name.endswith('.pt'):
                return os.path.join(self.model_path, name)
        # 任意 .pt
        for name in os.listdir(self.model_path):
            if name.endswith('.pt'):
                return os.path.join(self.model_path, name)
        # 任意 .onnx
        for name in os.listdir(self.model_path):
            if name.endswith('.onnx'):
                return os.path.join(self.model_path, name)
        return None

    # ------------------------------------------------------------------
    # 主检测接口
    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        检测帧中所有人的姿态关键点。
        内部 frame_skip: 每 N 帧推理一次,其他帧返回缓存。
        """
        self._pose_frame_count += 1
        # 跳帧时返回缓存结果
        if self._pose_frame_count % self.frame_skip != 0 and self._cached_result:
            return self._cached_result

        if self.model is None or not self.model_loaded:
            self.using_fallback = True
            return self._fallback_detect(frame)

        try:
            results = self.model(
                frame, conf=self.confidence_threshold,
                iou=self.iou_threshold, device=self.device,
                imgsz=416, half=True, verbose=False,
            )
            self._cached_result = self._parse_results(results, frame.shape)
            return self._cached_result
        except Exception as e:
            logger.error(f"YOLO-pose 检测错误: {str(e)}")
            return self._fallback_detect(frame)

    # ------------------------------------------------------------------
    # 结果解析
    # ------------------------------------------------------------------
    def _parse_results(self, results, frame_shape) -> List[Dict]:
        """将 Ultralytics Results 解析为兼容的关键点格式"""
        persons = []
        h, w = frame_shape[:2]

        for result in results:
            if result.keypoints is None:
                continue

            keypoints_data = result.keypoints.data  # (N, 17, 3) tensor
            boxes = result.boxes

            for i, kp_tensor in enumerate(keypoints_data):
                kp_np = kp_tensor.cpu().numpy()  # (17, 3)

                keypoints = self._coco_to_posenet_keypoints(kp_np, w, h)

                avg_conf = float(np.mean(kp_np[:, 2]))

                # 边界框
                bbox = None
                if boxes is not None and i < len(boxes):
                    xyxy = boxes[i].xyxy.cpu().numpy()[0]
                    bbox = (
                        int(xyxy[0]), int(xyxy[1]),
                        int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1])
                    )

                persons.append({
                    "keypoints": keypoints,
                    "confidence": avg_conf,
                    "bbox": bbox,
                })

        return persons

    def _coco_to_posenet_keypoints(
        self, kp_np: np.ndarray, frame_w: int, frame_h: int
    ) -> List[Dict]:
        """
        将 COCO 17 关键点 (17, 3) 映射为 PoseNet 风格的 18 关键点列表。
        - Neck 由左右肩中点合成
        - 缺失的 COCO 关键点（如耳朵）用鼻子位置近似
        """
        def _pt(coco_idx: int):
            """安全获取 COCO 关键点坐标和置信度"""
            if 0 <= coco_idx < len(kp_np):
                x, y, c = kp_np[coco_idx]
                return int(x), int(y), float(c)
            return 0, 0, 0.0

        # 读取 COCO 17 关键点
        nose = _pt(0)
        left_eye = _pt(1)
        right_eye = _pt(2)
        left_ear = _pt(3)
        right_ear = _pt(4)
        left_shoulder = _pt(5)
        right_shoulder = _pt(6)
        left_elbow = _pt(7)
        right_elbow = _pt(8)
        left_wrist = _pt(9)
        right_wrist = _pt(10)
        left_hip = _pt(11)
        right_hip = _pt(12)
        left_knee = _pt(13)
        right_knee = _pt(14)
        left_ankle = _pt(15)
        right_ankle = _pt(16)

        # 合成 Neck = 左右肩中点
        if left_shoulder[2] > 0 and right_shoulder[2] > 0:
            neck_x = (left_shoulder[0] + right_shoulder[0]) // 2
            neck_y = (left_shoulder[1] + right_shoulder[1]) // 2
            neck_conf = (left_shoulder[2] + right_shoulder[2]) / 2
        else:
            # 仅有一个肩部 → 用鼻子下移估算
            neck_x = nose[0]
            neck_y = nose[1] + int(frame_h * 0.08)
            neck_conf = nose[2] * 0.8

        keypoints = [
            {"part": "Nose",      "x": nose[0],  "y": nose[1],  "confidence": nose[2]},
            {"part": "Neck",      "x": neck_x,    "y": neck_y,    "confidence": neck_conf},
            {"part": "RShoulder", "x": right_shoulder[0], "y": right_shoulder[1], "confidence": right_shoulder[2]},
            {"part": "RElbow",    "x": right_elbow[0],    "y": right_elbow[1],    "confidence": right_elbow[2]},
            {"part": "RWrist",    "x": right_wrist[0],    "y": right_wrist[1],    "confidence": right_wrist[2]},
            {"part": "LShoulder", "x": left_shoulder[0],  "y": left_shoulder[1],  "confidence": left_shoulder[2]},
            {"part": "LElbow",    "x": left_elbow[0],     "y": left_elbow[1],     "confidence": left_elbow[2]},
            {"part": "LWrist",    "x": left_wrist[0],     "y": left_wrist[1],     "confidence": left_wrist[2]},
            {"part": "RHip",      "x": right_hip[0],      "y": right_hip[1],      "confidence": right_hip[2]},
            {"part": "RKnee",     "x": right_knee[0],     "y": right_knee[1],     "confidence": right_knee[2]},
            {"part": "RAnkle",    "x": right_ankle[0],    "y": right_ankle[1],    "confidence": right_ankle[2]},
            {"part": "LHip",      "x": left_hip[0],       "y": left_hip[1],       "confidence": left_hip[2]},
            {"part": "LKnee",     "x": left_knee[0],      "y": left_knee[1],      "confidence": left_knee[2]},
            {"part": "LAnkle",    "x": left_ankle[0],     "y": left_ankle[1],     "confidence": left_ankle[2]},
            {"part": "REye",      "x": right_eye[0],      "y": right_eye[1],      "confidence": right_eye[2]},
            {"part": "LEye",      "x": left_eye[0],       "y": left_eye[1],       "confidence": left_eye[2]},
            {"part": "REar",      "x": right_ear[0],      "y": right_ear[1],      "confidence": right_ear[2]},
            {"part": "LEar",      "x": left_ear[0],       "y": left_ear[1],       "confidence": left_ear[2]},
        ]
        return keypoints

    # ------------------------------------------------------------------
    # HOG 回退（与 SkeletonDetector._fallback_detect 一致）
    # ------------------------------------------------------------------
    def _fallback_detect(self, frame: np.ndarray) -> List[Dict]:
        """HOG 人体检测 + 合成关键点（无 YOLO 模型时使用）"""
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        boxes, _ = hog.detectMultiScale(
            gray, winStride=(8, 8), padding=(16, 16), scale=1.05
        )

        results = []
        for (x, y, w, h) in boxes:
            person = {
                "keypoints": [
                    {"part": "Neck",      "x": x + w // 2,     "y": y + h // 4,     "confidence": 0.8},
                    {"part": "RShoulder", "x": x + w // 4,     "y": y + h // 4,     "confidence": 0.7},
                    {"part": "LShoulder", "x": x + 3 * w // 4, "y": y + h // 4,     "confidence": 0.7},
                    {"part": "RHip",      "x": x + w // 4,     "y": y + 3 * h // 4, "confidence": 0.7},
                    {"part": "LHip",      "x": x + 3 * w // 4, "y": y + 3 * h // 4, "confidence": 0.7},
                    {"part": "RKnee",     "x": x + w // 4,     "y": y + h,          "confidence": 0.6},
                    {"part": "LKnee",     "x": x + 3 * w // 4, "y": y + h,          "confidence": 0.6},
                ],
                "confidence": 0.6,
                "bbox": (x, y, w, h),
            }
            results.append(person)

        return results

    # ------------------------------------------------------------------
    # 骨架绘制
    # ------------------------------------------------------------------
    def draw_skeleton(self, frame: np.ndarray, persons: List[Dict]) -> np.ndarray:
        """在帧上绘制骨架关键点和连接线"""
        frame = frame.copy()

        for person in persons:
            kp = {k["part"]: (k["x"], k["y"]) for k in person.get("keypoints", [])}

            for part_a, part_b in self.POSE_PAIRS:
                if part_a in kp and part_b in kp:
                    cv2.line(frame, kp[part_a], kp[part_b], (0, 255, 0), 2)

            for part, (px, py) in kp.items():
                cv2.circle(frame, (px, py), 5, (0, 0, 255), -1)

        return frame

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------
    def get_status(self) -> Dict:
        return {
            "loaded": self.model_loaded,
            "fallback": self.using_fallback,
            "message": self.status_message,
            "path": self.model_path,
        }

    def reset(self) -> None:
        """重置检测器状态"""
        pass
