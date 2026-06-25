"""
情绪识别模块 — EfficientNet-B0 (FER2013) + 启发式回退
优先使用训练好的 ONNX 模型，不可用时回退到启发式分析。
"""

import cv2
import numpy as np
import logging
import os
from typing import Dict, List

logger = logging.getLogger(__name__)


class EmotionRecognizer:
    """情绪识别器 (EfficientNet-B0 + 启发式回退)"""

    EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    EMOTION_CN = {
        "angry": "生气", "disgust": "反感", "fear": "恐惧",
        "happy": "高兴", "sad": "悲伤", "surprise": "惊讶",
        "neutral": "平静",
    }
    NEGATIVE = {"angry", "disgust", "fear", "sad"}

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.model_path = self.config.get(
            "model_path",
            os.path.join(os.path.dirname(__file__), "..", "..",
                         "models", "emotion", "efficient_emotion.onnx")
        )
        self.model_loaded = False
        self.onnx_session = None
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._load_model()
        if not self.model_loaded:
            self.using_fallback = True
            self.status_message = "EfficientNet 不可用, 使用启发式分析"
        else:
            self.using_fallback = False
            self.status_message = "EfficientNet-B0 就绪 (FER2013, 7类)"

    def _load_model(self):
        """加载 ONNX 模型 (优选 INT8 量化版)"""
        base = os.path.join(os.path.dirname(__file__), "..", "..", "models", "emotion")
        candidates = [
            os.path.join(base, "efficient_emotion_int8.onnx"),
            os.path.join(base, "efficient_emotion.onnx"),
            self.model_path,
        ]
        onnx_path = None
        for p in candidates:
            if os.path.exists(p):
                onnx_path = p; break
        if onnx_path is None:
            logger.info("情绪 ONNX 模型未找到")
            return
        try:
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.log_severity_level = 3
            try:
                self.onnx_session = ort.InferenceSession(
                    onnx_path,
                    sess_options=options,
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
                )
                logger.info(f"EfficientNet-B0 loaded with providers: {self.onnx_session.get_providers()}")
            except Exception as exc:
                logger.warning("CUDA 初始化失败，已回退到 CPU: %s", exc)
                self.onnx_session = ort.InferenceSession(
                    onnx_path,
                    sess_options=options,
                    providers=["CPUExecutionProvider"]
                )
                logger.info(f"EfficientNet-B0 loaded with providers: {self.onnx_session.get_providers()}")
            self.model_loaded = True
            logger.info(f"EfficientNet-B0 loaded: {onnx_path}")
        except Exception as e:
            logger.warning(f"ONNX load failed: {e}")

    def _preprocess_face(self, roi_bgr: np.ndarray) -> np.ndarray:
        """预处理人脸 ROI 为模型输入 (1, 3, 224, 224)"""
        roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)  # BGR→RGB
        roi = cv2.resize(roi, (224, 224))
        roi = roi.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        roi = (roi - mean) / std
        roi = np.transpose(roi, (2, 0, 1))  # HWC→CHW
        return np.expand_dims(roi, axis=0)   # → (1, 3, 224, 224)

    def analyze_face_roi(self, face_roi_bgr: np.ndarray) -> Dict:
        """直接分析人脸 ROI (跳过 Haar 检测, 配合 YOLO 人脸框使用)"""
        if face_roi_bgr is None or face_roi_bgr.size == 0:
            return {"emotion": "unknown", "emotion_cn": "无数据",
                    "confidence": 0.0, "scores": {}}

        # 尝试 ONNX 推理
        if self.onnx_session is not None:
            try:
                inp = self._preprocess_face(face_roi_bgr)
                out = self.onnx_session.run(None, {"input": inp})[0][0]
                out = np.exp(out) / np.sum(np.exp(out))  # softmax
                scores = {e: round(float(s), 3) for e, s in zip(self.EMOTIONS, out)}
                top = max(scores, key=scores.get)
                return {
                    "emotion": top,
                    "emotion_cn": self.EMOTION_CN.get(top, top),
                    "confidence": round(scores[top], 2),
                    "scores": scores,
                    "is_negative": top in self.NEGATIVE and scores[top] >= 0.6,
                    "method": "EfficientNet",
                }
            except Exception as e:
                logger.error(f"ONNX inference error: {e}")

        # 回退: 启发式
        gray_roi = cv2.cvtColor(face_roi_bgr, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray_roi) / 255.0
        texture = np.std(gray_roi) / 128.0
        scores = dict.fromkeys(self.EMOTIONS, 0.05)
        scores["neutral"] = 0.30
        if brightness < 0.35:
            scores["sad"] += 0.25
        if brightness > 0.65:
            scores["happy"] += 0.25
        if texture > 0.55:
            scores["fear"] += 0.15
            scores["surprise"] += 0.10
        total = sum(scores.values())
        scores = {k: round(v / max(total, 1e-6), 2) for k, v in scores.items()}
        top = max(scores, key=scores.get)
        return {
            "emotion": top,
            "emotion_cn": self.EMOTION_CN.get(top, top),
            "confidence": scores[top],
            "scores": scores,
            "is_negative": top in self.NEGATIVE and scores[top] >= 0.6,
            "method": "heuristic",
        }

    def analyze(self, frame: np.ndarray, keypoints: List = None) -> Dict:
        """分析情绪 (ONNX优先, 启发式回退)"""
        # 人脸检测
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        if len(faces) == 0:
            return {
                "emotion": "neutral", "emotion_cn": "未检测到人脸",
                "confidence": 0.0, "scores": {},
            }

        x, y, w, h = faces[0]
        face_roi = frame[y:y + h, x:x + w]

        # 尝试 ONNX 推理
        if self.onnx_session is not None:
            try:
                inp = self._preprocess_face(face_roi)
                out = self.onnx_session.run(None, {"input": inp})[0][0]
                out = np.exp(out) / np.sum(np.exp(out))  # softmax
                scores = {e: round(float(s), 3) for e, s in zip(self.EMOTIONS, out)}
                top = max(scores, key=scores.get)
                return {
                    "emotion": top,
                    "emotion_cn": self.EMOTION_CN.get(top, top),
                    "confidence": round(scores[top], 2),
                    "scores": scores,
                    "is_negative": top in self.NEGATIVE and scores[top] >= 0.6,
                    "method": "EfficientNet",
                }
            except Exception as e:
                logger.error(f"ONNX inference error: {e}")

        # 回退: 启发式
        gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY)
        brightness = np.mean(gray_roi) / 255.0
        texture = np.std(gray_roi) / 128.0

        scores = dict.fromkeys(self.EMOTIONS, 0.05)
        scores["neutral"] = 0.30
        if brightness < 0.35:
            scores["sad"] += 0.25
        if brightness > 0.65:
            scores["happy"] += 0.25
        if texture > 0.55:
            scores["fear"] += 0.15
            scores["surprise"] += 0.10

        total = sum(scores.values())
        scores = {k: round(v / max(total, 1e-6), 2) for k, v in scores.items()}
        top = max(scores, key=scores.get)
        return {
            "emotion": top,
            "emotion_cn": self.EMOTION_CN.get(top, top),
            "confidence": scores[top],
            "scores": scores,
            "is_negative": top in self.NEGATIVE and scores[top] >= 0.6,
            "method": "heuristic",
        }

    def get_status(self) -> Dict:
        return {
            "loaded": self.model_loaded,
            "fallback": not self.model_loaded,
            "message": self.status_message,
            "path": self.model_path,
        }
