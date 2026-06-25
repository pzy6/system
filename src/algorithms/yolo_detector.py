"""
YOLO 双模型检测器：人脸 + 烟雾

架构：
  - face_model: 人脸检测（face_yolov8n.pt / face_yolov8s.pt）
  - smoke_model: 烟雾检测（yolo11n_smoke.pt，训练后）
  - 回退: COCO person→face 映射 或 Haar Cascade

模型优先级（按文件名自动识别）：
  face_*.pt/onnx → 人脸模型
  smoke_*.pt/onnx → 烟雾模型
  yolo11n.pt/onnx → COCO 基础模型（person→face 映射，无烟雾）
  yolo11n_face_smoke.pt/onnx → 自定义双类模型
"""

import cv2
import numpy as np
import logging
import os
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

COCO_PERSON_CLASS_ID = 0  # COCO "person" class


class YOLODetector:
    """双模型人脸 + 烟雾检测器"""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.model_path = self.config.get("model_path", "./models/yolo_face_smoke/")
        self.confidence_threshold = self.config.get("confidence_threshold", 0.5)
        self.iou_threshold = self.config.get("iou_threshold", 0.45)
        self.device = self.config.get("device", 0)  # 默认 GPU
        self.frame_skip = self.config.get("frame_skip", 5)

        # 双模型
        self.face_model = None       # 人脸检测模型
        self.smoke_model = None      # 烟雾检测模型
        self.face_loaded = False
        self.smoke_loaded = False
        self.using_fallback = False
        self.status_message = "未加载任何模型"

        self._frame_count = 0
        self._last_result = None

        # Haar Cascade 终极回退
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        self._load_models()

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------
    def _load_models(self) -> None:
        """扫描模型目录，按文件名自动分配人脸/烟雾模型"""
        if not os.path.isdir(self.model_path):
            self.using_fallback = True
            self.status_message = f"模型目录不存在: {self.model_path}，使用 Haar Cascade 回退"
            logger.warning(self.status_message)
            return

        try:
            from ultralytics import YOLO
        except Exception as e:
            self.using_fallback = True
            self.status_message = f"ultralytics 导入失败: {str(e)[:80]}，使用 Haar Cascade 回退"
            logger.warning(self.status_message)
            return

        # OpenVINO IR 目录优先，.pt/.onnx 备选
        ov_dirs = [
            f for f in os.listdir(self.model_path)
            if 'openvino_model' in f.lower() and os.path.isdir(
                os.path.join(self.model_path, f)
            )
        ]
        pt_files = sorted([
            f for f in os.listdir(self.model_path)
            if f.endswith((".pt", ".onnx"))
        ])

        # 检测 ModelScope DAMOYOLO 烟火模型
        self.smoke_modelscope_pipe = None
        modelscope_dir = os.path.join(self.model_path, "modelscope")
        if os.path.isdir(modelscope_dir):
            self.smoke_modelscope_pipe = self._try_load_modelscope_smoke(modelscope_dir)

        if not pt_files and not ov_dirs and self.smoke_modelscope_pipe is None:
            self.using_fallback = True
            self.status_message = f"模型目录无模型文件: {self.model_path}"
            logger.warning(self.status_message)
            return

        parts = []

        # 加载 OpenVINO 模型
        for ov_dir in sorted(ov_dirs):
            ov_path = os.path.join(self.model_path, ov_dir)
            try:
                model = YOLO(ov_path, task="detect")
                ov_lower = ov_dir.lower()
                if "smoke" in ov_lower:
                    self.smoke_model = model
                    self.smoke_loaded = True
                    parts.append(f"烟雾={ov_dir}(OpenVINO)")
                elif "face" in ov_lower:
                    self.face_model = model
                    self.face_loaded = True
                    parts.append(f"人脸={ov_dir}(OpenVINO)")
                elif self.face_model is None:
                    self.face_model = model
                    self.face_loaded = True
                    parts.append(f"通用={ov_dir}(OpenVINO)")
            except Exception as e:
                logger.warning(f"加载 OpenVINO 模型失败 {ov_dir}: {e}")

        # 加载 ModelScope 烟雾模型
        if self.smoke_modelscope_pipe is not None:
            self.smoke_loaded = True
            parts.append("烟雾=DAMOYOLO(ModelScope)")

        # 加载 .pt/.onnx 文件
        for fname in pt_files:
            filepath = os.path.join(self.model_path, fname)
            fname_lower = fname.lower()
            is_onnx = fname.endswith(".onnx")

            try:
                model = YOLO(filepath, task="detect") if is_onnx else YOLO(filepath)
            except Exception as e:
                logger.warning(f"加载模型失败 {fname}: {e}")
                continue

            if "face_yolo" in fname_lower or "face_detect" in fname_lower:
                self.face_model = model
                self.face_loaded = True
                parts.append(f"人脸={fname}")
            elif "smoke" in fname_lower and self.smoke_modelscope_pipe is None:
                self.smoke_model = model
                self.smoke_loaded = True
                parts.append(f"烟雾={fname}")
            elif "yolo26n" in fname_lower:
                if self.face_model is None:
                    self.face_model = model
                    self.face_loaded = True
                parts.append(f"YOLO26→人脸={fname}")
            elif "yolo11n" in fname_lower or "yolov11n" in fname_lower:
                if self.face_model is None:
                    self.face_model = model
                    self.face_loaded = True
                parts.append(f"COCO→人脸={fname}")
            else:
                if self.face_model is None:
                    self.face_model = model
                    self.face_loaded = True
                parts.append(f"通用={fname}")

        if self.face_loaded or self.smoke_loaded:
            self.using_fallback = False
            self.status_message = " | ".join(parts)
            logger.info(f"YOLO 检测器就绪: {self.status_message}")
            self._warmup_models()  # GPU 预热
        else:
            self.using_fallback = True
            self.status_message = "所有模型加载失败，使用 Haar Cascade 回退"

    def _warmup_models(self):
        """GPU 预热: dummy 推理初始化 CUDA kernels"""
        import numpy as np
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        for name, model in [("face", self.face_model), ("smoke", self.smoke_model)]:
            if model is not None:
                try:
                    model(dummy, device=self.device, half=True, verbose=False)
                except Exception:
                    pass  # 预热失败不影响运行

    def _try_load_modelscope_smoke(self, modelscope_dir: str):
        """尝试加载 ModelScope DAMOYOLO 烟火模型"""
        try:
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks

            # 查找 damoyolo 模型目录
            for root, dirs, files in os.walk(modelscope_dir):
                for f in files:
                    if f.endswith('.pt') and ('smokefire' in f.lower() or 'smoke' in f.lower()):
                        model_dir = os.path.dirname(os.path.join(root, f))
                        pipe = pipeline(
                            Tasks.image_object_detection,
                            model=model_dir,
                            device='cpu',
                        )
                        logger.info(f"ModelScope DAMOYOLO 烟雾模型已加载: {model_dir}")
                        return pipe
        except ImportError:
            logger.info("modelscope 未安装，跳过 DAMOYOLO 模型")
        except Exception as e:
            logger.warning(f"ModelScope 模型加载失败: {e}")
        return None

    # ------------------------------------------------------------------
    # 主检测接口
    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> Dict:
        """
        返回: {face_bboxes, smoke_detected, smoke_bboxes, smoke_confidence}
        """
        self._frame_count += 1

        # 帧跳：人脸、烟雾都降频
        skip_face = (self.frame_skip > 0
                     and self._frame_count % self.frame_skip != 0)
        skip_smoke = (self.frame_skip > 0
                      and self._frame_count % self.frame_skip != 0)  # 烟雾与人脸同步跳帧

        if skip_face and skip_smoke and self._last_result is not None:
            return dict(self._last_result)

        if skip_face and self._last_result is not None:
            result = dict(self._last_result)
            if self.smoke_loaded and not skip_smoke:
                result.update(self._detect_smoke(frame))
            return result

        # 完整检测（人脸按 frame_skip，烟雾每 2 帧）
        face_result = self._detect_face(frame)
        if self.smoke_loaded and not skip_smoke:
            smoke_result = self._detect_smoke(frame)
        elif self._last_result is not None:
            smoke_result = {
                "smoke_detected": self._last_result.get("smoke_detected", False),
                "smoke_bboxes": self._last_result.get("smoke_bboxes", []),
                "smoke_confidence": self._last_result.get("smoke_confidence", 0.0),
                "fire_detected": self._last_result.get("fire_detected", False),
                "fire_bboxes": self._last_result.get("fire_bboxes", []),
                "fire_confidence": self._last_result.get("fire_confidence", 0.0),
            }
        else:
            smoke_result = {
                "smoke_detected": False, "smoke_bboxes": [], "smoke_confidence": 0.0,
                "fire_detected": False, "fire_bboxes": [], "fire_confidence": 0.0,
            }

        result = {**face_result, **smoke_result}
        self._last_result = result
        return result

    def _detect_face(self, frame: np.ndarray) -> Dict:
        """人脸检测：face_model > COCO person → Haar Cascade"""
        face_bboxes = []

        if self.face_model is not None:
            try:
                # 直接预测 (不用 tracker — 养老院人脸无需 ByteTrack 开销)
                results = self.face_model.predict(
                    frame,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    device=self.device,
                    imgsz=320,
                    half=True,
                    verbose=False
                )
                for r in results:
                    if r.boxes is None:
                        continue
                    for box in r.boxes:
                        cls_id = int(box.cls.cpu().item())
                        conf = float(box.conf.cpu().item())
                        xyxy = box.xyxy.cpu().numpy().flatten()
                        x, y = int(xyxy[0]), int(xyxy[1])
                        w, h = int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1])

                        # 人脸模型: class 0 = face (专用人脸检测器输出)
                        if cls_id == 0 and h < frame.shape[0] * 0.6:
                            face_bboxes.append((x, y, w, h))
            except Exception as e:
                logger.error(f"人脸检测错误: {e}")

        if not face_bboxes:
            # 回退 Haar Cascade
            face_bboxes = self._haar_detect(frame)

        return {"face_bboxes": face_bboxes}

    def _detect_smoke(self, frame: np.ndarray) -> Dict:
        """烟雾/火焰检测：ultralytics YOLO 或 ModelScope DAMOYOLO"""
        smoke_bboxes = []
        fire_bboxes = []
        smoke_confidence = 0.0
        fire_confidence = 0.0

        # ModelScope DAMOYOLO 后端
        if self.smoke_modelscope_pipe is not None:
            try:
                # ModelScope pipeline 需要 BGR 图像
                img_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                result = self.smoke_modelscope_pipe(img_bgr)
                # result = {'detection_boxes': [...], 'detection_scores': [...], 'detection_classes': [...]}
                boxes = result.get('detection_boxes', [])
                scores = result.get('detection_scores', [])
                classes = result.get('detection_classes', [])
                for bbox, score, cls in zip(boxes, scores, classes):
                    # DAMOYOLO boxes: [x1, y1, x2, y2] 归一化 [0,1]
                    h, w = frame.shape[:2]
                    x1, y1, x2, y2 = bbox
                    bx = int(x1 * w)
                    by = int(y1 * h)
                    bw = int((x2 - x1) * w)
                    bh = int((y2 - y1) * h)
                    # DAMOYOLO smokefire: class 0=fire, 1=smoke (不确定顺序，都收录)
                    if score >= self.confidence_threshold:
                        smoke_bboxes.append((bx, by, bw, bh))
                        smoke_confidence = max(smoke_confidence, float(score))
            except Exception as e:
                logger.error(f"ModelScope 烟雾检测错误: {e}")

        # ultralytics YOLO 后端
        elif self.smoke_model is not None:
            try:
                results = self.smoke_model(
                    frame, conf=self.confidence_threshold,
                    iou=self.iou_threshold, device=self.device,
                    imgsz=320, half=True, verbose=False,
                )
                for r in results:
                    if r.boxes is None:
                        continue
                    for box in r.boxes:
                        cls_id = int(box.cls.cpu().item())
                        conf = float(box.conf.cpu().item())
                        xyxy = box.xyxy.cpu().numpy().flatten()
                        x, y = int(xyxy[0]), int(xyxy[1])
                        w, h = int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1])
                        if cls_id == 0:  # fire
                            fire_bboxes.append((x, y, w, h))
                            fire_confidence = max(fire_confidence, conf)
                        else:  # smoke
                            smoke_bboxes.append((x, y, w, h))
                            smoke_confidence = max(smoke_confidence, conf)
            except Exception as e:
                logger.error(f"YOLO 烟雾检测错误: {e}")

        return {
            "smoke_detected": len(smoke_bboxes) > 0 or len(fire_bboxes) > 0,
            "smoke_bboxes": smoke_bboxes,
            "smoke_confidence": smoke_confidence,
            "fire_detected": len(fire_bboxes) > 0,
            "fire_bboxes": fire_bboxes,
            "fire_confidence": fire_confidence,
        }

    def _haar_detect(self, frame: np.ndarray) -> List[Tuple]:
        """Haar Cascade 人脸检测回退"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------
    def get_status(self) -> Dict:
        return {
            "loaded": self.face_loaded or self.smoke_loaded,
            "fallback": self.using_fallback,
            "message": self.status_message,
            "path": self.model_path,
        }

    def reset(self) -> None:
        self._frame_count = 0
        self._last_result = None
