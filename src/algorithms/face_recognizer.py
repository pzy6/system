"""
人脸识别模块
基于 insightface (ArcFace) 的人脸识别，支持人脸数据库注册与管理。

人脸数据库结构：
  data/face_db/
    index.json            # {"name": [embedding_list], ...}
    张大爷/
      img001.jpg
      img002.jpg
    李奶奶/
      img001.jpg

使用方式：
  recognizer = FaceRecognizer(config)
  identities = recognizer.recognize(frame, face_bboxes)
  # → [{'name': '张大爷', 'confidence': 0.92, 'bbox': (x,y,w,h), 'is_unknown': False}, ...]
"""

import cv2
import numpy as np
import logging
import os
import json
import time
from typing import List, Dict, Optional, Tuple
from collections import deque

logger = logging.getLogger(__name__)


class FaceRecognizer:
    """基于 ArcFace 的人脸识别器"""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.face_db_path = self.config.get("face_db_path", "./data/face_db/")
        self.recognition_threshold = self.config.get("recognition_threshold", 0.6)
        self.model_name = self.config.get("model_name", "buffalo_l")
        self.device = self.config.get("device", "cpu")

        self.model_loaded = False
        self.using_fallback = False
        self.status_message = "人脸识别模型未加载"
        self.recognition_model = None
        self._arcface_model = None
        self._arcface_loaded = False

        self.face_index: Dict[str, List[np.ndarray]] = {}
        self._unknown_cooldown: Dict[str, float] = {}
        self._unknown_cooldown_timeout = self.config.get("unknown_face_timeout", 30)

        # 仅加载 ArcFace (w600k_r50), 不加载 FaceAnalysis (SCRFD+landmark+genderage)
        self._init_raw_arcface()
        if self._arcface_loaded:
            self._load_face_db()

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------
    def _init_raw_arcface(self):
        """加载 ArcFace w600k_r50.onnx (CPU, 仅 embedding)"""
        import os as _os
        _os.environ.setdefault('ORT_DISABLE_CUDA', '1')  # 仅 CPU, 避免与 YOLO GPU 冲突
        try:
            import insightface
            model_dir = _os.path.join(
                _os.path.expanduser("~"), ".insightface", "models", "buffalo_l")
            onnx_path = _os.path.join(model_dir, "w600k_r50.onnx")
            if not _os.path.exists(onnx_path):
                logger.info(f"ArcFace ONNX 未找到: {onnx_path}")
                return
            self._arcface_model = insightface.model_zoo.get_model(onnx_path)
            self._arcface_model.prepare(ctx_id=-1)  # CPU
            self._arcface_loaded = True
            self.model_loaded = True
            self.using_fallback = False
            self.status_message = "ArcFace 直接推理 (w600k_r50, CPU)"
            logger.info(self.status_message)
        except Exception as e:
            logger.warning(f"ArcFace 加载失败: {e}")
            self._arcface_loaded = False

    def _lazy_load_faceanalysis(self):
        """懒加载 FaceAnalysis — 仅 enrollment 调用"""
        if self.recognition_model is not None:
            return
        import onnxruntime as _ort
        _ort.set_default_logger_severity(4)
        try:
            import insightface
            options = _ort.SessionOptions()
            options.log_severity_level = 3
            try:
                self.recognition_model = insightface.app.FaceAnalysis(
                    name=self.model_name,
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                )
                logger.info(f"FaceAnalysis loaded with CUDA providers")
            except Exception as exc:
                logger.warning("CUDA 初始化失败，FaceAnalysis 已回退到 CPU: %s", exc)
                self.recognition_model = insightface.app.FaceAnalysis(
                    name=self.model_name,
                    providers=["CPUExecutionProvider"],
                )
                logger.info(f"FaceAnalysis loaded with CPU provider")
            self.recognition_model.prepare(ctx_id=-1, det_size=(640, 480))
            logger.info("FaceAnalysis 懒加载完成 (enrollment)")
        except Exception as e:
            logger.error(f"FaceAnalysis 懒加载失败: {e}")
        except ImportError:
            self.using_fallback = True
            self.status_message = "insightface 未安装，人脸识别不可用（pip install insightface）"
            logger.warning(self.status_message)
        except Exception as e:
            self.using_fallback = True
            self.status_message = f"人脸识别模型加载失败: {str(e)}"
            logger.warning(self.status_message)

    # ------------------------------------------------------------------
    # 人脸数据库管理
    # ------------------------------------------------------------------
    def _get_index_path(self) -> str:
        return os.path.join(self.face_db_path, "index.json")

    def _load_face_db(self) -> None:
        """从 index.json 加载人脸嵌入索引"""
        index_path = self._get_index_path()
        if not os.path.exists(index_path):
            # 从图片目录重建索引
            logger.info("index.json 不存在，尝试从图片目录重建人脸索引...")
            self._rebuild_index_from_images()
            return

        try:
            with open(index_path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            self.face_index = {}
            for name, embeddings in raw.items():
                if embeddings:
                    self.face_index[name] = [
                        np.array(emb, dtype=np.float32) for emb in embeddings
                    ]

            enrolled = len(self.face_index)
            self.status_message = (
                f"人脸识别模型已加载 (insightface {self.model_name})，"
                f"已注册 {enrolled} 人"
            )
            logger.info(f"已加载 {enrolled} 个注册人脸")
        except Exception as e:
            logger.error(f"加载人脸索引失败: {str(e)}")
            self.face_index = {}

    def _save_face_db(self) -> None:
        """保存人脸嵌入索引到 index.json"""
        os.makedirs(self.face_db_path, exist_ok=True)
        index_path = self._get_index_path()

        serializable = {}
        for name, embeddings in self.face_index.items():
            serializable[name] = [emb.tolist() for emb in embeddings]

        try:
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, ensure_ascii=False, indent=2)
            logger.info(f"人脸索引已保存到 {index_path}")
        except Exception as e:
            logger.error(f"保存人脸索引失败: {str(e)}")

    def _rebuild_index_from_images(self) -> None:
        """从 face_db 目录中的图片重建嵌入索引"""
        if not os.path.isdir(self.face_db_path):
            logger.info(f"人脸数据库目录不存在: {self.face_db_path}")
            return

        for name in os.listdir(self.face_db_path):
            person_dir = os.path.join(self.face_db_path, name)
            if not os.path.isdir(person_dir):
                continue

            embeddings = []
            for fname in os.listdir(person_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    img_path = os.path.join(person_dir, fname)
                    img = cv2.imread(img_path)
                    if img is not None:
                        emb = self._extract_embedding(img)
                        if emb is not None:
                            embeddings.append(emb)

            if embeddings:
                self.face_index[name] = embeddings
                logger.info(f"从图片重建: {name} ({len(embeddings)} 张)")

        if self.face_index:
            self._save_face_db()

    def _extract_embedding(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        """从人脸图片提取 ArcFace 嵌入向量（512 维）"""
        if self.recognition_model is None and not self._arcface_loaded:
            return None

        try:
            # 优先使用原始 ArcFace 直接推理 (跳过 SCRFD 重检测, ~5ms)
            if self._arcface_loaded:
                return self._arcface_model.get_feat(face_img)

            # 回退: 完整 FaceAnalysis pipeline (含检测+识别)
            faces = self.recognition_model.get(face_img)
            if faces and len(faces) > 0:
                best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                return best.embedding
            return None
        except Exception as e:
            logger.error(f"嵌入提取失败: {str(e)}")
            return None

    def _extract_embedding_from_roi(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """从帧中裁剪人脸 ROI 并提取嵌入"""
        x, y, w, h = bbox
        fh, fw = frame.shape[:2]
        x, y = max(0, x), max(0, y)
        w, h = min(w, fw - x), min(h, fh - y)
        if w <= 10 or h <= 10:
            return None

        face_roi = frame[y:y + h, x:x + w]
        # insightface/ArcFace ONNX 需要 BGR 输入
        if face_roi.shape[2] == 3:
            face_roi = cv2.cvtColor(face_roi, cv2.COLOR_RGB2BGR)
        return self._extract_embedding(face_roi)

    # ------------------------------------------------------------------
    # 匹配
    # ------------------------------------------------------------------
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """计算两个向量的余弦相似度"""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def _find_best_match(self, embedding: np.ndarray) -> Tuple[Optional[str], float, str]:
        """查找最佳匹配，返回 (name, score, level)
        level: 'confirmed' (>0.50), 'possible' (0.42-0.50), 'unknown' (<0.42)
        """
        best_name = None
        best_score = 0.0

        for name, embeddings in self.face_index.items():
            for ref_emb in embeddings:
                score = self._cosine_similarity(embedding, ref_emb)
                if score > best_score:
                    best_score = score
                    best_name = name

        # 分级阈值 (文档建议)
        if best_score >= 0.50:
            return best_name, best_score, 'confirmed'
        elif best_score >= 0.42:
            return best_name, best_score, 'possible'
        else:
            return None, best_score, 'unknown'

    def _is_in_cooldown(self, name: str) -> bool:
        """检查该名称是否在冷却时间内（避免重复告警）"""
        now = time.time()
        last = self._unknown_cooldown.get(name, 0)
        if now - last < self._unknown_cooldown_timeout:
            return True
        return False

    # ------------------------------------------------------------------
    # 主识别接口
    # ------------------------------------------------------------------
    def recognize(
        self,
        frame: np.ndarray,
        face_bboxes: List[Tuple[int, int, int, int]]
    ) -> List[Dict]:
        """
        对检测到的人脸进行身份识别。

        参数：
          frame: RGB 图像
          face_bboxes: [(x, y, w, h), ...] 人脸边界框列表

        返回：
          [
            {
              'name': '张大爷' | 'unknown',
              'confidence': float,        # 余弦相似度
              'bbox': (x, y, w, h),
              'is_unknown': bool
            },
            ...
          ]
        """
        if not self.model_loaded or not face_bboxes:
            return []

        identities = []
        for bbox in face_bboxes:
            embedding = self._extract_embedding_from_roi(frame, bbox)
            if embedding is None:
                continue

            name, score, level = self._find_best_match(embedding)

            if level == 'confirmed':
                identities.append({
                    "name": name,
                    "confidence": round(score, 4),
                    "bbox": bbox,
                    "is_unknown": False,
                    "match_level": "confirmed",
                })
            elif level == 'possible':
                # 可能匹配: 记录但不保证, 降低陌生人告警
                identities.append({
                    "name": name or "possible_match",
                    "confidence": round(score, 4),
                    "bbox": bbox,
                    "is_unknown": False,
                    "match_level": "possible",
                })
            else:
                # 未知人脸 (score < 0.42)
                unknown_id = f"unknown_{bbox[0] // 100}_{bbox[1] // 100}"
                if not self._is_in_cooldown(unknown_id):
                    self._unknown_cooldown[unknown_id] = time.time()
                    identities.append({
                        "name": "unknown",
                        "confidence": round(score, 4),
                        "bbox": bbox,
                        "is_unknown": True,
                        "match_level": "unknown",
                    })

        return identities

    # ------------------------------------------------------------------
    # 人脸注册管理
    # ------------------------------------------------------------------
    def enroll(self, name: str, face_images: List[np.ndarray]) -> bool:
        if self.recognition_model is None and not self._arcface_loaded:
            logger.error("人脸识别模型未加载，无法注册")
            return False

        embeddings = []
        for img in face_images:
            emb = self._extract_embedding(img)  # ArcFace 直接推理或 FaceAnalysis 回退
            if emb is not None:
                embeddings.append(emb)

        if not embeddings:
            logger.warning(f"未能从 {len(face_images)} 张图片中提取人脸特征: {name}")
            return False

        # 保存到索引
        self.face_index[name] = embeddings
        self._save_face_db()
        logger.info(f"已注册人脸: {name} ({len(embeddings)} 个特征向量)")
        return True

    def remove_enrollment(self, name: str) -> bool:
        """删除已注册的人脸"""
        if name in self.face_index:
            del self.face_index[name]
            self._save_face_db()
            logger.info(f"已删除注册人脸: {name}")
            return True
        logger.warning(f"未找到注册人脸: {name}")
        return False

    def list_enrolled(self) -> List[str]:
        """列出所有已注册的身份"""
        return list(self.face_index.keys())

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------
    def get_status(self) -> Dict:
        return {
            "loaded": self.model_loaded,
            "fallback": self.using_fallback,
            "message": self.status_message,
            "path": self.face_db_path,
            "enrolled_count": len(self.face_index),
        }

    def reset(self) -> None:
        """重置识别器状态"""
        self._unknown_cooldown.clear()
