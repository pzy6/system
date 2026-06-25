"""
摔倒检测器: ST-GCN 深度学习 + 规则回退
优先使用 ST-GCN (UR Fall + Le2i 训练, 93.2% 准确率),
模型不可用时回退到规则算法。
"""

import numpy as np
import logging
import os
from typing import List, Dict, Tuple, Optional
from collections import deque

logger = logging.getLogger(__name__)

# 17 个关键点的 COCO→PoseNet 索引映射 (用于构建 ST-GCN 输入)
KP_ORDER = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist",
    "LShoulder", "LElbow", "LWrist", "RHip", "RKnee",
    "RAnkle", "LHip", "LKnee", "LAnkle", "REye", "LEye", "REar", "LEar"
]


class FallDetector:
    """ST-GCN 深度学习摔倒检测器 + 规则回退"""

    def __init__(self, threshold: float = 0.75, model_dir: str = None):
        self.threshold = threshold

        # ST-GCN
        self.model_loaded = False
        self.using_fallback = True
        self._onnx_session = None
        self._skel_buffer = deque(maxlen=60)  # 60 帧骨架窗口
        self._stgcn_call_count = 0            # ST-GCN 推理计数 (用于跳帧)

        # 防抖: 连续N帧确认 + 冷却
        self._fall_consecutive = 0          # 连续摔倒帧计数
        self._fall_required = 12            # 需要连续12帧确认 (更严格)
        self._last_alarm_time = 0.0         # 上次告警时间戳
        self._alarm_cooldown = 5.0          # 告警冷却(秒)
        self._upright_counter = 0           # 直立帧计数 (验证非误判)

        # 规则回退参数
        self.frame_buffer = deque(maxlen=30)
        self.vertical_angles = deque(maxlen=10)
        self.velocity_history = deque(maxlen=10)
        self.hw_ratios = deque(maxlen=10)
        self.ankle_hip_ratios = deque(maxlen=10)
        self.fall_detected = False
        self.fall_start_time = None

        self._load_stgcn(model_dir)
        status = "ST-GCN ONNX" if self.model_loaded else "规则算法 (HOG回退)"
        self.status_message = f"摔倒检测: {status}"

    # ------------------------------------------------------------------
    # ST-GCN 加载
    # ------------------------------------------------------------------
    def _load_stgcn(self, model_dir=None):
        if model_dir is None:
            model_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "models", "fall_detection"
            )
        # 优选 INT8 量化模型 (更小更快)
        onnx_path = os.path.join(model_dir, "stgcn_fall_int8.onnx")
        if not os.path.exists(onnx_path):
            onnx_path = os.path.join(model_dir, "stgcn_fall.onnx")
        if not os.path.exists(onnx_path):
            logger.info(f"ST-GCN 模型未找到: {onnx_path}, 使用规则回退")
            return
        try:
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.log_severity_level = 3
            try:
                self._onnx_session = ort.InferenceSession(
                    onnx_path,
                    sess_options=options,
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
                )
                logger.info(f"ST-GCN loaded with providers: {self._onnx_session.get_providers()}")
            except Exception as exc:
                logger.warning("CUDA 初始化失败，已回退到 CPU: %s", exc)
                self._onnx_session = ort.InferenceSession(
                    onnx_path,
                    sess_options=options,
                    providers=["CPUExecutionProvider"]
                )
                logger.info(f"ST-GCN loaded with providers: {self._onnx_session.get_providers()}")
            self.model_loaded = True
            self.using_fallback = False
            logger.info(f"ST-GCN 摔倒检测模型已加载: {onnx_path}")
        except Exception as e:
            logger.warning(f"ST-GCN 加载失败: {e}")

    # ------------------------------------------------------------------
    # 检测
    # ------------------------------------------------------------------
    def detect(
        self, keypoints: List[Dict], frame_timestamp: float = None
    ) -> Tuple[bool, float]:
        if not keypoints:
            self.fall_detected = False
            return False, 0.0

        # ST-GCN 推理 (每3帧跑一次, 60帧窗口仅变了1帧, 无需每帧推理)
        if self.model_loaded and self._onnx_session is not None:
            vec = self._keypoints_to_vec(keypoints)
            self._skel_buffer.append(vec)
            if len(self._skel_buffer) >= 60:
                self._stgcn_call_count += 1
                if self._stgcn_call_count % 3 == 0:
                    return self._stgcn_predict(frame_timestamp or 0.0)
                # 跳过帧: 保持上次检测状态
                return self.fall_detected, 0.0
            return False, 0.0

        # 回退规则
        return self._rule_detect(keypoints, frame_timestamp)

    # ------------------------------------------------------------------
    # ST-GCN 预测
    # ------------------------------------------------------------------
    def _stgcn_predict(self, frame_time: float = 0.0) -> Tuple[bool, float]:
        # 取最近60帧骨架
        skel = np.array(list(self._skel_buffer)[-60:], dtype=np.float32)

        # ── 严格有效性检查 ──
        # 检查1: 核心关键点 (Neck=1, Shoulders=2/5, Hips=8/11, Knees=9/12) 必须有足够置信度
        core_kps = [1, 2, 5, 8, 11, 9, 12]  # Neck/Shoulders/Hips/Knees
        recent_conf = skel[-10:, :, 2]
        valid_kps = (recent_conf > 0.15).sum(axis=1)  # 每帧有效关键点数 (提高阈值)
        core_valid = (recent_conf[:, core_kps] > 0.3).sum(axis=1)  # 核心关键点数

        # 条件: 平均 ≥10 个有效关键点 AND ≥4 个核心关键点
        if np.mean(valid_kps) < 10 or np.mean(core_valid) < 4:
            self._fall_consecutive = max(0, self._fall_consecutive - 2)
            self._upright_counter += 1
            if self._fall_consecutive == 0:
                self.fall_detected = False
            return self.fall_detected, 0.0

        # 检查2: 人体检测置信度 — 所有关键点平均置信度 > 0.15
        mean_person_conf = recent_conf.mean()
        if mean_person_conf < 0.15:
            self._fall_consecutive = max(0, self._fall_consecutive - 2)
            self._upright_counter += 1
            if self._fall_consecutive == 0:
                self.fall_detected = False
            return self.fall_detected, 0.0

        # 坐标缩放到训练数据范围 (~640px 宽度)
        xy = skel[:, :, :2]
        xy_max = xy.max()
        if xy_max > 700:
            scale = 640.0 / xy_max
            skel = skel.copy()
            skel[:, :, :2] *= scale

        # (T, 17, 3) → (1, 3, T, 17)  (B, C, T, V)
        inp = skel.transpose(2, 0, 1)[np.newaxis, ...]
        try:
            out = self._onnx_session.run(None, {"input": inp})[0]
            exp_out = np.exp(out - out.max(axis=1, keepdims=True))
            prob = float(exp_out[0, 1] / exp_out[0].sum())

            # 防抖: 连续N帧超过阈值才触发
            threshold = 0.85  # 提高到 0.85 (原 0.75)
            if prob >= threshold:
                self._fall_consecutive += 1
                self._upright_counter = 0
            else:
                self._fall_consecutive = max(0, self._fall_consecutive - 3)
                # 检测是否处于直立状态
                neck = skel[-1, 1, :2]
                hip = (skel[-1, 8, :2] + skel[-1, 11, :2]) / 2
                dx, dy = abs(hip[0] - neck[0]), abs(hip[1] - neck[1])
                angle = np.degrees(np.arctan(dy / dx)) if dx > 0.5 else 90.0
                if angle > 60:  # 接近垂直 → 直立
                    self._upright_counter += 1

            in_cooldown = (frame_time - self._last_alarm_time) < self._alarm_cooldown

            # ── 最终判断 ──
            # 需要: 连续12帧高置信度 + 不在冷却 + 非直立中突然跳变
            if self._fall_consecutive >= self._fall_required and not in_cooldown:
                if not self.fall_detected and self._upright_counter < 60:
                    self.fall_detected = True
                    self.fall_start_time = frame_time
                    self._last_alarm_time = frame_time
                    logger.warning(
                        f"ST-GCN 摔倒! conf={prob:.2f} "
                        f"(连续{self._fall_consecutive}帧, "
                        f"person_conf={mean_person_conf:.2f})"
                    )
            elif self._fall_consecutive == 0:
                self.fall_detected = False

            return self.fall_detected, prob
        except Exception as e:
            logger.error(f"ST-GCN 推理错误: {e}")
            return False, 0.0

    # ------------------------------------------------------------------
    # 关键点 → 向量
    # ------------------------------------------------------------------
    @staticmethod
    def _keypoints_to_vec(keypoints):
        kp_map = {k["part"]: (k["x"], k["y"], k.get("confidence", 0))
                  for k in keypoints}
        vec = np.zeros((17, 3), dtype=np.float32)
        for i, part in enumerate(KP_ORDER):
            if i >= 17:
                break
            if part in kp_map:
                x, y, c = kp_map[part]
                vec[i] = [x, y, c]
        return vec

    # ------------------------------------------------------------------
    # 规则回退 — 多特征融合 (躯干角度 + 下降速度 + 宽高比 + 踝髋比)
    # ------------------------------------------------------------------
    @staticmethod
    def _find_kp(keypoints, *names):
        """查找关键点坐标 (返回首个匹配的高置信度点)"""
        for name in names:
            for kp in keypoints:
                if kp["part"] == name and kp.get("confidence", 0) > 0.1:
                    return (kp["x"], kp["y"])
        return None

    @staticmethod
    def _get_kp_conf(keypoints, name):
        """获取关键点置信度"""
        for kp in keypoints:
            if kp["part"] == name:
                return kp.get("confidence", 0)
        return 0.0

    @staticmethod
    def _calc_angle_deg(a, b):
        """计算向量 a→b 与水平线的夹角 (度), 0=水平, 90=垂直"""
        dx, dy = b[0] - a[0], b[1] - a[1]
        if abs(dx) < 0.5:
            return 90.0
        return abs(np.degrees(np.arctan(dy / dx)))

    def _rule_detect(self, keypoints, frame_timestamp):
        self.frame_buffer.append({"keypoints": keypoints, "timestamp": frame_timestamp})

        # ---- 关键点有效性检查 ----
        required = ["Neck", "RHip", "LHip"]
        neck = self._find_kp(keypoints, "Neck")
        hip = self._find_kp(keypoints, "RHip", "LHip")
        if not neck or not hip:
            self.fall_detected = False
            return False, 0.0

        # ---- 特征1: 躯干角度 (躯干与水平线夹角) ----
        torso_angle = self._calc_angle_deg(neck, hip)
        self.vertical_angles.append(torso_angle)

        # ---- 特征2: 重心下降速度 (髋部y轴速度, 归一化) ----
        fall_speed = 0.0
        if len(self.frame_buffer) >= 2:
            prev = self.frame_buffer[-2]
            prev_hip = self._find_kp(prev["keypoints"], "RHip", "LHip")
            if prev_hip:
                # 像素/帧 → 归一化 (除以人体高度估算)
                body_h = abs(neck[1] - hip[1]) + 1
                fall_speed = (hip[1] - prev_hip[1]) / body_h
                self.velocity_history.append(fall_speed)

        # ---- 特征3: 人体宽高比 ----
        pts = [(k["x"], k["y"]) for k in keypoints if k.get("confidence", 0) > 0.1]
        if pts:
            xs, ys = zip(*pts)
            w, h = max(xs) - min(xs), max(ys) - min(ys)
            ratio = w / h if h > 0 else 1.0
        else:
            ratio = 1.0
        self.hw_ratios.append(ratio)

        # ---- 特征4: 踝-髋垂直比 (身体是否"塌缩") ----
        ankle = self._find_kp(keypoints, "RAnkle", "LAnkle")
        ankle_hip_ratio = 1.0
        if ankle and hip:
            torso_h = abs(neck[1] - hip[1]) + 1
            ankle_hip_ratio = abs(ankle[1] - hip[1]) / torso_h
        self.ankle_hip_ratios.append(ankle_hip_ratio)

        # ---- 加权评分 (使用滑动窗口均值) ----
        win = min(8, len(self.vertical_angles))
        avg_angle = np.mean(list(self.vertical_angles)[-win:])
        avg_ratio = np.mean(list(self.hw_ratios)[-win:])
        avg_speed = np.mean(list(self.velocity_history)[-win:]) if self.velocity_history else 0.0
        avg_ah = np.mean(list(self.ankle_hip_ratios)[-win:]) if self.ankle_hip_ratios else 1.0

        score = 0.0
        # 躯干水平 (权重 0.35): 角度 < 30° 满分, < 45° 半分的
        if avg_angle < 30:
            score += 0.35
        elif avg_angle < 45:
            score += 0.20
        # 下降速度 (权重 0.30): 髋部向下快速移动
        if avg_speed > 0.08:
            score += 0.30
        elif avg_speed > 0.04:
            score += 0.15
        # 宽高比 (权重 0.25): 宽>高说明身体水平
        if avg_ratio > 1.5:
            score += 0.25
        elif avg_ratio > 1.0:
            score += 0.15
        # 踝-髋塌缩 (权重 0.10): 踝髋比 < 0.3 说明身体折叠
        if avg_ah < 0.3:
            score += 0.10

        # ---- 时序确认: 连续帧 score >= 0.6 才触发 ----
        if score >= self.threshold:
            self._fall_consecutive += 1
        else:
            self._fall_consecutive = max(0, self._fall_consecutive - 2)

        in_cooldown = (frame_timestamp or 0) - self._last_alarm_time < self._alarm_cooldown

        if self._fall_consecutive >= self._fall_required and not in_cooldown:
            if not self.fall_detected:
                self.fall_detected = True
                self.fall_start_time = frame_timestamp
                self._last_alarm_time = frame_timestamp or 0
                logger.warning(
                    f"规则摔倒! score={score:.2f} angle={avg_angle:.0f}° "
                    f"speed={avg_speed:.3f} ratio={avg_ratio:.2f} "
                    f"(连续{self._fall_consecutive}帧)"
                )
            return True, score
        elif self._fall_consecutive == 0:
            self.fall_detected = False

        return self.fall_detected, score

    def get_status(self) -> Dict:
        return {
            "loaded": self.model_loaded,
            "fallback": self.using_fallback,
            "message": self.status_message,
            "path": os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "models", "fall_detection"
            ),
        }

    def reset(self):
        self._skel_buffer.clear()
        self.frame_buffer.clear()
        self.vertical_angles.clear()
        self.hw_ratios.clear()
        self._fall_consecutive = 0
        self._last_alarm_time = 0.0
        self.fall_detected = False
