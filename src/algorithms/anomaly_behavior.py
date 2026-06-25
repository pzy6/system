"""
异常行为分析器
基于质心跟踪 + 规则引擎，检测 5 类异常：
  滞留 / 徘徊 / 剧烈运动 / 静止不动 / 跌倒未恢复
无外部模型依赖。
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from collections import deque

logger = logging.getLogger(__name__)


class BehaviorAnalyzer:
    """质心跟踪 + 规则分析 异常行为检测器"""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.model_loaded = True
        self.using_fallback = False
        self.status_message = "异常行为检测使用质心跟踪与规则分析，无外部模型依赖"
        self.target_states = {}
        self.next_target_id = 1

        self.LINGER_THRESHOLD = self.config.get('linger_threshold', 1200)
        self.WANDER_THRESHOLD = self.config.get('wander_threshold', 300)
        self.WANDER_DISTANCE_RATIO = self.config.get('wander_distance_ratio', 0.3)
        self.VIOLENT_THRESHOLD = self.config.get('violent_threshold', 50)
        self.MOTIONLESS_THRESHOLD = self.config.get('motionless_threshold', 300)
        self.FALL_RECOVER_THRESHOLD = self.config.get('fall_recover_threshold', 120)

        # 目标年龄：移除过期目标前的帧数
        self.MAX_MISSING_FRAMES = 15
        self._missing_count: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------
    def _get_bbox_from_keypoints(self, keypoints: List[Dict]) -> Tuple[int, int, int, int]:
        if not keypoints:
            return (0, 0, 10, 10)
        xs = [kp['x'] for kp in keypoints]
        ys = [kp['y'] for kp in keypoints]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        return (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))

    def _get_center(self, keypoints: List[Dict]) -> Tuple[float, float]:
        if not keypoints:
            return (0.0, 0.0)
        xs = [kp['x'] for kp in keypoints if kp.get('confidence', 0) > 0.1]
        ys = [kp['y'] for kp in keypoints if kp.get('confidence', 0) > 0.1]
        if not xs:
            return (0.0, 0.0)
        return (float(np.mean(xs)), float(np.mean(ys)))

    def _iou(self, bbox_a: Tuple, bbox_b: Tuple) -> float:
        """计算两个边界框的 IoU"""
        xa, ya, wa, ha = bbox_a
        xb, yb, wb, hb = bbox_b
        xi1 = max(xa, xb)
        yi1 = max(ya, yb)
        xi2 = min(xa + wa, xb + wb)
        yi2 = min(ya + ha, yb + hb)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        area_a = wa * ha
        area_b = wb * hb
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _calculate_displacement(self, positions: List[Tuple[float, float]]) -> float:
        if len(positions) < 2:
            return 0.0
        start = positions[0]
        end = positions[-1]
        return float(np.linalg.norm(np.array(end) - np.array(start)))

    def _calculate_path_length(self, positions: List[Tuple[float, float]]) -> float:
        length = 0.0
        for i in range(1, len(positions)):
            length += np.linalg.norm(
                np.array(positions[i]) - np.array(positions[i - 1])
            )
        return length

    def _calculate_velocity_variance(
        self, keypoints: List[Dict], prev_keypoints: List[Dict]
    ) -> float:
        if not keypoints or not prev_keypoints:
            return 0.0
        cur_center = np.mean(
            [[kp['x'], kp['y']] for kp in keypoints if kp.get('confidence', 0) > 0.1], axis=0
        )
        prev_center = np.mean(
            [[kp['x'], kp['y']] for kp in prev_keypoints if kp.get('confidence', 0) > 0.1], axis=0
        )
        return float(np.linalg.norm(cur_center - prev_center))

    # ------------------------------------------------------------------
    # 主更新
    # ------------------------------------------------------------------
    def update(
        self, frame: np.ndarray, keypoints_list: List[Dict],
        timestamp: float, fall_detected: bool = False
    ) -> List[Dict]:
        anomalies = []

        # 标记所有目标为"本帧未见"
        for tid in list(self.target_states.keys()):
            self._missing_count[tid] = self._missing_count.get(tid, 0) + 1

        for person in keypoints_list:
            kps = person.get('keypoints', [])
            if not kps:
                continue

            bbox = self._get_bbox_from_keypoints(kps)
            center = self._get_center(kps)

            # 质心距离 + IoU 双重匹配
            best_tid = None
            best_score = 0.0

            for tid, state in self.target_states.items():
                prev_center = state['positions'][-1] if state['positions'] else (0, 0)
                dist = np.linalg.norm(np.array(center) - np.array(prev_center))

                # 距离 < 50px 视为同一目标
                if dist < 50:
                    iou = self._iou(bbox, state.get('current_bbox', bbox))
                    score = 1.0 - dist / 50.0 + iou
                    if score > best_score:
                        best_score = score
                        best_tid = tid

            if best_tid is not None:
                # 匹配到已有目标 → 更新
                state = self.target_states[best_tid]
                state['positions'].append(center)
                state['last_update'] = timestamp
                state['current_bbox'] = bbox
                state['keypoints'].append(kps)
                if len(state['keypoints']) > 10:
                    state['keypoints'].pop(0)
                self._missing_count[best_tid] = 0
            else:
                # 新目标
                tid = self.next_target_id
                self.next_target_id += 1
                self.target_states[tid] = {
                    'positions': deque([center], maxlen=60),
                    'start_time': timestamp,
                    'last_update': timestamp,
                    'current_bbox': bbox,
                    'keypoints': deque([kps], maxlen=10),
                    'fall_detected': False,
                    'fall_time': None,
                }
                self._missing_count[tid] = 0

        # 清理过期目标
        for tid in list(self.target_states.keys()):
            if self._missing_count.get(tid, 0) > self.MAX_MISSING_FRAMES:
                del self.target_states[tid]
                self._missing_count.pop(tid, None)

        # 规则分析
        for tid, state in self.target_states.items():
            duration = timestamp - state['start_time']

            if fall_detected:
                state['fall_detected'] = True
                state['fall_time'] = timestamp

            # 跌倒未恢复
            if state['fall_detected'] and state['fall_time']:
                fall_duration = timestamp - state['fall_time']
                if fall_duration > self.FALL_RECOVER_THRESHOLD:
                    anomalies.append({
                        'type': 'fall_not_recovered',
                        'target_id': tid,
                        'start_time': state['fall_time'],
                        'confidence': 0.95,
                        'duration': fall_duration,
                    })

            positions = list(state['positions'])
            if len(positions) < 30:
                continue

            displacement = self._calculate_displacement(positions)
            path_length = self._calculate_path_length(positions)

            # 滞留
            if duration > self.LINGER_THRESHOLD and displacement < 50:
                anomalies.append({
                    'type': 'lingering',
                    'target_id': tid,
                    'start_time': state['start_time'],
                    'confidence': 0.85,
                    'duration': duration,
                })

            # 徘徊
            if duration > self.WANDER_THRESHOLD:
                ratio = displacement / path_length if path_length > 0 else 1.0
                if path_length > 200 and ratio < self.WANDER_DISTANCE_RATIO:
                    anomalies.append({
                        'type': 'wandering',
                        'target_id': tid,
                        'start_time': state['start_time'],
                        'confidence': 0.8,
                        'duration': duration,
                    })

            # 剧烈运动
            kp_list = list(state['keypoints'])
            if len(kp_list) >= 2:
                velocity = self._calculate_velocity_variance(kp_list[-1], kp_list[-2])
                if velocity > self.VIOLENT_THRESHOLD:
                    anomalies.append({
                        'type': 'violent_movement',
                        'target_id': tid,
                        'start_time': timestamp,
                        'confidence': 0.75,
                        'duration': 0,
                    })

            # 静止不动
            if displacement < 10 and duration > self.MOTIONLESS_THRESHOLD:
                anomalies.append({
                    'type': 'motionless',
                    'target_id': tid,
                    'start_time': state['start_time'],
                    'confidence': 0.8,
                    'duration': duration,
                })

        return anomalies

    def reset(self):
        self.target_states.clear()
        self.next_target_id = 1
        self._missing_count.clear()

    def get_status(self) -> Dict:
        return {
            'loaded': self.model_loaded,
            'fallback': self.using_fallback,
            'message': self.status_message,
            'path': None,
        }
