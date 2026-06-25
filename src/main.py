# ═══════════════════════════════════════════════════════════
# 环境变量 + 警告抑制（必须在所有导入之前）
# ═══════════════════════════════════════════════════════════
import os as _os
_os.environ['OPENCV_LOG_LEVEL'] = 'FATAL'
_os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'
_os.environ['ULTRALYTICS_AUTO_UPDATE'] = '0'  # 禁用自动更新
# 修复 Windows 控制台中文乱码
_os.environ['PYTHONUTF8'] = '1'
_os.environ['PYTHONIOENCODING'] = 'utf-8'
# ONNX Runtime 模型自行选择 provider (ST-GCN/Emotion→GPU, insightface→CPU)
import warnings as _warnings
_warnings.filterwarnings('ignore')
import logging as _logging
_logging.captureWarnings(True)

import argparse
import yaml
import logging
import logging.handlers
import sys
import os
import time
import signal
import json
import threading
from collections import defaultdict, deque
from queue import Queue
import numpy as np
import cv2

# 确保 src/ 在 sys.path 中（兼容 python -m src.main 和直接运行）
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

def get_resource_path(relative_path):
    """获取资源文件的绝对路径，兼容打包和开发环境"""
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

def ensure_path_exists(path):
    """确保路径存在，不存在则创建"""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def normalize_resource_path(base_path, raw_path):
    if not raw_path:
        return None
    if os.path.isabs(raw_path):
        return raw_path
    return os.path.normpath(os.path.join(base_path, raw_path))

from perception.camera_manager import CameraManager
from perception.camera_worker import CameraWorker
from processing.frame_processor import FrameProcessor
from processing.thread_pool import PriorityThreadPool
from algorithms.yolo_pose_detector import YOLOPoseDetector
from algorithms.yolo_detector import YOLODetector
from algorithms.face_recognizer import FaceRecognizer
from algorithms.fall_detector import FallDetector
from algorithms.anomaly_behavior import BehaviorAnalyzer
from algorithms.emotion_recognition import EmotionRecognizer
from application.dashboard import Dashboard, DashboardUpdater, AlarmEvent
from application.modern_dashboard import ModernDashboard
from application.splash_screen import SplashScreen
from utils.audio import play_smoke_beep, play_fall_voice

logger = logging.getLogger(__name__)

DEFAULT_MODEL_LAYOUT = {
    'behavior': [],
    'fall_detection': [],
    'yolo_face_smoke': ['yolo26n_smoke.pt'],
    'yolo_pose': ['yolo26n-pose.pt'],
    'face_db': ['index.json'],
}

class AdaptiveFrameSkipper:
    """自适应帧跳过: 根据系统负载动态调整处理帧率 (文档 6.1)"""

    def __init__(self, target_fps=25):
        self.target_fps = target_fps
        self._process_times = []  # 最近处理时间 (ms)
        self._skip_count = 0

    def record_time(self, dt_ms):
        """记录一次处理耗时"""
        self._process_times.append(dt_ms)
        if len(self._process_times) > 60:
            self._process_times.pop(0)

    def should_process(self) -> bool:
        """返回 True 表示应处理当前帧"""
        if len(self._process_times) < 5:
            return True  # 预热阶段不跳帧
        avg_time = sum(self._process_times) / len(self._process_times)
        current_fps = 1000.0 / avg_time if avg_time > 0 else 30.0

        if current_fps >= self.target_fps:
            self._skip_count = 0
        else:
            ratio = self.target_fps / max(current_fps, 1.0)
            self._skip_count = max(0, int(ratio - 1))

        if self._skip_count > 0:
            self._skip_count -= 1
            return False
        return True

class AlarmDeduplicator:
    """告警去重: 同一摄像头+同一类型在冷却时间内只发一次"""
    # 告警类型中文名 → 内部键名
    _TYPE_MAP = {
        '跌倒检测': 'fall', '烟雾检测': 'smoke', '火焰检测': 'fire',
        '陌生人检测': 'stranger', '徘徊': 'wander', '滞留': 'linger',
        '剧烈运动': 'violent', '静止不动': 'motionless',
        'fall_not_recovered': 'fall',
    }

    def __init__(self, cooldown_config=None):
        defaults = {
            'fall': 300, 'smoke': 300, 'fire': 300,
            'stranger': 300, 'wander': 600, 'linger': 600,
            'violent': 600, 'motionless': 600,
        }
        self._cooldowns = {**defaults, **(cooldown_config or {})}
        self._last_emit = {}  # key=(camera_id, type_key) → timestamp

    def _get_key(self, alarm_type):
        return self._TYPE_MAP.get(alarm_type, alarm_type)

    def should_emit(self, camera_id, alarm_type, timestamp):
        """返回 True 表示应该发送告警"""
        type_key = self._get_key(alarm_type)
        cooldown = self._cooldowns.get(type_key, 300)
        key = (str(camera_id), type_key)
        last = self._last_emit.get(key, 0)
        if timestamp - last >= cooldown:
            self._last_emit[key] = timestamp
            return True
        return False

    def reset(self, camera_id=None, alarm_type=None):
        if camera_id and alarm_type:
            type_key = self._get_key(alarm_type)
            self._last_emit.pop((str(camera_id), type_key), None)
        elif camera_id:
            keys = [k for k in self._last_emit if k[0] == str(camera_id)]
            for k in keys:
                del self._last_emit[k]
        else:
            self._last_emit.clear()

class AlarmRecorder:
    def __init__(self, storage_config, base_path):
        self.base_path = base_path
        self.alarm_dir = self._resolve_path(storage_config.get('alarms', './data/alarms/'))
        self.screenshot_dir = self._resolve_path(
            storage_config.get('screenshots', './data/alarms/screenshots/')
        )
        self.video_dir = self._resolve_path(storage_config.get('videos', './data/alarms/videos/'))
        ensure_path_exists(self.alarm_dir)
        ensure_path_exists(self.screenshot_dir)
        ensure_path_exists(self.video_dir)
        self.history_path = os.path.join(self.alarm_dir, 'alarm_history.jsonl')
        self._lock = threading.Lock()

    def _resolve_path(self, path):
        if os.path.isabs(path):
            return path
        return os.path.join(self.base_path, path)

    def record(self, alarm, video_path=None):
        timestamp_text = time.strftime('%Y%m%d_%H%M%S', time.localtime(alarm.timestamp))
        safe_alarm_type = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in alarm.alarm_type)
        screenshot_path = None

        if alarm.frame is not None:
            filename = f"{timestamp_text}_{alarm.camera_id}_{safe_alarm_type}.jpg"
            screenshot_path = os.path.join(self.screenshot_dir, filename)
            cv2.imwrite(screenshot_path, alarm.frame)

        record = {
            'id': alarm.id,
            'type': alarm.alarm_type,
            'camera_id': alarm.camera_id,
            'camera_name': alarm.camera_name,
            'timestamp': alarm.timestamp,
            'time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alarm.timestamp)),
            'confidence': float(alarm.confidence),
            'level': alarm.level,
            'status': alarm.status,
            'screenshot': screenshot_path,
            'video': video_path
        }

        with self._lock:
            with open(self.history_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

        return record

    def get_video_path(self, alarm):
        timestamp_text = time.strftime('%Y%m%d_%H%M%S', time.localtime(alarm.timestamp))
        safe_alarm_type = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in alarm.alarm_type)
        return os.path.join(
            self.video_dir,
            f"{timestamp_text}_{alarm.camera_id}_{safe_alarm_type}.mp4"
        )


class AlarmVideoBuffer:
    def __init__(self, storage_config, base_path, seconds=10):
        self.base_path = base_path
        self.seconds = max(2, int(seconds))
        self.buffers = defaultdict(lambda: deque(maxlen=300))
        self.storage_config = storage_config or {}

    def add_frame(self, camera_id, frame, timestamp, fps=15):
        maxlen = max(30, int(self.seconds * max(fps, 1)))
        buffer = self.buffers[camera_id]
        if buffer.maxlen != maxlen:
            self.buffers[camera_id] = deque(buffer, maxlen=maxlen)
            buffer = self.buffers[camera_id]
        # 存储 640x360 缩略帧 (节省 ~70% 内存)
        small = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_NEAREST)
        buffer.append((timestamp, small))

    def export_alarm_clip(self, camera_id, output_path, alarm_time, fps=15, window_before=8, window_after=2):
        frames = list(self.buffers.get(camera_id, []))
        if not frames:
            return None

        selected = [
            frame for ts, frame in frames
            if ts >= alarm_time - window_before and ts <= alarm_time + window_after
        ]
        if not selected:
            selected = [frame for _, frame in frames]

        first_frame = selected[0]
        height, width = first_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, max(fps, 1), (width, height))
        try:
            for frame in selected:
                writer.write(frame)
        finally:
            writer.release()
        return output_path if os.path.exists(output_path) else None

class SilverGuardianSystem:
    def __init__(self, config_path, demo_mode=False):
        self.demo_mode = demo_mode
        
        if getattr(sys, 'frozen', False):
            self.base_path = os.path.dirname(sys.executable)
        else:
            self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        if config_path and not os.path.isabs(config_path):
            self.config_path = get_resource_path(config_path)
        else:
            self.config_path = config_path
        
        self.config = self.load_config()
        self.config['config_path'] = self.config_path
        
        self.running = False
        self.exit_event = False
        
        self.camera_manager = None
        self.frame_processor = None
        self.thread_pool = None
        
        self.yolo_pose_detector = None
        self.yolo_detector = None
        self.face_recognizer = None
        self.fall_detector = None
        self.behavior_analyzer = None
        self.alarm_recorder = None
        self.alarm_video_buffer = None
        self.model_status = {}
        
        self.raw_frame_queue = Queue(maxsize=100)
        self.processed_frame_queue = Queue(maxsize=100)
        self.dashboard_frame_queue = Queue(maxsize=50)
        self.alarm_queue = Queue(maxsize=50)
        self.face_identity_queue = Queue(maxsize=50)
        
        self.dashboard = None
        self.dashboard_updater = None
        self.app = None
        self.splash = None
        self.camera_workers = []
        
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0
        self._pose_frame = 0
        self._last_skeletons = []
        self._t_detect = self._t_pose = self._t_draw = 0  # 诊断计时
        
        self.init_logging()
    
    def load_config(self):
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
                self._ensure_runtime_paths(config)
                return config
        except Exception as e:
            logger.error(f"Failed to load config: {str(e)}")
            return {}

    def _ensure_runtime_paths(self, config):
        storage = config.setdefault('storage', {})
        for key, default_value in {
            'logs': './data/logs/',
            'alarms': './data/alarms/',
            'screenshots': './data/alarms/screenshots/',
            'videos': './data/alarms/videos/',
        }.items():
            storage.setdefault(key, default_value)
            ensure_path_exists(normalize_resource_path(self.base_path, storage[key]))

        model_paths = config.setdefault('model_paths', {})
        default_dirs = {
            'emotion': './models/emotion_model/',
            'vitals': './models/vitals_model/',
            'behavior': './models/behavior_model/',
            'fall_detection': './models/fall_detection/',
            'yolo_pose': './models/yolo_pose/',
            'yolo_face_smoke': './models/yolo_face_smoke/',
            'face_db': './data/face_db/',
        }
        for key, default_dir in default_dirs.items():
            model_paths.setdefault(key, default_dir)
            model_dir = normalize_resource_path(self.base_path, model_paths[key])
            ensure_path_exists(model_dir)
            for filename in DEFAULT_MODEL_LAYOUT.get(key, []):
                target_file = os.path.join(model_dir, filename)
                placeholder_file = f"{target_file}.placeholder"
                if not os.path.exists(target_file) and not os.path.exists(placeholder_file):
                    with open(placeholder_file, 'w', encoding='utf-8') as f:
                        f.write(
                            f'Place the required model file here: {filename}\n'
                            f'Directory: {model_dir}\n'
                        )
    
    def init_logging(self):
        log_path = self.config.get('storage', {}).get('logs', './data/logs/')

        if not os.path.isabs(log_path):
            log_path = os.path.join(self.base_path, log_path)

        ensure_path_exists(log_path)

        # 修复 Windows 控制台中文乱码
        import io as _io
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = _io.TextIOWrapper(
                sys.stdout.buffer, encoding='utf-8', errors='replace'
            )

        log_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        file_handler = logging.handlers.RotatingFileHandler(
            os.path.join(log_path, 'system.log'),
            maxBytes=1024*1024*10,
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(logging.INFO)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_formatter)
        console_handler.setLevel(logging.DEBUG)
        
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        logger.info(f"Logging initialized at {log_path}")
    
    def _splash_progress(self, step, msg):
        """更新开屏画面进度"""
        if self.splash:
            self.splash.set_progress(step, msg)

    def init_components(self):
        logger.info("Initializing components...")
        app = self.app  # QApplication 引用，用于刷新 UI

        self.camera_manager = CameraManager()
        self.camera_manager.stop_scanning()

        processing_config = self.config.get('processing', {})
        input_width = processing_config.get('input_size', {}).get('width', 640)
        input_height = processing_config.get('input_size', {}).get('height', 480)
        self.frame_processor = FrameProcessor(input_width, input_height)
        self.frame_processor.set_input_queue(self.raw_frame_queue)
        self.frame_processor.set_output_queue(self.processed_frame_queue)

        num_threads = processing_config.get('num_threads', 4)
        self.thread_pool = PriorityThreadPool(max_workers=num_threads)

        model_paths = self.config.get('model_paths', {})
        processing_config = self.config.get('processing', {})
        alarm_thresholds = self.config.get('alarm_thresholds', {})

        # --- 并行加载: YOLO 姿态 + YOLO 人脸/烟雾 (两个线程同时加载 ~5s→~5s 并行) ---
        self._splash_progress(2, "正在加载检测模型...")
        yolo_pose_config = {
            'model_path': normalize_resource_path(
                self.base_path, model_paths.get('yolo_pose', './models/yolo_pose/')),
            'confidence_threshold': 0.5, 'iou_threshold': 0.5,
            'device': processing_config.get('yolo_device', 0),
        }
        yolo_detect_config = {
            'model_path': normalize_resource_path(
                self.base_path, model_paths.get('yolo_face_smoke', './models/yolo_face_smoke/')),
            'confidence_threshold': 0.5, 'iou_threshold': 0.45,
            'device': processing_config.get('yolo_device', 0),
            'frame_skip': processing_config.get('yolo_face_skip_frames', 2),
        }

        # 两个 YOLO 模型并行加载 (不共用 CUDA context, 安全并行)
        import threading as _th
        pose_loaded = [None]
        detect_loaded = [None]
        def _load_pose():
            pose_loaded[0] = YOLOPoseDetector(yolo_pose_config)
        def _load_detect():
            detect_loaded[0] = YOLODetector(yolo_detect_config)
        t1 = _th.Thread(target=_load_pose, daemon=True); t1.start()
        t2 = _th.Thread(target=_load_detect, daemon=True); t2.start()
        # 等待期间刷新 splash 动画
        while t1.is_alive() or t2.is_alive():
            app.processEvents()
            _th.Event().wait(0.05)
        self.yolo_pose_detector = pose_loaded[0]
        self.yolo_detector = detect_loaded[0]

        # --- 人脸识别器 ---
        self._splash_progress(4, "正在加载人脸识别模型...")
        app.processEvents()
        face_recognition_config = {
            'face_db_path': normalize_resource_path(
                self.base_path, model_paths.get('face_db', './data/face_db/')),
            'recognition_threshold': alarm_thresholds.get('face_recognition', 0.6),
            'model_name': 'buffalo_l',
            'device': processing_config.get('yolo_device', 0),
            'unknown_face_timeout': alarm_thresholds.get('unknown_face_timeout', 30),
        }
        self.face_recognizer = FaceRecognizer(face_recognition_config)

        # --- 摔倒 + 情绪并行加载 ---
        self._splash_progress(5, "正在加载摔倒+情绪模型...")
        app.processEvents()
        fall_threshold = alarm_thresholds.get('fall_detection', 0.75)
        fall_loaded = [None]; emo_loaded = [None]
        def _load_fall():
            fall_loaded[0] = FallDetector(threshold=fall_threshold)
        def _load_emo():
            emo_loaded[0] = EmotionRecognizer()
        t3 = _th.Thread(target=_load_fall, daemon=True); t3.start()
        t4 = _th.Thread(target=_load_emo, daemon=True); t4.start()
        while t3.is_alive() or t4.is_alive():
            app.processEvents()
            _th.Event().wait(0.03)
        self.fall_detector = fall_loaded[0]
        self.emotion_recognizer = emo_loaded[0]

        # --- 行为分析器 (纯规则, 无模型) ---
        self._splash_progress(6, "正在初始化异常行为分析...")
        app.processEvents()
        behavior_config = {
            'linger_threshold': 1200, 'wander_threshold': 300,
            'wander_distance_ratio': 0.3, 'violent_threshold': 50,
            'motionless_threshold': 300, 'fall_recover_threshold': 120,
        }
        self.behavior_analyzer = BehaviorAnalyzer(behavior_config)

        # 告警去重器 (同摄像头+同类型5分钟内只报一次)
        cooldown_cfg = self.config.get('alarm_cooldown', {})
        self.alarm_dedup = AlarmDeduplicator(cooldown_cfg)
        self._frame_skipper = AdaptiveFrameSkipper(target_fps=10)  # 保守目标, <10fps 才跳帧

        self.alarm_recorder = AlarmRecorder(self.config.get('storage', {}), self.base_path)
        self.alarm_video_buffer = AlarmVideoBuffer(
            self.config.get('storage', {}),
            self.base_path,
            seconds=10
        )
        self.model_status = self.collect_model_status()

        # Create camera workers for each configured camera
        self._splash_progress(7, f"已创建 {len(self.config.get('cameras', []))} 个摄像头")
        app.processEvents()
        self.camera_workers = []
        for cam_cfg in self.config.get('cameras', []):
            worker = CameraWorker(
                camera_id=cam_cfg['id'],
                camera_name=cam_cfg['name'],
                source=str(cam_cfg['source']),
                width=cam_cfg.get('resolution', {}).get('width', 640),
                height=cam_cfg.get('resolution', {}).get('height', 480),
                fps=cam_cfg.get('fps', 15)
            )
            worker.set_queue(self.raw_frame_queue)
            self.camera_workers.append(worker)

        self._splash_progress(8, "初始化完成")
        app.processEvents()
        logger.info("All components initialized")

    def collect_model_status(self):
        status = {}
        components = {
            'yolo_pose': self.yolo_pose_detector,
            'yolo_face_smoke': self.yolo_detector,
            'face_recognition': self.face_recognizer,
            'fall_detection': self.fall_detector,
            'behavior': self.behavior_analyzer,
        }
        for key, component in components.items():
            if component and hasattr(component, 'get_status'):
                status[key] = component.get_status()
            else:
                status[key] = {
                    'loaded': False,
                    'fallback': True,
                    'message': '组件未初始化',
                    'path': None,
                }
        return status
    
    def init_dashboard(self):
        self.dashboard = ModernDashboard(
            auto_camera_scan=not self.demo_mode and len(self.camera_workers) == 0,
            config=self.config,
            config_path=self.config_path,
            model_status=self.model_status,
            frame_queue=self.dashboard_frame_queue,
        )

        if self.splash:
            self.splash.set_progress(8, "系统就绪")
            self.dashboard.show()
            self.splash.finish(self.dashboard)
        else:
            self.dashboard.show()

        logger.info("Dashboard initialized")

    def get_runtime_summary(self):
        return {
            'config_path': self.config_path,
            'model_status': self.model_status,
            'camera_workers': [worker.get_stats() for worker in self.camera_workers],
        }

    def process_frame(self, frame_item):
        try:
            camera_id = frame_item['camera_id']
            camera_name = frame_item['camera_name']
            frame = frame_item.get('original_frame', frame_item['frame'])
            processed_frame = frame_item.get('processed_frame', frame_item['frame'])
            timestamp = frame_item['timestamp']
            self.alarm_video_buffer.add_frame(
                camera_id,
                frame,
                timestamp,
                fps=self._get_camera_fps(camera_id)
            )

            # 自适应帧跳过: 系统负载高时自动跳帧 (文档 6.1)
            _proc_start = time.perf_counter()
            if not self._frame_skipper.should_process():
                return  # 跳过当前帧, 保持告警缓冲区更新

            # ============================================================
            # Step 1: YOLO 人脸 + 烟雾检测
            # ============================================================
            _t0 = time.perf_counter()
            yolo_result = self.yolo_detector.detect(processed_frame)
            self._t_detect += time.perf_counter() - _t0
            face_bboxes = yolo_result.get('face_bboxes', [])
            smoke_detected = yolo_result.get('smoke_detected', False)
            smoke_confidence = yolo_result.get('smoke_confidence', 0.0)

            # ============================================================
            # Step 2: 姿态检测 (检测器内部 frame_skip=5 管理跳帧)
            # ============================================================
            _tp = time.perf_counter()
            skeletons = self.yolo_pose_detector.detect(processed_frame)
            self._t_pose += time.perf_counter() - _tp

            # ============================================================
            # Step 3: 人脸识别（仅当 YOLO 检测到人脸时运行）
            # ============================================================
            face_identities = []
            # 人脸识别每5帧跑一次（insightface ArcFace 较慢，跳帧省算力）
            if (face_bboxes and self.face_recognizer.model_loaded
                    and self.frame_count % 5 == 0):  # 每5帧 (~3Hz) 识别
                face_identities = self.face_recognizer.recognize(
                    processed_frame, face_bboxes
                )
                # 情绪识别 (复用 YOLO 人脸框, 避免 Haar 重复检测)
                for i, bbox in enumerate(face_bboxes[:2]):  # 最多分析2张脸
                    try:
                        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                        face_roi = processed_frame[max(0, y):y+h, max(0, x):x+w]
                        if face_roi.size > 0:
                            emotion = self.emotion_recognizer.analyze_face_roi(face_roi)
                            if i < len(face_identities):
                                face_identities[i]['emotion'] = emotion
                    except Exception:
                        pass

            # ============================================================
            # Step 4: 摔倒检测
            # ============================================================
            fall_detected = False
            fall_confidence = 0.0
            all_keypoints = []

            if skeletons:
                for person in skeletons:
                    all_keypoints.extend(person['keypoints'])

                fall_detected, fall_confidence = self.fall_detector.detect(
                    all_keypoints, timestamp
                )

            # ============================================================
            # Step 5: 异常行为分析
            # ============================================================
            anomalies = self.behavior_analyzer.update(
                frame, skeletons, timestamp, fall_detected
            )


            # ============================================================
            # 告警发射
            # ============================================================

            # 摔倒告警
            if fall_detected and fall_confidence >= self.fall_detector.threshold:
                if self.alarm_dedup.should_emit(camera_id, '跌倒检测', timestamp):
                    alarm = AlarmEvent(
                        alarm_type='跌倒检测',
                        camera_id=camera_id,
                        camera_name=camera_name,
                        timestamp=timestamp,
                        confidence=fall_confidence,
                        frame=frame.copy()
                    )
                    self._emit_alarm(alarm)
                    # 语音告警：有人脸库中的老人摔倒
                    has_known_person = any(
                        not i.get('is_unknown', True) for i in face_identities
                    )
                    if has_known_person:
                        play_fall_voice()
                    logger.warning(
                        f"Fall alarm triggered: {camera_name} - {fall_confidence:.2f}"
                    )

            # 烟雾告警
            smoke_threshold = self.config.get('alarm_thresholds', {}).get(
                'smoke_detection', 0.5
            )
            if smoke_detected and smoke_confidence >= smoke_threshold:
                if self.alarm_dedup.should_emit(camera_id, '烟雾检测', timestamp):
                    alarm = AlarmEvent(
                        alarm_type='烟雾检测',
                        camera_id=camera_id,
                        camera_name=camera_name,
                        timestamp=timestamp,
                        confidence=smoke_confidence,
                        frame=frame.copy()
                    )
                    self._emit_alarm(alarm)
                    play_smoke_beep()
                    logger.warning(
                        f"Smoke alarm triggered: {camera_name} - {smoke_confidence:.2f}"
                    )

            # 火焰告警
            fire_detected = yolo_result.get('fire_detected', False)
            fire_confidence = yolo_result.get('fire_confidence', 0.0)
            fire_threshold = self.config.get('alarm_thresholds', {}).get(
                'fire_detection', 0.5
            )
            if fire_detected and fire_confidence >= fire_threshold:
                if self.alarm_dedup.should_emit(camera_id, '火焰检测', timestamp):
                    alarm = AlarmEvent(
                        alarm_type='火焰检测',
                        camera_id=camera_id,
                        camera_name=camera_name,
                        timestamp=timestamp,
                        confidence=fire_confidence,
                        frame=frame.copy()
                    )
                    self._emit_alarm(alarm)
                    play_smoke_beep()

            # 陌生人告警
            for face_id in face_identities:
                if face_id.get('is_unknown'):
                    if self.alarm_dedup.should_emit(camera_id, '陌生人检测', timestamp):
                        alarm = AlarmEvent(
                            alarm_type='陌生人检测',
                            camera_id=camera_id,
                            camera_name=camera_name,
                            timestamp=timestamp,
                            confidence=face_id['confidence'],
                            frame=frame.copy()
                        )
                        self._emit_alarm(alarm)
                    logger.warning(
                        f"Unknown face detected: {camera_name}"
                    )

            # 异常行为告警
            for anomaly in anomalies:
                if anomaly['confidence'] >= 0.7:
                    if self.alarm_dedup.should_emit(camera_id, anomaly['type'], timestamp):
                        alarm = AlarmEvent(
                            alarm_type=anomaly['type'],
                            camera_id=camera_id,
                            camera_name=camera_name,
                            timestamp=anomaly['start_time'],
                            confidence=anomaly['confidence'],
                            frame=frame.copy()
                        )
                        self._emit_alarm(alarm)

            # ============================================================
            # 队列输出
            # ============================================================
            if face_identities:
                self.face_identity_queue.put({
                    'camera_id': camera_id,
                    'identities': face_identities,
                    'timestamp': timestamp,
                })

            # === 可视化标注 ===
            # 计算坐标缩放比 (processed → original)
            ph, pw = processed_frame.shape[:2]
            fh, fw = frame.shape[:2]
            sx, sy = fw / pw, fh / ph

            _td = time.perf_counter()
            try:
                display_frame = self._draw_overlays(
                    frame, face_bboxes, face_identities,
                    yolo_result.get('smoke_bboxes', []),
                    yolo_result.get('fire_bboxes', []),
                    skeletons, fall_detected, fall_confidence,
                    sx, sy)
            except Exception:
                display_frame = frame
            self._t_draw += time.perf_counter() - _td

            # 非阻塞: UI 跟不上时丢帧, 不阻塞处理流水线
            try:
                self.dashboard_frame_queue.put({
                    'camera_id': camera_id,
                    'camera_name': camera_name,
                    'frame': display_frame,
                    'skeletons': skeletons,
                    'fall_detected': fall_detected,
                    'timestamp': timestamp
                }, block=False)
            except Exception:
                pass

            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_fps_time >= 3.0:
                saved_fps = self.frame_count / 3.0
                saved_count = self.frame_count
                saved_prev_time = self.last_fps_time
                self.fps = saved_fps
                self.frame_count = 0
                self.last_fps_time = current_time
                logger.info(f"FPS: {self.fps:.0f} | "
                           f"detect={self._t_detect*1000:.0f}ms "
                           f"pose={self._t_pose*1000:.0f}ms "
                           f"draw={self._t_draw*1000:.0f}ms")
                self._t_detect = self._t_pose = self._t_draw = 0
                # 通知自适应跳帧器 (用保存的值, 非重置后的)
                total_ms = (current_time - saved_prev_time) / max(saved_count, 1) * 1000
                self._frame_skipper.record_time(total_ms)

        except Exception as e:
            import traceback as _tb
            logger.error(f"Frame processing error: {e}\n{_tb.format_exc()}")

    def _emit_alarm(self, alarm):
        # 录制和截图放到后台线程，避免阻塞 process_frame
        recorder = self.alarm_recorder
        video_buffer = self.alarm_video_buffer
        fps = self._get_camera_fps(alarm.camera_id)

        def _record_bg():
            video_path = None
            if video_buffer and recorder:
                try:
                    video_path = video_buffer.export_alarm_clip(
                        alarm.camera_id, recorder.get_video_path(alarm),
                        alarm.timestamp, fps=fps)
                except Exception as e:
                    logger.error(f"Video export failed {alarm.id}: {e}")
            if recorder:
                try:
                    recorder.record(alarm, video_path=video_path)
                except Exception as e:
                    logger.error(f"Record failed {alarm.id}: {e}")

        # 用线程池复用线程，避免告警风暴时线程爆炸
        self.thread_pool.submit(_record_bg, priority='low')

        try:
            self.alarm_queue.put(alarm, block=False)
        except Exception:
            logger.warning("Alarm queue full, dropping alarm %s", alarm.id)

    def _get_camera_fps(self, camera_id):
        for camera in self.config.get('cameras', []):
            if camera.get('id') == camera_id:
                return int(camera.get('fps', 15))
        return 15

    def _draw_overlays(self, frame, face_bboxes, face_identities, smoke_bboxes,
                       fire_bboxes, skeletons, fall_detected, fall_confidence,
                       scale_x=1.0, scale_y=1.0):
        """在帧上绘制检测标注。scale_x/y 用于将 processed_frame→frame 坐标缩放。"""
        display = frame  # 直接绘制，不复制（省 2.7MB/帧）

        # 人脸框 (绿色) + 身份 — 需要缩放到原始帧坐标
        for bbox in face_bboxes:
            try:
                fx = int(int(bbox[0]) * scale_x)
                fy = int(int(bbox[1]) * scale_y)
                fw = int(int(bbox[2]) * scale_x)
                fh = int(int(bbox[3]) * scale_y)
            except (ValueError, TypeError, IndexError):
                continue
            cv2.rectangle(display, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
            # 匹配身份
            for ident in face_identities:
                ib = ident.get('bbox')
                if ib is None:
                    continue
                try:
                    bx, by, bw, bh = int(ib[0]), int(ib[1]), int(ib[2]), int(ib[3])
                except (ValueError, TypeError, IndexError):
                    continue
                if abs(bx - fx) < 15 and abs(by - fy) < 15:
                    label = '?' if ident.get('is_unknown') else ident.get('name', '?')
                    color = (0, 165, 255) if ident.get('is_unknown') else (0, 255, 0)
                    cv2.putText(display, label, (fx, fy - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    break

        # 火焰框 (橙色) — 缩放到原始帧坐标
        for bbox in fire_bboxes:
            try:
                fx = int(int(bbox[0]) * scale_x)
                fy = int(int(bbox[1]) * scale_y)
                fw = int(int(bbox[2]) * scale_x)
                fh = int(int(bbox[3]) * scale_y)
            except (ValueError, TypeError, IndexError):
                continue
            cv2.rectangle(display, (fx, fy), (fx + fw, fy + fh), (0, 165, 255), 2)
            cv2.putText(display, 'FIRE', (fx, fy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # 烟雾框 (红色)
        for bbox in smoke_bboxes:
            try:
                sx = int(int(bbox[0]) * scale_x)
                sy = int(int(bbox[1]) * scale_y)
                sw = int(int(bbox[2]) * scale_x)
                sh = int(int(bbox[3]) * scale_y)
            except (ValueError, TypeError, IndexError):
                continue
            cv2.rectangle(display, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)
            cv2.putText(display, 'SMOKE', (sx, sy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 骨架
        POSE_PAIRS = [
            ("Neck", "RShoulder"), ("Neck", "LShoulder"), ("RShoulder", "RElbow"),
            ("RElbow", "RWrist"), ("LShoulder", "LElbow"), ("LElbow", "LWrist"),
            ("Neck", "RHip"), ("RHip", "RKnee"), ("RKnee", "RAnkle"),
            ("Neck", "LHip"), ("LHip", "LKnee"), ("LKnee", "LAnkle"),
            ("Neck", "Nose"),
        ]
        for person in (skeletons or []):
            try:
                kps = person.get('keypoints', [])
                if not kps:
                    continue
                kp = {
                    str(k.get('part', '')): (
                        int(int(k.get('x', 0)) * scale_x),
                        int(int(k.get('y', 0)) * scale_y))
                    for k in kps if 'x' in k and 'y' in k
                }
                for a, b in POSE_PAIRS:
                    if a in kp and b in kp:
                        cv2.line(display, kp[a], kp[b], (0, 255, 255), 2)
                for px, py in kp.values():
                    cv2.circle(display, (px, py), 3, (0, 255, 255), -1)
            except Exception:
                continue

        # 摔倒告警
        if fall_detected:
            h, w = display.shape[:2]
            cv2.putText(display, 'FALL!', (w // 2 - 60, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # FPS
        cv2.putText(display, f'FPS:{self.fps}', (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return display


    def run_demo_mode(self):
        logger.info("Running in demo mode")
        
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(test_frame, "银发守护者 - 演示模式", (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        while self.running:
            try:
                for camera in self.config.get('cameras', []):
                    frame_item = {
                        'camera_id': camera['id'],
                        'camera_name': camera['name'],
                        'frame': test_frame.copy(),
                        'processed_frame': test_frame.copy(),
                        'timestamp': time.time()
                    }
                    self.process_frame(frame_item)
                
                time.sleep(1.0 / 15)
            except Exception as e:
                logger.error(f"Demo mode error: {str(e)}")

    def run(self):
        logger.info("Starting Silver Guardian system...")
        
        self.running = True
        
        if self.demo_mode:
            if self.camera_manager:
                self.camera_manager.stop_scanning()
            self.init_dashboard()
            self.start_dashboard_updater()
            
            import threading
            demo_thread = threading.Thread(target=self.run_demo_mode, daemon=True)
            demo_thread.start()
            
            self.app.exec_()
        else:
            self.frame_processor.start()
            
            # Start camera workers
            for worker in self.camera_workers:
                if worker.connect():
                    worker.start()
                    logger.info(f"Camera worker {worker.camera_id} started")
                else:
                    logger.warning(f"Failed to connect camera worker {worker.camera_id}")
            
            self.init_dashboard()
            
            # Start DashboardUpdater to bridge processing queues to UI
            self.start_dashboard_updater()
            
            import threading
            processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
            processing_thread.start()
            
            self.app.exec_()

    def start_dashboard_updater(self):
        self.dashboard_updater = DashboardUpdater(
            self.alarm_queue,
            self.face_identity_queue
        )
        if hasattr(self.dashboard, 'add_alarm'):
            self.dashboard_updater.alarm_signal.connect(self.dashboard.add_alarm)
        if hasattr(self.dashboard, 'update_status'):
            self.dashboard_updater.status_signal.connect(self.dashboard.update_status)
        if hasattr(self.dashboard, 'update_face_identity'):
            self.dashboard_updater.face_identity_signal.connect(
                self.dashboard.update_face_identity
            )
        self.dashboard_updater.start()
        logger.info("Dashboard updater started")
    
    def processing_loop(self):
        import queue
        while self.running:
            try:
                # 阻塞等待帧 (事件驱动, 无 busy-wait CPU消耗)
                frame_item = self.processed_frame_queue.get(timeout=0.1)
                self.process_frame(frame_item)
            except queue.Empty:
                continue
            except Exception as e:
                if not self.running:
                    break
                logger.error(f"Processing loop error: {str(e)}")
    
    def stop(self):
        logger.info("Stopping Silver Guardian system...")
        
        self.running = False
        
        if self.dashboard_updater:
            self.dashboard_updater.stop()
        
        for worker in self.camera_workers:
            worker.stop()
        
        if self.thread_pool:
            self.thread_pool.shutdown()
        
        if self.frame_processor:
            self.frame_processor.stop()
        
        if self.camera_manager:
            self.camera_manager.release_all()
        
        if self.app:
            self.app.quit()
        
        logger.info("System stopped successfully")

def signal_handler(signal, frame):
    logger.info("Received exit signal")
    if 'system' in globals():
        system.stop()
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description='银发守护者 AI 视觉应用系统')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--demo', action='store_true', default=False,
                        help='Run in demo mode without real cameras')

    args = parser.parse_args()

    # 创建 QApplication（必须在 init_components 前，以便显示 splash）
    import sys as _sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(_sys.argv)

    global system
    system = SilverGuardianSystem(args.config, args.demo)
    system.app = app  # 传入已有的 QApplication

    # 开屏加载画面 — 在模型加载前显示
    splash_bg = os.path.join(system.base_path, '开屏加载背景画面.png')
    system.splash = SplashScreen(splash_bg, total_steps=8)
    system.splash.show()
    app.processEvents()

    system.init_components()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        system.run()
    except Exception as e:
        logger.error(f"System error: {str(e)}", exc_info=True)
        system.stop()

if __name__ == '__main__':
    main()
