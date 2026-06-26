import cv2
import numpy as np
import threading
import time
import logging
import os
from typing import Optional, Tuple, Any
from utils.common import put_latest

# 抑制 OpenCV 摄像头探测时的 WARN/ERROR 刷屏
os.environ['OPENCV_LOG_LEVEL'] = 'FATAL'

logger = logging.getLogger(__name__)

class CameraWorker:
    def __init__(self, camera_id: str, camera_name: str, source: Any, 
                 width: int = 1920, height: int = 1080, fps: int = 30):
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.connected = False
        self.frame_queue = None
        self.thread: Optional[threading.Thread] = None
        self.last_frame_time = 0
        self.frame_count = 0
        self.drop_count = 0
        self.reconnect_interval = 3
        self.last_reconnect_attempt = 0
        self.last_error = None
        self._consecutive_fails = 0  # 连续读取失败计数

    def set_queue(self, frame_queue):
        self.frame_queue = frame_queue

    def _open_capture(self):
        if isinstance(self.source, str) and self.source.startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://')
        ):
            for backend in (cv2.CAP_FFMPEG, cv2.CAP_ANY):
                try:
                    cap = cv2.VideoCapture(self.source, backend)
                    if cap.isOpened():
                        return cap, backend, None
                    cap.release()
                except Exception as exc:
                    self.last_error = str(exc)
            return None, None, self.last_error

        if isinstance(self.source, str) and not self.source.isdigit():
            try:
                cap = cv2.VideoCapture(self.source, cv2.CAP_ANY)
                if cap.isOpened():
                    return cap, cv2.CAP_ANY, None
                cap.release()
            except Exception as exc:
                self.last_error = str(exc)
            return None, None, self.last_error or f"invalid_source:{self.source}"

        camera_index = int(self.source)
        # 优先 DSHOW，回退 MSMF
        attempted_backends = (cv2.CAP_DSHOW, cv2.CAP_MSMF)
        backend_errors = []
        for backend in attempted_backends:
            try:
                cap = cv2.VideoCapture(camera_index, backend)
                if cap.isOpened():
                    return cap, backend, None
                cap.release()
            except Exception as exc:
                backend_errors.append(f"{backend}:{exc}")
        return None, None, "; ".join(backend_errors) or f"camera_index_unavailable:{camera_index}"

    def connect(self) -> bool:
        try:
            self.cap, backend, open_error = self._open_capture()
            
            if self.cap is None or not self.cap.isOpened():
                self.last_error = open_error or "capture_not_opened"
                logger.error(
                    f"Failed to open camera {self.camera_id}: {self.camera_name}; "
                    f"source={self.source}; error={self.last_error}"
                )
                return False
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            self.connected = True
            self._consecutive_fails = 0
            self.last_error = None
            logger.info(
                f"Successfully connected to camera {self.camera_id}: {self.camera_name} "
                f"(source={self.source}, backend={backend})"
            )
            return True
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Connection error for camera {self.camera_id}: {str(e)}")
            return False

    def disconnect(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.connected = False
        logger.info(f"Disconnected camera {self.camera_id}: {self.camera_name}")

    def capture_frame(self) -> Optional[Tuple[np.ndarray, float]]:
        if not self.connected or self.cap is None:
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            self._consecutive_fails += 1
            if self._consecutive_fails >= 3:
                self.connected = False
                self.last_error = "read_failed_3x"
                logger.warning(f"Camera {self.camera_id}: 连续 {self._consecutive_fails} 次读取失败，断开")
            return None
        else:
            self._consecutive_fails = 0
        
        timestamp = time.time()
        return (frame, timestamp)

    def run(self):
        self.running = True
        frame_interval = 1.0 / self.fps
        
        while self.running:
            current_time = time.time()
            
            if not self.connected:
                if current_time - self.last_reconnect_attempt >= self.reconnect_interval:
                    logger.info(f"Attempting to reconnect camera {self.camera_id}")
                    self.last_reconnect_attempt = current_time
                    self.connect()
                time.sleep(0.1)
                continue
            
            if current_time - self.last_frame_time >= frame_interval:
                result = self.capture_frame()
                if result is not None:
                    frame, timestamp = result
                    self.last_frame_time = current_time
                    self.frame_count += 1
                    
                    if self.frame_queue is not None:
                        item = {
                            'camera_id': self.camera_id,
                            'camera_name': self.camera_name,
                            'frame': frame,
                            'timestamp': timestamp,
                            'frame_count': self.frame_count
                        }
                        dropped = put_latest(self.frame_queue, item)
                        if dropped:
                            self.drop_count += dropped
                            logger.warning(
                                f"Frame queue full for camera {self.camera_id}, "
                                f"dropped {dropped} stale frames"
                            )
                else:
                    time.sleep(0.01)
            else:
                time.sleep(0.001)

    def start(self):
        if not self.thread or not self.thread.is_alive():
            self.thread = threading.Thread(target=self.run, name=f"CameraWorker-{self.camera_id}", daemon=True)
            self.thread.start()
            logger.info(f"Started camera worker thread for {self.camera_id}")

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        self.disconnect()
        logger.info(f"Stopped camera worker for {self.camera_id}")

    def get_stats(self) -> dict:
        return {
            'camera_id': self.camera_id,
            'camera_name': self.camera_name,
            'connected': self.connected,
            'frame_count': self.frame_count,
            'drop_count': self.drop_count,
            'fps': self.fps,
            'last_error': self.last_error
        }
