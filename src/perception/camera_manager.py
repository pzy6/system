import os
import cv2
os.environ.setdefault('OPENCV_LOG_LEVEL', 'ERROR')
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, QTimer
import threading
import time

class CameraDevice:
    def __init__(self, index, name):
        self.index = index
        self.name = name
        self.resolution = None
        self.fps = None
        self.connected = False
        self.cap = None
        self.backend = None
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._capture_thread = None
        self._capture_running = False

    def connect(self):
        backends = [
            cv2.CAP_ANY,
            cv2.CAP_DSHOW,
            cv2.CAP_MSMF,
            cv2.CAP_VFW
        ]

        for backend in backends:
            try:
                self.cap = cv2.VideoCapture(self.index, backend)
                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    self.connected = True
                    self.backend = backend
                    self._get_device_info()
                    return True
                self.cap.release()
            except Exception as e:
                continue

        return False

    def _get_device_info(self):
        if self.cap:
            width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps_val = self.cap.get(cv2.CAP_PROP_FPS)

            if width > 0 and height > 0:
                self.resolution = f"{int(width)}x{int(height)}"
            else:
                self.resolution = "未知"

            if fps_val > 0:
                self.fps = int(fps_val)
            else:
                self.fps = 30

    def disconnect(self):
        self._capture_running = False
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=1.5)
            self._capture_thread = None
        if self.cap:
            self.cap.release()
            self.cap = None
        self.connected = False
        self.backend = None
        with self._frame_lock:
            self._latest_frame = None

    def start_capture_thread(self):
        if self._capture_running:
            return
        self._capture_running = True
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True,
            name=f"CameraCap-{self.index}"
        )
        self._capture_thread.start()

    def _capture_loop(self):
        while self._capture_running and self.connected:
            if self.cap is None:
                break
            grabbed = self.cap.grab()
            if not grabbed:
                time.sleep(0.005)
                continue
            ret, frame = self.cap.retrieve()
            if ret and frame is not None:
                with self._frame_lock:
                    self._latest_frame = frame
        self._capture_running = False

    def get_latest_frame(self):
        with self._frame_lock:
            frame = self._latest_frame
            self._latest_frame = None
        return frame

    def read_frame(self):
        if self.connected and self.cap:
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None

class CameraManager(QObject):
    device_connected = pyqtSignal(object)
    device_disconnected = pyqtSignal(int)
    frame_ready = pyqtSignal(int, np.ndarray)
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.devices = {}
        self.scan_timer = QTimer()
        self.scan_timer.timeout.connect(self.scan_devices)
        self.scan_timer.start(3000)

        self.capture_timers = {}
        self._capture_active_indices = set()
        self._display_timer = QTimer()
        self._display_timer.timeout.connect(self._push_latest_frames)
        self._display_timer.start(33)

    def stop_scanning(self):
        if self.scan_timer.isActive():
            self.scan_timer.stop()

    def _push_latest_frames(self):
        for idx in list(self._capture_active_indices):
            device = self.devices.get(idx)
            if device and device.connected:
                frame = device.get_latest_frame()
                if frame is not None:
                    self.frame_ready.emit(idx, frame)

    def scan_devices(self):
        if self._capture_active_indices:
            return
        try:
            found_indices = set()

            for i in range(10):
                success = self._test_camera(i)
                if success:
                    found_indices.add(i)

            current_indices = set(self.devices.keys())

            for index in found_indices - current_indices:
                self._add_device(index)

            for index in current_indices - found_indices:
                self._remove_device(index)

        except Exception as e:
            self.error_occurred.emit(f"扫描设备失败: {str(e)}")

    def _test_camera(self, index):
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]

        for backend in backends:
            try:
                cap = cv2.VideoCapture(index, backend)
                if cap.isOpened():
                    cap.release()
                    return True
            except:
                continue

        return False

    def _add_device(self, index):
        device = CameraDevice(index, f"摄像头 {index + 1}")
        if device.connect():
            self.devices[index] = device
            self.device_connected.emit(device)

    def _remove_device(self, index):
        if index in self.devices:
            self.devices[index].disconnect()
            del self.devices[index]
            self.device_disconnected.emit(index)

    def get_devices(self):
        return list(self.devices.values())

    def get_device(self, index):
        return self.devices.get(index)

    def start_capture(self, index):
        if index not in self.devices:
            return None

        device = self.devices[index]
        if not device.connected:
            return None

        device.start_capture_thread()
        self._capture_active_indices.add(index)
        return True

    def stop_capture(self, index):
        if index in self._capture_active_indices:
            device = self.devices.get(index)
            if device:
                device.disconnect()
            self._capture_active_indices.discard(index)

    def release_all(self):
        for idx in list(self._capture_active_indices):
            self.stop_capture(idx)
        if self.scan_timer.isActive():
            self.scan_timer.stop()
        if self._display_timer.isActive():
            self._display_timer.stop()
        for device in self.devices.values():
            device.disconnect()
        self.devices.clear()
        self.capture_timers.clear()

def test_camera_access():
    """测试摄像头访问权限和可用性"""
    results = {
        'has_camera': False,
        'devices': [],
        'errors': []
    }
    
    print("正在检测摄像头设备...")
    
    for i in range(10):
        device = CameraDevice(i, f"设备 {i}")
        success = device.connect()
        
        if success:
            print(f"[OK] 摄像头 {i}: {device.resolution} @ {device.fps}fps")
            results['has_camera'] = True
            results['devices'].append({
                'index': i,
                'name': device.name,
                'resolution': device.resolution,
                'fps': device.fps
            })
            device.disconnect()
        else:
            results['errors'].append(f"摄像头 {i} 无法访问")
    
    if not results['has_camera']:
        print("[FAIL] 未检测到可用摄像头")
        print("\n可能的原因：")
        print("1. 摄像头未连接或已被其他应用占用")
        print("2. 缺少摄像头驱动程序")
        print("3. 应用程序权限不足")
        print("4. 系统摄像头服务未启动")
    
    return results

if __name__ == "__main__":
    test_camera_access()
