import cv2
import threading
import time
import logging
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

logger = logging.getLogger(__name__)

class CameraDevice:
    def __init__(self, index, name, resolution=None, fps=None):
        self.index = index
        self.name = name
        self.resolution = resolution
        self.fps = fps
        self.connected = True
    
    def to_dict(self):
        return {
            'index': self.index,
            'name': self.name,
            'resolution': self.resolution,
            'fps': self.fps,
            'connected': self.connected
        }

class CameraDetector(QObject):
    devices_changed = pyqtSignal(list)
    device_connected = pyqtSignal(CameraDevice)
    device_disconnected = pyqtSignal(int)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.devices = []
        self.monitoring = False
        self.monitor_thread = None
        self.last_scan_time = 0
        self.scan_interval = 2.0
    
    def scan_cameras(self):
        """扫描系统中可用的摄像头设备"""
        new_devices = []
        max_index = 10
        
        for index in range(max_index):
            try:
                cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
                if cap.isOpened():
                    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    resolution = None
                    if width > 0 and height > 0:
                        resolution = f"{int(width)}x{int(height)}"
                    
                    fps_value = None
                    if fps > 0:
                        fps_value = int(fps)
                    
                    device_name = self._get_device_name(index, cap)
                    
                    device = CameraDevice(
                        index=index,
                        name=device_name,
                        resolution=resolution,
                        fps=fps_value
                    )
                    new_devices.append(device)
                    
                    cap.release()
            except Exception as e:
                logger.debug(f"Failed to open camera {index}: {str(e)}")
        
        return new_devices
    
    def _get_device_name(self, index, cap):
        """获取摄像头设备名称"""
        try:
            backends = [cv2.CAP_ANY, cv2.CAP_DSHOW, cv2.CAP_MSMF]
            for backend in backends:
                temp_cap = cv2.VideoCapture(index, backend)
                if temp_cap.isOpened():
                    name = temp_cap.get(cv2.CAP_PROP_DEVICE_NAME)
                    temp_cap.release()
                    if name and name != "":
                        return str(name)
        except Exception:
            pass
        
        return f"摄像头 {index + 1}"
    
    def get_available_devices(self):
        """获取当前可用的摄像头列表"""
        return [d.to_dict() for d in self.devices]
    
    def start_monitoring(self):
        """开始监测设备状态变化"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Camera device monitoring started")
    
    def stop_monitoring(self):
        """停止设备状态监测"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        logger.info("Camera device monitoring stopped")
    
    def _monitor_loop(self):
        """设备状态监测循环"""
        while self.monitoring:
            current_time = time.time()
            if current_time - self.last_scan_time >= self.scan_interval:
                try:
                    new_devices = self.scan_cameras()
                    self._update_devices(new_devices)
                    self.last_scan_time = current_time
                except Exception as e:
                    logger.error(f"Camera monitoring error: {str(e)}")
                    self.error_occurred.emit(f"设备监测错误: {str(e)}")
            
            time.sleep(0.5)
    
    def _update_devices(self, new_devices):
        """更新设备列表并发出信号"""
        old_indices = set(d.index for d in self.devices)
        new_indices = set(d.index for d in new_devices)
        
        for device in new_devices:
            if device.index not in old_indices:
                self.device_connected.emit(device)
        
        for device in self.devices:
            if device.index not in new_indices:
                self.device_disconnected.emit(device.index)
        
        self.devices = new_devices
        self.devices_changed.emit([d.to_dict() for d in self.devices])
    
    def test_device(self, index):
        """测试指定摄像头设备是否可用"""
        try:
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                return ret is True
            cap.release()
            return False
        except Exception as e:
            logger.error(f"Test device {index} failed: {str(e)}")
            return False
    
    def release_all(self):
        """释放所有资源"""
        self.stop_monitoring()

class CameraPermissionError(Exception):
    """摄像头权限错误"""
    pass

class CameraNotFoundError(Exception):
    """摄像头未找到错误"""
    pass