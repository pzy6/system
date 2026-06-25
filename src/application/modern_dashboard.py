import sys
import os
import cv2
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QFrame, QListWidget, QListWidgetItem,
    QGroupBox, QScrollArea, QSizePolicy, QSpacerItem
)
from PyQt5.QtGui import QFont, QColor, QPalette, QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer, QRect, pyqtSignal, pyqtSlot
import time

from .config_dialog import ConfigDialog
from .toast_widget import ToastWidget
from .alarm_panel import CollapsibleAlarmPanel
from perception.camera_manager import CameraManager, CameraDevice

class ModernDashboard(QMainWindow):
    def __init__(self, auto_camera_scan=True, config=None, config_path=None, model_status=None, frame_queue=None):
        super().__init__()
        self.setWindowTitle("银发守护者系统")
        self.setGeometry(0, 0, 1920, 1080)
        self.setMinimumSize(1440, 810)
        self.auto_camera_scan = auto_camera_scan
        self.config = config or {}
        self.config_path = config_path or os.path.join(os.getcwd(), 'config', 'config.yaml')
        self.model_status = model_status or {}
        self._frame_queue = frame_queue  # 直接轮询队列 (避免信号序列化)

        self.camera_manager = None
        if self.auto_camera_scan:
            self.camera_manager = CameraManager()
            self.camera_manager.device_connected.connect(self.on_camera_connected)
            self.camera_manager.device_disconnected.connect(self.on_camera_disconnected)
            self.camera_manager.frame_ready.connect(self.on_frame_ready)

        self.camera_frames = {}
        self.camera_timers = {}
        self.camera_status_dots = {}
        self.camera_res_labels = {}
        self.active_cameras = set()
        self._camera_id_map = {}  # camera_id字符串→显示索引
        self._setup_frame_poller()

        # 手动检测模块
        self._latest_frame = None
        self._emotion_checker = None  # EmotionRecognizer 惰性加载

        # Toast + 折叠报警栏
        self.toast = ToastWidget(self)

        self.demo_timer = QTimer()
        self.demo_timer.timeout.connect(self.update_demo_frames)
        self.demo_timer.start(66)

        self.init_ui()

    def update_demo_frames(self):
        for index in range(4):
            if index in self.camera_frames and index not in self.active_cameras:
                label = self.camera_frames[index]
                if label.width() > 0 and label.height() > 0:
                    demo_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(demo_frame, f"Camera {index+1} - Waiting...", (80, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 150, 200), 2)
                    frame_rgb = cv2.cvtColor(demo_frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = frame_rgb.shape
                    bytes_per_line = ch * w
                    q_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(q_image).scaled(
                        label.width(), label.height(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
                    label.setPixmap(pixmap)

    def init_ui(self):
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #0A1628, stop:1 #1A2A4A);
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)

        main_layout.addLayout(self.create_header())

        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)

        content_layout.addWidget(self.create_video_panel(), 8)
        content_layout.addLayout(self.create_side_panel(), 2)

        main_layout.addLayout(content_layout)

    def create_header(self):
        header_layout = QHBoxLayout()
        header_layout.setSpacing(24)

        title_label = QLabel("银发守护者")
        title_label.setFont(QFont("Microsoft YaHei", 42, QFont.Bold))
        title_label.setStyleSheet("color: #4A90E2; background: transparent;")
        header_layout.addWidget(title_label)

        header_layout.addStretch()

        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(20)

        self.elder_count_card = self.create_stat_card("老人总数", "256", "#4A90E2")
        self.device_count_card = self.create_stat_card("智能设备", "128", "#38A169")

        stats_layout.addWidget(self.elder_count_card)
        stats_layout.addWidget(self.device_count_card)

        header_layout.addLayout(stats_layout)

        return header_layout

    def create_stat_card(self, label, value, color):
        card = QFrame()
        card.setFixedSize(180, 90)
        card.setStyleSheet("""
            QFrame {
                background: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 16px;
            }
        """)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)

        label_widget = QLabel(label)
        label_widget.setFont(QFont("Microsoft YaHei", 14))
        label_widget.setStyleSheet("color: #8FA3BF;")
        label_widget.setAlignment(Qt.AlignCenter)
        layout.addWidget(label_widget)

        value_widget = QLabel(value)
        value_widget.setFont(QFont("Microsoft YaHei", 32, QFont.Bold))
        value_widget.setStyleSheet(f"color: {color};")
        value_widget.setAlignment(Qt.AlignCenter)
        layout.addWidget(value_widget)

        return card

    def create_video_panel(self):
        panel = QFrame()
        panel.setStyleSheet("""
            QFrame {
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 20px;
            }
        """)
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        header_layout = QHBoxLayout()

        title = QLabel("实时监控")
        title.setFont(QFont("Microsoft YaHei", 26, QFont.Bold))
        title.setStyleSheet("color: #FFFFFF;")
        header_layout.addWidget(title)

        header_layout.addStretch()

        camera_btn = QPushButton("📹 摄像头联动")
        camera_btn.setFont(QFont("Microsoft YaHei", 16))
        camera_btn.setStyleSheet("""
            QPushButton {
                background: linear-gradient(135deg, #4A90E2 0%, #3A80D2 100%);
                color: white;
                border: none;
                border-radius: 12px;
                padding: 12px 24px;
            }
            QPushButton:hover {
                background: linear-gradient(135deg, #3A80D2 0%, #2A70C2 100%);
            }
            QPushButton:pressed {
                background: linear-gradient(135deg, #2A70C2 0%, #1A60B2 100%);
            }
        """)
        camera_btn.clicked.connect(self.scan_and_connect_cameras)
        header_layout.addWidget(camera_btn)

        config_btn = QPushButton("配置")
        config_btn.setFont(QFont("Microsoft YaHei", 16))
        config_btn.setStyleSheet("""
            QPushButton {
                background: rgba(255, 255, 255, 0.12);
                color: white;
                border: 1px solid rgba(255, 255, 255, 0.18);
                border-radius: 12px;
                padding: 12px 20px;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 0.18);
            }
        """)
        config_btn.clicked.connect(self.show_config_dialog)
        header_layout.addWidget(config_btn)

        enroll_btn = QPushButton("人脸录入")
        enroll_btn.setFont(QFont("Microsoft YaHei", 11))
        enroll_btn.setStyleSheet("""
            QPushButton {
                background: rgba(74, 144, 226, 0.25);
                color: #4A90E2;
                border: 1px solid #4A90E2;
                border-radius: 8px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background: rgba(74, 144, 226, 0.40);
            }
        """)
        enroll_btn.clicked.connect(self._open_face_enroll)
        header_layout.addWidget(enroll_btn)

        layout.addLayout(header_layout)

        grid = QGridLayout()
        grid.setSpacing(16)
        grid.setContentsMargins(0, 0, 0, 0)

        positions = [(0, 0)]  # 单路，占满整个监控区域
        for idx, (row, col) in enumerate(positions):
            cam_widget = QFrame()
            cam_widget.setStyleSheet("""
                QFrame {
                    background: rgba(0, 0, 0, 0.3);
                    border: 2px solid rgba(255, 255, 255, 0.1);
                    border-radius: 16px;
                }
            """)
            cam_widget.setSizePolicy(
                QSizePolicy.Expanding, QSizePolicy.Expanding
            )
            cam_layout = QVBoxLayout(cam_widget)
            cam_layout.setContentsMargins(0, 0, 0, 0)
            cam_layout.setSpacing(0)

            cam_label = QLabel()
            cam_label.setAlignment(Qt.AlignCenter)
            cam_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            cam_label.setScaledContents(True)  # 让画面随 label 缩放撑满
            cam_layout.addWidget(cam_label, stretch=1)
            self.camera_frames[idx] = cam_label

            status_bar = QHBoxLayout()
            status_bar.setContentsMargins(16, 10, 16, 10)

            dot = QFrame()
            dot.setFixedSize(10, 10)
            dot.setStyleSheet("background-color: #E53E3E; border-radius: 50%;")
            self.camera_status_dots[idx] = dot
            status_bar.addWidget(dot)

            res_label = QLabel("Waiting")
            res_label.setFont(QFont("Microsoft YaHei", 12))
            res_label.setStyleSheet("color: #8FA3BF;")
            self.camera_res_labels[idx] = res_label
            status_bar.addWidget(res_label)

            status_bar.addStretch()
            cam_layout.addLayout(status_bar)

            grid.addWidget(cam_widget, row, col)

        layout.addLayout(grid)
        return panel

    def create_side_panel(self):
        side_layout = QVBoxLayout()
        side_layout.setSpacing(16)

        side_layout.addWidget(self.create_chart_panel(), 2)
        side_layout.addWidget(self.create_alarm_panel(), 3)
        side_layout.addStretch(1)  # vitals panel removed
        side_layout.addWidget(self.create_emotion_panel(), 1)

        return side_layout

    def create_chart_panel(self):
        panel = QFrame()
        panel.setStyleSheet("""
            QFrame {
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 20px;
            }
        """)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        title = QLabel("走访情况统计分析")
        title.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        title.setStyleSheet("color: #FFFFFF;")
        layout.addWidget(title)

        chart_container = QFrame()
        chart_container.setStyleSheet("background: rgba(0, 0, 0, 0.3); border-radius: 16px;")
        chart_layout = QVBoxLayout(chart_container)
        chart_layout.setContentsMargins(20, 20, 20, 20)
        chart_layout.setSpacing(16)

        bars_layout = QHBoxLayout()
        bars_layout.setSpacing(16)
        bars_layout.setAlignment(Qt.AlignBottom)

        data = [
            ("Mian", 200, "#4A90E2"),
            ("Tou", 150, "#FF9F43"),
            ("Wen", 100, "#38A169"),
            ("Fai", 50, "#ECC94B")
        ]

        max_value = 200

        for name, value, color in data:
            bar_group = QVBoxLayout()
            bar_group.setSpacing(8)
            bar_group.setAlignment(Qt.AlignCenter)

            bar_frame = QFrame()
            bar_height = int((value / max_value) * 160)
            bar_frame.setFixedSize(50, bar_height)
            bar_frame.setStyleSheet(f"""
                background: linear-gradient(180deg, {color} 0%, {color}88 100%);
                border-radius: 8px 8px 0 0;
            """)
            bar_group.addWidget(bar_frame)

            value_label = QLabel(str(value))
            value_label.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
            value_label.setStyleSheet(f"color: {color};")
            bar_group.addWidget(value_label)

            name_label = QLabel(name)
            name_label.setFont(QFont("Microsoft YaHei", 12))
            name_label.setStyleSheet("color: #8FA3BF;")
            bar_group.addWidget(name_label)

            bars_layout.addLayout(bar_group)

        chart_layout.addLayout(bars_layout)

        layout.addWidget(chart_container)

        legend_layout = QHBoxLayout()
        legend_layout.setSpacing(12)

        for name, _, color in data:
            legend_item = QHBoxLayout()
            legend_item.setSpacing(6)

            dot = QFrame()
            dot.setFixedSize(10, 10)
            dot.setStyleSheet(f"background-color: {color}; border-radius: 50%;")

            label = QLabel(name)
            label.setFont(QFont("Microsoft YaHei", 12))
            label.setStyleSheet("color: #8FA3BF;")

            legend_item.addWidget(dot)
            legend_item.addWidget(label)
            legend_layout.addLayout(legend_item)

        legend_layout.addStretch()
        layout.addLayout(legend_layout)

        return panel

    def create_alarm_panel(self):
        panel = QFrame()
        panel.setStyleSheet("""
            QFrame {
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 20px;
            }
        """)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        header_layout = QHBoxLayout()
        title = QLabel("报警信息")
        title.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        title.setStyleSheet("color: #FFFFFF;")
        header_layout.addWidget(title)

        badge = QFrame()
        badge.setFixedSize(28, 28)
        badge.setStyleSheet("background-color: #E53E3E; border-radius: 50%;")
        badge_label = QLabel("0")
        badge_label.setParent(badge)
        badge_label.setGeometry(QRect(0, 0, 28, 28))
        badge_label.setAlignment(Qt.AlignCenter)
        badge_label.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        badge_label.setStyleSheet("color: white;")
        header_layout.addWidget(badge)

        header_layout.addStretch()
        layout.addLayout(header_layout)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("border: none; background: transparent;")

        list_widget = QListWidget()
        list_widget.setStyleSheet("""
            QListWidget {
                border: none;
                background: transparent;
            }
            QListWidget::item {
                padding: 14px;
                border-bottom: 1px solid rgba(255, 255, 255, 0.05);
                color: #C8D4E3;
                font-family: Microsoft YaHei;
                font-size: 14px;
            }
            QListWidget::item:last-child {
                border-bottom: none;
            }
            QListWidget::item:hover {
                background: rgba(255, 255, 255, 0.05);
            }
        """)

        self.alarm_list = list_widget
        self.alarm_badge = badge
        self.alarm_badge_label = badge_label
        self.alarm_count = 0

        scroll_area.setWidget(list_widget)
        layout.addWidget(scroll_area)

        return panel

    def create_emotion_panel(self):
        panel = QFrame()
        panel.setStyleSheet("""
            QFrame {
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 20px;
            }
        """)

        layout = QHBoxLayout(panel)
        layout.setContentsMargins(20, 14, 20, 14)
        layout.setSpacing(12)

        title = QLabel("情绪状态")
        title.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        title.setStyleSheet("color: #FFFFFF;")
        layout.addWidget(title)

        layout.addStretch()

        self.emotion_label = QLabel("检测中...")
        self.emotion_label.setFont(QFont("Microsoft YaHei", 16))
        self.emotion_label.setStyleSheet("color: #8FA3BF;")
        layout.addWidget(self.emotion_label)

        self.emotion_indicator = QFrame()
        self.emotion_indicator.setFixedSize(16, 16)
        self.emotion_indicator.setStyleSheet("background-color: #ECC94B; border-radius: 50%;")
        layout.addWidget(self.emotion_indicator)

        self.emotion_btn = QPushButton("检测情绪")
        self.emotion_btn.setFont(QFont("Microsoft YaHei", 10))
        self.emotion_btn.setStyleSheet(
            "QPushButton { background: #4A90E2; color: #fff; border-radius: 8px; padding: 6px 12px; }"
            "QPushButton:hover { background: #357ABD; }"
        )
        self.emotion_btn.clicked.connect(self._start_emotion_check)
        layout.addWidget(self.emotion_btn)

        return panel

    def create_system_status_panel(self):
        panel = QFrame()
        panel.setStyleSheet("""
            QFrame {
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 16px;
            }
        """)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        status_title = QLabel("系统状态")
        status_title.setFont(QFont("Microsoft YaHei", 14))
        status_title.setStyleSheet("color: #8FA3BF;")
        layout.addWidget(status_title)

        status_value = QLabel("✔ 运行正常")
        status_value.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        status_value.setStyleSheet("color: #38A169;")
        layout.addWidget(status_value)

        return panel

    # --- Data display slots (connected from DashboardUpdater) ---

    def add_alarm(self, alarm_event):
        """Receive alarm from processing pipeline and display in alarm list."""
        # Toast 弹出
        face = getattr(alarm_event, 'face_name', '') or ''
        self.toast.show_alarm(alarm_event.alarm_type, alarm_event.camera_name, face)

        from datetime import datetime as dt
        time_str = dt.fromtimestamp(alarm_event.timestamp).strftime("%H:%M:%S")
        alarm_text = f"[{time_str}] {alarm_event.alarm_type} - {alarm_event.camera_name}"
        item = QListWidgetItem(alarm_text)
        if alarm_event.alarm_type in ("跌倒检测", "fall_not_recovered"):
            item.setForeground(QColor("#E53E3E"))
        elif alarm_event.alarm_type in ("心率异常", "呼吸异常"):
            item.setForeground(QColor("#FF9F43"))
        else:
            item.setForeground(QColor("#ED8936"))
        self.alarm_list.insertItem(0, item)
        self.alarm_count += 1
        self.alarm_badge_label.setText(str(min(self.alarm_count, 99)))
        while self.alarm_list.count() > 50:
            self.alarm_list.takeItem(self.alarm_list.count() - 1)

    def update_emotion_display(self, emotion):
        """Update emotion status indicator."""
        emo = emotion.get("emotion", "neutral")
        conf = emotion.get("confidence", 0.0)
        emo_cn = {"angry": "生气", "disgust": "反感", "fear": "恐惧",
                   "happy": "高兴", "sad": "悲伤", "surprise": "惊讶",
                   "neutral": "平静"}
        display = emo_cn.get(emo, emo)
        self.emotion_label.setText(f"{display} ({conf:.0%})")
        if emotion.get("is_negative"):
            self.emotion_indicator.setStyleSheet("background-color: #E53E3E; border-radius: 50%;")
            self.emotion_label.setStyleSheet("color: #E53E3E;")
        else:
            self.emotion_indicator.setStyleSheet("background-color: #38A169; border-radius: 50%;")
            self.emotion_label.setStyleSheet("color: #38A169;")

    # ------------------------------------------------------------------
    # 手动检测 — 生命体征
    # ------------------------------------------------------------------



    def _start_emotion_check(self):
        """单击检测当前帧的情绪"""
        if self._latest_frame is None:
            self.emotion_label.setText("无画面")
            return
        if self._emotion_checker is None:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
            from algorithms.emotion_recognition import EmotionRecognizer
            self._emotion_checker = EmotionRecognizer()
        result = self._emotion_checker.analyze(self._latest_frame)
        display = result.get("emotion_cn", "未知")
        conf = result.get("confidence", 0)
        self.emotion_label.setText(f"{display} ({conf:.0%})")
        if result.get("is_negative"):
            self.emotion_indicator.setStyleSheet(
                "background-color: #E53E3E; border-radius: 50%;"
            )
        else:
            self.emotion_indicator.setStyleSheet(
                "background-color: #38A169; border-radius: 50%;"
            )

    def _open_face_enroll(self):
        """打开人脸录入独立程序"""
        from application.face_enroll_app import FaceEnrollApp
        self._enroll_window = FaceEnrollApp()
        self._enroll_window.show()

    def show_config_dialog(self):
        dialog = ConfigDialog(
            config=self.config,
            config_path=self.config_path,
            model_status=self.model_status,
            parent=self
        )
        if dialog.exec_():
            self.config = dialog.config
            self.model_status = dialog.model_status

    def scan_and_connect_cameras(self):
        if self.camera_manager is None:
            for i in range(4):
                self.camera_status_dots[i].setStyleSheet("background-color: #38A169; border-radius: 50%;")
                self.camera_res_labels[i].setText("Pipeline managed")
            return

        print("Scanning and connecting cameras...")
        self.camera_manager.scan_devices()

        device_count = len(self.camera_manager.get_devices())
        print(f"Found {device_count} camera device(s)")

        if device_count == 0:
            for i in range(4):
                self.camera_status_dots[i].setStyleSheet("background-color: #ECC94B; border-radius: 50%;")
                self.camera_res_labels[i].setText("No camera detected")
            QTimer.singleShot(3000, self._reset_camera_status)
            return

        connected_count = 0
        for device in self.camera_manager.get_devices()[:4]:
            if device.index < 4:
                self.camera_status_dots[device.index].setStyleSheet("background-color: #38A169; border-radius: 50%;")
                self.camera_res_labels[device.index].setText(device.resolution or "Connected")
                timer = self.camera_manager.start_capture(device.index)
                if timer:
                    self.camera_timers[device.index] = timer
                    connected_count += 1
                    print(f"Camera {device.index} capturing started")

        if connected_count == 0:
            for i in range(4):
                self.camera_status_dots[i].setStyleSheet("background-color: #E53E3E; border-radius: 50%;")
                self.camera_res_labels[i].setText("Connect failed")

    def _reset_camera_status(self):
        for i in range(4):
            if i in self.camera_status_dots:
                self.camera_status_dots[i].setStyleSheet("background-color: #E53E3E; border-radius: 50%;")
                self.camera_res_labels[i].setText("Waiting")

    def showEvent(self, event):
        super().showEvent(event)
        if self.auto_camera_scan:
            QTimer.singleShot(1000, self.auto_connect_cameras)

    def auto_connect_cameras(self):
        if self.camera_manager is None:
            return
        print("Auto-scanning and connecting cameras...")
        self.camera_manager.scan_devices()

    def on_camera_connected(self, device):
        print(f"Camera connected: {device.name}, resolution: {device.resolution}")
        self.active_cameras.add(device.index)

        if device.index < 4:
            self.camera_status_dots[device.index].setStyleSheet("background-color: #38A169; border-radius: 50%;")
            self.camera_res_labels[device.index].setText(device.resolution or "已连接")

            timer = self.camera_manager.start_capture(device.index)
            if timer:
                self.camera_timers[device.index] = timer
                print(f"Camera {device.index} capture started")

    def on_camera_disconnected(self, index):
        print(f"Camera disconnected: {index}")
        if index in self.camera_status_dots:
            self.camera_status_dots[index].setStyleSheet("background-color: #E53E3E; border-radius: 50%;")
            self.camera_res_labels[index].setText("已断开")

    def _setup_frame_poller(self):
        """QTimer 轮询帧队列 (主线程，避免 pyqtSignal 序列化 ndarray)"""
        if self._frame_queue is None:
            return
        self._frame_timer = QTimer()
        self._frame_timer.timeout.connect(self._poll_frames)
        self._frame_timer.start(33)  # ~30fps 刷新率

    def _poll_frames(self):
        """直接从队列读取帧并渲染 (无跨线程拷贝)"""
        q = self._frame_queue
        if q is None:
            return
        # 确保 camera_id 映射已初始化
        if not self._camera_id_map:
            for i, cam in enumerate(self.config.get('cameras', [])):
                self._camera_id_map[cam['id']] = i
        try:
            while not q.empty():
                item = q.get_nowait()
                cam_id = item['camera_id']
                idx = self._camera_id_map.get(cam_id, 0)
                self.on_frame_ready(idx, item['frame'])
        except Exception:
            pass

    def on_frame_ready(self, index, frame):
        if index not in self.camera_frames:
            return

        self._latest_frame = frame  # 保存最新帧供手动检测使用
        self.active_cameras.add(index)
        if index in self.camera_status_dots:
            self.camera_status_dots[index].setStyleSheet("background-color: #38A169; border-radius: 50%;")
        if index in self.camera_res_labels:
            self.camera_res_labels[index].setText(f"{frame.shape[1]}x{frame.shape[0]}")

        label = self.camera_frames[index]

        if label.width() <= 0 or label.height() <= 0:
            return

        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).copy()
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            q_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

            pixmap = QPixmap.fromImage(q_image).scaled(
                label.width(),
                label.height(),
                Qt.IgnoreAspectRatio,
                Qt.FastTransformation
            )
            label.setPixmap(pixmap)
        except Exception as e:
            print(f"Error displaying frame for camera {index}: {e}")

def main():
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(10, 22, 40))
    app.setPalette(palette)

    window = ModernDashboard()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
