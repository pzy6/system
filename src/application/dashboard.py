import sys
import cv2
import numpy as np
import logging
import time
import os
from datetime import datetime
from queue import Queue
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QListWidget, QListWidgetItem, QPushButton, QSystemTrayIcon,
    QMenu, QAction, QDialog, QTextEdit, QSplitter, QFrame, QProgressBar,
    QTabWidget, QGroupBox, QScrollArea
)
from PyQt5.QtGui import QImage, QPixmap, QIcon, QColor, QPalette, QBrush, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QTimer, QSize

from .styles import Colors, Fonts, Sizes, StyleSheet
from .config_dialog import ConfigDialog

def get_resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base_path, relative_path)

def ensure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path

logger = logging.getLogger(__name__)

class AlarmEvent:
    def __init__(self, alarm_type, camera_id, camera_name, timestamp, confidence, frame=None, level='warning'):
        self.alarm_type = alarm_type
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.timestamp = timestamp
        self.confidence = confidence
        self.frame = frame
        self.status = 'unhandled'
        self.id = f"{alarm_type}_{int(timestamp*1000)}"
        self.level = level

class VideoWidget(QWidget):
    def __init__(self, camera_id, camera_name):
        super().__init__()
        self.camera_id = camera_id
        self.camera_name = camera_name
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet(f"background-color: #1a1a1a; border-radius: {Sizes.ROUNDING_MEDIUM}px;")
        
        self.info_label = QLabel(camera_name)
        self.info_label.setStyleSheet(f"""
            color: {Colors.NEUTRAL_WHITE}; 
            font-size: {Fonts.SIZE_CAPTION}px; 
            background-color: rgba(0,0,0,0.7);
            padding: {Sizes.SPACING_XS}px {Sizes.SPACING_SM}px;
            border-radius: 0 0 {Sizes.ROUNDING_MEDIUM}px {Sizes.ROUNDING_MEDIUM}px;
        """)
        self.info_label.setAlignment(Qt.AlignCenter)
        
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.info_label)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setLayout(layout)
        
        self.current_frame = None
        self.detected_skeletons = []
        self.fall_detected = False
    
    @pyqtSlot(np.ndarray, list, bool)
    def update_frame(self, frame, skeletons=None, fall_detected=False):
        self.current_frame = frame
        self.detected_skeletons = skeletons or []
        self.fall_detected = fall_detected
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).copy()
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        q_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image).scaled(
            self.image_label.width(), self.image_label.height(),
            Qt.KeepAspectRatio, Qt.FastTransformation
        )
        self.image_label.setPixmap(pixmap)
        
        status_text = self.camera_name
        if fall_detected:
            status_text += " - 跌倒告警!"
            self.info_label.setStyleSheet(f"""
                color: {Colors.NEUTRAL_WHITE}; 
                font-size: {Fonts.SIZE_CAPTION}px; 
                font-weight: {Fonts.WEIGHT_BOLD};
                background-color: {Colors.ALERT_RED};
                padding: {Sizes.SPACING_XS}px {Sizes.SPACING_SM}px;
                border-radius: 0 0 {Sizes.ROUNDING_MEDIUM}px {Sizes.ROUNDING_MEDIUM}px;
            """)
        else:
            self.info_label.setStyleSheet(f"""
                color: {Colors.NEUTRAL_WHITE}; 
                font-size: {Fonts.SIZE_CAPTION}px; 
                background-color: rgba(0,0,0,0.7);
                padding: {Sizes.SPACING_XS}px {Sizes.SPACING_SM}px;
                border-radius: 0 0 {Sizes.ROUNDING_MEDIUM}px {Sizes.ROUNDING_MEDIUM}px;
            """)
        self.info_label.setText(status_text)

class AlarmListWidget(QListWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet(StyleSheet.get_alarm_list_style())
        self.setSelectionMode(QListWidget.SingleSelection)
        self.setSpacing(Sizes.SPACING_XS)

class AlarmCardWidget(QWidget):
    def __init__(self, alarm):
        super().__init__()
        self.alarm = alarm
        
        layout = QHBoxLayout()
        layout.setContentsMargins(Sizes.SPACING_SM, Sizes.SPACING_SM, Sizes.SPACING_SM, Sizes.SPACING_SM)
        layout.setSpacing(Sizes.SPACING_MD)
        
        color_stripe = QFrame()
        color_stripe.setFixedWidth(8)
        color_stripe.setStyleSheet(f"""
            background-color: {Colors.get_color_by_alarm_level(alarm.level)};
            border-radius: {Sizes.ROUNDING_SMALL}px;
        """)
        layout.addWidget(color_stripe)
        
        icon_label = QLabel()
        icon_label.setFixedSize(48, 48)
        icon_label.setStyleSheet(f"""
            background-color: {Colors.get_color_by_alarm_level(alarm.level)};
            border-radius: 50%;
        """)
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setText(self._get_alarm_icon(alarm.alarm_type))
        layout.addWidget(icon_label)
        
        info_layout = QVBoxLayout()
        info_layout.setSpacing(Sizes.SPACING_XS)
        
        title_label = QLabel(alarm.alarm_type)
        title_label.setStyleSheet(f"""
            font-family: {Fonts.FAMILY_BODY};
            font-size: {Fonts.SIZE_BODY}px;
            font-weight: {Fonts.WEIGHT_SEMIBOLD};
            color: {Colors.NEUTRAL_DARK};
        """)
        
        time_str = datetime.fromtimestamp(alarm.timestamp).strftime("%H:%M:%S")
        time_label = QLabel(f"{time_str} - {alarm.camera_name}")
        time_label.setStyleSheet(f"""
            font-family: {Fonts.FAMILY_MONO};
            font-size: {Fonts.SIZE_CAPTION}px;
            color: {Colors.NEUTRAL_MEDIUM};
        """)
        
        info_layout.addWidget(title_label)
        info_layout.addWidget(time_label)
        layout.addLayout(info_layout)
        
        if alarm.frame is not None:
            thumbnail = self._create_thumbnail(alarm.frame)
            layout.addWidget(thumbnail)
        
        view_btn = QPushButton("立即查看")
        view_btn.setStyleSheet(StyleSheet.get_button_style('primary'))
        view_btn.clicked.connect(self._on_view_clicked)
        layout.addWidget(view_btn)
        
        self.setLayout(layout)
        
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {Colors.NEUTRAL_WHITE};
                border-radius: {Sizes.ROUNDING_MEDIUM}px;
                border: 1px solid {Colors.NEUTRAL_LIGHT};
            }}
        """)
    
    def _get_alarm_icon(self, alarm_type):
        icons = {
            '跌倒检测': '🙋',
            'fall_not_recovered': '😰',
            'lingering': '⏳',
            'wandering': '🔄',
            'violent_movement': '💨',
            'motionless': '😴',
            '心率异常': '❤️',
            '呼吸异常': '💨',
            '情绪异常': '😔'
        }
        return icons.get(alarm_type, '⚠')
    
    def _create_thumbnail(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (80, 80))
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        q_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        label = QLabel()
        label.setPixmap(pixmap)
        label.setStyleSheet(f"border-radius: {Sizes.ROUNDING_SMALL}px;")
        return label
    
    def _on_view_clicked(self):
        self.alarm.status = 'handled'
        self.setStyleSheet(f"""
            QWidget {{
                background-color: rgba({Colors.STATUS_GREEN[1:]}, 0.1);
                border-radius: {Sizes.ROUNDING_MEDIUM}px;
                border: 2px solid {Colors.STATUS_GREEN};
            }}
        """)

class VitalsWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(Sizes.SPACING_MD, Sizes.SPACING_MD, Sizes.SPACING_MD, Sizes.SPACING_MD)
        self.layout.setSpacing(Sizes.SPACING_MD)
        
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {Colors.NEUTRAL_WHITE};
                border-radius: {Sizes.ROUNDING_MEDIUM}px;
                border: 1px solid {Colors.NEUTRAL_LIGHT};
            }}
        """)
        
        title_label = QLabel("<b>生命体征监测</b>")
        title_label.setStyleSheet(f"""
            font-family: {Fonts.FAMILY_BODY};
            font-size: {Fonts.SIZE_H3}px;
            font-weight: {Fonts.WEIGHT_BOLD};
            color: {Colors.NEUTRAL_DARK};
        """)
        self.layout.addWidget(title_label)
        
        self.heart_rate_label = QLabel("心率: --")
        self.heart_rate_label.setStyleSheet(f"""
            font-family: {Fonts.FAMILY_BODY};
            font-size: {Fonts.SIZE_BODY}px;
            color: {Colors.NEUTRAL_DARK};
        """)
        
        self.heart_rate_bar = QProgressBar()
        self.heart_rate_bar.setRange(40, 150)
        self.heart_rate_bar.setValue(70)
        self.heart_rate_bar.setFormat("%v bpm")
        self.heart_rate_bar.setStyleSheet(StyleSheet.get_progress_bar_style())
        
        self.resp_rate_label = QLabel("呼吸率: --")
        self.resp_rate_label.setStyleSheet(f"""
            font-family: {Fonts.FAMILY_BODY};
            font-size: {Fonts.SIZE_BODY}px;
            color: {Colors.NEUTRAL_DARK};
        """)
        
        self.resp_rate_bar = QProgressBar()
        self.resp_rate_bar.setRange(5, 35)
        self.resp_rate_bar.setValue(15)
        self.resp_rate_bar.setFormat("%v rpm")
        self.resp_rate_bar.setStyleSheet(StyleSheet.get_progress_bar_style())
        
        self.layout.addWidget(self.heart_rate_label)
        self.layout.addWidget(self.heart_rate_bar)
        self.layout.addWidget(self.resp_rate_label)
        self.layout.addWidget(self.resp_rate_bar)
        
        emotion_title = QLabel("<b>情绪状态</b>")
        emotion_title.setStyleSheet(f"""
            font-family: {Fonts.FAMILY_BODY};
            font-size: {Fonts.SIZE_H3}px;
            font-weight: {Fonts.WEIGHT_BOLD};
            color: {Colors.NEUTRAL_DARK};
            margin-top: {Sizes.SPACING_MD}px;
        """)
        self.layout.addWidget(emotion_title)
        
        self.emotion_label = QLabel("情绪: 正常")
        self.emotion_label.setStyleSheet(f"""
            font-family: {Fonts.FAMILY_BODY};
            font-size: {Fonts.SIZE_BODY}px;
            color: {Colors.STATUS_GREEN};
        """)
        self.layout.addWidget(self.emotion_label)
        
        self.setLayout(self.layout)
    
    @pyqtSlot(dict)
    def update_vitals(self, vitals):
        if vitals.get('heart_rate'):
            self.heart_rate_label.setText(f"心率: {vitals['heart_rate']} bpm")
            self.heart_rate_bar.setValue(int(vitals['heart_rate']))
            
            if vitals.get('heart_rate_alarm'):
                self.heart_rate_bar.setStyleSheet(StyleSheet.get_progress_bar_style('danger'))
                self.heart_rate_label.setStyleSheet(f"""
                    font-family: {Fonts.FAMILY_BODY};
                    font-size: {Fonts.SIZE_BODY}px;
                    color: {Colors.ALERT_RED};
                    font-weight: {Fonts.WEIGHT_BOLD};
                """)
            else:
                self.heart_rate_bar.setStyleSheet(StyleSheet.get_progress_bar_style('success'))
                self.heart_rate_label.setStyleSheet(f"""
                    font-family: {Fonts.FAMILY_BODY};
                    font-size: {Fonts.SIZE_BODY}px;
                    color: {Colors.NEUTRAL_DARK};
                """)
        
        if vitals.get('respiratory_rate'):
            self.resp_rate_label.setText(f"呼吸率: {vitals['respiratory_rate']} rpm")
            self.resp_rate_bar.setValue(int(vitals['respiratory_rate']))
            
            if vitals.get('respiratory_rate_alarm'):
                self.resp_rate_bar.setStyleSheet(StyleSheet.get_progress_bar_style('danger'))
                self.resp_rate_label.setStyleSheet(f"""
                    font-family: {Fonts.FAMILY_BODY};
                    font-size: {Fonts.SIZE_BODY}px;
                    color: {Colors.ALERT_RED};
                    font-weight: {Fonts.WEIGHT_BOLD};
                """)
            else:
                self.resp_rate_bar.setStyleSheet(StyleSheet.get_progress_bar_style('success'))
                self.resp_rate_label.setStyleSheet(f"""
                    font-family: {Fonts.FAMILY_BODY};
                    font-size: {Fonts.SIZE_BODY}px;
                    color: {Colors.NEUTRAL_DARK};
                """)
    
    @pyqtSlot(dict)
    def update_emotion(self, emotion):
        if emotion.get('is_negative'):
            self.emotion_label.setText(f"情绪: {emotion['emotion']} (异常)")
            self.emotion_label.setStyleSheet(f"""
                font-family: {Fonts.FAMILY_BODY};
                font-size: {Fonts.SIZE_BODY}px;
                color: {Colors.ALERT_YELLOW};
                font-weight: {Fonts.WEIGHT_BOLD};
            """)
        else:
            self.emotion_label.setText(f"情绪: {emotion['emotion']}")
            self.emotion_label.setStyleSheet(f"""
                font-family: {Fonts.FAMILY_BODY};
                font-size: {Fonts.SIZE_BODY}px;
                color: {Colors.STATUS_GREEN};
            """)

class DashboardCard(QWidget):
    def __init__(self, title, value, unit, color, trend=0):
        super().__init__()
        self.setFixedSize(Sizes.CARD_DASHBOARD_WIDTH, Sizes.CARD_DASHBOARD_HEIGHT)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(Sizes.SPACING_MD, Sizes.SPACING_MD, Sizes.SPACING_MD, Sizes.SPACING_MD)
        layout.setSpacing(Sizes.SPACING_SM)
        
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {Colors.NEUTRAL_WHITE};
                border-radius: {Sizes.ROUNDING_MEDIUM}px;
                border: 1px solid {Colors.NEUTRAL_LIGHT};
            }}
        """)
        
        title_label = QLabel(title)
        title_label.setStyleSheet(f"""
            font-family: {Fonts.FAMILY_BODY};
            font-size: {Fonts.SIZE_CAPTION}px;
            color: {Colors.NEUTRAL_MEDIUM};
        """)
        layout.addWidget(title_label)
        
        value_layout = QHBoxLayout()
        value_layout.setAlignment(Qt.AlignBaseline)
        
        value_label = QLabel(str(value))
        value_label.setStyleSheet(f"""
            font-family: {Fonts.FAMILY_MONO};
            font-size: {Fonts.SIZE_DATA}px;
            font-weight: {Fonts.WEIGHT_BOLD};
            color: {color};
        """)
        value_layout.addWidget(value_label)
        
        unit_label = QLabel(unit)
        unit_label.setStyleSheet(f"""
            font-family: {Fonts.FAMILY_BODY};
            font-size: {Fonts.SIZE_CAPTION}px;
            color: {Colors.NEUTRAL_MEDIUM};
            margin-left: {Sizes.SPACING_XS}px;
        """)
        value_layout.addWidget(unit_label)
        
        layout.addLayout(value_layout)
        
        if trend != 0:
            trend_icon = "↑" if trend > 0 else "↓"
            trend_color = Colors.STATUS_GREEN if trend > 0 else Colors.ALERT_RED
            trend_label = QLabel(f"{trend_icon} {abs(trend)}%")
            trend_label.setStyleSheet(f"""
                font-family: {Fonts.FAMILY_MONO};
                font-size: {Fonts.SIZE_CAPTION}px;
                color: {trend_color};
            """)
            layout.addWidget(trend_label)
        
        self.setLayout(layout)
        
        self.value_label = value_label
    
    def update_value(self, value, color):
        self.value_label.setText(str(value))
        self.value_label.setStyleSheet(f"""
            font-family: {Fonts.FAMILY_MONO};
            font-size: {Fonts.SIZE_DATA}px;
            font-weight: {Fonts.WEIGHT_BOLD};
            color: {color};
        """)

class StatusBarWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QHBoxLayout()
        self.layout.setContentsMargins(Sizes.SPACING_MD, Sizes.SPACING_SM, Sizes.SPACING_MD, Sizes.SPACING_SM)
        self.layout.setSpacing(Sizes.SPACING_LG)
        
        self.setStyleSheet(StyleSheet.get_status_bar_style())
        
        self.cpu_label = QLabel("CPU: --%")
        self.fps_label = QLabel("FPS: --")
        self.camera_count_label = QLabel("摄像头: 0/0")
        self.alarm_count_label = QLabel("告警: 0")
        
        self.layout.addWidget(self.cpu_label)
        self.layout.addWidget(self.fps_label)
        self.layout.addWidget(self.camera_count_label)
        self.layout.addWidget(self.alarm_count_label)
        
        self.layout.addStretch()
        
        self.camera_btn = QPushButton("📷 摄像头")
        self.camera_btn.setStyleSheet(StyleSheet.get_button_style())
        
        self.config_btn = QPushButton("配置")
        self.config_btn.setStyleSheet(StyleSheet.get_button_style('primary'))
        
        self.log_btn = QPushButton("日志")
        self.log_btn.setStyleSheet(StyleSheet.get_button_style())
        
        self.layout.addWidget(self.camera_btn)
        self.layout.addWidget(self.config_btn)
        self.layout.addWidget(self.log_btn)
        
        self.setLayout(self.layout)

class LogDialog(QDialog):
    def __init__(self, log_path):
        super().__init__()
        self.setWindowTitle("系统日志")
        self.setGeometry(100, 100, 800, 600)
        
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setStyleSheet(f"""
            QTextEdit {{
                background-color: {Colors.NEUTRAL_WHITE};
                border: 1px solid {Colors.NEUTRAL_MEDIUM};
                border-radius: {Sizes.ROUNDING_MEDIUM}px;
                font-family: {Fonts.FAMILY_MONO};
                font-size: {Fonts.SIZE_CAPTION}px;
            }}
        """)
        
        layout = QVBoxLayout()
        layout.addWidget(self.text_edit)
        self.setLayout(layout)
        
        self.load_log(log_path)
    
    def load_log(self, log_path):
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                self.text_edit.setText(f.read())
        except Exception as e:
            self.text_edit.setText(f"无法加载日志文件: {str(e)}")

class Dashboard(QMainWindow):
    frame_received = pyqtSignal(str, np.ndarray, list, bool)
    alarm_received = pyqtSignal(AlarmEvent)
    vitals_received = pyqtSignal(dict)
    emotion_received = pyqtSignal(dict)
    status_updated = pyqtSignal(dict)
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.setWindowTitle("银发守护者 - 护理站管理系统")
        self.setGeometry(0, 0, 1920, 1080)
        
        self.alarm_queue = Queue()
        self.video_widgets = {}
        self.alarms = []
        
        self.init_ui()
        self.init_tray()
        
        self.frame_received.connect(self.handle_frame)
        self.alarm_received.connect(self.handle_alarm)
        self.vitals_received.connect(self.vitals_widget.update_vitals)
        self.emotion_received.connect(self.vitals_widget.update_emotion)
        self.status_updated.connect(self.update_status)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_alarm_queue)
        self.timer.start(100)
    
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(Sizes.SPACING_MD, Sizes.SPACING_MD, Sizes.SPACING_MD, Sizes.SPACING_MD)
        main_layout.setSpacing(Sizes.SPACING_MD)
        
        self.setStyleSheet(StyleSheet.get_main_window_style())
        
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(Sizes.SPACING_LG)
        
        logo_label = QLabel()
        logo_label.setFixedSize(48, 48)
        logo_label.setStyleSheet(f"""
            background-color: {Colors.PRIMARY_BLUE};
            border-radius: {Sizes.ROUNDING_MEDIUM}px;
        """)
        logo_label.setAlignment(Qt.AlignCenter)
        logo_label.setText("👁❤️🛡")
        header_layout.addWidget(logo_label)
        
        title_label = QLabel("银发守护者系统")
        title_label.setStyleSheet(f"""
            font-family: {Fonts.FAMILY_HEADING};
            font-size: {Fonts.SIZE_H1}px;
            font-weight: {Fonts.WEIGHT_BOLD};
            color: {Colors.NEUTRAL_DARK};
        """)
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        self.dashboard_cards = []
        self.dashboard_cards.append(DashboardCard("今日告警", 0, "条", Colors.ALERT_RED))
        self.dashboard_cards.append(DashboardCard("已处置", 0, "条", Colors.STATUS_GREEN))
        self.dashboard_cards.append(DashboardCard("平均响应", "0s", "", Colors.PRIMARY_BLUE))
        self.dashboard_cards.append(DashboardCard("在线摄像头", 0, "个", Colors.NEUTRAL_MEDIUM))
        
        for card in self.dashboard_cards:
            header_layout.addWidget(card)
        
        main_layout.addLayout(header_layout)
        
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setHandleWidth(1)
        
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)
        self.left_layout.setContentsMargins(0, 0, 0, 0)
        self.left_layout.setSpacing(Sizes.SPACING_MD)
        
        self.video_grid = QGridLayout()
        self.video_grid.setSpacing(Sizes.SPACING_MD)
        self.left_layout.addLayout(self.video_grid)
        
        self.splitter.addWidget(self.left_panel)
        
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        self.right_layout.setSpacing(Sizes.SPACING_MD)
        
        alarm_group = QGroupBox()
        alarm_group.setTitle("告警事件")
        alarm_group.setStyleSheet(f"""
            QGroupBox {{
                font-family: {Fonts.FAMILY_BODY};
                font-size: {Fonts.SIZE_H3}px;
                font-weight: {Fonts.WEIGHT_BOLD};
                color: {Colors.NEUTRAL_DARK};
                border: 1px solid {Colors.NEUTRAL_MEDIUM};
                border-radius: {Sizes.ROUNDING_MEDIUM}px;
                margin-top: {Sizes.SPACING_LG}px;
                padding-top: {Sizes.SPACING_MD}px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: {Sizes.SPACING_MD}px;
                padding: 0 {Sizes.SPACING_SM}px;
            }}
        """)
        
        alarm_layout = QVBoxLayout()
        alarm_layout.setContentsMargins(Sizes.SPACING_SM, 0, Sizes.SPACING_SM, Sizes.SPACING_SM)
        
        self.alarm_scroll = QScrollArea()
        self.alarm_scroll.setWidgetResizable(True)
        self.alarm_scroll.setStyleSheet(f"""
            QScrollArea {{
                background-color: {Colors.NEUTRAL_WHITE};
                border: none;
            }}
        """)
        
        self.alarm_container = QWidget()
        self.alarm_container_layout = QVBoxLayout()
        self.alarm_container_layout.setSpacing(Sizes.SPACING_MD)
        self.alarm_container.setLayout(self.alarm_container_layout)
        self.alarm_scroll.setWidget(self.alarm_container)
        
        alarm_layout.addWidget(self.alarm_scroll)
        alarm_group.setLayout(alarm_layout)
        
        self.right_layout.addWidget(alarm_group)
        
        self.vitals_widget = VitalsWidget()
        self.right_layout.addWidget(self.vitals_widget)
        
        self.splitter.addWidget(self.right_panel)
        self.splitter.setSizes([1200, 720])
        
        main_layout.addWidget(self.splitter)
        
        self.status_bar = StatusBarWidget()
        main_layout.addWidget(self.status_bar)
        
        self.status_bar.config_btn.clicked.connect(self.show_config)
        self.status_bar.log_btn.clicked.connect(self.show_log)
        self.status_bar.camera_btn.clicked.connect(self.show_camera_selector)
    
    def init_tray(self):
        self.tray_icon = QSystemTrayIcon(QIcon())
        self.tray_icon.setIcon(QIcon())
        
        tray_menu = QMenu()
        show_action = QAction("显示窗口", self)
        exit_action = QAction("退出", self)
        
        show_action.triggered.connect(self.show)
        exit_action.triggered.connect(self.close)
        
        tray_menu.addAction(show_action)
        tray_menu.addAction(exit_action)
        
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()
    
    def add_camera_view(self, camera_id, camera_name):
        video_widget = VideoWidget(camera_id, camera_name)
        self.video_widgets[camera_id] = video_widget
        
        row = len(self.video_widgets) // 2
        col = len(self.video_widgets) % 2
        self.video_grid.addWidget(video_widget, row, col)
    
    @pyqtSlot(str, np.ndarray, list, bool)
    def handle_frame(self, camera_id, frame, skeletons=None, fall_detected=False):
        if camera_id in self.video_widgets:
            self.video_widgets[camera_id].update_frame(frame, skeletons, fall_detected)
    
    def add_alarm(self, alarm_event):
        self.alarm_queue.put(alarm_event)
    
    def process_alarm_queue(self):
        while not self.alarm_queue.empty():
            alarm = self.alarm_queue.get()
            self.alarms.append(alarm)
            
            alarm_card = AlarmCardWidget(alarm)
            self.alarm_container_layout.insertWidget(0, alarm_card)
            
            while self.alarm_container_layout.count() > 10:
                item = self.alarm_container_layout.takeAt(10)
                if item.widget():
                    item.widget().deleteLater()
            
            self.tray_icon.showMessage(
                "告警",
                f"{alarm.alarm_type} - {alarm.camera_name}",
                QSystemTrayIcon.Warning,
                3000
            )
            
            self.dashboard_cards[0].update_value(len(self.alarms), Colors.ALERT_RED)
            self.status_bar.alarm_count_label.setText(f"告警: {len(self.alarms)}")
    
    @pyqtSlot(AlarmEvent)
    def handle_alarm(self, alarm):
        self.add_alarm(alarm)
    
    @pyqtSlot(dict)
    def update_status(self, status):
        if 'cpu' in status:
            self.status_bar.cpu_label.setText(f"CPU: {status['cpu']}%")
        if 'fps' in status:
            self.status_bar.fps_label.setText(f"FPS: {status['fps']}")
        if 'camera_count' in status:
            self.status_bar.camera_count_label.setText(f"摄像头: {status['camera_count']}")
            self.dashboard_cards[3].update_value(status['camera_count'], Colors.NEUTRAL_MEDIUM)

    def update_face_identity(self, identity_data):
        """接收人脸识别结果（由 face_identity_signal 触发）"""
        identities = identity_data.get('identities', [])
        camera_id = identity_data.get('camera_id', '')
        for ident in identities:
            name = ident.get('name', 'unknown')
            conf = ident.get('confidence', 0)
            is_unknown = ident.get('is_unknown', True)
            status_text = f"陌生人 ({conf:.2f})" if is_unknown else f"{name} ({conf:.2f})"
            logger.debug(f"Face identity [{camera_id}]: {status_text}")

    def show_config(self):
        dialog = ConfigDialog(
            config=self.config,
            config_path=self.config.get('config_path', 'config/config.yaml'),
            model_status=self.config.get('model_status', {}),
            parent=self
        )
        if dialog.exec_():
            self.config = dialog.config
    
    def show_log(self):
        log_path = os.path.join(self.config.get('storage', {}).get('logs', './data/logs/'), 'system.log')
        dialog = LogDialog(log_path)
        dialog.exec_()
    
    def show_camera_selector(self):
        from perception.camera_detector import CameraDetector
        from .camera_selector import CameraSelectorDialog
        
        detector = CameraDetector()
        
        dialog = CameraSelectorDialog(detector, self)
        dialog.camera_selected.connect(self.on_camera_selected)
        
        result = dialog.exec_()
        if result == QDialog.Accepted:
            logger.info("Camera selection dialog accepted")
    
    def on_camera_selected(self, camera_index):
        from perception.camera_manager import CameraManager
        
        camera_manager = CameraManager(self.config.get('config_path', 'config/config.yaml'))
        success = camera_manager.update_camera_config(camera_index)
        
        if success:
            QMessageBox.information(self, "配置更新", "摄像头配置已更新，请重启应用生效")
        else:
            QMessageBox.warning(self, "配置失败", "无法更新摄像头配置")

    def closeEvent(self, event):
        event.ignore()
        self.hide()
        self.tray_icon.showMessage("银发守护者", "系统已最小化到托盘", QSystemTrayIcon.Information, 2000)

class DashboardUpdater(QThread):
    """告警 + 人脸身份更新线程 (帧传输已由 ModernDashboard QTimer 直接轮询)"""
    alarm_signal = pyqtSignal(AlarmEvent)
    status_signal = pyqtSignal(dict)
    face_identity_signal = pyqtSignal(dict)

    def __init__(self, alarm_queue, face_identity_queue=None):
        super().__init__()
        self.alarm_queue = alarm_queue
        self.face_identity_queue = face_identity_queue
        self.running = True

    def run(self):
        while self.running:
            try:
                while not self.alarm_queue.empty():
                    alarm = self.alarm_queue.get_nowait()
                    self.alarm_signal.emit(alarm)

                if self.face_identity_queue is not None:
                    while not self.face_identity_queue.empty():
                        identity_data = self.face_identity_queue.get_nowait()
                        self.face_identity_signal.emit(identity_data)

                time.sleep(0.05)  # 告警不需要高频轮询
            except Exception as e:
                logger.error(f"Dashboard updater error: {str(e)}")

    def stop(self):
        self.running = False
        self.wait()
