"""可折叠报警侧栏 — 玻璃拟态, 按摄像头+人脸分类"""
from PyQt5.QtWidgets import (
    QFrame, QVBoxLayout, QLabel, QPushButton, QScrollArea,
    QWidget, QHBoxLayout, QGridLayout, QSizePolicy, QGroupBox
)
from PyQt5.QtGui import QFont, QPixmap, QImage, QColor
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QPropertyAnimation, QEasingCurve
from collections import defaultdict
import cv2, numpy as np


class CollapsibleAlarmPanel(QFrame):
    """可折叠报警侧栏"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("alarmPanel")
        self.setFixedWidth(340)
        self.setStyleSheet("""
            QFrame#alarmPanel {
                background: rgba(10, 22, 40, 0.92);
                border-left: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 16px 0 0 16px;
            }
        """)

        self._alarms = []  # 报警列表
        self._grouped = defaultdict(list)  # 分组

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 20, 16, 20)
        layout.setSpacing(12)

        # 标题行
        header = QHBoxLayout()
        self.title_label = QLabel("📋 报警记录")
        self.title_label.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        self.title_label.setStyleSheet("color: #FFFFFF;")
        header.addWidget(self.title_label)
        header.addStretch()

        self.count_label = QLabel("0")
        self.count_label.setFont(QFont("Microsoft YaHei", 13, QFont.Bold))
        self.count_label.setStyleSheet(
            "color: #FFF; background: #E53E3E; border-radius: 10px; "
            "padding: 2px 10px;"
        )
        self.count_label.setAlignment(Qt.AlignCenter)
        header.addWidget(self.count_label)
        layout.addLayout(header)

        # 分类标签
        self.filter_row = QHBoxLayout()
        self.filter_row.setSpacing(6)
        self.btn_all = self._make_filter_btn("全部")
        self.btn_fall = self._make_filter_btn("跌倒")
        self.btn_fire = self._make_filter_btn("火/烟")
        self.filter_row.addWidget(self.btn_all)
        self.filter_row.addWidget(self.btn_fall)
        self.filter_row.addWidget(self.btn_fire)
        self.filter_row.addStretch()
        layout.addLayout(self.filter_row)

        # 滚动区域
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setStyleSheet(
            "QScrollArea { background: transparent; }"
            "QScrollBar:vertical { width: 4px; background: transparent; }"
            "QScrollBar::handle:vertical { background: rgba(255,255,255,0.2); "
            "border-radius: 2px; }"
        )
        self.scroll_content = QWidget()
        self.scroll_content.setStyleSheet("background: transparent;")
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_layout.setSpacing(8)
        self.scroll_layout.addStretch()
        self.scroll.setWidget(self.scroll_content)
        layout.addWidget(self.scroll, stretch=1)

    def _make_filter_btn(self, text):
        btn = QPushButton(text)
        btn.setFont(QFont("Microsoft YaHei", 10))
        btn.setFixedHeight(28)
        btn.setStyleSheet("""
            QPushButton {
                background: rgba(255,255,255,0.08);
                color: #8FA3BF;
                border: none;
                border-radius: 8px;
                padding: 4px 10px;
            }
            QPushButton:hover { background: rgba(255,255,255,0.15); }
        """)
        return btn

    def add_alarm(self, alarm_event):
        """添加报警记录"""
        self._alarms.append(alarm_event)
        # 按摄像头+人脸分组
        key = f"{alarm_event.camera_name}|{getattr(alarm_event, 'face_name', '')}"
        self._grouped[key].append(alarm_event)

        # 添加到滚动列表
        card = self._make_alarm_card(alarm_event)
        # 插入到 stretch 之前
        self.scroll_layout.insertWidget(
            max(0, self.scroll_layout.count() - 1), card
        )

        # 更新计数
        self.count_label.setText(str(len(self._alarms)))

        # 限制数量
        if len(self._alarms) > 100:
            old = self._alarms.pop(0)
            old_key = f"{old.camera_name}|{getattr(old, 'face_name', '')}"
            self._grouped[old_key].remove(old)

    def _make_alarm_card(self, alarm):
        """创建报警卡片"""
        card = QFrame()
        card.setStyleSheet(
            "QFrame { background: rgba(255,255,255,0.06); border-radius: 12px; "
            "border: 1px solid rgba(255,255,255,0.08); }"
            "QFrame:hover { background: rgba(255,255,255,0.10); }"
        )
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(12, 10, 12, 10)
        card_layout.setSpacing(4)

        import time as _time
        time_str = _time.strftime("%H:%M:%S", _time.localtime(alarm.timestamp))

        # 类型 + 时间
        row1 = QHBoxLayout()
        type_label = QLabel(f"🔴 {alarm.alarm_type}")
        type_label.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        type_label.setStyleSheet("color: #FF6B6B;")
        row1.addWidget(type_label)
        row1.addStretch()
        time_label = QLabel(time_str)
        time_label.setFont(QFont("Consolas", 9))
        time_label.setStyleSheet("color: #8FA3BF;")
        row1.addWidget(time_label)
        card_layout.addLayout(row1)

        # 摄像头 + 人脸
        detail = alarm.camera_name
        if hasattr(alarm, 'face_name') and alarm.face_name:
            detail += f" | 👤 {alarm.face_name}"
        detail_label = QLabel(detail)
        detail_label.setFont(QFont("Microsoft YaHei", 10))
        detail_label.setStyleSheet("color: #B0B8C8;")
        card_layout.addWidget(detail_label)

        # 截图缩略图
        if alarm.frame is not None:
            thumb = cv2.resize(alarm.frame, (120, 68))
            thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
            h, w, ch = thumb_rgb.shape
            q_img = QImage(thumb_rgb.data, w, h, w * 3, QImage.Format_RGB888)
            pix = QPixmap.fromImage(q_img)
            img_label = QLabel()
            img_label.setPixmap(pix)
            img_label.setFixedSize(120, 68)
            img_label.setStyleSheet("border-radius: 8px;")
            card_layout.addWidget(img_label)

        return card
