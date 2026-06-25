"""报警弹出 Toast — 右下角滑入, 3秒自动消失"""
from PyQt5.QtWidgets import QFrame, QVBoxLayout, QLabel, QHBoxLayout, QGraphicsOpacityEffect
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, pyqtProperty
import time


class ToastWidget(QFrame):
    """报警弹窗"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(320, 80)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)

        self.setStyleSheet("""
            QFrame#toast {
                background: rgba(30, 30, 50, 0.95);
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 16px;
            }
        """)
        self.setObjectName("toast")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(10)

        # 图标
        self.icon_label = QLabel("🚨")
        self.icon_label.setFont(QFont("Segoe UI Emoji", 22))
        layout.addWidget(self.icon_label)

        # 文字
        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)
        self.title_label = QLabel()
        self.title_label.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        self.title_label.setStyleSheet("color: #FF6B6B;")
        text_layout.addWidget(self.title_label)

        self.detail_label = QLabel()
        self.detail_label.setFont(QFont("Microsoft YaHei", 10))
        self.detail_label.setStyleSheet("color: #B0B8C8;")
        text_layout.addWidget(self.detail_label)

        layout.addLayout(text_layout)
        layout.addStretch()

        # 透明度动画
        self._opacity_effect = QGraphicsOpacityEffect(self)
        self._opacity_effect.setOpacity(1.0)
        self.setGraphicsEffect(self._opacity_effect)

        # 自动消失定时器
        self._hide_timer = QTimer(self)
        self._hide_timer.timeout.connect(self._fade_out)
        self._hide_timer.setSingleShot(True)

        self._fade_anim = None

        self.hide()

    def show_alarm(self, alarm_type: str, camera_name: str, face_name: str = ""):
        """显示报警"""
        icons = {
            "跌倒检测": "🚨", "烟雾检测": "💨", "火焰检测": "🔥",
            "陌生人检测": "🕵️", "异常行为": "⚠️",
        }
        self.icon_label.setText(icons.get(alarm_type, "🔔"))
        self.title_label.setText(alarm_type)
        detail = camera_name
        if face_name:
            detail += f" | {face_name}"
        self.detail_label.setText(detail)

        # 定位到右下角
        if self.parent():
            pw = self.parent().width()
            ph = self.parent().height()
            self.move(pw - self.width() - 20, ph - self.height() - 20)

        self._opacity_effect.setOpacity(1.0)
        self.show()
        self.raise_()
        self._hide_timer.start(3000)

    def _fade_out(self):
        """淡出动画"""
        self._fade_anim = QPropertyAnimation(self._opacity_effect, b"opacity")
        self._fade_anim.setDuration(500)
        self._fade_anim.setStartValue(1.0)
        self._fade_anim.setEndValue(0.0)
        self._fade_anim.setEasingCurve(QEasingCurve.OutCubic)
        self._fade_anim.finished.connect(self.hide)
        self._fade_anim.start()
