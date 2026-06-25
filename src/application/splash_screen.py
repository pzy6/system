"""开屏加载画面 — 系统启动时显示，含加载动画和进度条"""
import os
from PyQt5.QtWidgets import QSplashScreen, QProgressBar, QLabel
from PyQt5.QtGui import QPixmap, QPainter, QFont, QColor, QLinearGradient, QBrush
from PyQt5.QtCore import Qt, QTimer, QRectF, pyqtSignal

class SplashScreen(QSplashScreen):
    progress_updated = pyqtSignal(int, str)

    def __init__(self, bg_image_path: str, total_steps: int = 10):
        # 加载背景图
        self.bg_pixmap = QPixmap(bg_image_path)
        if self.bg_pixmap.isNull():
            # 回退：纯色背景
            self.bg_pixmap = QPixmap(800, 500)
            self.bg_pixmap.fill(QColor("#1a2a4a"))

        # 缩放到合适尺寸（最大 70% 屏幕）
        from PyQt5.QtWidgets import QApplication
        screen = QApplication.primaryScreen().availableGeometry()
        max_w = int(screen.width() * 0.7)
        max_h = int(screen.height() * 0.7)
        scaled = self.bg_pixmap.scaled(
            max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        super().__init__(scaled)

        self.total = total_steps
        self.current = 0
        self.message = "正在启动..."

        # 加载动画角度
        self._angle = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._rotate)
        self._timer.start(30)  # ~33 fps

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, total_steps)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: rgba(255,255,255,0.15);
                border: none;
                border-radius: 3px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #2563EB, stop:0.5 #4A90E2, stop:1 #60A5FA);
                border-radius: 3px;
            }
        """)

    def set_progress(self, step: int, message: str):
        """更新进度和状态文字"""
        self.current = min(step, self.total)
        self.progress_bar.setValue(self.current)
        self.message = message
        self.repaint()

    def _rotate(self):
        self._angle = (self._angle + 6) % 360
        self.repaint()

    def drawContents(self, painter: QPainter):
        """绘制加载动画、进度条和状态文字"""
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        # 半透明遮罩
        painter.fillRect(0, int(h * 0.85), w, int(h * 0.15),
                         QColor(0, 0, 0, 80))

        # 加载圆环（右上角）
        cx = w - 30
        cy = int(h * 0.1) + 15
        r = 14
        painter.setPen(Qt.NoPen)
        for i in range(12):
            alpha = max(30, 255 - i * 18)
            painter.setBrush(QColor(74, 144, 226, alpha))
            angle = (self._angle + i * 30) * 3.14159 / 180
            dx = int(cx + r * 0.6 * __import__('math').cos(angle))
            dy = int(cy - r * 0.6 * __import__('math').sin(angle))
            painter.drawEllipse(dx - 2, dy - 2, 4, 4)

        # 进度条
        bar_w = int(w * 0.7)
        bar_x = (w - bar_w) // 2
        bar_y = int(h * 0.88)
        self.progress_bar.setGeometry(bar_x, bar_y, bar_w, 6)
        # render progress bar manually
        frac = self.current / max(self.total, 1)
        # bg
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 255, 255, 38))
        painter.drawRoundedRect(QRectF(bar_x, bar_y, bar_w, 6), 3, 3)
        # fill
        grad = QLinearGradient(bar_x, 0, bar_x + bar_w, 0)
        grad.setColorAt(0.0, QColor("#2563EB"))
        grad.setColorAt(0.5, QColor("#4A90E2"))
        grad.setColorAt(1.0, QColor("#60A5FA"))
        painter.setBrush(QBrush(grad))
        painter.drawRoundedRect(QRectF(bar_x, bar_y, bar_w * frac, 6), 3, 3)

        # 状态文字
        font = QFont("Microsoft YaHei", 10)
        painter.setFont(font)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(0, bar_y + 22, w, 22, Qt.AlignCenter, self.message)

        # 标题
        title_font = QFont("Microsoft YaHei", 16, QFont.Bold)
        painter.setFont(title_font)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(0, int(h * 0.08), w, 30, Qt.AlignCenter,
                         "银发守护者 AI 视觉监护系统")
