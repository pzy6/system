"""
人脸录入独立程序 — 玻璃拟态 + 扁平化设计
与银发守护者系统 UI 风格统一 (深蓝渐变背景, 守护蓝 #4A90E2)
"""

import sys, os, time, cv2, json
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QSpinBox, QFrame, QMessageBox,
    QListWidget, QListWidgetItem, QSizePolicy
)
from PyQt5.QtGui import (
    QFont, QPixmap, QImage, QPainter, QColor, QBrush,
    QLinearGradient, QPalette, QIcon
)
from PyQt5.QtCore import Qt, QTimer, QRect, QSize


# ═══════════════════════════════════════════════════════════
# 全局样式
# ═══════════════════════════════════════════════════════════

GLASS_STYLE = """
QFrame#glassCard {
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 20px;
}
QLabel {
    color: #E8EDF5;
    font-family: "Microsoft YaHei";
}
QLineEdit {
    background: rgba(255, 255, 255, 0.10);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 12px;
    padding: 12px 16px;
    color: #FFFFFF;
    font-size: 15px;
    font-family: "Microsoft YaHei";
}
QLineEdit:focus {
    border-color: #4A90E2;
}
QSpinBox {
    background: rgba(255, 255, 255, 0.10);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 12px;
    padding: 10px 12px;
    color: #FFFFFF;
    font-size: 14px;
    font-family: "Microsoft YaHei";
}
QPushButton {
    border-radius: 12px;
    padding: 12px 24px;
    font-size: 14px;
    font-family: "Microsoft YaHei";
    font-weight: bold;
    border: none;
}
QPushButton#btnPrimary {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #2563EB, stop:1 #4A90E2);
    color: #FFFFFF;
}
QPushButton#btnPrimary:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #1D4ED8, stop:1 #3B82F6);
}
QPushButton#btnDanger {
    background: rgba(229, 62, 62, 0.8);
    color: #FFFFFF;
}
QPushButton#btnDanger:hover {
    background: rgba(229, 62, 62, 1.0);
}
QListWidget {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 14px;
    color: #E8EDF5;
    font-size: 13px;
}
QListWidget::item {
    padding: 8px 12px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}
"""


class FaceEnrollApp(QMainWindow):
    """人脸录入独立应用"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("人脸录入 — 银发守护者系统")
        self.resize(1050, 700)
        self.setMinimumSize(900, 600)

        # 人脸数据库
        self.face_db = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "face_db"
        )
        os.makedirs(self.face_db, exist_ok=True)

        # 摄像头
        self.cap = None
        self.camera_available = False
        self.recognizer = None  # insightface 惰性加载

        # 已捕获的人脸
        self.captured_faces = []  # [(name, frame), ...]

        # 初始化 UI
        self._init_ui()
        self._init_camera()

        # 定时器刷新预览
        self._timer = QTimer()
        self._timer.timeout.connect(self._update_preview)
        self._timer.start(50)  # 20 FPS

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------
    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        # 玻璃拟态背景 (深蓝渐变)
        palette = QPalette()
        gradient = QLinearGradient(0, 0, 1, 1)
        gradient.setCoordinateMode(QLinearGradient.ObjectMode)
        gradient.setColorAt(0.0, QColor("#0A1628"))
        gradient.setColorAt(1.0, QColor("#1A2A4A"))
        palette.setBrush(QPalette.Window, QBrush(gradient))
        central.setPalette(palette)
        central.setAutoFillBackground(True)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(20)

        # ── 左侧: 摄像头预览 ──
        left_card = QFrame(objectName="glassCard")
        left_layout = QVBoxLayout(left_card)
        left_layout.setContentsMargins(16, 16, 16, 16)
        left_layout.setSpacing(12)

        title_preview = QLabel("📷 摄像头预览")
        title_preview.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        title_preview.setStyleSheet("color: #FFFFFF;")
        left_layout.addWidget(title_preview)

        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(480, 360)
        self.preview_label.setStyleSheet(
            "background: rgba(0, 0, 0, 0.4); border-radius: 16px;"
        )
        self.preview_label.setText("摄像头未连接")
        self.preview_label.setFont(QFont("Microsoft YaHei", 14))
        left_layout.addWidget(self.preview_label, stretch=1)

        # 状态标签
        self.status_label = QLabel("")
        self.status_label.setFont(QFont("Microsoft YaHei", 11))
        self.status_label.setStyleSheet("color: #8FA3BF;")
        left_layout.addWidget(self.status_label)

        main_layout.addWidget(left_card, stretch=3)

        # ── 右侧: 控制面板 ──
        right_card = QFrame(objectName="glassCard")
        right_layout = QVBoxLayout(right_card)
        right_layout.setContentsMargins(20, 20, 20, 20)
        right_layout.setSpacing(14)

        # 标题
        title_ctrl = QLabel("录入信息")
        title_ctrl.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        title_ctrl.setStyleSheet("color: #FFFFFF;")
        right_layout.addWidget(title_ctrl)

        # 姓名
        lbl_name = QLabel("老人姓名")
        lbl_name.setFont(QFont("Microsoft YaHei", 12))
        lbl_name.setStyleSheet("color: #8FA3BF;")
        right_layout.addWidget(lbl_name)

        self.input_name = QLineEdit()
        self.input_name.setPlaceholderText("请输入姓名，如：张大爷")
        right_layout.addWidget(self.input_name)

        # 采集数量
        lbl_count = QLabel("采集数量")
        lbl_count.setFont(QFont("Microsoft YaHei", 12))
        lbl_count.setStyleSheet("color: #8FA3BF;")
        right_layout.addWidget(lbl_count)

        count_row = QHBoxLayout()
        self.spin_count = QSpinBox()
        self.spin_count.setRange(1, 30)
        self.spin_count.setValue(10)
        count_row.addWidget(self.spin_count)

        self.btn_capture = QPushButton("📸 开始采集")
        self.btn_capture.setObjectName("btnPrimary")
        self.btn_capture.clicked.connect(self._toggle_capture)
        count_row.addWidget(self.btn_capture)
        right_layout.addLayout(count_row)

        # 采集列表
        lbl_list = QLabel("已采集人脸")
        lbl_list.setFont(QFont("Microsoft YaHei", 12))
        lbl_list.setStyleSheet("color: #8FA3BF;")
        right_layout.addWidget(lbl_list)

        self.list_faces = QListWidget()
        right_layout.addWidget(self.list_faces, stretch=1)

        # 操作按钮
        btn_row = QHBoxLayout()
        self.btn_save = QPushButton("💾 保存到数据库")
        self.btn_save.setObjectName("btnPrimary")
        self.btn_save.clicked.connect(self._save_to_db)
        btn_row.addWidget(self.btn_save)

        self.btn_clear = QPushButton("🗑 清空")
        self.btn_clear.setObjectName("btnDanger")
        self.btn_clear.clicked.connect(self._clear_faces)
        btn_row.addWidget(self.btn_clear)
        right_layout.addLayout(btn_row)

        # 已注册列表
        lbl_enrolled = QLabel("已注册人员")
        lbl_enrolled.setFont(QFont("Microsoft YaHei", 12))
        lbl_enrolled.setStyleSheet("color: #8FA3BF;")
        right_layout.addWidget(lbl_enrolled)

        self.list_enrolled = QListWidget()
        self.list_enrolled.setMaximumHeight(100)
        right_layout.addWidget(self.list_enrolled)

        self._refresh_enrolled_list()

        main_layout.addWidget(right_card, stretch=2)

        # 应用全局样式
        self.setStyleSheet(GLASS_STYLE)

    # ------------------------------------------------------------------
    # 摄像头
    # ------------------------------------------------------------------
    def _init_camera(self):
        try:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.camera_available = True
                self.status_label.setText("✅ 摄像头已连接")
            else:
                self.cap.release()
                self.cap = None
                self.status_label.setText("⚠️ 摄像头不可用 — 可手动导入图片")
        except Exception:
            self.cap = None
            self.status_label.setText("⚠️ 摄像头不可用 — 可手动导入图片")

    def _update_preview(self):
        if not self.camera_available or self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 画人脸框
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)

        h, w, ch = frame_rgb.shape
        q_img = QImage(frame_rgb.data, w, h, w * 3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(q_img).scaled(
            self.preview_label.width(), self.preview_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.preview_label.setPixmap(pix)

        # 采集模式: 自动保存含人脸的帧
        name = self.input_name.text().strip()
        if self._capturing and faces and name:
            self._captured_count += 1
            self.captured_faces.append((f"{name}_{self._captured_count:03d}", frame.copy()))
            self.list_faces.addItem(
                f"📷 {name}_{self._captured_count:03d}  ({len(faces)}人脸)"
            )
            self.status_label.setText(
                f"📸 已采集: {self._captured_count}/{self.spin_count.value()}"
            )
            if self._captured_count >= self.spin_count.value():
                self._stop_capture()

    # ------------------------------------------------------------------
    # 采集控制
    # ------------------------------------------------------------------
    _capturing = False
    _captured_count = 0

    def _toggle_capture(self):
        if not self._capturing:
            name = self.input_name.text().strip()
            if not name:
                QMessageBox.warning(self, "提示", "请先输入老人姓名")
                return
            self._start_capture()
        else:
            self._stop_capture()

    def _start_capture(self):
        self._capturing = True
        self._captured_count = 0
        self.captured_faces.clear()
        self.list_faces.clear()
        self.btn_capture.setText("⏹ 停止采集")
        self.btn_capture.setObjectName("btnDanger")
        self.btn_capture.setStyleSheet(
            "background: rgba(229, 62, 62, 0.8); color: #FFF;"
            "border-radius: 12px; padding: 12px 24px; font-weight: bold;"
        )
        self.status_label.setText("📸 采集中... 请面对摄像头")

    def _stop_capture(self):
        self._capturing = False
        self.btn_capture.setText("📸 开始采集")
        self.btn_capture.setObjectName("btnPrimary")
        self.btn_capture.setStyleSheet("")
        n = len(self.captured_faces)
        self.status_label.setText(f"✅ 采集完成: {n} 张人脸")

    # ------------------------------------------------------------------
    # 保存/清空
    # ------------------------------------------------------------------
    def _save_to_db(self):
        name = self.input_name.text().strip()
        if not name:
            QMessageBox.warning(self, "提示", "请先输入姓名")
            return

        if not self.captured_faces:
            QMessageBox.warning(self, "提示", "没有采集到人脸图片")
            return

        # 保存图片到 face_db/{name}/
        person_dir = os.path.join(self.face_db, name)
        os.makedirs(person_dir, exist_ok=True)

        embeddings = []
        for fname, frame in self.captured_faces:
            img_path = os.path.join(person_dir, f"{fname}.jpg")
            cv2.imwrite(img_path, frame)

            # 提取 ArcFace 嵌入
            emb = self._extract_embedding(frame)
            if emb is not None:
                embeddings.append(emb)

        if not embeddings:
            QMessageBox.warning(self, "提示", "未能从图片中提取人脸特征")
            return

        # 更新 index.json
        index_path = os.path.join(self.face_db, "index.json")
        index = {}
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)

        index[name] = [e.tolist() for e in embeddings]

        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

        QMessageBox.information(
            self, "保存成功",
            f"已注册 {name}\n{len(embeddings)} 个特征向量\n{len(self.captured_faces)} 张图片保存"
        )
        self._refresh_enrolled_list()
        self._clear_faces()

    def _extract_embedding(self, frame_bgr):
        """提取 ArcFace 嵌入"""
        try:
            if self.recognizer is None:
                import insightface
                self.recognizer = insightface.app.FaceAnalysis(
                    name="buffalo_l", providers=["CPUExecutionProvider"]
                )
                self.recognizer.prepare(ctx_id=0, det_size=(640, 480))
            faces = self.recognizer.get(frame_bgr)
            if faces:
                best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                return best.embedding
        except Exception as e:
            print(f"Embedding error: {e}")
        return None

    def _clear_faces(self):
        self.captured_faces.clear()
        self.list_faces.clear()
        self._captured_count = 0

    def _refresh_enrolled_list(self):
        self.list_enrolled.clear()
        index_path = os.path.join(self.face_db, "index.json")
        if not os.path.exists(index_path):
            return
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        for name, embs in index.items():
            self.list_enrolled.addItem(f"👤 {name} ({len(embs)} 特征)")

    # ------------------------------------------------------------------
    # 清理
    # ------------------------------------------------------------------
    def closeEvent(self, event):
        self._timer.stop()
        if self.cap:
            self.cap.release()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = FaceEnrollApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
