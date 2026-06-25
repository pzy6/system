import copy
import os

import yaml
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


class ConfigDialog(QDialog):
    def __init__(self, config, config_path, model_status=None, parent=None):
        super().__init__(parent)
        self.original_config = copy.deepcopy(config or {})
        self.config = copy.deepcopy(config or {})
        self.config_path = config_path
        self.model_status = model_status or {}
        self.camera_fields = []

        self.setWindowTitle("系统配置")
        self.setMinimumSize(860, 680)
        self._init_ui()
        self._load_from_config()

    def _init_ui(self):
        self.setStyleSheet(
            """
            QDialog { background-color: #f4f7fb; }
            QGroupBox {
                border: 1px solid #d8e0ea;
                border-radius: 8px;
                margin-top: 12px;
                font-weight: 600;
                background: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px;
            }
            QLabel { color: #243447; }
            QLineEdit, QSpinBox, QDoubleSpinBox {
                min-height: 30px;
                border: 1px solid #c8d2de;
                border-radius: 6px;
                padding: 4px 8px;
                background: #ffffff;
            }
            QPushButton {
                min-height: 32px;
                border-radius: 6px;
                padding: 4px 12px;
            }
            """
        )

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(12)

        self.tabs = QTabWidget()
        root_layout.addWidget(self.tabs)

        self.general_tab = self._build_general_tab()
        self.camera_tab = self._build_camera_tab()
        self.model_tab = self._build_model_tab()

        self.tabs.addTab(self.general_tab, "系统")
        self.tabs.addTab(self.camera_tab, "摄像头")
        self.tabs.addTab(self.model_tab, "模型状态")

        button_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self._save_config)
        button_box.rejected.connect(self.reject)
        root_layout.addWidget(button_box)

    def _wrap_scroll(self, widget):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(widget)
        return scroll

    def _build_general_tab(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        processing_group = QGroupBox("处理参数")
        processing_form = QFormLayout(processing_group)
        self.input_width = QSpinBox()
        self.input_width.setRange(160, 3840)
        self.input_height = QSpinBox()
        self.input_height.setRange(120, 2160)
        self.num_threads = QSpinBox()
        self.num_threads.setRange(1, 32)
        self.queue_size = QSpinBox()
        self.queue_size.setRange(1, 2000)
        processing_form.addRow("输入宽度", self.input_width)
        processing_form.addRow("输入高度", self.input_height)
        processing_form.addRow("处理线程数", self.num_threads)
        processing_form.addRow("队列容量", self.queue_size)
        layout.addWidget(processing_group)

        alarm_group = QGroupBox("告警阈值")
        alarm_form = QFormLayout(alarm_group)
        self.fall_threshold = QDoubleSpinBox()
        self.fall_threshold.setRange(0.0, 1.0)
        self.fall_threshold.setSingleStep(0.05)
        self.behavior_threshold = QDoubleSpinBox()
        self.behavior_threshold.setRange(0.0, 1.0)
        self.behavior_threshold.setSingleStep(0.05)
        self.emotion_threshold = QDoubleSpinBox()
        self.emotion_threshold.setRange(0.0, 1.0)
        self.emotion_threshold.setSingleStep(0.05)
        self.hr_min = QSpinBox()
        self.hr_min.setRange(20, 200)
        self.hr_max = QSpinBox()
        self.hr_max.setRange(20, 200)
        self.rr_min = QSpinBox()
        self.rr_min.setRange(5, 60)
        self.rr_max = QSpinBox()
        self.rr_max.setRange(5, 60)
        alarm_form.addRow("跌倒阈值", self.fall_threshold)
        alarm_form.addRow("异常行为阈值", self.behavior_threshold)
        alarm_form.addRow("情绪阈值", self.emotion_threshold)
        alarm_form.addRow("最小心率", self.hr_min)
        alarm_form.addRow("最大心率", self.hr_max)
        alarm_form.addRow("最小呼吸频率", self.rr_min)
        alarm_form.addRow("最大呼吸频率", self.rr_max)
        layout.addWidget(alarm_group)

        system_group = QGroupBox("系统与存储")
        system_form = QFormLayout(system_group)
        self.auto_save_interval = QSpinBox()
        self.auto_save_interval.setRange(10, 86400)
        self.max_alarm_history = QSpinBox()
        self.max_alarm_history.setRange(10, 100000)
        self.enable_audio_alarm = QCheckBox("启用声音告警")
        self.logs_path = QLineEdit()
        self.alarms_path = QLineEdit()
        self.screenshots_path = QLineEdit()
        self.videos_path = QLineEdit()
        system_form.addRow("自动保存间隔(秒)", self.auto_save_interval)
        system_form.addRow("历史告警上限", self.max_alarm_history)
        system_form.addRow("声音告警", self.enable_audio_alarm)
        system_form.addRow("日志目录", self.logs_path)
        system_form.addRow("告警目录", self.alarms_path)
        system_form.addRow("截图目录", self.screenshots_path)
        system_form.addRow("视频目录", self.videos_path)
        layout.addWidget(system_group)
        layout.addStretch()

        return self._wrap_scroll(container)

    def _build_camera_tab(self):
        container = QWidget()
        self.camera_layout = QVBoxLayout(container)
        self.camera_layout.setContentsMargins(8, 8, 8, 8)
        self.camera_layout.setSpacing(12)
        self.camera_layout.addStretch()
        return self._wrap_scroll(container)

    def _build_model_tab(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        path_group = QGroupBox("模型路径")
        path_form = QFormLayout(path_group)
        self.model_path_inputs = {}
        for key in ["skeleton", "fall_detection", "behavior", "vitals", "emotion",
                     "yolo_face_smoke", "yolo_pose", "face_db"]:
            field = QLineEdit()
            self.model_path_inputs[key] = field
            path_form.addRow(key, field)
        layout.addWidget(path_group)

        status_group = QGroupBox("加载状态")
        status_layout = QVBoxLayout(status_group)
        self.model_status_grid = QGridLayout()
        self.model_status_grid.setHorizontalSpacing(16)
        self.model_status_grid.setVerticalSpacing(10)
        self.model_status_labels = {}
        self.model_status_badges = {}
        self.model_status_messages = {}
        for row, key in enumerate(["yolo_pose", "yolo_face_smoke", "face_recognition",
                                    "skeleton", "fall_detection", "behavior",
                                    "vitals", "emotion"]):
            name_label = QLabel(key)
            status_label = QLabel("未知")
            message_label = QLabel("-")
            message_label.setWordWrap(True)
            self.model_status_labels[key] = name_label
            self.model_status_badges[key] = status_label
            self.model_status_messages[key] = message_label
            self.model_status_grid.addWidget(name_label, row, 0)
            self.model_status_grid.addWidget(status_label, row, 1)
            self.model_status_grid.addWidget(message_label, row, 2)
        status_layout.addLayout(self.model_status_grid)

        refresh_btn = QPushButton("刷新状态")
        refresh_btn.clicked.connect(self._refresh_model_status_view)
        status_layout.addWidget(refresh_btn, alignment=Qt.AlignLeft)
        layout.addWidget(status_group)
        layout.addStretch()

        return self._wrap_scroll(container)

    def _load_from_config(self):
        processing = self.config.get("processing", {})
        input_size = processing.get("input_size", {})
        thresholds = self.config.get("alarm_thresholds", {})
        heart_rate = thresholds.get("heart_rate", {})
        respiratory_rate = thresholds.get("respiratory_rate", {})
        system_cfg = self.config.get("system", {})
        storage = self.config.get("storage", {})
        model_paths = self.config.get("model_paths", {})

        self.input_width.setValue(int(input_size.get("width", 640)))
        self.input_height.setValue(int(input_size.get("height", 480)))
        self.num_threads.setValue(int(processing.get("num_threads", 4)))
        self.queue_size.setValue(int(processing.get("queue_size", 100)))
        self.fall_threshold.setValue(float(thresholds.get("fall_detection", 0.75)))
        self.behavior_threshold.setValue(float(thresholds.get("abnormal_behavior", 0.7)))
        self.emotion_threshold.setValue(float(thresholds.get("emotion_confidence", 0.8)))
        self.hr_min.setValue(int(heart_rate.get("min", 60)))
        self.hr_max.setValue(int(heart_rate.get("max", 100)))
        self.rr_min.setValue(int(respiratory_rate.get("min", 12)))
        self.rr_max.setValue(int(respiratory_rate.get("max", 20)))
        self.auto_save_interval.setValue(int(system_cfg.get("auto_save_interval", 300)))
        self.max_alarm_history.setValue(int(system_cfg.get("max_alarm_history", 100)))
        self.enable_audio_alarm.setChecked(bool(system_cfg.get("enable_audio_alarm", True)))
        self.logs_path.setText(storage.get("logs", "./data/logs/"))
        self.alarms_path.setText(storage.get("alarms", "./data/alarms/"))
        self.screenshots_path.setText(storage.get("screenshots", "./data/alarms/screenshots/"))
        self.videos_path.setText(storage.get("videos", "./data/alarms/videos/"))

        for key, field in self.model_path_inputs.items():
            field.setText(model_paths.get(key, ""))

        self._rebuild_camera_fields()
        self._refresh_model_status_view()

    def _rebuild_camera_fields(self):
        while self.camera_layout.count():
            item = self.camera_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.camera_fields = []
        for index, camera in enumerate(self.config.get("cameras", [])):
            group = QGroupBox(f"摄像头 {index + 1}")
            form = QFormLayout(group)
            name = QLineEdit(camera.get("name", ""))
            source = QLineEdit(str(camera.get("source", "")))
            width = QSpinBox()
            width.setRange(160, 3840)
            width.setValue(int(camera.get("resolution", {}).get("width", 640)))
            height = QSpinBox()
            height.setRange(120, 2160)
            height.setValue(int(camera.get("resolution", {}).get("height", 480)))
            fps = QSpinBox()
            fps.setRange(1, 120)
            fps.setValue(int(camera.get("fps", 15)))
            form.addRow("名称", name)
            form.addRow("视频源", source)
            form.addRow("宽度", width)
            form.addRow("高度", height)
            form.addRow("帧率", fps)
            self.camera_layout.addWidget(group)
            self.camera_fields.append(
                {
                    "name": name,
                    "source": source,
                    "width": width,
                    "height": height,
                    "fps": fps,
                }
            )

        self.camera_layout.addStretch()

    def _refresh_model_status_view(self):
        for key in self.model_status_badges:
            status = self.model_status.get(key, {})
            loaded = bool(status.get("loaded"))
            fallback = bool(status.get("fallback"))
            message = status.get("message", "未提供状态")
            if loaded and not fallback:
                text = "已加载"
                color = "#2f855a"
            elif loaded and fallback:
                text = "降级"
                color = "#dd6b20"
            else:
                text = "未加载"
                color = "#c53030"
            self.model_status_badges[key].setText(text)
            self.model_status_badges[key].setStyleSheet(
                f"color: #ffffff; background: {color}; border-radius: 10px; padding: 2px 8px;"
            )
            self.model_status_messages[key].setText(message)

    def set_model_status(self, model_status):
        self.model_status = model_status or {}
        self._refresh_model_status_view()

    def _collect_config(self):
        config = copy.deepcopy(self.config)
        config.setdefault("processing", {})
        config["processing"].setdefault("input_size", {})
        config.setdefault("alarm_thresholds", {})
        config["alarm_thresholds"].setdefault("heart_rate", {})
        config["alarm_thresholds"].setdefault("respiratory_rate", {})
        config.setdefault("system", {})
        config.setdefault("storage", {})
        config.setdefault("model_paths", {})

        config["processing"]["input_size"]["width"] = self.input_width.value()
        config["processing"]["input_size"]["height"] = self.input_height.value()
        config["processing"]["num_threads"] = self.num_threads.value()
        config["processing"]["queue_size"] = self.queue_size.value()

        config["alarm_thresholds"]["fall_detection"] = round(self.fall_threshold.value(), 2)
        config["alarm_thresholds"]["abnormal_behavior"] = round(self.behavior_threshold.value(), 2)
        config["alarm_thresholds"]["emotion_confidence"] = round(self.emotion_threshold.value(), 2)
        config["alarm_thresholds"]["heart_rate"]["min"] = self.hr_min.value()
        config["alarm_thresholds"]["heart_rate"]["max"] = self.hr_max.value()
        config["alarm_thresholds"]["respiratory_rate"]["min"] = self.rr_min.value()
        config["alarm_thresholds"]["respiratory_rate"]["max"] = self.rr_max.value()

        config["system"]["auto_save_interval"] = self.auto_save_interval.value()
        config["system"]["max_alarm_history"] = self.max_alarm_history.value()
        config["system"]["enable_audio_alarm"] = self.enable_audio_alarm.isChecked()

        config["storage"]["logs"] = self.logs_path.text().strip()
        config["storage"]["alarms"] = self.alarms_path.text().strip()
        config["storage"]["screenshots"] = self.screenshots_path.text().strip()
        config["storage"]["videos"] = self.videos_path.text().strip()

        for key, field in self.model_path_inputs.items():
            config["model_paths"][key] = field.text().strip()

        for camera, fields in zip(config.get("cameras", []), self.camera_fields):
            camera["name"] = fields["name"].text().strip()
            source_text = fields["source"].text().strip()
            if source_text.isdigit():
                camera["source"] = int(source_text)
            else:
                camera["source"] = source_text
            camera.setdefault("resolution", {})
            camera["resolution"]["width"] = fields["width"].value()
            camera["resolution"]["height"] = fields["height"].value()
            camera["fps"] = fields["fps"].value()

        return config

    def _save_config(self):
        updated_config = self._collect_config()
        try:
            config_dir = os.path.dirname(self.config_path)
            if config_dir:
                os.makedirs(config_dir, exist_ok=True)
            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(updated_config, f, allow_unicode=True, sort_keys=False)
        except Exception as exc:
            QMessageBox.critical(self, "保存失败", f"无法写入配置文件:\n{exc}")
            return

        self.config = updated_config
        self.accept()
        QMessageBox.information(self, "保存成功", "配置已写入。重启应用后可完全生效。")
