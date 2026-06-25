import sys
import os
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QLabel, QGroupBox, QScrollArea, QWidget, QMessageBox,
    QProgressBar, QFrame
)
from PyQt5.QtGui import QIcon, QFont, QColor
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QTimer

from .styles import Colors, Fonts, Sizes, StyleSheet

class CameraListWidget(QListWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet(StyleSheet.get_camera_list_style())
        self.setSelectionMode(QListWidget.SingleSelection)
        self.setSpacing(Sizes.SPACING_SM)

class CameraDeviceWidget(QWidget):
    selected = pyqtSignal(int)
    
    def __init__(self, device_info):
        super().__init__()
        self.device_info = device_info
        
        layout = QHBoxLayout()
        layout.setContentsMargins(Sizes.SPACING_MD, Sizes.SPACING_SM, Sizes.SPACING_MD, Sizes.SPACING_SM)
        layout.setSpacing(Sizes.SPACING_MD)
        
        status_icon = QLabel()
        status_icon.setFixedSize(24, 24)
        if device_info.get('connected', True):
            status_icon.setStyleSheet(f"""
                background-color: {Colors.STATUS_GREEN};
                border-radius: 12px;
            """)
        else:
            status_icon.setStyleSheet(f"""
                background-color: {Colors.STATUS_RED};
                border-radius: 12px;
            """)
        layout.addWidget(status_icon)
        
        info_layout = QVBoxLayout()
        info_layout.setSpacing(Sizes.SPACING_XS)
        
        name_label = QLabel(device_info.get('name', 'Unknown Device'))
        name_label.setStyleSheet(f"""
            font-family: {Fonts.FAMILY_BODY};
            font-size: {Fonts.SIZE_BODY}px;
            font-weight: {Fonts.WEIGHT_SEMIBOLD};
            color: {Colors.NEUTRAL_DARK};
        """)
        info_layout.addWidget(name_label)
        
        properties = []
        if device_info.get('resolution'):
            properties.append(f"分辨率: {device_info['resolution']}")
        if device_info.get('fps'):
            properties.append(f"帧率: {device_info['fps']} FPS")
        properties.append(f"设备ID: {device_info['index']}")
        
        props_label = QLabel(" | ".join(properties))
        props_label.setStyleSheet(f"""
            font-family: {Fonts.FAMILY_MONO};
            font-size: {Fonts.SIZE_CAPTION}px;
            color: {Colors.NEUTRAL_MEDIUM};
        """)
        info_layout.addWidget(props_label)
        
        layout.addLayout(info_layout)
        layout.addStretch()
        
        select_btn = QPushButton("选择")
        select_btn.setStyleSheet(StyleSheet.get_button_style('primary'))
        select_btn.clicked.connect(lambda: self.selected.emit(device_info['index']))
        layout.addWidget(select_btn)
        
        self.setLayout(layout)
        
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {Colors.NEUTRAL_WHITE};
                border-radius: {Sizes.ROUNDING_MEDIUM}px;
                border: 2px solid transparent;
            }}
            QWidget:hover {{
                border-color: {Colors.PRIMARY_BLUE};
            }}
        """)

class CameraSelectorDialog(QDialog):
    camera_selected = pyqtSignal(int)
    
    def __init__(self, detector, parent=None):
        super().__init__(parent)
        self.detector = detector
        self.selected_device = None
        
        self.setWindowTitle("选择摄像头设备")
        self.setGeometry(300, 200, 600, 500)
        self.setStyleSheet(f"background-color: {Colors.NEUTRAL_LIGHT};")
        
        self.init_ui()
        self.setup_connections()
        
        self.detector.start_monitoring()
    
    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(Sizes.SPACING_LG, Sizes.SPACING_LG, Sizes.SPACING_LG, Sizes.SPACING_LG)
        main_layout.setSpacing(Sizes.SPACING_MD)
        
        header_layout = QHBoxLayout()
        header_layout.setSpacing(Sizes.SPACING_MD)
        
        icon_label = QLabel()
        icon_label.setFixedSize(48, 48)
        icon_label.setStyleSheet(f"""
            background-color: {Colors.PRIMARY_BLUE};
            border-radius: {Sizes.ROUNDING_MEDIUM}px;
        """)
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setText("📷")
        header_layout.addWidget(icon_label)
        
        title_label = QLabel("摄像头设备选择")
        title_label.setStyleSheet(f"""
            font-family: {Fonts.FAMILY_HEADING};
            font-size: {Fonts.SIZE_H2}px;
            font-weight: {Fonts.WEIGHT_BOLD};
            color: {Colors.NEUTRAL_DARK};
        """)
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        refresh_btn = QPushButton("🔄 刷新")
        refresh_btn.setStyleSheet(StyleSheet.get_button_style('secondary'))
        refresh_btn.clicked.connect(self.refresh_devices)
        header_layout.addWidget(refresh_btn)
        
        main_layout.addLayout(header_layout)
        
        self.scan_progress = QProgressBar()
        self.scan_progress.setRange(0, 0)
        self.scan_progress.setVisible(False)
        main_layout.addWidget(self.scan_progress)
        
        device_group = QGroupBox()
        device_group.setTitle("可用设备")
        device_group.setStyleSheet(f"""
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
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet(f"""
            QScrollArea {{
                background-color: {Colors.NEUTRAL_WHITE};
                border: none;
                border-radius: {Sizes.ROUNDING_MEDIUM}px;
            }}
        """)
        
        self.device_container = QWidget()
        self.device_container_layout = QVBoxLayout()
        self.device_container_layout.setSpacing(Sizes.SPACING_MD)
        self.device_container.setLayout(self.device_container_layout)
        self.scroll_area.setWidget(self.device_container)
        
        device_layout = QVBoxLayout()
        device_layout.addWidget(self.scroll_area)
        device_group.setLayout(device_layout)
        
        main_layout.addWidget(device_group)
        
        self.empty_state = None
        self.show_empty_state("正在扫描摄像头设备...")
        
        button_layout = QHBoxLayout()
        button_layout.setSpacing(Sizes.SPACING_MD)
        button_layout.addStretch()
        
        cancel_btn = QPushButton("取消")
        cancel_btn.setStyleSheet(StyleSheet.get_button_style('danger'))
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        self.confirm_btn = QPushButton("确认选择")
        self.confirm_btn.setStyleSheet(StyleSheet.get_button_style('primary'))
        self.confirm_btn.clicked.connect(self.confirm_selection)
        self.confirm_btn.setEnabled(False)
        button_layout.addWidget(self.confirm_btn)
        
        main_layout.addLayout(button_layout)
        
        self.setLayout(main_layout)
    
    def show_empty_state(self, text):
        """显示空状态提示"""
        if self.empty_state is None:
            self.empty_state = QLabel()
            self.empty_state.setAlignment(Qt.AlignCenter)
            self.empty_state.setFixedHeight(100)
        self.empty_state.setText(text)
        self.empty_state.setStyleSheet(f"""
            font-family: {Fonts.FAMILY_BODY};
            font-size: {Fonts.SIZE_BODY}px;
            color: {Colors.NEUTRAL_MEDIUM};
        """)
        self.device_container_layout.addWidget(self.empty_state)
    
    def setup_connections(self):
        self.detector.devices_changed.connect(self.update_device_list)
        self.detector.device_connected.connect(self.on_device_connected)
        self.detector.device_disconnected.connect(self.on_device_disconnected)
        self.detector.error_occurred.connect(self.show_error)
    
    def refresh_devices(self):
        """刷新设备列表"""
        self.scan_progress.setVisible(True)
        self.empty_state.setText("正在扫描...")
        
        def do_scan():
            devices = self.detector.scan_cameras()
            self.detector._update_devices(devices)
        
        import threading
        thread = threading.Thread(target=do_scan, daemon=True)
        thread.start()
    
    @pyqtSlot(list)
    def update_device_list(self, devices):
        """更新设备列表显示"""
        self.scan_progress.setVisible(False)
        
        for i in reversed(range(self.device_container_layout.count())):
            item = self.device_container_layout.takeAt(i)
            if item.widget():
                item.widget().deleteLater()
        
        if not devices:
            if self.empty_state is None:
                self.empty_state = QLabel()
                self.empty_state.setAlignment(Qt.AlignCenter)
                self.empty_state.setFixedHeight(100)
            self.empty_state.setText("未检测到摄像头设备\n\n请确保摄像头已正确连接\n并授予应用访问权限")
            self.empty_state.setStyleSheet(f"""
                font-family: {Fonts.FAMILY_BODY};
                font-size: {Fonts.SIZE_BODY}px;
                color: {Colors.ALERT_YELLOW};
            """)
            self.device_container_layout.addWidget(self.empty_state)
            self.confirm_btn.setEnabled(False)
            return
        
        if self.empty_state is not None:
            self.empty_state.deleteLater()
            self.empty_state = None
        
        for device in devices:
            device_widget = CameraDeviceWidget(device)
            device_widget.selected.connect(self.select_device)
            self.device_container_layout.addWidget(device_widget)
        
        self.confirm_btn.setEnabled(True)
    
    def select_device(self, device_index):
        """选择摄像头设备"""
        self.selected_device = device_index
        QMessageBox.information(self, "设备选择", f"已选择设备: {device_index}")
    
    def confirm_selection(self):
        """确认选择"""
        if self.selected_device is not None:
            self.camera_selected.emit(self.selected_device)
            self.accept()
        else:
            QMessageBox.warning(self, "提示", "请先选择一个摄像头设备")
    
    @pyqtSlot(object)
    def on_device_connected(self, device):
        """设备连接事件"""
        QMessageBox.information(self, "设备已连接", f"检测到新设备: {device.name}")
    
    @pyqtSlot(int)
    def on_device_disconnected(self, device_index):
        """设备断开事件"""
        QMessageBox.warning(self, "设备已断开", f"设备 {device_index} 已断开连接")
    
    @pyqtSlot(str)
    def show_error(self, error_message):
        """显示错误信息"""
        QMessageBox.critical(self, "错误", error_message)
    
    def closeEvent(self, event):
        """关闭对话框时停止监测"""
        self.detector.stop_monitoring()
        event.accept()