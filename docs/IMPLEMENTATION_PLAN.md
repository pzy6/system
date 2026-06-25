# 银发守护者系统功能完善计划

## 已核对范围

- 项目说明：`PROJECT_SUMMARY.md`、`README_PACKAGE.md`
- 运行配置：`config/config.yaml`
- 主链路：`src/main.py`
- 摄像头、帧处理、算法、桌面界面相关模块

## 发现的未完善功能

### P0：实机处理链路字段不一致

`FrameProcessor` 输出字段为 `original_frame` 和 `frame`，但主程序读取 `processed_frame`，导致实机模式拿到第一帧后进入异常分支，算法链路无法持续工作。

处理状态：已修复。主程序现在兼容 `processed_frame`、`frame`、`original_frame` 三类字段。

### P0：新版界面和主程序重复占用摄像头

主程序已经通过 `CameraWorker` 读取摄像头，新版 `ModernDashboard` 初始化后还会自动扫描并打开摄像头，可能造成设备占用冲突，也会让 UI 显示绕开算法处理链路。

处理状态：已修复。`ModernDashboard` 增加 `auto_camera_scan` 参数，由主程序驱动时禁用界面内部扫描；单独运行 `modern_dashboard.py` 时仍保留自动扫描能力。

### P1：告警历史和截图未按配置落盘

文档和配置承诺了 `data/alarms/`、`data/alarms/screenshots/`、告警历史记录，但主程序原先只把告警推送到 UI 队列，应用退出后无法追溯。

处理状态：已修复。新增 `AlarmRecorder`，每条告警写入 `alarm_history.jsonl`，并在存在告警帧时保存截图。

### P1：演示模式没有启动 UI 队列桥接

演示模式会调用算法处理并推送队列，但原先没有启动 `DashboardUpdater`，视频、生命体征、情绪、告警无法稳定进入界面。

处理状态：已修复。演示模式和实机模式共用 `start_dashboard_updater()`。

## 后续建议

### P1：配置界面仍是占位

旧版 `Dashboard.show_config()` 仍显示“配置界面开发中”。建议下一步实现摄像头、阈值、存储路径的可视化编辑，并写回 `config/config.yaml`。

### P2：算法模型多数为降级实现

骨骼检测缺少模型时使用 HOG 降级，情绪、生命体征也以轻量逻辑为主。建议补充模型存在性检查、模型加载状态展示和推理失败降级提示。

### P2：告警视频目录尚未使用

配置包含 `storage.videos`，但当前只保存截图和 JSONL 历史。建议增加告警前后若干秒视频片段缓存。

### P2：包验证脚本编码和校验深度不足

`verify_package.py` 的中文显示存在编码异常，且没有校验新增的告警截图、历史文件、主链路字段兼容。建议后续扩展为完整健康检查。
