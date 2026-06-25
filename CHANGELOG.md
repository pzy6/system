# Changelog

## v2.0.0 (2026-06-20)

### Added
- YOLOv11n-pose 姿态检测，替换 HOG 人体检测回退
- YOLOv8n 人脸检测 (WIDER Face 预训练) + YOLOv11n 烟雾检测 (Home Fire 训练, mAP 89.7%)
- insightface ArcFace 人脸识别 + `data/face_db/` 人脸数据库
- `scripts/enroll_face.py` 人脸注册工具
- `scripts/training/train_yolo_smoke.py` 烟雾检测训练脚本
- `scripts/download_models.py` 模型下载工具
- `scripts/test_face_model.py` / `test_smoke_model.py` / `test_pose_model.py` 模型测试脚本
- 双模型架构 (`yolo_detector.py`): face_model + smoke_model 独立推理
- 烟雾检测告警 + 陌生人检测告警 (共 7 种告警类型)
- GPU 训练支持 (CUDA 12.4)
- 帧跳机制: 人脸/姿态隔帧检测降低 CPU 负载
- 质心跟踪替换 KCF 跟踪器 (`anomaly_behavior.py`)
- `docs/REQUIREMENTS.md` 需求规格说明书
- `docs/ARCHITECTURE.md` 架构设计文档
- `VERSION` + `CHANGELOG.md`

### Changed
- `fall_detector.py` 增强: 新增踝髋倒置检测，权重调整为 0.30/0.30/0.20/0.20
- `vitals_monitor.py` / `emotion_recognition.py`: 接受外部 face_bboxes 参数
- `skeleton_detector.py`: 跳过 .placeholder 占位文件静默回退
- 模型加载优先级: .pt > .onnx (兼容 CUDA 12.4)
- `requirements.txt`: ultralytics 8.0→8.3, 新增 insightface/onnx 依赖

### Fixed
- `cv2.TrackerKCF_create()` → 质心跟踪 (OpenCV 4.13 兼容)
- BOM 字符清理 (`main.py`, `camera_manager.py`, `camera_worker.py`, `config.yaml`)
- `cv2.imshow()` 冲突 (多版本 OpenCV 共存的 DLL 冲突)
- Windows 控制台日志中文乱码 (stdout UTF-8 wrapper)

### Removed
- 根目录 pip 残留文件 (`=*`)
- 重复模型 (`yolo26n.pt`, `yolo11n.pt` root copy)
- 空目录 (18 个), `__pycache__/` (46 .pyc), `data/DFireDataset/.git/`
- 训练 epoch checkpoint (5 × 21MB)
- `models/yolo_face_smoke/smoke_fire.pt` (旧预训练模型)
- `data/face_smoke/` 空数据集目录

---

## v1.x (2025) — 原始版本

- PyQt5 仪表盘 GUI
- HOG 人体检测 + 规则摔倒检测
- KCF 跟踪 + 5 类异常行为
- rPPG 生命体征 + Caffe 情绪识别
- 多摄像头 + RTSP 支持
