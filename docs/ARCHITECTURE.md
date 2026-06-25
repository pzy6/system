# 银发守护者系统 — 架构设计文档

## 1. 分层架构

```
┌─────────────────────────────────────────────┐
│              应用层 (Application)           │
│  ModernDashboard  │  ConfigDialog           │
│  DashboardUpdater │  AlarmEvent             │
└──────────────────┬──────────────────────────┘
                   │ Qt Signals
┌──────────────────▼──────────────────────────┐
│              算法层 (Algorithms)            │
│  YOLODetector      YOLOPoseDetector         │
│  FaceRecognizer    FallDetector             │
│  BehaviorAnalyzer  VitalsMonitor            │
│  EmotionRecognizer SkeletonDetector (回退)   │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│            数据处理层 (Processing)           │
│  FrameProcessor   │  PriorityThreadPool     │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│             感知层 (Perception)              │
│  CameraManager  │  CameraWorker (多线程)    │
│  CameraDetector │   RTSP/USB 支持           │
└─────────────────────────────────────────────┘
```

## 2. 数据流

```
CameraWorker[i]                   ModernDashboard
    │ (raw BGR)                         ▲
    ▼                                   │ Qt Signal
raw_frame_queue (max=100)               │
    │                                   │
    ▼                                   │
FrameProcessor                      DashboardUpdater
  · resize→640×480                     │ (QThread 轮询)
  · CLAHE 增强                         │
  · 高斯去噪                           │
  · BGR→RGB                            │
    │                                   │
    ▼                                   │
processed_frame_queue                   │
    │                                   │
    ▼                                   │
process_frame() ───┬── YOLODetector ── face_bboxes ── FaceRecognizer
    │              │                       │
    │              ├── YOLODetector ── smoke_detected
    │              │
    │              ├── YOLOPoseDetector ── skeletons ── FallDetector
    │              │
    │              ├── BehaviorAnalyzer ── anomalies[]
    │              │
    │              ├── VitalsMonitor ── heart_rate, resp_rate
    │              │
    │              └── EmotionRecognizer ── emotion
    │
    ├── alarm_queue (max=50) ── AlarmRecorder (JSONL+截图+视频)
    ├── vitals_queue (max=50)
    ├── emotion_queue (max=50)
    ├── face_identity_queue (max=50)
    └── dashboard_frame_queue (max=50)
```

## 3. 模型架构

```
models/
├── yolo_face_smoke/
│   ├── face_yolov8n.pt        WIDER Face 预训练, 人脸检测
│   ├── yolo11n_smoke.pt       Home Fire 训练, fire+smoke (mAP 89.7%)
│   └── yolo11n.pt             COCO 80类, person→face 映射(回退)
│
├── yolo_pose/
│   ├── yolo11n-pose.pt        COCO 17关键点, 摔倒检测
│   └── yolo11n-pose.onnx      ONNX 导出 (备选)
│
└── ~/.insightface/models/buffalo_l/
    ├── det_10g.onnx            人脸检测 (SCRFD)
    ├── w600k_r50.onnx          ArcFace 512维嵌入
    └── ...                     关键点/性别年龄模型
```

**模型加载优先级** (yolo_detector.py):
1. `face_yolov8n.pt` → 人脸模型
2. `yolo11n_smoke.pt` → 烟雾模型
3. `yolo11n.pt` → COCO person→face 回退
4. Haar Cascade → 终极回退

## 4. 告警管线

```
AlarmEvent 生成
    │
    ├── AlarmRecorder.record()
    │   ├── data/alarms/alarm_history.jsonl  (JSON 行)
    │   └── data/alarms/screenshots/         (JPEG)
    │
    ├── AlarmVideoBuffer.export_alarm_clip()
    │   └── data/alarms/videos/              (MP4, 前后8秒)
    │
    └── alarm_queue → DashboardUpdater → ModernDashboard.add_alarm()
```

## 5. 并发模型

```
主线程 (Main Thread)
├── PyQt5 QApplication (事件循环, UI 渲染)
├── DashboardUpdater (QThread) — 队列→Qt Signal 桥接
│
├── CameraWorker[0..N] (threading.Thread, 每摄像头一个)
│   └── 推帧到 raw_frame_queue
│
├── FrameProcessor (threading.Thread)
│   └── raw_frame_queue → 预处理 → processed_frame_queue
│
└── processing_loop (threading.Thread)
    └── processed_frame_queue → process_frame() → 各算法队列
```

## 6. 人脸数据库

```
data/face_db/
├── index.json              {name: [512-dim embedding]}
├── 张三/                   注册图片目录
└── 李四/
```

- 注册: `scripts/enroll_face.py` → insightface 提取嵌入 → 写入 index.json
- 识别: 每帧检测到的 face_bbox → w600k_r50 提取嵌入 → 余弦匹配 → 阈值 0.6

## 7. 配置管理

`config/config.yaml`:
- `cameras[]` — 摄像头列表 (id/name/source/resolution/fps)
- `alarm_thresholds` — 各检测器阈值
- `model_paths` — 模型目录
- `processing` — 帧尺寸/线程数/跳帧参数
- `storage` — 日志/告警/截图/视频路径
- `system` — 声音告警/历史上限

## 8. 回退策略

| 组件 | 正常路径 | 回退路径 |
|------|----------|----------|
| 人脸检测 | face_yolov8n.pt | yolo11n person→face | Haar Cascade |
| 烟雾检测 | yolo11n_smoke.pt | 无 (暂缺) |
| 姿态检测 | yolo11n-pose.pt | HOG 人体检测+合成关键点 |
| 人脸识别 | insightface buffalo_l | 所有人标记 unknown |
| 情绪识别 | Caffe 模型 | 软编码回退 |
| 生命体征 | Haar Cascade 脸 | 跳过该帧 |

## 9. 部署拓扑

```
┌──────────────┐     ┌──────────────┐
│ 摄像头 1-4    │────▶│ Windows 10/11│
│ (USB/RTSP)   │     │ i7 RTX 4070  │
└──────────────┘     │ 16GB RAM     │
                     │              │
                     │ PyQt5 GUI    │────▶ 护理站显示器
                     │              │
                     │ data/        │────▶ 本地存储
                     └──────────────┘
```
