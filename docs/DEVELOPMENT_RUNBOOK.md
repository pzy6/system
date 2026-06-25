# 银发守护者系统开发与调试操作文档

## 1. 适用场景

本文档用于开发环境下完成以下工作：

- 放置模型文件并校验模型路径
- 配置 USB 摄像头或 RTSP 摄像头
- 排查摄像头无法调用问题
- 验证系统是否按设计文档进入可运行状态

## 2. 模型文件准备

### 2.1 目录约定

系统启动后会自动创建以下目录：

- `models/skeleton_model/`
- `models/emotion_model/`
- `models/behavior_model/`
- `models/fall_model/`
- `models/vitals_model/`

如果某个目录没有真实模型文件，系统会在目录中生成 `.placeholder` 占位文件。

### 2.2 需要手动放置的模型文件

骨骼检测模型：

- `models/skeleton_model/frozen_inference_graph.pb`
- `models/skeleton_model/graph_opt.pbtxt`

情绪识别模型：

- `models/emotion_model/deploy.prototxt.txt`
- `models/emotion_model/emotion_net.caffemodel`

说明：

- `fall_detection`、`behavior`、`vitals` 当前实现允许不依赖外部深度学习模型运行。
- 放置真实模型后，可删除同目录下对应的 `.placeholder` 文件。

## 3. 摄像头配置

### 3.1 USB 摄像头

编辑 `config/config.yaml` 中的 `cameras`：

```yaml
cameras:
  - id: camera_1
    name: 主厅摄像头
    source: 0
    resolution:
      width: 1280
      height: 720
    fps: 15
```

建议：

- 开发调试先从单路摄像头开始
- 先用 `1280x720 @ 15fps`
- 如果多个摄像头混用，索引一般是 `0/1/2...`

### 3.2 RTSP 摄像头

```yaml
source: rtsp://username:password@ip:port/stream
```

建议：

- 开发阶段先在 VLC 或其他播放器中确认 RTSP 地址可播放
- 如果存在认证或网络问题，优先排查摄像头厂商提供的原始流地址

## 4. 摄像头排查步骤

### 4.1 检测开发机可见摄像头

运行：

```powershell
python src\perception\camera_manager.py
```

输出说明：

- `[OK] 摄像头 N`：说明设备和当前 OpenCV 后端可访问
- `[FAIL] 未检测到可用摄像头`：说明设备不可见、被占用，或驱动/权限有问题

### 4.2 常见故障

1. 摄像头被其他程序占用
   关闭微信、钉钉、浏览器会议页面、相机程序。

2. 设备索引配置错误
   将 `source` 改为 `0`，确认第一路可用后再尝试 `1`、`2`。

3. 分辨率或帧率过高
   先使用 `1280x720` 和 `15fps`。

4. 驱动未加载或系统权限不足
   在 Windows 相机应用中先确认系统能正常打开设备。

## 5. 运行与验证

### 5.1 演示模式

```powershell
python src\main.py --demo
```

用于验证：

- UI 是否正常启动
- 配置界面是否可打开
- 模型状态展示是否正常

### 5.2 实机模式

```powershell
python src\main.py
```

用于验证：

- 摄像头是否接入主处理链
- 告警截图/视频目录是否自动生成
- `data/logs/system.log` 是否记录设备连接结果

### 5.3 包校验

```powershell
python verify_package.py
```

重点关注：

- `model_paths` 对应目录是否存在
- `screenshots`、`videos`、`alarm_history.jsonl` 是否符合预期
- 开发环境下 `silver_guardian.exe` 缺失属于正常现象，打包后再校验

## 6. 与设计文档对照检查

对照 `docs/VISUAL_DESIGN_GUIDE.md`，当前开发时应重点检查：

- 主色是否以守护蓝 `#4A90E2` 为主
- 告警色是否区分红色/黄色层级
- 告警列表是否倒序显示
- 视频区是否为 2x2 栅格
- 配置与模型状态是否可见且可操作

## 7. 当前实现边界

- 如果 `models/` 中没有真实模型文件，系统会降级到回退逻辑，不会自动下载模型
- 摄像头可用性依赖本机 OpenCV、Windows 驱动和设备权限
- 若需部署到新机器，必须手动复制模型文件和配置文件
