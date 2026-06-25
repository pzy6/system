# 银发守护者系统运行日志问题解决方案

## 结论

这次日志里没有看到程序崩溃，系统已经启动并连接摄像头。主要问题是：

1. ONNX Runtime 试图启用 CUDA，但 CUDA/cuDNN/MSVC 依赖不匹配，自动回退到 CPU。
2. 因为回退 CPU，检测耗时偏高，FPS 只有 7~10。
3. Windows 控制台/IDE 把 ONNX Runtime 的彩色警告和 UTF-8 输出显示成乱码。
4. `Unknown face detected` 是业务告警，表示当前画面里出现了未注册人脸，不是启动失败。

## 问题 1：`Failed to create CUDAExecutionProvider`

日志关键点：

```text
Failed to create CUDAExecutionProvider. Require cuDNN 9.* and CUDA 13.* ...
Applied providers: ['CPUExecutionProvider']
```

含义：你安装了 GPU 版 ONNX Runtime，但当前电脑没有满足它要求的 CUDA、cuDNN 或 MSVC 运行时，所以程序回退到 CPU 推理。

### 方案 A：不需要 GPU，只稳定运行并消除警告

适合：电脑没有 NVIDIA 独显，或能接受 CPU 推理。

```powershell
D:\anaconda\python.exe -m pip uninstall -y onnxruntime-gpu
D:\anaconda\python.exe -m pip install -U onnxruntime
```

代码里创建 ONNX Session 时只使用 CPU：

```python
import onnxruntime as ort

session = ort.InferenceSession(
    model_path,
    providers=["CPUExecutionProvider"],
)
```

建议封装为统一函数：

```python
import logging
import onnxruntime as ort

logger = logging.getLogger(__name__)


def create_onnx_session(model_path: str) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.log_severity_level = 3

    try:
        session = ort.InferenceSession(
            model_path,
            sess_options=options,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
    except Exception as exc:
        logger.warning("CUDA 初始化失败，已回退到 CPU: %s", exc)
        session = ort.InferenceSession(
            model_path,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

    logger.info("ONNX providers: %s", session.get_providers())
    return session
```

如果完全不想尝试 CUDA，把 providers 固定成 `["CPUExecutionProvider"]` 即可。

### 方案 B：需要 GPU 加速，提高 FPS

适合：电脑有 NVIDIA 显卡，希望提升实时识别速度。

先检查：

```powershell
nvidia-smi
D:\anaconda\python.exe -c "import onnxruntime as ort; print('ort=', ort.__version__); print(ort.get_available_providers())"
```

当前日志显示你的 ONNX Runtime GPU 包要求：

```text
CUDA 13.* + cuDNN 9.* + 最新 MSVC runtime
```

所以必须保证三者匹配：

1. NVIDIA 驱动正常，`nvidia-smi` 能看到显卡。
2. 安装日志要求的 CUDA Toolkit。
3. 安装匹配的 cuDNN，并把 cuDNN 的 `bin` 加入 PATH。
4. 安装最新版 Microsoft Visual C++ Redistributable。
5. 重启终端/Trae 后再运行。

如果你本机已经是 CUDA 12.x，通常更建议把 `onnxruntime-gpu` 换成匹配 CUDA 12/cuDNN 9 的版本，而不是强行安装 CUDA 13：

```powershell
D:\anaconda\python.exe -m pip uninstall -y onnxruntime onnxruntime-gpu
D:\anaconda\python.exe -m pip install onnxruntime-gpu==1.20.1
```

安装后验证：

```powershell
D:\anaconda\python.exe -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
```

期望看到：

```text
['CUDAExecutionProvider', 'CPUExecutionProvider']
```

## 问题 2：乱码 `N0R... [0;93m ...`

原因通常是 Windows 控制台/IDE 对 ONNX Runtime 的 ANSI 彩色日志和 UTF-8 编码处理不好。

启动前设置：

```powershell
chcp 65001
$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"
D:\anaconda\python.exe "D:\Trae文件\银发守护者系统\src\main.py"
```

如果仍有乱码，根因大概率还是 CUDA Provider 初始化失败时 ONNX Runtime C++ 层直接输出警告。优先按“问题 1”修复 provider，或在代码里设置：

```python
options = ort.SessionOptions()
options.log_severity_level = 3
```

## 问题 3：FPS 低

日志显示：

```text
FPS: 7~10 | detect=686~856ms pose=195~279ms
```

性能瓶颈主要在检测和姿态估计，且当前已经回退到 CPU。

优化优先级：

1. 优先修复 GPU Provider，让 ONNX 模型走 CUDA。
2. 降低摄像头输入分辨率，例如 1280x720 改 640x480。
3. 不要每帧都做人脸、烟雾、姿态、情绪全量检测，改为隔帧检测。
4. 检测结果用 tracker 复用，例如每 5 帧检测一次，中间帧只跟踪。
5. 人脸识别、情绪识别只在人脸区域变化明显或每 N 秒执行一次。

示例策略：

```python
if frame_index % 5 == 0:
    detections = detector.detect(frame)
else:
    detections = tracker.update(frame)

if frame_index % 10 == 0:
    face_result = face_recognizer.recognize(face_crop)
```

## 问题 4：`Unknown face detected: 主厅摄像头`

这不是程序错误，而是系统识别到未注册人脸。

处理方式：

1. 如果是家人/护理人员，进入注册流程，增加人脸样本。
2. 注册样本建议包含正脸、侧脸、不同光照，每人 5~10 张。
3. 检查人脸库路径是否正确。日志显示只加载了 1 个注册人脸：

```text
已加载 1 个注册人脸
```

4. 如果误报频繁，调高识别阈值或做人脸质量过滤。
5. 加告警冷却时间，避免同一个摄像头连续刷屏。

示例冷却逻辑：

```python
import time

unknown_face_last_warn = {}
UNKNOWN_FACE_WARN_INTERVAL = 10


def warn_unknown_face(camera_id: str, camera_name: str) -> None:
    now = time.time()
    last = unknown_face_last_warn.get(camera_id, 0)
    if now - last >= UNKNOWN_FACE_WARN_INTERVAL:
        logger.warning("Unknown face detected: %s", camera_name)
        unknown_face_last_warn[camera_id] = now
```

## 推荐落地顺序

1. 先决定是否需要 GPU：
   - 不需要：卸载 `onnxruntime-gpu`，改 CPU provider。
   - 需要：安装匹配的 CUDA/cuDNN/MSVC，确保 `CUDAExecutionProvider` 可用。
2. 修复乱码：设置 UTF-8 环境变量，并降低 ONNX Runtime 日志等级。
3. 提升 FPS：先解决 GPU，再做隔帧检测/降低分辨率。
4. 处理 Unknown face：补充人脸库，增加告警冷却。
5. 验证运行，确认不再出现 CUDA Provider 创建失败，FPS 有改善。

## 最小验证清单

```powershell
chcp 65001
$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"

D:\anaconda\python.exe -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
D:\anaconda\python.exe "D:\Trae文件\银发守护者系统\src\main.py"
```

通过标准：

- 不再出现 `Failed to create CUDAExecutionProvider`，或明确只使用 CPU。
- 中文日志正常显示。
- FPS 稳定，GPU 模式应明显高于当前 7~10 FPS。
- Unknown face 告警不再连续刷屏，或已通过注册人脸解决。
