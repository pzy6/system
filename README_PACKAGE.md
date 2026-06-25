# 银发守护者系统 - 打包部署指南

## 系统要求

- Windows 10 或更高版本
- 64位操作系统
- 至少 4GB 可用内存
- 至少 2GB 可用磁盘空间

## 一、开发环境打包

### 1. 环境准备

确保已安装以下软件：

1. **Python 3.9** 或更高版本
2. **Git**（可选，用于版本控制）

### 2. 安装依赖

在项目根目录下运行：

```bash
# 创建虚拟环境（推荐）
python -m venv venv
venv\Scripts\activate

# 安装项目依赖
pip install -r requirements.txt

# 安装打包工具
pip install pyinstaller pywin32
```

### 3. 打包应用程序

运行打包脚本：

```bash
scripts\build_app.bat
```

或手动执行：

```bash
# 创建必要的目录
mkdir data
mkdir data\logs
mkdir data\alarms
mkdir models

# 使用PyInstaller打包
pyinstaller --clean silver_guardian.spec
```

### 4. 打包结果

打包完成后，在 `dist` 目录下将生成：

```
dist/
├── silver_guardian.exe    # 主程序
├── config/                # 配置文件目录
│   └── config.yaml
├── src/                   # 源代码（自动包含）
├── data/                  # 数据目录
│   ├── logs/
│   └── alarms/
└── models/                # 模型目录
```

## 二、独立运行版本

### 1. 目录结构

将以下内容打包到一个文件夹中：

```
银发守护者/
├── silver_guardian.exe
├── config/
│   └── config.yaml
├── data/
│   ├── logs/
│   ├── alarms/
│   └── screenshots/
├── models/
├── 启动应用.bat
├── 配置说明.txt
└── 使用说明.txt
```

### 2. 配置说明

`config/config.yaml` 文件包含所有系统配置：

```yaml
cameras:
  - id: camera_1
    name: 主厅摄像头
    type: highres_rgb
    source: 0
    resolution:
      width: 1920
      height: 1080
    fps: 30

alarm_thresholds:
  fall_detection: 0.75
  heart_rate:
    min: 60
    max: 100

storage:
  logs: ./data/logs/
  alarms: ./data/alarms/
```

## 三、使用说明

### 1. 首次运行

1. 双击 `启动应用.bat` 或 `silver_guardian.exe`
2. 系统将自动检测摄像头并启动界面
3. 默认会打开演示模式（--demo）

### 2. 运行参数

```bash
# 演示模式（无摄像头）
silver_guardian.exe --demo

# 指定配置文件
silver_guardian.exe --config "C:\路径\config.yaml"
```

### 3. 基本操作

- **查看摄像头**: 在主界面查看实时视频
- **告警列表**: 右侧显示触发的告警
- **生命体征**: 心率、呼吸率实时数据
- **系统状态**: 底部显示系统运行状态
- **最小化**: 窗口关闭会最小化到系统托盘

### 4. 配置摄像头

编辑 `config/config.yaml` 中的摄像头配置：

```yaml
# USB摄像头
source: 0

# RTSP流
source: rtsp://username:password@ip:port/stream
```

## 四、常见问题

### 1. 摄像头无法连接

- 检查摄像头是否被其他程序占用
- 确认摄像头设备ID是否正确
- 查看日志文件 `data/logs/system.log`

### 2. 程序无法启动

- 检查是否有杀毒软件拦截
- 确保系统满足最低要求
- 以管理员身份运行

### 3. 性能问题

- 降低摄像头分辨率
- 减少同时处理的摄像头数量
- 降低帧率设置

## 五、技术支持

如遇到问题，请检查：

1. 日志文件：`data/logs/system.log`
2. 配置文件：`config/config.yaml`
3. 系统要求是否满足

## 六、卸载

直接删除应用目录即可，不会在系统中留下残留文件。
