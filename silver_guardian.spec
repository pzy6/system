# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置 — 银发守护者系统 v2.0"""
import sys, os

block_cipher = None
project_root = os.path.abspath('.')

# 模型和数据文件
datas = [
    ('config/config.yaml', 'config'),
    ('models', 'models'),
    ('data/face_db', 'data/face_db'),
    ('assets', 'assets'),
    ('VERSION', '.'),
]

# 排除不需要的
excludes = [
    'matplotlib.tests', 'numpy.tests', 'scipy.tests',
    'pandas.tests', 'torch.utils.tensorboard',
    'tensorboard', 'jupyter', 'IPython',
]

# 隐藏导入
hiddenimports = [
    # GUI
    'PyQt5', 'PyQt5.QtCore', 'PyQt5.QtGui', 'PyQt5.QtWidgets',
    'PyQt5.sip',
    # CV & ML
    'cv2', 'cv2.cv2',
    'numpy', 'numpy.core', 'numpy.core.multiarray',
    'scipy', 'scipy.signal', 'scipy.fft',
    'torch', 'torchvision',
    'ultralytics', 'ultralytics.nn', 'ultralytics.utils',
    'ultralytics.engine', 'ultralytics.data',
    # ONNX
    'onnxruntime', 'onnxruntime.capi',
    # Face recognition
    'insightface', 'insightface.app', 'insightface.model_zoo',
    'insightface.utils',
    # Audio
    'pyttsx3', 'pyttsx3.drivers', 'pyttsx3.drivers.sapi5',
    'comtypes', 'comtypes.client',
    # YAML
    'yaml', 'pyyaml',
    # Misc
    'huggingface_hub',
    'winsound',
]

a = Analysis(
    ['src/main.py'],
    pathex=[project_root, 'src'],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='银发守护者系统',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/logo/logo_square.svg',
)
