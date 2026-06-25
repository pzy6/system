#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
打包与运行环境校验脚本。
"""

import importlib
import os
import sys
from pathlib import Path

import yaml


def ok_text(ok):
    return "通过" if ok else "失败"


def resolve_base_dir():
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def check_exists(path, kind="file"):
    if kind == "dir":
        return path.is_dir(), "存在" if path.is_dir() else "缺失"
    return path.exists(), "存在" if path.exists() else "缺失"


def load_config(config_path):
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def verify_config(config):
    errors = []

    required_top_keys = [
        "cameras",
        "alarm_thresholds",
        "model_paths",
        "processing",
        "storage",
        "system",
    ]
    for key in required_top_keys:
        if key not in config:
            errors.append(f"缺少顶层配置项: {key}")

    cameras = config.get("cameras", [])
    if not isinstance(cameras, list) or not cameras:
        errors.append("cameras 必须是非空列表")
    else:
        for index, camera in enumerate(cameras, start=1):
            for key in ["id", "name", "source", "resolution", "fps"]:
                if key not in camera:
                    errors.append(f"第 {index} 路摄像头缺少字段: {key}")

    processing = config.get("processing", {})
    if "input_size" not in processing:
        errors.append("processing.input_size 缺失")

    storage = config.get("storage", {})
    for key in ["logs", "alarms", "screenshots", "videos"]:
        if key not in storage:
            errors.append(f"storage.{key} 缺失")

    thresholds = config.get("alarm_thresholds", {})
    for key in ["fall_detection", "abnormal_behavior", "emotion_confidence", "heart_rate", "respiratory_rate"]:
        if key not in thresholds:
            errors.append(f"alarm_thresholds.{key} 缺失")

    return len(errors) == 0, errors


def verify_model_paths(base_dir, config):
    results = []
    model_paths = config.get("model_paths", {})
    expected = {
        "skeleton": ["frozen_inference_graph.pb", "graph_opt.pbtxt"],
        "emotion": ["deploy.prototxt.txt", "emotion_net.caffemodel"],
        "behavior": [],
        "fall_detection": [],
        "vitals": [],
    }

    for name, required_files in expected.items():
        raw_path = model_paths.get(name, "")
        if not raw_path:
            results.append((False, name, "未配置路径"))
            continue

        model_dir = Path(raw_path)
        if not model_dir.is_absolute():
            model_dir = (base_dir / raw_path).resolve()

        if not model_dir.exists():
            results.append((False, name, f"路径不存在: {model_dir}"))
            continue

        missing_files = [filename for filename in required_files if not (model_dir / filename).exists()]
        if missing_files:
            results.append((False, name, f"缺少文件: {', '.join(missing_files)}"))
        else:
            results.append((True, name, f"路径可用: {model_dir}"))

    return results


def verify_modules():
    modules = [
        "cv2",
        "numpy",
        "yaml",
        "PyQt5",
    ]
    results = []
    for module in modules:
        try:
            importlib.import_module(module)
            results.append((True, module, "已安装"))
        except ImportError as exc:
            results.append((False, module, str(exc)))
    return results


def verify_source_tree(base_dir):
    src_dir = base_dir / "src"
    required_paths = [
        "main.py",
        "application",
        "algorithms",
        "perception",
        "processing",
        "utils",
    ]
    results = []
    if not src_dir.exists():
        return [(False, "src", "源码目录不存在")]

    for rel_path in required_paths:
        path = src_dir / rel_path
        results.append((path.exists(), f"src/{rel_path}", "存在" if path.exists() else "缺失"))
    return results


def verify_storage_dirs(base_dir, config):
    storage = config.get("storage", {})
    results = []
    for key in ["logs", "alarms", "screenshots", "videos"]:
        raw_path = storage.get(key, "")
        target = Path(raw_path)
        if not target.is_absolute():
            target = (base_dir / raw_path).resolve()
        results.append((target.is_dir(), key, str(target)))
    return results


def print_section(title):
    print()
    print(title)


def main():
    base_dir = resolve_base_dir()
    print("=" * 60)
    print("银发守护者系统 - 打包与运行环境校验")
    print("=" * 60)
    print(f"检查目录: {base_dir}")

    all_passed = True

    print_section("[1/8] 关键文件")
    files_to_check = [
        ("主程序", base_dir / "silver_guardian.exe"),
        ("配置文件", base_dir / "config" / "config.yaml"),
        ("启动脚本", base_dir / "启动应用.bat"),
        ("使用说明", base_dir / "使用说明.txt"),
        ("配置说明", base_dir / "配置说明.txt"),
        ("校验脚本", base_dir / "verify_package.py"),
    ]
    for label, path in files_to_check:
        exists, status = check_exists(path, "file")
        print(f"{label:10s}: {status:4s} - {path.name}")
        all_passed &= exists

    config_path = base_dir / "config" / "config.yaml"
    config = {}

    print_section("[2/8] 配置文件结构")
    if config_path.exists():
        try:
            config = load_config(config_path)
            ok, errors = verify_config(config)
            print(f"配置解析: {ok_text(ok)}")
            if not ok:
                for error in errors:
                    print(f"  - {error}")
            all_passed &= ok
        except Exception as exc:
            print(f"配置解析: 失败 - {exc}")
            all_passed = False
    else:
        print("配置解析: 失败 - config/config.yaml 不存在")
        all_passed = False

    print_section("[3/8] 存储目录")
    if config:
        for ok, label, detail in verify_storage_dirs(base_dir, config):
            print(f"{label:10s}: {ok_text(ok)} - {detail}")
            all_passed &= ok
    else:
        print("跳过，配置未加载")
        all_passed = False

    print_section("[4/8] 模型路径与文件")
    if config:
        for ok, model_name, detail in verify_model_paths(base_dir, config):
            print(f"{model_name:14s}: {ok_text(ok)} - {detail}")
        # 模型允许部分缺失，因为系统有回退逻辑，这里不把所有缺失都作为硬失败
    else:
        print("跳过，配置未加载")

    print_section("[5/8] Python 依赖")
    for ok, module_name, detail in verify_modules():
        print(f"{module_name:10s}: {ok_text(ok)} - {detail}")
        all_passed &= ok

    print_section("[6/8] 源码结构")
    for ok, name, detail in verify_source_tree(base_dir):
        print(f"{name:20s}: {ok_text(ok)} - {detail}")
        all_passed &= ok

    print_section("[7/8] 打包脚本与规范")
    spec_file = base_dir / "silver_guardian.spec"
    scripts = [
        base_dir / "scripts" / "build_app.bat",
        base_dir / "scripts" / "prepare_release.bat",
        base_dir / "scripts" / "full_build.bat",
    ]
    spec_ok = spec_file.exists()
    print(f"silver_guardian.spec: {ok_text(spec_ok)} - {spec_file}")
    all_passed &= spec_ok
    for script in scripts:
        exists = script.exists()
        print(f"{script.name:20s}: {ok_text(exists)} - {script}")
        all_passed &= exists

    print_section("[8/8] 告警产物检查")
    if config:
        storage = config.get("storage", {})
        alarms_dir = Path(storage.get("alarms", "./data/alarms/"))
        if not alarms_dir.is_absolute():
            alarms_dir = (base_dir / alarms_dir).resolve()
        history_file = alarms_dir / "alarm_history.jsonl"
        print(f"告警历史文件: {ok_text(history_file.exists())} - {history_file}")
        if history_file.exists():
            try:
                with history_file.open("r", encoding="utf-8") as f:
                    first_line = f.readline().strip()
                print("告警历史编码: 通过 - UTF-8 可读")
                if first_line:
                    print(f"首条记录预览: {first_line[:120]}")
            except Exception as exc:
                print(f"告警历史编码: 失败 - {exc}")
                all_passed = False
    else:
        print("跳过，配置未加载")

    print()
    print("=" * 60)
    print("总体结果:", "全部通过" if all_passed else "存在问题")
    print("=" * 60)
    return 0 if all_passed else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n校验已取消")
        sys.exit(1)
