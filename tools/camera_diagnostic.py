#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
摄像头诊断工具 - 系统性排查摄像头画面无法显示的问题
"""

import cv2
import os
import platform
import subprocess
import sys

def check_system_info():
    """检查系统信息"""
    print("=" * 60)
    print("系统信息检查")
    print("=" * 60)
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python版本: {sys.version}")
    print(f"OpenCV版本: {cv2.__version__}")
    print()

def test_camera_backends():
    """测试不同的摄像头后端"""
    print("=" * 60)
    print("摄像头后端测试")
    print("=" * 60)
    
    backends = [
        ("CAP_ANY", cv2.CAP_ANY),
        ("CAP_DSHOW", cv2.CAP_DSHOW),
        ("CAP_MSMF", cv2.CAP_MSMF),
        ("CAP_VFW", cv2.CAP_VFW),
        ("CAP_AVFOUNDATION", cv2.CAP_AVFOUNDATION),
        ("CAP_GSTREAMER", cv2.CAP_GSTREAMER),
    ]
    
    results = {}
    for name, backend in backends:
        try:
            cap = cv2.VideoCapture(0, backend)
            if cap.isOpened():
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = cap.get(cv2.CAP_PROP_FPS)
                results[name] = {
                    'success': True,
                    'resolution': f"{int(width)}x{int(height)}",
                    'fps': int(fps) if fps > 0 else "未知"
                }
                cap.release()
            else:
                results[name] = {'success': False, 'error': "无法打开设备"}
        except Exception as e:
            results[name] = {'success': False, 'error': str(e)}
    
    for name, result in results.items():
        status = "✓" if result['success'] else "✗"
        if result['success']:
            print(f"{status} {name}: 成功 - 分辨率: {result['resolution']}, FPS: {result['fps']}")
        else:
            print(f"{status} {name}: 失败 - {result.get('error', '未知错误')}")
    print()
    
    return results

def scan_camera_devices():
    """扫描所有可用的摄像头设备"""
    print("=" * 60)
    print("摄像头设备扫描")
    print("=" * 60)
    
    found_cameras = []
    
    for i in range(10):
        success = False
        info = {}
        
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        
        for backend in backends:
            try:
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    info = {
                        'index': i,
                        'backend': backend,
                        'resolution': f"{int(width)}x{int(height)}" if width > 0 else "未知",
                        'fps': int(fps) if fps > 0 else 30
                    }
                    found_cameras.append(info)
                    cap.release()
                    success = True
                    break
            except Exception as e:
                continue
        
        if success:
            print(f"✓ 摄像头 {i}: {info['resolution']} @ {info['fps']}fps")
        else:
            print(f"✗ 摄像头 {i}: 不可用")
    
    print()
    return found_cameras

def check_device_manager_windows():
    """检查Windows设备管理器中的摄像头状态"""
    if platform.system() != 'Windows':
        print("注意: 设备管理器检查仅适用于Windows系统")
        return
    
    print("=" * 60)
    print("设备管理器检查")
    print("=" * 60)
    
    try:
        # 使用WMIC命令检查图像处理设备
        result = subprocess.run(
            ['wmic', 'path', 'Win32_PnPEntity', 'where', 
             'PNPClass="Image"', 'get', 'Name,Status'],
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        print(result.stdout)
        
        if result.returncode != 0:
            print("警告: 无法访问设备管理器信息")
    except Exception as e:
        print(f"错误: {e}")
    print()

def check_privacy_settings_windows():
    """检查Windows隐私设置"""
    if platform.system() != 'Windows':
        return
    
    print("=" * 60)
    print("隐私设置检查")
    print("=" * 60)
    
    try:
        # 检查摄像头访问权限
        import winreg
        key_path = r"SOFTWARE\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\webcam"
        
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
            try:
                value, _ = winreg.QueryValueEx(key, "Value")
                print(f"摄像头权限状态: {value}")
            except FileNotFoundError:
                print("摄像头权限状态: 未设置")
        
        print("\n提示: 请确保在 Windows 设置 > 隐私和安全性 > 摄像头 中启用权限")
    except Exception as e:
        print(f"无法检查隐私设置: {e}")
    print()

def test_camera_capture():
    """测试摄像头捕获功能"""
    print("=" * 60)
    print("摄像头捕获测试")
    print("=" * 60)
    
    cameras = scan_camera_devices()
    
    if not cameras:
        print("未检测到任何摄像头设备")
        return
    
    print(f"\n测试第一个摄像头 (索引 {cameras[0]['index']})...")
    
    cap = cv2.VideoCapture(cameras[0]['index'], cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("✗ 无法打开摄像头")
        return
    
    print("✓ 摄像头已打开")
    
    # 读取几帧测试
    success_count = 0
    for i in range(10):
        ret, frame = cap.read()
        if ret:
            success_count += 1
            if i == 0:
                height, width = frame.shape[:2]
                print(f"✓ 成功读取帧: {width}x{height}")
    
    cap.release()
    
    print(f"\n读取结果: {success_count}/10 帧成功")
    
    if success_count == 0:
        print("\n可能的原因:")
        print("1. 摄像头被其他应用占用")
        print("2. 摄像头驱动问题")
        print("3. 硬件故障")
    elif success_count < 10:
        print("\n警告: 部分帧读取失败，可能存在连接不稳定")
    else:
        print("\n✓ 摄像头工作正常")

def generate_report(results):
    """生成诊断报告"""
    print("=" * 60)
    print("诊断报告")
    print("=" * 60)
    
    cameras = results['cameras']
    backends = results['backends']
    
    if not cameras:
        print("结论: 未检测到可用摄像头")
        print("\n建议:")
        print("1. 检查摄像头硬件连接")
        print("2. 在设备管理器中检查驱动状态")
        print("3. 确保摄像头未被其他应用占用")
        print("4. 尝试重新安装摄像头驱动")
    else:
        print(f"结论: 检测到 {len(cameras)} 个摄像头设备")
        print("\n建议:")
        print("1. 摄像头硬件正常")
        print("2. 检查应用程序是否有权限访问摄像头")
        print("3. 确保没有其他应用占用摄像头")
    
    working_backends = [name for name, info in backends.items() if info['success']]
    if working_backends:
        print(f"\n推荐使用的后端: {', '.join(working_backends)}")
    
    print("\n" + "=" * 60)

def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("摄像头诊断工具 v1.0")
    print("=" * 60)
    print("正在进行系统性摄像头排查...")
    print()
    
    results = {}
    
    # 1. 检查系统信息
    check_system_info()
    
    # 2. 测试摄像头后端
    results['backends'] = test_camera_backends()
    
    # 3. 扫描摄像头设备
    results['cameras'] = scan_camera_devices()
    
    # 4. 检查设备管理器(Windows)
    check_device_manager_windows()
    
    # 5. 检查隐私设置(Windows)
    check_privacy_settings_windows()
    
    # 6. 测试摄像头捕获
    test_camera_capture()
    
    # 7. 生成报告
    generate_report(results)

if __name__ == "__main__":
    main()
