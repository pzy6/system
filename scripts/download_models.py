"""
下载 YOLOv11n 预训练权重并导出 ONNX 格式。
放置到 models/ 目录供系统加载。

用法：
  python scripts/download_models.py                # 下载全部
  python scripts/download_models.py --pose-only    # 仅姿态模型
  python scripts/download_models.py --detect-only  # 仅检测模型
"""

import argparse
import os
import sys


def get_project_root():
    """获取项目根目录"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def download_yolo_pose(output_dir: str) -> str:
    """下载 YOLOv11n-pose 预训练权重并导出 ONNX"""
    os.makedirs(output_dir, exist_ok=True)
    pt_path = os.path.join(output_dir, "yolo11n-pose.pt")
    onnx_path = os.path.join(output_dir, "yolo11n-pose.onnx")

    print("=" * 60)
    print("下载 YOLOv11n-pose (姿态估计) 模型...")
    print("=" * 60)

    try:
        from ultralytics import YOLO

        # 下载预训练权重
        print(f"  下载预训练权重到 {pt_path} ...")
        model = YOLO("yolo11n-pose.pt")  # Ultralytics 自动下载
        print("  ✓ 预训练权重下载完成")

        # 导出 ONNX
        print(f"  导出 ONNX 到 {onnx_path} ...")
        model.export(format="onnx", imgsz=640, opset=12, simplify=True)
        print("  ✓ ONNX 导出完成")

        return pt_path

    except ImportError:
        print("  [错误] ultralytics 未安装。请运行: pip install ultralytics>=8.3.0")
        sys.exit(1)
    except Exception as e:
        print(f"  [错误] 下载失败: {str(e)}")
        sys.exit(1)


def download_yolo_detect(output_dir: str) -> str:
    """下载 YOLOv11n 预训练权重（用于后续微调）并导出 ONNX"""
    os.makedirs(output_dir, exist_ok=True)
    pt_path = os.path.join(output_dir, "yolo11n.pt")
    onnx_path = os.path.join(output_dir, "yolo11n.onnx")

    print("=" * 60)
    print("下载 YOLOv11n (目标检测) 基础模型...")
    print("=" * 60)
    print("  注意: 这是 COCO 预训练的基础模型，未针对人脸+烟雾优化。")
    print("  使用 scripts/training/train_yolo_face_smoke.py 进行微调。")
    print()

    try:
        from ultralytics import YOLO

        print(f"  下载预训练权重到 {pt_path} ...")
        model = YOLO("yolo11n.pt")
        print("  ✓ 预训练权重下载完成")

        print(f"  导出 ONNX 到 {onnx_path} ...")
        model.export(format="onnx", imgsz=640, opset=12, simplify=True)
        print("  ✓ ONNX 导出完成")

        return pt_path

    except ImportError:
        print("  [错误] ultralytics 未安装。请运行: pip install ultralytics>=8.3.0")
        sys.exit(1)
    except Exception as e:
        print(f"  [错误] 下载失败: {str(e)}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="下载 YOLOv11n 模型权重")
    parser.add_argument("--pose-only", action="store_true", help="仅下载姿态模型")
    parser.add_argument("--detect-only", action="store_true", help="仅下载检测模型")
    args = parser.parse_args()

    root = get_project_root()
    pose_dir = os.path.join(root, "models", "yolo_pose")
    detect_dir = os.path.join(root, "models", "yolo_face_smoke")

    if not args.detect_only:
        download_yolo_pose(pose_dir)

    if not args.pose_only:
        download_yolo_detect(detect_dir)

    print()
    print("=" * 60)
    print("全部下载完成!")
    print("=" * 60)
    print(f"  姿态模型: {pose_dir}/")
    print(f"  检测模型: {detect_dir}/")
    print()
    print("下一步:")
    print("  1. 运行 scripts/training/train_yolo_face_smoke.py 微调人脸+烟雾模型")
    print("  2. 运行 scripts/enroll_face.py 注册人脸")
    print("  3. 启动系统: python src/main.py")


if __name__ == "__main__":
    main()
