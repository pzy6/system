"""
训练/微调 YOLOv11n-pose 姿态估计模型。

默认策略：
  直接使用 YOLOv11n-pose 在 COCO 上的预训练权重（已支持 17 个关键点）。
  可选: 用自定义摔倒/老年姿态数据集微调。

数据集要求（YOLO 姿态格式）：
  目录结构：
    data/pose/
      dataset.yaml
      images/train/
      images/val/
      labels/train/       # YOLO 姿态标签
      labels/val/

  YOLO 姿态标签格式（每行一个关键点）：
    <class_id> <cx> <cy> <w> <h> <px1> <py1> <v1> ... <px17> <py17> <v17>
    - class_id: 0 (person)
    - v=0: 不可见, v=1: 遮挡, v=2: 可见

用法：
  # 仅下载预训练权重，不训练
  python scripts/training/train_yolo_pose.py --download-only

  # 微调
  python scripts/training/train_yolo_pose.py --data ./data/pose/dataset.yaml --epochs 50
"""

import argparse
import os
import sys
import yaml


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def download_and_export(output_dir: str):
    """下载 YOLOv11n-pose 预训练权重并导出 ONNX"""
    os.makedirs(output_dir, exist_ok=True)
    pt_path = os.path.join(output_dir, "yolo11n-pose.pt")
    onnx_path = os.path.join(output_dir, "yolo11n-pose.onnx")

    print("=" * 60)
    print("下载 YOLOv11n-pose 预训练权重")
    print("=" * 60)

    try:
        from ultralytics import YOLO

        model = YOLO("yolo11n-pose.pt")
        print(f"  ✓ 预训练权重: {pt_path}")

        print(f"  导出 ONNX 到 {onnx_path} ...")
        model.export(format="onnx", imgsz=640, opset=12, simplify=True)
        print("  ✓ ONNX 导出完成")

        return pt_path
    except ImportError:
        print("[错误] ultralytics 未安装。pip install ultralytics>=8.3.0")
        sys.exit(1)
    except Exception as e:
        print(f"[错误] {str(e)}")
        sys.exit(1)


def finetune(data_yaml: str, epochs: int, batch: int, imgsz: int, output_dir: str):
    """微调 YOLOv11n-pose"""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[错误] ultralytics 未安装。")
        sys.exit(1)

    if not os.path.exists(data_yaml):
        print(f"[错误] 数据集配置不存在: {data_yaml}")
        sys.exit(1)

    print("=" * 60)
    print("微调 YOLOv11n-pose 姿态估计模型")
    print("=" * 60)
    print(f"  数据集:   {data_yaml}")
    print(f"  训练轮数: {epochs}")
    print(f"  批次大小: {batch}")
    print(f"  图像尺寸: {imgsz}")
    print()

    os.makedirs(output_dir, exist_ok=True)

    model = YOLO("yolo11n-pose.pt")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project=output_dir,
        name="yolo11n_pose_finetune",
        exist_ok=True,
        patience=15,
        save=True,
        save_period=10,
        val=True,
        plots=True,
        device="cpu",
        workers=4,
        pretrained=True,
        optimizer="auto",
        # 姿态相关增强
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=45.0,            # 大幅度旋转（模拟摔倒的各种角度）
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,              # 姿态标注对 mixup 敏感
    )

    # 验证
    print()
    print("验证最佳模型...")
    best_pt = os.path.join(output_dir, "yolo11n_pose_finetune", "weights", "best.pt")
    if os.path.exists(best_pt):
        model = YOLO(best_pt)
        metrics = model.val()
        print(f"  mAP50:    {metrics.box.map50:.4f}")
        print(f"  mAP50-95: {metrics.box.map:.4f}")

        # 导出 ONNX
        print()
        print("导出 ONNX...")
        onnx_path = model.export(format="onnx", imgsz=640, opset=12, simplify=True)
        print(f"  ONNX 模型: {onnx_path}")

        # 复制到 models 目录
        import shutil
        models_dir = os.path.join(get_project_root(), "models", "yolo_pose")
        os.makedirs(models_dir, exist_ok=True)
        shutil.copy(best_pt, os.path.join(models_dir, "yolo11n_pose.pt"))
        print(f"  模型已复制到: {models_dir}/")

    print()
    print("训练完成!")


def main():
    parser = argparse.ArgumentParser(description="训练 YOLOv11n-pose 姿态估计模型")
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="仅下载预训练权重，不进行训练",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="dataset.yaml 路径（微调时必填）",
    )
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch", type=int, default=16, help="批次大小")
    parser.add_argument("--imgsz", type=int, default=640, help="图像尺寸")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出目录（默认: models/yolo_pose/training_output）",
    )
    args = parser.parse_args()

    root = get_project_root()
    models_dir = os.path.join(root, "models", "yolo_pose")

    if args.download_only:
        download_and_export(models_dir)
        print()
        print("预训练模型已就绪，无需额外训练即可使用。")
        print(f"模型路径: {models_dir}/")
        print()
        print("如需微调，请准备数据集后运行:")
        print("  python scripts/training/train_yolo_pose.py --data <dataset.yaml>")
        return

    if args.data is None:
        print("[提示] 未指定 --data，将仅下载预训练权重。")
        print("YOLOv11n-pose 的 COCO 预训练权重已可直接用于摔倒检测。")
        print()
        print("如需微调，请准备数据集并使用 --data 参数。")
        print("或使用 --download-only 仅下载权重。")
        download_and_export(models_dir)
        return

    if args.output is None:
        args.output = os.path.join(models_dir, "training_output")

    finetune(args.data, args.epochs, args.batch, args.imgsz, args.output)


if __name__ == "__main__":
    main()
