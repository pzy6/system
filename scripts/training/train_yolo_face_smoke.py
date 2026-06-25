"""
训练 YOLOv11n 人脸+烟雾检测模型（2 类：face, smoke）。

数据集要求：
  目录结构（YOLO 格式）：
    data/face_smoke/
      dataset.yaml        # 数据集配置
      images/train/       # 训练图片
      images/val/         # 验证图片
      labels/train/       # 训练标签（每图对应一个 .txt）
      labels/val/         # 验证标签

  dataset.yaml 内容：
    path: ./data/face_smoke
    train: images/train
    val: images/val
    names:
      0: face
      1: smoke
    nc: 2

  标签格式（YOLO 归一化）：
    <class_id> <cx> <cy> <w> <h>
    - class_id: 0=face, 1=smoke
    - 坐标归一化到 [0, 1]

推荐数据集来源：
  - 人脸: WIDER Face (http://shuoyang1213.me/WIDERFACE/)
  - 烟雾: Kaggle Smoke Detection, Roboflow Smoke datasets
  - 自行标注: LabelImg, Roboflow Annotate

用法：
  python scripts/training/train_yolo_face_smoke.py
  python scripts/training/train_yolo_face_smoke.py --epochs 200 --batch 32
  python scripts/training/train_yolo_face_smoke.py --data ./my_dataset/dataset.yaml
"""

import argparse
import os
import sys
import yaml


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_default_dataset_yaml(data_dir: str) -> str:
    """创建默认的 dataset.yaml（如果不存在）"""
    yaml_path = os.path.join(data_dir, "dataset.yaml")
    if os.path.exists(yaml_path):
        return yaml_path

    os.makedirs(data_dir, exist_ok=True)
    # 创建子目录
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        os.makedirs(os.path.join(data_dir, sub), exist_ok=True)

    dataset_config = {
        "path": data_dir,
        "train": "images/train",
        "val": "images/val",
        "names": {0: "face", 1: "smoke"},
        "nc": 2,
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dataset_config, f, allow_unicode=True)

    print(f"已创建默认 dataset.yaml: {yaml_path}")
    print("请将训练/验证图片放入对应目录，并确保 labels/ 中有对应的 .txt 标注文件。")
    return yaml_path


def train(data_yaml: str, epochs: int, batch: int, imgsz: int, output_dir: str):
    """训练 YOLOv11n 人脸+烟雾检测模型"""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[错误] ultralytics 未安装。pip install ultralytics>=8.3.0")
        sys.exit(1)

    if not os.path.exists(data_yaml):
        print(f"[错误] 数据集配置文件不存在: {data_yaml}")
        sys.exit(1)

    print("=" * 60)
    print("训练 YOLOv11n 人脸+烟雾检测模型")
    print("=" * 60)
    print(f"  数据集:     {data_yaml}")
    print(f"  训练轮数:   {epochs}")
    print(f"  批次大小:   {batch}")
    print(f"  图像尺寸:   {imgsz}")
    print(f"  输出目录:   {output_dir}")
    print()

    os.makedirs(output_dir, exist_ok=True)

    # 加载预训练权重，如果不存在则自动下载
    model = YOLO("yolo11n.pt")

    print("开始训练...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project=output_dir,
        name="yolo11n_face_smoke",
        exist_ok=True,
        patience=20,              # 早停
        save=True,
        save_period=10,           # 每 10 轮保存一次
        val=True,
        plots=True,
        device="cpu",             # 或 "cuda:0"
        workers=4,
        pretrained=True,
        optimizer="auto",
        verbose=True,
        # 数据增强
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,            # 轻度旋转（人脸不需要大幅度旋转）
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
    )

    # 验证
    print()
    print("验证最佳模型...")
    best_pt = os.path.join(output_dir, "yolo11n_face_smoke", "weights", "best.pt")
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
        models_dir = os.path.join(get_project_root(), "models", "yolo_face_smoke")
        os.makedirs(models_dir, exist_ok=True)
        shutil.copy(best_pt, os.path.join(models_dir, "yolov11n_face_smoke.pt"))
        print(f"  模型已复制到: {models_dir}/")
    else:
        print("  [警告] 未找到 best.pt，请检查训练输出")

    print()
    print("训练完成!")


def main():
    parser = argparse.ArgumentParser(description="训练 YOLOv11n 人脸+烟雾检测模型")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="dataset.yaml 文件路径（默认: data/face_smoke/dataset.yaml）",
    )
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch", type=int, default=16, help="批次大小")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图像尺寸")
    parser.add_argument(
        "--fast", action="store_true",
        help="CPU 优化模式（imgsz=320, batch=8, epochs=30）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出目录（默认: models/yolo_face_smoke/training_output）",
    )
    args = parser.parse_args()

    if args.fast:
        args.imgsz = 320
        args.batch = 8
        args.epochs = 30
        print("[快速模式] imgsz=320 batch=8 epochs=30 (CPU 优化)")

    root = get_project_root()

    if args.data is None:
        args.data = os.path.join(root, "data", "face_smoke", "dataset.yaml")
        if not os.path.exists(args.data):
            args.data = create_default_dataset_yaml(
                os.path.join(root, "data", "face_smoke")
            )
            print()
            print("[提示] 请准备数据集后再运行训练。")
            print("  1. 将训练图片放入 data/face_smoke/images/train/")
            print("  2. 将验证图片放入 data/face_smoke/images/val/")
            print("  3. 将 YOLO 格式标签放入对应 labels/ 目录")
            print("  4. 编辑 data/face_smoke/dataset.yaml 确认路径")
            print()
            if not os.path.exists(os.path.join(root, "data", "face_smoke", "images", "train")):
                print("[警告] 训练数据目录为空，训练将无法进行。")
                print("请准备数据集后重新运行此脚本。")
                return

    if args.output is None:
        args.output = os.path.join(root, "models", "yolo_face_smoke", "training_output")

    train(args.data, args.epochs, args.batch, args.imgsz, args.output)


if __name__ == "__main__":
    main()
