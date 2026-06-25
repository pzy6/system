"""
训练 YOLOv11n 烟雾检测模型（基于 D-Fire 数据集或通用数据集）

D-Fire 数据集结构（默认）：
  data/DFireDataset/
    train/images/   train/labels/
    val/images/     val/labels/
  类别: 0=fire, 1=smoke  →  本脚本提取 smoke 类（class 1）并重映射为 class 0

用法：
  # 默认使用 D-Fire 数据集
  python scripts/training/train_yolo_smoke.py --fast

  # 使用自定义数据集
  python scripts/training/train_yolo_smoke.py --data ./my_data/smoke.yaml --fast
"""

import argparse
import os
import sys
import shutil
import yaml
import random


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ====================================================================
# D-Fire 数据集预处理（fire+smoke → smoke only）
# ====================================================================

def prepare_dfire_dataset(dfire_dir: str, output_dir: str, train_ratio: float = 0.85):
    """将 D-Fire 数据集转换为仅烟雾的单类 YOLO 数据集"""
    print("=" * 60)
    print("准备 D-Fire → 烟雾检测数据集")
    print("=" * 60)

    # 收集所有 smoke 类样本
    smoke_samples = {"train": [], "val": []}

    for split in ["train", "val"]:
        img_dir = os.path.join(dfire_dir, split, "images")
        lbl_dir = os.path.join(dfire_dir, split, "labels")

        if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
            print(f"  跳过 {split}: 目录不存在")
            continue

        for lbl_name in os.listdir(lbl_dir):
            if not lbl_name.endswith(".txt"):
                continue
            lbl_path = os.path.join(lbl_dir, lbl_name)

            # 检查是否包含 smoke 类（class 1）
            has_smoke = False
            has_fire = False
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cls_id = int(parts[0])
                    if cls_id == 1:
                        has_smoke = True
                    elif cls_id == 0:
                        has_fire = True

            if has_smoke:
                base = os.path.splitext(lbl_name)[0]
                # 查找对应图片
                for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                    img_path = os.path.join(img_dir, base + ext)
                    if os.path.exists(img_path):
                        smoke_samples[split].append({
                            "img": img_path,
                            "lbl": lbl_path,
                            "has_fire": has_fire,
                            "smoke_only": not has_fire,
                        })
                        break

    total = len(smoke_samples["train"]) + len(smoke_samples["val"])
    print(f"  含 smoke 样本: train={len(smoke_samples['train'])}, "
          f"val={len(smoke_samples['val'])}, total={total}")

    if total == 0:
        print("[错误] 未找到 smoke 标注样本")
        return False

    # 合并所有样本并重新分割
    all_samples = smoke_samples["train"] + smoke_samples["val"]
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * train_ratio)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    # 写入输出目录
    for split_name, samples in [("train", train_samples), ("val", val_samples)]:
        img_out = os.path.join(output_dir, "images", split_name)
        lbl_out = os.path.join(output_dir, "labels", split_name)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for i, s in enumerate(samples):
            ext = os.path.splitext(s["img"])[1]
            new_name = f"smoke_{i:05d}"
            shutil.copy2(s["img"], os.path.join(img_out, f"{new_name}{ext}"))

            # 重映射标签: smoke (class 1) → class 0
            dst_lbl = os.path.join(lbl_out, f"{new_name}.txt")
            with open(s["lbl"], "r") as src, open(dst_lbl, "w") as dst:
                for line in src:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cls_id = int(parts[0])
                    if cls_id == 1:  # smoke only
                        parts[0] = "0"
                        dst.write(" ".join(parts) + "\n")

    # 生成 dataset.yaml
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    dataset_yaml = {
        "path": os.path.abspath(output_dir).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "smoke"},
        "nc": 1,
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dataset_yaml, f, allow_unicode=True, sort_keys=False)

    print(f"  输出: train={len(train_samples)}, val={len(val_samples)}")
    print(f"  配置: {yaml_path}")
    return True


# ====================================================================
# 训练
# ====================================================================

def train_smoke(data_yaml: str, epochs: int, batch: int, imgsz: int, output_dir: str):
    """训练 YOLOv11n 烟雾检测（单类）"""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[错误] ultralytics 未安装")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("训练 YOLOv11n 烟雾检测模型")
    print("=" * 60)
    print(f"  数据集:   {data_yaml}")
    print(f"  Epochs:   {epochs}")
    print(f"  Batch:    {batch}")
    print(f"  Image:    {imgsz}")
    print()

    os.makedirs(output_dir, exist_ok=True)

    model = YOLO("yolo26n.pt")

    model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project=output_dir,
        name="yolo26n_smoke",
        exist_ok=True,
        patience=15,
        save=True,
        save_period=10,
        val=True,
        plots=True,
        device=0,
        workers=4,
        amp=False,
        pretrained=True,
        optimizer="auto",
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=5.0,
        translate=0.1,
        scale=0.3,
        fliplr=0.5,
        mosaic=0.8,
        mixup=0.1,
    )

    # 导出模型
    best_pt = os.path.join(output_dir, "yolo26n_smoke", "weights", "best.pt")
    if os.path.exists(best_pt):
        print("\n导出 ONNX...")
        model = YOLO(best_pt)
        model.export(format="onnx", imgsz=imgsz, opset=12, simplify=True)

        # 复制到 models 目录
        models_dir = os.path.join(get_project_root(), "models", "yolo_face_smoke")
        os.makedirs(models_dir, exist_ok=True)
        shutil.copy(best_pt, os.path.join(models_dir, "yolo26n_smoke.pt"))
        print(f"模型已复制到: {models_dir}/")

    print("\n训练完成!")


# ====================================================================
# 主入口
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description="训练 YOLOv11n 烟雾检测模型")
    parser.add_argument("--data", type=str, default=None, help="dataset.yaml 路径")
    parser.add_argument("--dfire", type=str, default=None,
                        help="D-Fire 数据集目录（默认: data/DFireDataset）")
    parser.add_argument("--epochs", type=int, default=80, help="训练轮数")
    parser.add_argument("--batch", type=int, default=16, help="批次大小")
    parser.add_argument("--imgsz", type=int, default=640, help="图像尺寸")
    parser.add_argument("--fast", action="store_true",
                        help="CPU 快速模式（imgsz=320, batch=8, epochs=30）")
    parser.add_argument("--output", type=str, default=None,
                        help="输出目录")
    args = parser.parse_args()

    root = get_project_root()

    if args.fast:
        args.imgsz = 320
        args.batch = 8
        args.epochs = 30
        print("[快速模式] imgsz=320 batch=8 epochs=30")

    if args.output is None:
        args.output = os.path.join(root, "models", "yolo_face_smoke", "training_output")

    # 确定数据源
    if args.data:
        # 直接使用指定的 dataset.yaml
        train_smoke(args.data, args.epochs, args.batch, args.imgsz, args.output)
        return

    # 尝试 D-Fire → 转换为烟雾数据集
    if args.dfire is None:
        args.dfire = os.path.join(root, "data", "DFireDataset")

    smoke_dir = os.path.join(root, "data", "smoke_dataset")
    if not os.path.isdir(args.dfire):
        print(f"[错误] D-Fire 数据集未找到: {args.dfire}")
        print()
        print("下载方式：")
        print(f"  git clone https://github.com/gaiasd/DFireDataset {args.dfire}")
        print("  或 Google Drive: https://drive.google.com/drive/folders/1TbaHn5hxP5hGsrUXNhQ-Dw3sx_L6D0pn")
        print()
        print("也可以手动准备数据集后使用 --data 参数：")
        print("  python scripts/training/train_yolo_smoke.py --data ./my_data/smoke.yaml")
        return

    success = prepare_dfire_dataset(args.dfire, smoke_dir)
    if not success:
        return

    data_yaml = os.path.join(smoke_dir, "dataset.yaml")
    train_smoke(data_yaml, args.epochs, args.batch, args.imgsz, args.output)


if __name__ == "__main__":
    main()
