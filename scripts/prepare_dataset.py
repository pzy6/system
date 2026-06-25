"""
人脸+烟雾数据集准备工具

支持三种模式：
  1. --auto: 自动从 Roboflow 下载公开数据集（需 ROBOFLOW_API_KEY）
  2. --merge: 合并已有的人脸和烟雾 YOLO 数据集
  3. --from-images: 从原始图片目录生成空白 YOLO 标注（用于手动标注）

用法：
  # 自动下载（需先注册 Roboflow 获取 API key）
  python scripts/prepare_dataset.py --auto --api-key YOUR_KEY

  # 合并已有数据集
  python scripts/prepare_dataset.py --merge --face-dir ./data/face_dataset --smoke-dir ./data/smoke_dataset

  # 生成空标注模板
  python scripts/prepare_dataset.py --from-images --image-dir ./images/
"""

import argparse
import os
import sys
import shutil
import random
import yaml
from typing import List, Tuple


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ====================================================================
# 模式 1: 自动下载（Roboflow Universe）
# ====================================================================

# 公开数据集（workspace/project_name/version 格式）
PUBLIC_FACE_DATASETS = [
    ("foduucom", "face-detection-f67c2", 1),
]
PUBLIC_SMOKE_DATASETS = [
    ("shubhampardeshi", "smoke-detection-tqy3d", 1),
    ("roboflow-100", "smoke-detection-vaede", 1),
]


def download_roboflow_dataset(api_key: str, workspace: str, project_name: str,
                               version: int, output_dir: str) -> bool:
    """通过 Roboflow API 下载公开数据集（YOLOv11 格式）"""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("  [提示] roboflow 未安装。pip install roboflow")
        return False

    rf = Roboflow(api_key=api_key)
    try:
        ws = rf.workspace(workspace)
        project = ws.project(project_name)
        version_obj = project.version(version)
        dataset = version_obj.download("yolov11", location=output_dir)
        print(f"  下载完成: {workspace}/{project_name} v{version}")
        return True
    except Exception as e:
        msg = str(e)[:120]
        print(f"  下载失败 {workspace}/{project_name}: {msg}")
        return False


def auto_download(api_key: str, output_dir: str) -> bool:
    """自动下载人脸+烟雾数据集并合并"""
    root = get_project_root()
    data_dir = os.path.join(root, "data", "face_smoke")
    tmp_dir = os.path.join(root, "data", "_tmp_downloads")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    print("=" * 60)
    print("自动下载人脸+烟雾数据集")
    print("=" * 60)

    # 下载人脸
    print("\n[1/2] 下载人脸检测数据集...")
    face_ok = False
    face_dir = os.path.join(tmp_dir, "face")
    for ws, proj, ver in PUBLIC_FACE_DATASETS:
        if download_roboflow_dataset(api_key, ws, proj, ver, face_dir):
            face_ok = True
            break

    if not face_ok:
        print("\n所有公开人脸数据集下载失败。请手动下载：")
        print("  https://universe.roboflow.com/ → 搜索 'face detection' → YOLOv11 格式")
        return False

    # 下载烟雾
    print("\n[2/2] 下载烟雾检测数据集...")
    smoke_ok = False
    smoke_dir = os.path.join(tmp_dir, "smoke")
    for ws, proj, ver in PUBLIC_SMOKE_DATASETS:
        if download_roboflow_dataset(api_key, ws, proj, ver, smoke_dir):
            smoke_ok = True
            break

    if not smoke_ok:
        print("\n所有公开烟雾数据集下载失败。请手动下载：")
        print("  https://universe.roboflow.com/ → 搜索 'smoke detection' → YOLOv11 格式")
        return False

    # 合并
    print("\n合并数据集...")
    merge_datasets(
        face_dir=face_dir,
        smoke_dir=smoke_dir,
        output_dir=data_dir,
        face_class_id=0,
        smoke_class_id=1,
    )

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("\n完成! 数据集已保存到:", data_dir)
    return True


# ====================================================================
# 模式 2: 合并已有数据集
# ====================================================================

def find_yolo_images_and_labels(dataset_dir: str) -> Tuple[str, str]:
    """在 YOLO 格式数据目录中查找 images 和 labels 子目录"""
    images_dir = None
    labels_dir = None

    for root, dirs, files in os.walk(dataset_dir):
        for d in dirs:
            if d.lower() in ("images", "train", "test", "val"):
                test_path = os.path.join(root, d)
                # 检查是否包含图片
                for f in os.listdir(test_path)[:5]:
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                        images_dir = test_path
                        break
            if d.lower() == "labels":
                labels_dir = os.path.join(root, d)

    # 如果没找到明确的 labels 目录，尝试在 images 同级找
    if images_dir and labels_dir is None:
        parent = os.path.dirname(images_dir)
        for d in os.listdir(parent):
            if d.lower() == "labels":
                labels_dir = os.path.join(parent, d)
                break

    return images_dir or dataset_dir, labels_dir or dataset_dir


def merge_datasets(face_dir: str, smoke_dir: str, output_dir: str,
                   face_class_id: int = 0, smoke_class_id: int = 1,
                   train_ratio: float = 0.85):
    """合并人脸和烟雾 YOLO 数据集"""
    # 创建输出目录
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

    face_img_dir, face_lbl_dir = find_yolo_images_and_labels(face_dir)
    smoke_img_dir, smoke_lbl_dir = find_yolo_images_and_labels(smoke_dir)

    print(f"  人脸数据: images={face_img_dir}, labels={face_lbl_dir}")
    print(f"  烟雾数据: images={smoke_img_dir}, labels={smoke_lbl_dir}")

    # 收集所有图片文件
    face_images = _collect_images(face_img_dir)
    smoke_images = _collect_images(smoke_img_dir)

    print(f"  人脸图片: {len(face_images)} 张")
    print(f"  烟雾图片: {len(smoke_images)} 张")

    if len(face_images) == 0 and len(smoke_images) == 0:
        print("[错误] 未找到任何图片文件")
        return

    # 合并并分配 train/val
    all_files = []  # [(img_path, lbl_path, class_prefix)]

    for img_path in face_images:
        lbl_path = _find_label(img_path, face_img_dir, face_lbl_dir)
        all_files.append((img_path, lbl_path, "face"))

    for img_path in smoke_images:
        lbl_path = _find_label(img_path, smoke_img_dir, smoke_lbl_dir)
        all_files.append((img_path, lbl_path, "smoke"))

    random.shuffle(all_files)
    split_idx = int(len(all_files) * train_ratio)

    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    # 复制文件并重映射类别
    for split_name, file_list in [("train", train_files), ("val", val_files)]:
        for i, (img_path, lbl_path, prefix) in enumerate(file_list):
            # 图片
            ext = os.path.splitext(img_path)[1]
            new_name = f"{prefix}_{i:05d}"
            new_img_path = os.path.join(
                output_dir, "images", split_name, f"{new_name}{ext}"
            )
            shutil.copy2(img_path, new_img_path)

            # 标签（重映射类别 ID）
            new_lbl_path = os.path.join(
                output_dir, "labels", split_name, f"{new_name}.txt"
            )
            _remap_labels(
                lbl_path, new_lbl_path,
                old_class_id=0,
                new_class_id=(
                    face_class_id if prefix == "face" else smoke_class_id
                ),
            )

    print(f"  训练集: {len(train_files)} 张")
    print(f"  验证集: {len(val_files)} 张")

    # 生成 dataset.yaml
    dataset_yaml = {
        "path": os.path.abspath(output_dir).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {face_class_id: "face", smoke_class_id: "smoke"},
        "nc": 2,
    }
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dataset_yaml, f, allow_unicode=True, sort_keys=False)

    print(f"  dataset.yaml: {yaml_path}")


def _collect_images(directory: str) -> List[str]:
    """收集目录中的所有图片文件"""
    images = []
    if not os.path.isdir(directory):
        return images
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                images.append(os.path.join(root, f))
    return images


def _find_label(img_path: str, img_base: str, lbl_base: str) -> str:
    """查找图片对应的 YOLO 标签文件"""
    # 尝试多种标签路径推理方式
    rel = os.path.relpath(img_path, img_base)
    name_no_ext = os.path.splitext(rel)[0]

    candidates = [
        os.path.join(lbl_base, f"{name_no_ext}.txt"),
        os.path.join(lbl_base.replace("images", "labels"), f"{name_no_ext}.txt"),
    ]

    for c in candidates:
        if os.path.exists(c):
            return c

    # 返回一个不存在的路径（无标注文件时跳过）
    return candidates[0]


def _remap_labels(src_path: str, dst_path: str, old_class_id: int, new_class_id: int):
    """复制标签文件并重映射类别 ID"""
    if not os.path.exists(src_path):
        # 无标签文件 → 创建空文件
        open(dst_path, "w").close()
        return
    with open(src_path, "r") as f:
        lines = f.readlines()
    with open(dst_path, "w") as f:
        for line in lines:
            parts = line.strip().split()
            if parts:
                # 将原有的 class_id 替换为新 ID
                parts[0] = str(new_class_id)
                f.write(" ".join(parts) + "\n")


# ====================================================================
# 模式 3: 从图片目录生成模板
# ====================================================================

def from_images(image_dir: str):
    """从原始图片目录生成 YOLO 格式空标注模板"""
    root = get_project_root()
    data_dir = os.path.join(root, "data", "face_smoke")

    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        os.makedirs(os.path.join(data_dir, sub), exist_ok=True)

    images = _collect_images(image_dir)
    if not images:
        print(f"[错误] 目录中未找到图片: {image_dir}")
        return

    random.shuffle(images)
    split_idx = int(len(images) * 0.85)

    for split, imgs in [("train", images[:split_idx]), ("val", images[split_idx:])]:
        for i, img_path in enumerate(imgs):
            ext = os.path.splitext(img_path)[1]
            new_name = f"img_{i:05d}"
            shutil.copy2(
                img_path,
                os.path.join(data_dir, "images", split, f"{new_name}{ext}"),
            )
            # 创建空标签文件（待手动标注）
            open(
                os.path.join(data_dir, "labels", split, f"{new_name}.txt"), "w"
            ).close()

    print(f"已创建 {len(images)} 个空标注模板到: {data_dir}")
    print("请使用 LabelImg 或 Roboflow Annotate 进行标注。")


# ====================================================================
# 主入口
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description="人脸+烟雾数据集准备工具")
    parser.add_argument("--auto", action="store_true", help="自动从 Roboflow 下载")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Roboflow API key（也可设置环境变量 ROBOFLOW_API_KEY）")
    parser.add_argument("--merge", action="store_true", help="合并已有数据集")
    parser.add_argument("--face-dir", type=str, default=None, help="人脸数据集目录")
    parser.add_argument("--smoke-dir", type=str, default=None, help="烟雾数据集目录")
    parser.add_argument("--from-images", action="store_true", help="从图片生成标注模板")
    parser.add_argument("--image-dir", type=str, default=None, help="原始图片目录")
    parser.add_argument("--output", type=str, default=None, help="输出目录")
    args = parser.parse_args()

    root = get_project_root()

    if args.auto:
        api_key = args.api_key or os.environ.get("ROBOFLOW_API_KEY")
        if not api_key:
            print("=" * 60)
            print("需要 Roboflow API Key 才能自动下载")
            print("=" * 60)
            print()
            print("获取步骤：")
            print("  1. 访问 https://app.roboflow.com/ 注册免费账号")
            print("  2. 进入 Settings → API Keys → 复制密钥")
            print("  3. 运行: python scripts/prepare_dataset.py --auto --api-key YOUR_KEY")
            print()
            print("或者手动下载（推荐，更稳定）：")
            print("  1. 人脸: https://universe.roboflow.com/ → 搜索 'face detection'")
            print("  2. 烟雾: https://universe.roboflow.com/ → 搜索 'smoke detection'")
            print("  3. 都选 YOLOv11 格式下载")
            print(f"  4. 解压后运行: python scripts/prepare_dataset.py --merge --face-dir <人脸目录> --smoke-dir <烟雾目录>")
            return

        output_dir = args.output or os.path.join(root, "data", "face_smoke")
        auto_download(api_key, output_dir)
        return

    if args.merge:
        if not args.face_dir or not args.smoke_dir:
            print("[错误] --merge 需要 --face-dir 和 --smoke-dir")
            return
        output_dir = args.output or os.path.join(root, "data", "face_smoke")
        merge_datasets(args.face_dir, args.smoke_dir, output_dir)
        return

    if args.from_images:
        if not args.image_dir:
            print("[错误] --from-images 需要 --image-dir")
            return
        from_images(args.image_dir)
        return

    parser.print_help()
    print()
    print("示例:")
    print("  python scripts/prepare_dataset.py --auto --api-key YOUR_KEY")
    print("  python scripts/prepare_dataset.py --merge --face-dir ./face_data --smoke-dir ./smoke_data")
    print("  python scripts/prepare_dataset.py --from-images --image-dir ./my_images/")


if __name__ == "__main__":
    main()
