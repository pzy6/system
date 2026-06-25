"""
人脸注册工具 —— 将人员人脸录入系统数据库。

支持两种输入方式：
  1. 从摄像头实时采集人脸
  2. 从图片目录批量导入

人脸数据库结构:
  data/face_db/
    index.json          # 嵌入向量索引
    张大爷/
      img_001.jpg       # 注册图片
      img_002.jpg
      ...

用法：
  # 从摄像头注册（采集 10 张人脸）
  python scripts/enroll_face.py 张大爷 --count 10

  # 从图片目录注册
  python scripts/enroll_face.py 李奶奶 --input ./photos/李奶奶/

  # 列出已注册人员
  python scripts/enroll_face.py --list

  # 删除注册
  python scripts/enroll_face.py --remove 张大爷
"""

import argparse
import os
import sys
import json
import time
import cv2
import numpy as np


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_face_db_path():
    return os.path.join(get_project_root(), "data", "face_db")


def load_recognizer():
    """加载 insightface 识别器"""
    try:
        import insightface
        model = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
        )
        model.prepare(ctx_id=0, det_size=(640, 480))
        return model
    except ImportError:
        print("[错误] insightface 未安装。pip install insightface onnxruntime")
        sys.exit(1)
    except Exception as e:
        print(f"[错误] 模型加载失败: {str(e)}")
        sys.exit(1)


def extract_face_embedding(model, face_img: np.ndarray):
    """从人脸图片提取嵌入向量"""
    faces = model.get(face_img)
    if not faces:
        return None
    # 使用最大的人脸
    best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return best.embedding


def enroll_from_camera(name: str, count: int):
    """从摄像头采集人脸并注册"""
    model = load_recognizer()
    face_db_path = get_face_db_path()
    person_dir = os.path.join(face_db_path, name)
    os.makedirs(person_dir, exist_ok=True)

    # 摄像头优化：低分辨率 + DShow 后端
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        # DShow 可能不可用，回退默认
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[错误] 无法打开摄像头")
            sys.exit(1)

    print(f"正在从摄像头采集 {name} 的人脸...")
    print(f"目标: {count} 张有效人脸图片")
    print("按 's' 保存当前人脸，按 'q' 退出")
    print()

    embeddings = []
    saved_count = 0
    frame_idx = 0
    detect_interval = 3  # 每 3 帧检测一次
    cached_faces = []    # 缓存最近一次检测结果

    while saved_count < count:
        ret, frame = cap.read()
        if not ret:
            continue

        display = frame.copy()
        frame_idx += 1

        # 跳帧检测：每隔 detect_interval 帧才运行 insightface
        if frame_idx % detect_interval == 1:
            cached_faces = model.get(frame)

        # 用缓存的人脸框绘制（无检测开销）
        for face in cached_faces:
            bbox = face.bbox.astype(int)
            cv2.rectangle(
                display,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                (0, 255, 0),
                2,
            )

        # 显示计数和提示
        cv2.putText(
            display,
            f"Saved: {saved_count}/{count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            display,
            "Press 's' to save | 'q' to quit",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
        )

        cv2.imshow("Face Enrollment", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s") and cached_faces:
            # 保存当前帧，用于嵌入提取时用全分辨率
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            img_path = os.path.join(person_dir, f"img_{timestamp}_{saved_count:03d}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"  [{saved_count + 1}/{count}] 已保存: {img_path}")

            # 用原始帧重新提取嵌入（不用缓存，确保精度）
            emb = extract_face_embedding(model, frame)
            if emb is not None:
                embeddings.append(emb)

            saved_count += 1

        elif key == ord("q"):
            print("用户取消")
            break

    cap.release()
    cv2.destroyAllWindows()

    if embeddings:
        save_to_index(name, embeddings)
        print(f"✓ 注册完成: {name} ({len(embeddings)} 个特征向量)")
    else:
        print("[警告] 未采集到有效人脸特征")


def enroll_from_directory(name: str, input_dir: str):
    """从图片目录批量注册人脸"""
    if not os.path.isdir(input_dir):
        print(f"[错误] 目录不存在: {input_dir}")
        sys.exit(1)

    model = load_recognizer()
    face_db_path = get_face_db_path()
    person_dir = os.path.join(face_db_path, name)
    os.makedirs(person_dir, exist_ok=True)

    # 收集所有图片文件
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    img_files = [
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in img_extensions
    ]

    if not img_files:
        print(f"[错误] 目录中未找到图片文件: {input_dir}")
        sys.exit(1)

    print(f"从 {input_dir} 注册 {name} 的人脸...")
    print(f"找到 {len(img_files)} 张图片")
    print()

    embeddings = []
    success = 0

    for i, fname in enumerate(img_files):
        img_path = os.path.join(input_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"  [跳过] 无法读取: {fname}")
            continue

        emb = extract_face_embedding(model, img)
        if emb is not None:
            embeddings.append(emb)
            # 复制图片到数据库
            import shutil
            shutil.copy(img_path, os.path.join(person_dir, fname))
            success += 1
            print(f"  [{success}] ✓ {fname}")
        else:
            print(f"  [跳过] 未检测到人脸: {fname}")

    if embeddings:
        save_to_index(name, embeddings)
        print(f"✓ 注册完成: {name} ({len(embeddings)} 个特征向量, {success} 张图片)")
    else:
        print("[警告] 未从任何图片中检测到人脸")


def save_to_index(name: str, embeddings: list):
    """保存嵌入向量到 index.json"""
    face_db_path = get_face_db_path()
    os.makedirs(face_db_path, exist_ok=True)
    index_path = os.path.join(face_db_path, "index.json")

    # 加载现有索引
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
    else:
        index = {}

    # 更新
    index[name] = [emb.tolist() for emb in embeddings]

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"  索引已保存: {index_path}")


def list_enrolled():
    """列出已注册人员"""
    index_path = os.path.join(get_face_db_path(), "index.json")
    if not os.path.exists(index_path):
        print("人脸数据库为空")
        return

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    print("=" * 40)
    print("已注册人员:")
    print("=" * 40)
    for name, embeddings in index.items():
        img_count = len(embeddings)
        # 统计实际图片数量
        person_dir = os.path.join(get_face_db_path(), name)
        file_count = (
            len([f for f in os.listdir(person_dir)
                 if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))])
            if os.path.isdir(person_dir) else 0
        )
        print(f"  {name}: {img_count} 个特征向量, {file_count} 张图片")
    print(f"共 {len(index)} 人")


def remove_enrollment(name: str):
    """删除已注册人员"""
    index_path = os.path.join(get_face_db_path(), "index.json")
    if not os.path.exists(index_path):
        print("人脸数据库为空")
        return

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    if name not in index:
        print(f"未找到注册人员: {name}")
        return

    del index[name]
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    # 删除图片目录
    import shutil
    person_dir = os.path.join(get_face_db_path(), name)
    if os.path.isdir(person_dir):
        shutil.rmtree(person_dir)

    print(f"已删除: {name}")


def main():
    parser = argparse.ArgumentParser(description="人脸注册工具")
    parser.add_argument("name", nargs="?", type=str, help="人员姓名")
    parser.add_argument(
        "--count", type=int, default=10, help="摄像头采集数量（默认 10）"
    )
    parser.add_argument(
        "--input", type=str, default=None, help="图片目录路径（从目录注册时使用）"
    )
    parser.add_argument(
        "--list", action="store_true", help="列出已注册人员"
    )
    parser.add_argument(
        "--remove", type=str, default=None, help="删除指定人员的注册信息"
    )
    args = parser.parse_args()

    if args.list:
        list_enrolled()
        return

    if args.remove:
        remove_enrollment(args.remove)
        return

    if not args.name:
        parser.print_help()
        print()
        print("示例:")
        print("  python scripts/enroll_face.py 张大爷 --count 10")
        print("  python scripts/enroll_face.py 李奶奶 --input ./photos/李奶奶/")
        print("  python scripts/enroll_face.py --list")
        print("  python scripts/enroll_face.py --remove 张大爷")
        return

    if args.input:
        enroll_from_directory(args.name, args.input)
    else:
        enroll_from_camera(args.name, args.count)


if __name__ == "__main__":
    main()
