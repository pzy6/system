"""银发守护者系统 性能基准测试"""
import time, sys, os, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from algorithms.yolo_pose_detector import YOLOPoseDetector
from algorithms.yolo_detector import YOLODetector
from algorithms.fall_detector import FallDetector


def bench(name, fn, warmup=3, runs=20):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    avg = np.mean(times)
    f = runs / sum(t / 1000 for t in times)  # FPS
    print(f"  {name:28s} {avg:6.1f} ms  ({f:.1f} FPS)")
    return avg, times


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 128

    print("=" * 50)
    print("银发守护者 性能基准测试")
    print("=" * 50)

    # YOLO 人脸+烟雾
    det = YOLODetector({
        'model_path': os.path.join(root, 'models', 'yolo_face_smoke'),
        'frame_skip': 1, 'device': 0,
    })
    bench("YOLO26 人脸+烟雾检测", lambda: det.detect(frame))
    print(f"       ({det.get_status()['message'][:50]})")

    # YOLO 姿态
    pose = YOLOPoseDetector({
        'model_path': os.path.join(root, 'models', 'yolo_pose'),
        'device': 0,
    })
    bench("YOLO26 姿态检测", lambda: pose.detect(frame))
    print(f"       ({pose.get_status()['message'][:50]})")

    # 摔倒
    fd = FallDetector()
    kps = [{'part': 'Neck', 'x': 320, 'y': 150, 'confidence': 0.9},
           {'part': 'RHip', 'x': 350, 'y': 280, 'confidence': 0.9},
           {'part': 'LHip', 'x': 290, 'y': 280, 'confidence': 0.9}]
    bench("摔倒检测 (纯规则)", lambda: fd.detect(kps, 0))

    print()
    print("注: 值已预热，GPU 模式，640x480")


if __name__ == '__main__':
    main()
