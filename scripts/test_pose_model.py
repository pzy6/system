"""测试姿态检测模型 yolo11n-pose"""
from ultralytics import YOLO
import cv2, os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = YOLO(os.path.join(root, 'models', 'yolo_pose', 'yolo11n-pose.pt'))

img = cv2.imread(os.path.join(root, 'test_pose.jpg'))
if img is None:
    print("请先放一张 test_pose.jpg 到项目根目录")
    exit(1)

results = model(img, conf=0.5)

for r in results:
    kps = r.keypoints
    if kps is not None and len(kps.data) > 0:
        n_persons = len(kps.data)
        n_kps = kps.data.shape[1]
        print(f'检测到 {n_persons} 人，每人 {n_kps} 个关键点')
        for i, kp in enumerate(kps.data[0][:5]):
            print(f'  关键点{i}: x={kp[0]:.0f} y={kp[1]:.0f} conf={kp[2]:.2f}')
    else:
        print('未检测到人体 — 确保图片中有人且光线充足')

results[0].save(os.path.join(root, 'test_pose_result.jpg'))
print('骨架图已保存: test_pose_result.jpg')
