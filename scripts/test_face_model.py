"""测试人脸检测模型 face_yolov8n.pt"""
from ultralytics import YOLO
import cv2, os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = YOLO(os.path.join(root, 'models', 'yolo_face_smoke', 'face_yolov8n.pt'))

img = cv2.imread(os.path.join(root, 'test_face.jpg'))
if img is None:
    print("请先放一张 test_face.jpg 到项目根目录（D:\\Trae文件\\银发守护者系统\\）")
    exit(1)

results = model(img, conf=0.25)

for r in results:
    n = len(r.boxes) if r.boxes is not None else 0
    print(f'检测到 {n} 个人脸')
    if r.boxes is not None:
        for box in r.boxes:
            conf = float(box.conf[0])
            print(f'  置信度: {conf:.2f}')

results[0].save(os.path.join(root, 'test_face_result.jpg'))
print('标注图已保存: test_face_result.jpg')
