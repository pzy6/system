"""测试烟雾检测模型 smoke_fire.pt"""
from ultralytics import YOLO
import cv2, os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = YOLO(os.path.join(root, 'models', 'yolo_face_smoke', 'yolo26n_smoke.pt'))

img = cv2.imread(os.path.join(root, 'test_smoke.jpg'))
if img is None:
    print("请先放一张 test_smoke.jpg 到项目根目录")
    exit(1)

results = model(img, conf=0.25)

found = False
for r in results:
    if r.boxes is not None and len(r.boxes) > 0:
        for b in r.boxes:
            cls_name = model.names[int(b.cls[0])]
            conf = float(b.conf[0])
            print(f'检测: {cls_name}  置信度: {conf:.2f}')
            found = True

if not found:
    print('未检测到烟雾/火焰 — 换一张烟雾更明显的图片')

results[0].save(os.path.join(root, 'test_smoke_result.jpg'))
print('标注图已保存: test_smoke_result.jpg')
