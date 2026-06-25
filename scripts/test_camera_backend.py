"""测试哪个摄像头后端稳定"""
import cv2, time

backends = [(cv2.CAP_DSHOW, 'DSHOW'), (cv2.CAP_MSMF, 'MSMF'), (0, 'ANY')]

for backend, name in backends:
    cap = cv2.VideoCapture(0, backend)
    ok = cap.isOpened()
    if ok:
        fails = 0
        for i in range(10):
            ret, _ = cap.read()
            if not ret:
                fails += 1
        print(f'{name}: connected=OK, read_fails={fails}/10')
    else:
        print(f'{name}: FAILED to open')
    cap.release()
