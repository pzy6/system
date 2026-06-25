"""多数据集摔倒预处理: UR Fall + Le2i → 骨架序列"""
import os, sys, argparse, numpy as np, cv2

root = lambda: os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def detect_pose_from_frames(frame_dir, pose_model, max_frames=60):
    frames_list = sorted([f for f in os.listdir(frame_dir) if f.endswith(('.png','.jpg','.jpeg'))])
    if not frames_list: return None
    all_skel = []
    step = max(1, len(frames_list)//max_frames)
    for i in range(0, len(frames_list), step):
        if len(all_skel) >= max_frames: break
        fpath = os.path.join(frame_dir, frames_list[i])
        frame = cv2.imread(fpath)
        if frame is None: continue
        try:
            res = pose_model(frame, conf=0.5, verbose=False, device=0)
            if res[0].keypoints is not None and len(res[0].keypoints.data) > 0:
                all_skel.append(res[0].keypoints.data[0].cpu().numpy())
            else:
                all_skel.append(np.zeros((17,3), dtype=np.float32))
        except:
            all_skel.append(np.zeros((17,3), dtype=np.float32))
    if not all_skel: return None
    skel = np.array(all_skel, dtype=np.float32)
    T = skel.shape[0]
    if T < 60:
        pad = 60 - T
        last = skel[-1:] if T > 0 else np.zeros((1,17,3), dtype=np.float32)
        skel = np.concatenate([skel, np.tile(last, (pad,1,1))])
    elif T > 60:
        skel = skel[:60]
    return skel

def process_le2i(raw_dir, pose_model):
    all_X, all_y = [], []
    for scene in os.listdir(raw_dir):
        sd = os.path.join(raw_dir, scene)
        if not os.path.isdir(sd): continue
        for sub in os.listdir(sd):
            sd2 = os.path.join(sd, sub)
            if not os.path.isdir(sd2): continue
            vd = os.path.join(sd2, 'Videos')
            ad = os.path.join(sd2, 'Annotation_files')
            if not os.path.isdir(vd): continue
            for vf in os.listdir(vd):
                if not vf.endswith('.avi'): continue
                vp = os.path.join(vd, vf)
                base = os.path.splitext(vf)[0]
                af = os.path.join(ad, base + '.txt')
                fall_ranges = []
                if os.path.exists(af):
                    lines = open(af).readlines()
                    if len(lines) >= 2:
                        try:
                            fs = int(lines[0].strip().split()[0])
                            fe = int(lines[1].strip().split()[0])
                            fall_ranges.append((fs, fe))
                        except: pass
                has_fall = len(fall_ranges) > 0
                cap = cv2.VideoCapture(vp)
                skeletons = []; fc = 0
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    fc += 1
                    if fc % 3 != 0: continue
                    try:
                        res = pose_model(frame, conf=0.5, verbose=False, device=0)
                        kp = res[0].keypoints.data[0].cpu().numpy() if (res[0].keypoints is not None and len(res[0].keypoints.data)>0) else np.zeros((17,3),dtype=np.float32)
                        skeletons.append((fc, kp))
                    except:
                        skeletons.append((fc, np.zeros((17,3),dtype=np.float32)))
                cap.release()
                if len(skeletons) < 30: continue
                # 滑窗采样
                for start in range(0, len(skeletons)-60+1, 30):
                    seg = skeletons[start:start+60]
                    skel_arr = np.array([s[1] for s in seg], dtype=np.float32)
                    seg_fall = any(fs<=s[0]<=fe for s in seg for fs,fe in fall_ranges)
                    all_X.append(skel_arr); all_y.append(1 if seg_fall else 0)
                # 摔倒段3倍采样
                for fs, fe in fall_ranges:
                    fall_seg = [(fc,kp) for fc,kp in skeletons if fs<=fc<=fe]
                    if len(fall_seg)<30: continue
                    for _ in range(3):
                        idx = sorted(np.random.choice(len(fall_seg), min(60,len(fall_seg)), replace=True))
                        skel_arr = np.array([fall_seg[i][1] for i in idx[:60]], dtype=np.float32)
                        if len(skel_arr)<60:
                            skel_arr=np.concatenate([skel_arr, np.tile(skel_arr[-1:],(60-len(skel_arr),1,1))])
                        all_X.append(skel_arr); all_y.append(1)
    if not all_X: return None, None
    return np.array(all_X, dtype=np.float32), np.array(all_y, dtype=np.int64)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    r = root()
    out_dir = args.output or os.path.join(r, "data", "fall_dataset", "processed")
    os.makedirs(out_dir, exist_ok=True)

    from ultralytics import YOLO
    pm = YOLO(os.path.join(r, "models", "yolo_pose", "yolo26n-pose.pt"))
    print(f"YOLO-pose loaded")

    all_X, all_y = [], []

    # UR Fall (帧格式)
    ur = os.path.join(r, "data", "fall_dataset", "raw", "UR_fall_detection_dataset_cam0_rgb")
    if os.path.isdir(ur):
        print("\n===== UR Fall =====")
        fd, nd = [], []
        for d in os.listdir(ur):
            dp = os.path.join(ur, d)
            if os.path.isdir(dp):
                if d.lower().startswith('fall'): fd.append((dp,1))
                elif d.lower().startswith('adl'): nd.append((dp,0))
        for dp, lb in fd + nd:
            sk = detect_pose_from_frames(dp, pm)
            if sk is not None: all_X.append(np.expand_dims(sk,0)); all_y.append(lb)
        print(f"UR: {len(fd)}F+{len(nd)}N = {len(all_X)}")

    # Le2i
    le2i = os.path.join(r, "data", "fall_dataset", "raw_le2i")
    if os.path.isdir(le2i):
        print("\n===== Le2i (AVI) =====")
        Xl, yl = process_le2i(le2i, pm)
        if Xl is not None:
            all_X.append(Xl); all_y.append(yl)
            print(f"Le2i: {len(Xl)} seq (fall={yl.sum()})")

    if not all_X:
        print("No data found"); return

    Xa = np.concatenate([np.atleast_3d(x) for x in all_X])
    ya = np.concatenate([np.atleast_1d(y) for y in all_y]).flatten()
    print(f"\nTotal: {len(Xa)} seq (fall={ya.sum()})")

    # 平衡+分割
    fi = np.where(ya==1)[0]; ni = np.where(ya==0)[0]
    mn = min(len(fi), len(ni))
    fi = np.random.choice(fi, mn, replace=False)
    ni = np.random.choice(ni, mn, replace=False)
    idx = np.concatenate([fi, ni]); np.random.shuffle(idx)
    Xa, ya = Xa[idx], ya[idx]

    n = len(Xa); s = int(n*0.8)
    perm = np.random.permutation(n)
    tr, vl = perm[:s], perm[s:]
    np.save(os.path.join(out_dir,"X_train.npy"), Xa[tr])
    np.save(os.path.join(out_dir,"y_train.npy"), ya[tr])
    np.save(os.path.join(out_dir,"X_val.npy"), Xa[vl])
    np.save(os.path.join(out_dir,"y_val.npy"), ya[vl])
    print(f"Saved: train={len(tr)}, val={len(vl)} → {out_dir}/")

if __name__=="__main__":
    main()
