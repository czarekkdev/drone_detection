"""Full pipeline, frame by frame (online, same logic as the Pi will run):
  1. detector on the whole frame
  2. PREDICTION crops: for each tracked drone not detected near its predicted spot -> full-res crop there, model checks it
  3. MOTION crops: moving spots not explained by 1-2, nearest to a predicted path first, then strongest -> crop + check
  4. tracker (ids 1/2, coasting, re-id)
Offline mode reuses cached whole-frame detections (--dets) and motion blobs (--motion) so experiments are fast;
crops are always computed live.
python pipeline.py --videos ../test_sample_clean --dets preds/hf_skyguard --motion motion --out preds/full [--no-pred] [--no-motion]"""
import argparse, os, glob, time
import cv2, numpy as np, torch
from ultralytics import YOLO
from tracker import TwoDroneTracker

p = argparse.ArgumentParser()
p.add_argument("--videos", required=True); p.add_argument("--out", required=True)
p.add_argument("--dets", default=""); p.add_argument("--motion", default="")
p.add_argument("--weights", default="../pretrained/drone-detection-yolov11/best.pt")
p.add_argument("--imgsz", type=int, default=1280, help="whole-frame size (live mode)")
p.add_argument("--crop", type=int, default=320); p.add_argument("--crop-imgsz", type=int, default=320)
p.add_argument("--motion-topk", type=int, default=4)
p.add_argument("--min-persist", type=int, default=1, help="motion spot must be seen in this many of the 2 previous frames")
p.add_argument("--no-pred", action="store_true"); p.add_argument("--no-motion", action="store_true")
p.add_argument("--device", default="")
a = p.parse_args()
dev = a.device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
model = YOLO(a.weights); os.makedirs(a.out, exist_ok=True)


def read(path):
    D = {}
    if path and os.path.exists(path):
        for l in open(path):
            v = l.strip().split(",")
            if len(v) > 5:
                D.setdefault(int(float(v[0])), []).append([float(x) for x in v[1:]])
    return D


def iou(a_, b):
    ix = max(0, min(a_[2], b[2]) - max(a_[0], b[0])); iy = max(0, min(a_[3], b[3]) - max(a_[1], b[1]))
    i = ix * iy; return i / ((a_[2] - a_[0]) * (a_[3] - a_[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def run_crops(img, centers, size):
    H, W = img.shape[:2]; tiles, offs = [], []
    for cx, cy in centers:
        x0 = int(min(max(cx - size / 2, 0), max(W - size, 0))); y0 = int(min(max(cy - size / 2, 0), max(H - size, 0)))
        tiles.append(img[y0:y0 + size, x0:x0 + size]); offs.append((x0, y0))
    out = []
    if tiles:
        for r, (x0, y0) in zip(model.predict(tiles, imgsz=a.crop_imgsz, conf=0.05, device=dev, verbose=False), offs):
            for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                out.append((x1 + x0, y1 + y0, x2 + x0, y2 + y0, c))
    return out


def merge(dets, new):
    for b in new:
        same = [i for i, d in enumerate(dets) if iou(b, d) > 0.3]
        if same:
            i = same[0]
            if b[4] > dets[i][4]:
                dets[i] = b
        else:
            dets.append(b)
    return dets


stats = {}
for vdir in sorted(glob.glob(os.path.join(a.videos, "*/"))):
    name = os.path.basename(os.path.normpath(vdir))
    D = {k: [(x, y, x + w, y + h, c) for _, x, y, w, h, c in v] for k, v in read(os.path.join(a.dets, name + ".txt") if a.dets else "").items()}
    M = read(os.path.join(a.motion, name + ".txt") if a.motion else "")
    cap = cv2.VideoCapture(os.path.join(vdir, "video.mp4"))
    T = TwoDroneTracker(); lines = []; n = 0; npred = nmot = 0; t0 = time.time()
    while True:
        ok, img = cap.read()
        if not ok:
            break
        n += 1
        dets = list(D.get(n, [])) if a.dets else [(*b, c) for b, c in zip(*[(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()) for r in [model.predict(img, imgsz=a.imgsz, conf=0.01, device=dev, verbose=False)[0]]][0])]
        preds = T.predictions()
        # 2) prediction crops
        if not a.no_pred:
            centers = []
            for tid, pb, miss in preds:
                if any(iou(pb, d) > 0.2 and d[4] >= T.p["low"] for d in dets):
                    continue  # detector already sees it
                centers.append(((pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2))
            if centers:
                npred += len(centers); dets = merge(dets, run_crops(img, centers, a.crop))
        # 3) motion crops
        if not a.no_motion and M:
            pc = [((pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2) for _, pb, _ in preds]
            cand = []
            def seen(f, cx, cy):
                return any(abs(cx - (b[0] + b[2] / 2)) < 60 and abs(cy - (b[1] + b[3] / 2)) < 60 for b in M.get(f, []))
            for x, y, w, h, s, *_ in M.get(n, []):
                cx, cy = x + w / 2, y + h / 2
                if a.min_persist and (seen(n - 1, cx, cy) + seen(n - 2, cx, cy)) < a.min_persist:
                    continue  # not moving steadily: flicker / leaves
                if any(d[0] - 10 <= cx <= d[2] + 10 and d[1] - 10 <= cy <= d[3] + 10 for d in dets if d[4] >= 0.05):
                    continue
                dist = min([np.hypot(cx - u, cy - v) for u, v in pc], default=1e9)
                cand.append((0 if dist < 300 else 1, -s, cx, cy))
            cand.sort(); centers = []
            for _, _, cx, cy in cand:
                if any(abs(cx - u) < a.crop / 3 and abs(cy - v) < a.crop / 3 for u, v in centers):
                    continue
                centers.append((cx, cy))
                if len(centers) >= a.motion_topk:
                    break
            if centers:
                nmot += len(centers); dets = merge(dets, run_crops(img, centers, a.crop))
        for f2, i, (x1, y1, x2, y2), c in T.step(n, dets):
            lines.append(f"{f2},{i},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{c:.4f}")
    lines.sort(key=lambda l: (int(l.split(",")[0]), int(l.split(",")[1])))
    open(os.path.join(a.out, name + ".txt"), "w").write("\n".join(lines) + "\n")
    print(f"{name}: {n} frames, prediction crops/frame {npred / n:.2f}, motion crops/frame {nmot / n:.2f}, {n / (time.time() - t0):.1f} fps")
