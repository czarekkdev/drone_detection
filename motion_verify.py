"""Add motion-based detections: each moving blob (from motion.py output) that the detector has not already
covered is cut out of the full-resolution frame and checked by the model. Confirmed crops become extra
detections. Output = detection file in the usual format (feed it to run_tracker.py).
python motion_verify.py --dets preds/hf_skyguard --motion motion --videos ../test_sample_clean --out preds/sky_motion"""
import argparse, os, glob, time
import cv2, numpy as np, torch
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("--dets", required=True); p.add_argument("--motion", required=True)
p.add_argument("--videos", required=True); p.add_argument("--out", required=True)
p.add_argument("--weights", default="../pretrained/drone-detection-yolov11/best.pt")
p.add_argument("--crop", type=int, default=320); p.add_argument("--imgsz", type=int, default=320)
p.add_argument("--topk", type=int, default=8, help="max crops per frame (strongest blobs first)")
p.add_argument("--covered", type=float, default=0.05, help="detections above this conf count as covering a blob")
p.add_argument("--conf", type=float, default=0.05)
p.add_argument("--device", default="")
a = p.parse_args()
dev = a.device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
model = YOLO(a.weights); os.makedirs(a.out, exist_ok=True)


def read(path, n_after=1):
    D = {}
    if os.path.exists(path):
        for l in open(path):
            v = l.strip().split(",")
            if len(v) > 5:
                D.setdefault(int(float(v[0])), []).append([float(x) for x in v[n_after:]])
    return D


def iou(a_, b):
    ix = max(0, min(a_[2], b[2]) - max(a_[0], b[0])); iy = max(0, min(a_[3], b[3]) - max(a_[1], b[1]))
    i = ix * iy; return i / ((a_[2] - a_[0]) * (a_[3] - a_[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


for f in sorted(glob.glob(os.path.join(a.dets, "*.txt"))):
    name = os.path.basename(f)[:-4]
    D = {k: [(x, y, x + w, y + h, c) for _, x, y, w, h, c in v] for k, v in read(f).items()}
    M = read(os.path.join(a.motion, name + ".txt"))
    cap = cv2.VideoCapture(os.path.join(a.videos, name, "video.mp4"))
    out, n, added, crops_total, t0 = [], 0, 0, 0, time.time()
    while True:
        ok, img = cap.read()
        if not ok:
            break
        n += 1; H, W = img.shape[:2]
        dets = list(D.get(n, []))
        cover = [d for d in dets if d[4] >= a.covered]
        blobs = sorted(M.get(n, []), key=lambda b: -b[4])
        centers = []
        for x, y, w, h, s in blobs:
            cx, cy = x + w / 2, y + h / 2
            if any(d[0] - 10 <= cx <= d[2] + 10 and d[1] - 10 <= cy <= d[3] + 10 for d in cover):
                continue
            if any(abs(cx - u) < a.crop / 3 and abs(cy - v) < a.crop / 3 for u, v in centers):
                continue  # already inside another crop
            centers.append((cx, cy))
            if len(centers) >= a.topk:
                break
        if centers:
            tiles, offs = [], []
            for cx, cy in centers:
                x0 = int(min(max(cx - a.crop / 2, 0), W - a.crop)); y0 = int(min(max(cy - a.crop / 2, 0), H - a.crop))
                tiles.append(img[y0:y0 + a.crop, x0:x0 + a.crop]); offs.append((x0, y0))
            crops_total += len(tiles)
            for r, (x0, y0) in zip(model.predict(tiles, imgsz=a.imgsz, conf=a.conf, device=dev, verbose=False), offs):
                for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                    b = (x1 + x0, y1 + y0, x2 + x0, y2 + y0, c)
                    # skip boxes touching the crop border (probably cut-off objects) unless at image border
                    if any(iou(b, d) > 0.3 for d in dets):
                        continue
                    dets.append(b); added += 1
        for x1, y1, x2, y2, c in dets:
            out.append(f"{n},1,{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{c:.4f}")
    open(os.path.join(a.out, name + ".txt"), "w").write("\n".join(out) + "\n")
    print(f"{name}: {n} frames, {crops_total / max(n, 1):.1f} crops/frame, {added} boxes added, {n / (time.time() - t0):.1f} fps")
