"""Cache whole-frame SkyGuard detections (imgsz 1280, conf 0.01, class drone) for Anti-UAV-RGBT visible videos.
python cache_auav.py --src "../Anti-UAV-RGBT training data/val" --out preds/auav_val_sky --max-frames 500 --part 0/2"""
import argparse, os, glob, time
import cv2, torch
from ultralytics import YOLO
p = argparse.ArgumentParser()
p.add_argument("--src", required=True); p.add_argument("--out", required=True)
p.add_argument("--weights", default="../pretrained/drone-detection-yolov11/best.pt")
p.add_argument("--max-frames", type=int, default=500); p.add_argument("--part", default="0/1")
a = p.parse_args()
k, K = map(int, a.part.split("/"))
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = YOLO(a.weights); classes = [i for i, nm in model.names.items() if nm.lower() == "drone"]
os.makedirs(a.out, exist_ok=True)
vids = sorted(glob.glob(os.path.join(a.src, "*/visible.mp4")))[k::K]
for vp in vids:
    name = vp.split(os.sep)[-2]; op = os.path.join(a.out, name + ".txt")
    if os.path.exists(op):
        continue
    cap = cv2.VideoCapture(vp); lines = []; n = 0; t0 = time.time()
    while n < a.max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        n += 1
        r = model.predict(fr, imgsz=1280, conf=0.01, device=dev, classes=classes, verbose=False)[0]
        for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
            lines.append(f"{n},1,{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{c:.4f}")
    open(op + ".tmp", "w").write("\n".join(lines) + ("\n" if lines else "")); os.rename(op + ".tmp", op)
    print(f"{name}: {n} frames {len(lines)} boxes {n / (time.time() - t0):.1f} fps", flush=True)
