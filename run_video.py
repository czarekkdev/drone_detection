"""Run a model on every frame of each video and write result files.
python run_video.py --weights runs/rgb/weights/best.pt --videos <test_sample_clean> --out preds/rgb"""
import argparse, os, time, glob
import cv2
from ultralytics import YOLO
import mlayer  # noqa: F401  (needed to unpickle the multi-layer model)

p = argparse.ArgumentParser()
p.add_argument("--weights", required=True)
p.add_argument("--videos", required=True)
p.add_argument("--out", required=True)
p.add_argument("--imgsz", type=int, default=640)
p.add_argument("--conf", type=float, default=0.01)
p.add_argument("--device", default="", help="mps / cuda / cpu (default: best available)")
p.add_argument("--keep", default="", help="comma-separated class names to keep (default: all)")
a = p.parse_args()
import torch
dev = a.device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
os.makedirs(a.out, exist_ok=True)
model = YOLO(a.weights)
classes = None
if a.keep:
    want = {k.strip().lower() for k in a.keep.split(",") if k.strip()}
    classes = [i for i, nm in model.names.items() if nm.lower() in want]
    missing = want - {model.names[i].lower() for i in classes}
    if missing or not classes:
        raise SystemExit(f"--keep {sorted(missing)} not in model classes {model.names}")
    print("keeping classes:", {i: model.names[i] for i in classes})
for vdir in sorted(glob.glob(os.path.join(a.videos, "*/"))):
    name = os.path.basename(os.path.normpath(vdir))
    cap = cv2.VideoCapture(os.path.join(vdir, "video.mp4"))
    lines, n, t0 = [], 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1  # frame numbers start at 1
        r = model.predict(frame, imgsz=a.imgsz, conf=a.conf, device=dev, classes=classes, verbose=False)[0]
        for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
            lines.append(f"{n},1,{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{c:.4f}")
    with open(os.path.join(a.out, name + ".txt"), "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    print(f"{name}: {n} frames, {len(lines)} boxes, {n / (time.time() - t0):.1f} fps (this machine)")
