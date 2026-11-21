"""Fine-tune the pretrained SkyGuard YOLO11n on hackathon_dataset + Anti-UAV.
python finetune.py --weights ../pretrained/drone-detection-yolov11/best.pt --epochs 15 --imgsz 960"""
import argparse, os
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("--weights", required=True)
p.add_argument("--data", default="drone_ft.yaml")
p.add_argument("--epochs", type=int, default=15)
p.add_argument("--imgsz", type=int, default=960)
p.add_argument("--batch", type=int, default=16)
p.add_argument("--device", default="mps")
p.add_argument("--fraction", type=float, default=1.0)
p.add_argument("--name", default="ft_skyguard")
a = p.parse_args()
YOLO(a.weights).train(
    data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device, fraction=a.fraction,
    project=os.path.abspath("runs"), name=a.name, exist_ok=True, seed=0, workers=6, plots=False,
    optimizer="SGD", lr0=0.002, lrf=0.1, warmup_epochs=1, cos_lr=True,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.5, scale=0.5, fliplr=0.5, mosaic=1.0, close_mosaic=3,
)
