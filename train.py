"""Train one variant.  python train.py --variant rgb|ml --data drone.yaml"""
import argparse, os
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from mlayer import make_multilayer


class MLTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        m = super().get_model(cfg, weights, verbose)  # pretrained weights loaded first
        return make_multilayer(m)


p = argparse.ArgumentParser()
p.add_argument("--variant", choices=["rgb", "ml"], required=True)
p.add_argument("--data", default="drone.yaml")
p.add_argument("--model", default="yolo11n.pt")
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--imgsz", type=int, default=640)
p.add_argument("--batch", type=int, default=16)
p.add_argument("--device", default="mps")
p.add_argument("--fraction", type=float, default=1.0)
a = p.parse_args()

YOLO(a.model).train(
    data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
    fraction=a.fraction, project=os.path.abspath("runs"), name=a.variant, exist_ok=True, seed=0,
    deterministic=False, workers=4, patience=100, plots=False,
    # identical augmentation for both: color jitter + some grayscale
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.5, degrees=0, scale=0.5, fliplr=0.5, mosaic=1.0,
    close_mosaic=5,
    trainer=MLTrainer if a.variant == "ml" else DetectionTrainer,
)
