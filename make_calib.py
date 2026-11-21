"""Collect calibration images for the Hailo compiler (it uses them to pick int8 ranges, so they must look like real input).
python make_calib.py --videos ../test_sample_clean --images drone_data/val/images --out calib --n 600
Takes evenly spaced frames from every video.mp4 under --videos plus random images from --images, saves them as .jpg."""
import argparse, glob, os, random
import cv2

p = argparse.ArgumentParser()
p.add_argument("--videos", default="", help="folder of <video folder>/video.mp4 (or Anti-UAV folders with visible.mp4)")
p.add_argument("--images", default="", help="folder of training/val images")
p.add_argument("--out", default="calib")
p.add_argument("--n", type=int, default=600)
a = p.parse_args()
os.makedirs(a.out, exist_ok=True); random.seed(0)
vids = sorted(glob.glob(os.path.join(a.videos, "**", "*.mp4"), recursive=True)) if a.videos else []
imgs = sorted(glob.glob(os.path.join(a.images, "*.jpg")) + glob.glob(os.path.join(a.images, "*.png"))) if a.images else []
n_vid = a.n // 2 if (vids and imgs) else (a.n if vids else 0)
k = 0
for v in vids:
    cap = cv2.VideoCapture(v); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); per = max(n_vid // len(vids), 1)
    for i in range(per):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((i + 0.5) * total / per)); ok, f = cap.read()
        if ok:
            cv2.imwrite(os.path.join(a.out, f"v{k:05d}.jpg"), f); k += 1
for path in random.sample(imgs, min(a.n - k, len(imgs))):
    f = cv2.imread(path)
    if f is not None:
        cv2.imwrite(os.path.join(a.out, f"i{k:05d}.jpg"), f); k += 1
print(f"saved {k} calibration images to {a.out}/")
