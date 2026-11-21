"""Offline: apply the tracker to a folder of detection result files.
python run_tracker.py --dets preds/hf_skyguard --out preds/trk [--videos test_sample_clean  (enables appearance re-id)]"""
import argparse, os, glob
from tracker import TwoDroneTracker, appearance

p = argparse.ArgumentParser()
p.add_argument("--dets", required=True); p.add_argument("--out", required=True)
p.add_argument("--videos", default="")
for k, v in dict(high=0.6, low=0.05, new=0.3, confirm=3, coast=3, gate=2.5, w_feat=1.0, w_pos=1.0, new_id_cost=1.0).items():
    p.add_argument("--" + k, type=type(v), default=v)
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
for f in sorted(glob.glob(os.path.join(a.dets, "*.txt"))):
    name = os.path.basename(f)[:-4]; D = {}
    for line in open(f):
        v = line.split(",")
        if len(v) >= 7:
            x, y, w, h = map(float, v[2:6]); D.setdefault(int(v[0]), []).append((x, y, x + w, y + h, float(v[6])))
    cap = None
    if a.videos:
        import cv2
        cap = cv2.VideoCapture(os.path.join(a.videos, name, "video.mp4"))
    T = TwoDroneTracker(high=a.high, low=a.low, new=a.new, confirm=a.confirm, coast=a.coast, gate=a.gate,
                        w_feat=a.w_feat, w_pos=a.w_pos, new_id_cost=a.new_id_cost)
    n = max(D) if D else 0
    if cap is not None:
        n = int(cap.get(7)) or n
    lines = []
    for fr in range(1, n + 1):
        dets = D.get(fr, [])
        feats = None
        if cap is not None:
            ok, img = cap.read()
            if ok:
                feats = [appearance(img, d[:4]) if d[4] >= a.low else None for d in dets]
        for f2, i, (x1, y1, x2, y2), c in T.step(fr, dets, feats):
            lines.append(f"{f2},{i},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{c:.4f}")
    lines.sort(key=lambda l: (int(l.split(",")[0]), int(l.split(",")[1])))
    open(os.path.join(a.out, name + ".txt"), "w").write("\n".join(lines) + "\n")
