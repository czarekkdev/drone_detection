"""Motion detector with camera-shake compensation. CPU only, results cached to a text file.
python motion.py --videos ../test_sample_clean --out motion
For each frame t: align frames t-2 and t-1 onto t (feature tracking + RANSAC homography),
take |t - (t-1)| AND |t - (t-2)| (three-frame difference, removes ghosts), threshold,
clean up, and write each moving blob as a box:  frame,x,y,w,h,strength  (full-res pixels).
Frames where the camera alignment fails are written with no blobs."""
import argparse, os, glob, time
import cv2
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--videos", required=True)
p.add_argument("--out", default="motion")
p.add_argument("--width", type=int, default=960, help="work at this width (speed)")
p.add_argument("--thr", type=float, default=4.0, help="threshold = thr * robust noise level of the diff")
p.add_argument("--min-area", type=int, default=6)
p.add_argument("--max-area", type=int, default=6000)
p.add_argument("--max-blobs", type=int, default=30)
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)


def align(src, dst, pts_src):
    """Homography mapping src -> dst from sparse optical flow; None if unreliable."""
    if pts_src is None or len(pts_src) < 20:
        return None
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(src, dst, pts_src, None, winSize=(21, 21), maxLevel=3)
    ok = st.ravel() == 1
    if ok.sum() < 20:
        return None
    H, inl = cv2.findHomography(pts_src[ok], nxt[ok], cv2.RANSAC, 2.0)
    if H is None or inl.sum() < 15:
        return None
    return H


for vdir in sorted(glob.glob(os.path.join(a.videos, "*/"))):
    name = os.path.basename(os.path.normpath(vdir))
    cap = cv2.VideoCapture(os.path.join(vdir, "video.mp4"))
    hist, lines, n, t0 = [], [], 0, time.time()  # hist: list of (gray, features)
    s = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        if s is None:
            s = a.width / frame.shape[1]
        g = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA),
                                          cv2.COLOR_BGR2GRAY), (3, 3), 0)
        feats = cv2.goodFeaturesToTrack(g, 800, 0.01, 8)
        if len(hist) == 2:
            h, w = g.shape
            diffs = []
            for pg, pf in hist:
                H = align(pg, g, pf)
                if H is None:
                    break
                warped = cv2.warpPerspective(pg, H, (w, h), borderMode=cv2.BORDER_REPLICATE)
                valid = cv2.warpPerspective(np.ones_like(pg), H, (w, h)) > 0
                d = cv2.absdiff(g, warped).astype(np.float32)
                d[~valid] = 0
                diffs.append(d)
            if len(diffs) == 2:
                d = np.minimum(diffs[0], diffs[1])
                noise = np.median(d) * 1.4826 + 1.0
                m = (d > a.thr * noise).astype(np.uint8)
                m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
                m = cv2.dilate(m, np.ones((5, 5), np.uint8))
                k, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
                blobs = []
                for i in range(1, k):
                    x, y, bw, bh, area = stats[i]
                    if a.min_area <= area <= a.max_area and bw < w * 0.3 and bh < h * 0.3:
                        strength = float(d[lab == i].mean() / noise)
                        blobs.append((strength, x, y, bw, bh))
                blobs.sort(reverse=True)
                for st_, x, y, bw, bh in blobs[: a.max_blobs]:
                    lines.append(f"{n},{x / s:.1f},{y / s:.1f},{bw / s:.1f},{bh / s:.1f},{st_:.2f}")
        hist = (hist + [(g, feats)])[-2:]
    with open(os.path.join(a.out, name + ".txt"), "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    print(f"{name}: {n} frames, {len(lines)} motion blobs, {n / (time.time() - t0):.1f} fps")
