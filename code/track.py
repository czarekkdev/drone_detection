"""Tracking + motion post-processing on top of any detector's result files.
python track.py --dets preds/hf_skyguard --motion motion --out preds/hf_skyguard_trk [--mode offline|causal]

Input:  <dets>/<video>.txt   frame,1,x,y,w,h,conf   (as written by run_video.py, ideally with --conf 0.01)
        <motion>/<video>.txt frame,x,y,w,h,strength (from motion.py; optional)
Output: same format as the input, so score.py works unchanged.

What it does, per video, frame by frame:
1. Kalman tracks (constant velocity on the box centre, smoothed size). Matching uses centre distance
   relative to the box size, because drones move many box-widths per second and IoU breaks.
   Confident detections match first, then low-confidence ones (only to existing tracks, tighter gate),
   so weak detections that continue a good track are kept and promoted.
2. Motion: a detection with a moving blob under it gets a small confidence bonus. When a track gets
   no detection, a moving blob near the predicted position keeps it alive (box = blob centre, track size).
3. Rescoring: each box's confidence is mixed with the confidence of its whole track, so boxes on long,
   steady tracks rank high and one-off blips rank low.
   --mode causal  : only uses the past (what a live system could do).
   --mode offline : uses the whole video (track score over the full track, gaps between detections
                    filled by interpolation, trailing coasted boxes dropped).
"""
import argparse, os, glob
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--dets", required=True)
p.add_argument("--motion", default="")
p.add_argument("--out", required=True)
p.add_argument("--mode", choices=["offline", "causal"], default="offline")
p.add_argument("--high", type=float, default=0.25, help="detections >= this can start tracks")
p.add_argument("--gate", type=float, default=2.5, help="max centre distance / box size to match")
p.add_argument("--gate-low", type=float, default=1.2, help="same, for low-confidence detections")
p.add_argument("--max-age", type=int, default=15, help="frames a track survives without detections")
p.add_argument("--min-hits", type=int, default=3, help="hits before a track counts as confirmed")
p.add_argument("--motion-bonus", type=float, default=0.15)
p.add_argument("--no-motion", action="store_true")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)


def load(path, ncol):
    d = {}
    if not path or not os.path.exists(path):
        return d
    for line in open(path):
        v = line.strip().split(",")
        if len(v) < ncol:
            continue
        v = list(map(float, v))
        if ncol == 7:  # frame,1,x,y,w,h,conf
            fr, x, y, w, h, c = int(v[0]), v[2], v[3], v[4], v[5], v[6]
        else:          # frame,x,y,w,h,strength
            fr, x, y, w, h, c = int(v[0]), v[1], v[2], v[3], v[4], v[5]
        d.setdefault(fr, []).append(np.array([x + w / 2, y + h / 2, w, h, c]))
    return d


class Track:
    F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], float)
    Hm = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)

    def __init__(self, det, tid):
        self.id = tid
        self.x = np.array([det[0], det[1], 0.0, 0.0]); self.wh = det[2:4].copy()
        s = max(np.sqrt(det[2] * det[3]), 10.0)
        self.P = np.diag([s, s, 4 * s, 4 * s]) ** 2
        self.hits, self.miss, self.confs = 1, 0, [det[4]]
        self.boxes = []  # (frame, cx, cy, w, h, raw_conf, kind)  kind: det / motion / coast

    def size(self):
        return max(np.sqrt(self.wh[0] * self.wh[1]), 12.0)

    def predict(self):
        s = self.size()
        Q = np.diag([0.5 * s, 0.5 * s, 0.5 * s, 0.5 * s]) ** 2
        self.x = self.F @ self.x; self.P = self.F @ self.P @ self.F.T + Q

    def update(self, z, wh=None, r=0.15):
        R = np.eye(2) * (r * self.size()) ** 2
        S = self.Hm @ self.P @ self.Hm.T + R; K = self.P @ self.Hm.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.Hm @ self.x); self.P = (np.eye(4) - K @ self.Hm) @ self.P
        if wh is not None:
            self.wh = 0.6 * self.wh + 0.4 * wh

    def score(self, upto=None):
        c = np.array(self.confs if upto is None else self.confs[:upto])
        top = np.sort(c)[::-1][: max(5, len(c) // 2)]  # mean of the better half
        return float(top.mean()) * min(1.0, len(c) / a.min_hits)


def dist(t, det):
    return np.hypot(t.x[0] - det[0], t.x[1] - det[1]) / max(t.size(), np.sqrt(det[2] * det[3]), 12.0)


def greedy(tracks, dets, gate):
    pairs = sorted((dist(t, d), i, j) for i, t in enumerate(tracks) for j, d in enumerate(dets))
    ut, ud, m = set(), set(), []
    for c, i, j in pairs:
        if c > gate:
            break
        if i not in ut and j not in ud:
            ut.add(i); ud.add(j); m.append((i, j))
    return m


def has_motion(det, blobs):
    x0, y0 = det[0] - det[2] * 0.75, det[1] - det[3] * 0.75
    x1, y1 = det[0] + det[2] * 0.75, det[1] + det[3] * 0.75
    return any(x0 <= b[0] <= x1 and y0 <= b[1] <= y1 for b in blobs)


for path in sorted(glob.glob(os.path.join(a.dets, "*.txt"))):
    name = os.path.basename(path)[:-4]
    D = load(path, 7)
    M = {} if a.no_motion else load(os.path.join(a.motion, name + ".txt") if a.motion else "", 6)
    last = max(list(D) + list(M) + [0])
    tracks, dead, loose, nid = [], [], [], 0  # loose: detections not on any track
    for fr in range(1, last + 1):
        dets, blobs = D.get(fr, []), M.get(fr, [])
        for d in dets:
            if blobs and has_motion(d, blobs):
                d[4] = min(1.0, d[4] + a.motion_bonus * (1 - d[4]))
        for t in tracks:
            t.predict()
        hi = [d for d in dets if d[4] >= a.high]; lo = [d for d in dets if d[4] < a.high]
        used_t, used_hi, used_lo = set(), set(), set()
        for i, j in greedy(tracks, hi, a.gate):
            used_t.add(i); used_hi.add(j)
        rest = [k for k in range(len(tracks)) if k not in used_t and tracks[k].hits >= a.min_hits]
        m_lo = greedy([tracks[k] for k in rest], lo, a.gate_low)
        matches = [(i, hi[j]) for i, j in greedy(tracks, hi, a.gate)] + [(rest[i], lo[j]) for i, j in m_lo]
        used_t |= {rest[i] for i, _ in m_lo}; used_lo = {j for _, j in m_lo}
        for i, d in matches:
            t = tracks[i]
            t.update(d[:2], d[2:4]); t.hits += 1; t.miss = 0; t.confs.append(d[4])
            t.boxes.append((fr, d[0], d[1], d[2], d[3], d[4], "det", t.score()))
        free_blobs = list(blobs)
        for k, t in enumerate(tracks):
            if k in used_t:
                continue
            t.miss += 1
            kind = "coast"
            if free_blobs and t.hits >= a.min_hits:
                ds = [np.hypot(t.x[0] - b[0], t.x[1] - b[1]) / t.size() for b in free_blobs]
                bi = int(np.argmin(ds))
                if ds[bi] < 1.5:
                    t.update(free_blobs.pop(bi)[:2], r=0.3); kind = "motion"
            t.boxes.append((fr, t.x[0], t.x[1], t.wh[0], t.wh[1], t.miss, kind, t.score()))  # miss count in conf slot
        for j, d in enumerate(hi):
            if j not in used_hi:
                t = Track(d, nid); nid += 1
                t.boxes.append((fr, d[0], d[1], d[2], d[3], d[4], "det", t.score())); tracks.append(t)
        for j, d in enumerate(lo):
            if j not in used_lo:
                loose.append((fr, d))
        alive = []
        for t in tracks:
            (alive if t.miss <= a.max_age else dead).append(t)
        tracks = alive
    dead += tracks

    out = []  # (frame, x, y, w, h, conf)
    for fr, d in loose:  # weak detections that never joined a track
        out.append((fr, d[0], d[1], d[2], d[3], d[4] * 0.5))
    for t in dead:
        bx = t.boxes
        if a.mode == "offline":
            last_det = max(k for k, b in enumerate(bx) if b[6] == "det")
            bx = bx[: last_det + 1]  # drop coasting after the track was lost
            ts = t.score()
            idx = [k for k, b in enumerate(bx) if b[6] == "det"]
            for k, b in enumerate(bx):
                fr, cx, cy, w, h, c, kind, _ = b
                if kind == "det":
                    conf = 0.5 * c + 0.5 * ts if t.hits >= a.min_hits else 0.5 * c
                else:  # interpolate between the surrounding detections
                    k0 = max(i for i in idx if i < k); k1 = min(i for i in idx if i > k)
                    f = (k - k0) / (k1 - k0)
                    cx, cy, w, h = [(1 - f) * bx[k0][q] + f * bx[k1][q] for q in (1, 2, 3, 4)]
                    if t.hits < a.min_hits:
                        continue
                    conf = ts * (0.5 if kind == "motion" else 0.35)
                out.append((fr, cx, cy, w, h, conf))
        else:
            for fr, cx, cy, w, h, c, kind, ts in bx:  # ts = track score at that moment (past only)
                if kind == "det":
                    conf = 0.5 * c + 0.5 * ts
                elif kind == "motion":
                    conf = ts * 0.5
                elif c <= 5:  # coast a few frames only when live (c holds frames since last detection)
                    conf = ts * 0.3
                else:
                    continue
                out.append((fr, cx, cy, w, h, conf))
    out.sort()
    with open(os.path.join(a.out, name + ".txt"), "w") as f:
        for fr, cx, cy, w, h, c in out:
            f.write(f"{fr},1,{cx - w / 2:.2f},{cy - h / 2:.2f},{w:.2f},{h:.2f},{c:.4f}\n")
    print(f"{name}: {len(dead)} tracks, {len(out)} boxes")
