"""The 0.883 extras as one reusable stage, shared by pi_run.py (real time, Pi) and identical to pipeline_boost.py's best flags:
  --no-motion --layer-width 480 --verify --keeper2 --agree-boost 0.1 --boost-to 0.65 --crop-scale 2
Per frame, given the whole-frame detections and the tracker:
  1 prediction crops: 320 px crops at predicted spots the whole-frame pass missed, enlarged 2x (cubic), crop model at 640
  2 shape layer (layers.py + consensus.py at 480 px, CPU)
  3 verify: weak boxes (< 0.5) with no shape support -> conf x0.3
  4 agreement booster: boxes with conf 0.10-0.30 and a same-place, same-size blob -> 0.65
  5 keeper2: keep a track <= 5 frames if a blob near its prediction moves like the track (conf 0.25)
Returns the detections to give to TwoDroneTracker.step."""
import time
import cv2
import numpy as np
import layers as LY
from consensus import Consensus


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy; return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


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


def same_place(d, b):
    dcx, dcy = (d[0] + d[2]) / 2, (d[1] + d[3]) / 2; bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    inside = (b[0] - 4 <= dcx <= b[2] + 4 and b[1] - 4 <= dcy <= b[3] + 4) or (d[0] <= bcx <= d[2] and d[1] <= bcy <= d[3])
    ds, bs = max(d[2] - d[0], d[3] - d[1], 1), max(b[2] - b[0], b[3] - b[1], 1)
    return inside and 0.5 <= ds / bs <= 2.0


class Booster:
    def __init__(self, crop_backend, crop=320, crop_imgsz=320, crop_scale=2.0, layer_width=480,
                 verify=True, verify_below=0.5, verify_mult=0.3, agree_lo=0.1, boost_to=0.65,
                 keeper2=True, keeper_conf=0.25, keeper_max=5, use_layers=True, use_crops=True, cmc=False):
        self.cb = crop_backend; self.crop, self.crop_imgsz, self.k = crop, crop_imgsz, crop_scale
        self.verify, self.vb, self.vm = verify, verify_below, verify_mult
        self.lo, self.hi = agree_lo, boost_to
        self.keeper2, self.kconf, self.kmax = keeper2, keeper_conf, keeper_max
        self.use_layers, self.use_crops = use_layers, use_crops
        self.cmc = cmc  # camera-motion compensation: shift the tracker with the camera (measured in layers.py)
        self.lw = layer_width
        self.reset()

    def reset(self):
        """call at the start of each video"""
        self.LM = LY.LayerMaker(width=self.lw); self.CS = Consensus(near=60 * self.lw / 960)
        self.krun = {}; self.prev_bl = []
        self.t = {"crops": [], "layers": [], "logic": []}

    def _crops(self, img, centers):
        H, W = img.shape[:2]; s = self.crop; tiles, offs = [], []
        for cx, cy in centers:
            x0 = int(min(max(cx - s / 2, 0), max(W - s, 0))); y0 = int(min(max(cy - s / 2, 0), max(H - s, 0)))
            t = img[y0:y0 + s, x0:x0 + s]
            if self.k != 1:
                t = cv2.resize(t, None, fx=self.k, fy=self.k, interpolation=cv2.INTER_CUBIC)
            tiles.append(t); offs.append((x0, y0))
        out = []; k = self.k
        if tiles:
            for ds, (x0, y0) in zip(self.cb.infer(tiles, imgsz=int(round(self.crop_imgsz * k)), conf=0.05), offs):
                for x1, y1, x2, y2, c in ds:
                    out.append((x1 / k + x0, y1 / k + y0, x2 / k + x0, y2 / k + y0, c))
        return out

    def step(self, img, dets, T):
        dets = list(dets); preds = T.predictions(); low = T.p["low"]
        if self.cmc and self.use_layers:  # shape layer first: it measures the camera motion, then the tracker follows it
            t1 = time.time()
            L = self.LM.step(img, [pb for _, pb, _ in preds]); cbl = self.CS.step(L, self.LM.s)
            if len(self.LM.Hs) and len(self.LM.grays) > 1:
                s = self.LM.s; S = np.diag([s, s, 1.0])
                T.apply_camera(np.linalg.inv(S) @ self.LM.Hs[-1] @ S)
                preds = T.predictions()
            self.t["layers"].append(time.time() - t1)
            return self._rest(img, dets, T, preds, low, cbl)
        t0 = time.time()
        if self.use_crops:  # 1 prediction crops
            centers = [((pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2) for _, pb, _ in preds
                       if not any(iou(pb, d) > 0.2 and d[4] >= low for d in dets)]
            if centers:
                dets = merge(dets, self._crops(img, centers))
        t1 = time.time(); self.t["crops"].append(t1 - t0)
        if not self.use_layers:
            return dets
        L = self.LM.step(img, [pb for _, pb, _ in preds])  # 2 shape layer
        cbl = self.CS.step(L, self.LM.s)
        t2 = time.time(); self.t["layers"].append(t2 - t1)
        return self._logic(dets, T, preds, low, cbl, t2)

    def _rest(self, img, dets, T, preds, low, cbl):
        t0 = time.time()
        if self.use_crops:
            centers = [((pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2) for _, pb, _ in preds
                       if not any(iou(pb, d) > 0.2 and d[4] >= low for d in dets)]
            if centers:
                dets = merge(dets, self._crops(img, centers))
        t2 = time.time(); self.t["crops"].append(t2 - t0)
        return self._logic(dets, T, preds, low, cbl, t2)

    def _logic(self, dets, T, preds, low, cbl, t2):
        if self.verify:  # 3
            mk = self.CS.mask; Hm, Wm = mk.shape; nd = []
            for d in dets:
                if d[4] < self.vb:
                    x1, y1, x2, y2 = [int(v * self.LM.s) for v in d[:4]]
                    x1, y1 = max(x1 - 2, 0), max(y1 - 2, 0); x2, y2 = min(x2 + 3, Wm), min(y2 + 3, Hm)
                    if x2 <= x1 or y2 <= y1 or not mk[y1:y2, x1:x2].any():
                        d = (*d[:4], d[4] * self.vm)
                nd.append(d)
            dets = nd
        if self.lo:  # 4
            dets = [(*d[:4], max(d[4], self.hi)) if self.lo <= d[4] < 0.3 and any(same_place(d, b) for b in cbl) else d
                    for d in dets]
        if self.keeper2:  # 5
            vel = {t.id: (t.kf.x[4], t.kf.x[5]) for t in T.active if t.id is not None}
            for tid, pb, miss in preds:
                if any(iou(pb, d) > 0.2 and d[4] >= low for d in dets):
                    self.krun[tid] = 0; continue
                if self.krun.get(tid, 0) >= self.kmax or tid not in vel:
                    continue
                pw, ph = pb[2] - pb[0], pb[3] - pb[1]; r = max(pw, ph, 20)
                pcx, pcy = (pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2; vx, vy = vel[tid]; vs = np.hypot(vx, vy)
                ok_b = None
                for b in cbl:
                    bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                    if np.hypot(bcx - pcx, bcy - pcy) >= r:
                        continue
                    bsz = max(b[2] - b[0], b[3] - b[1], 1)
                    if not 0.5 <= bsz / max(pw, ph, 1) <= 2.0:
                        continue
                    prv = [q for q in self.prev_bl if np.hypot((q[0] + q[2]) / 2 - bcx, (q[1] + q[3]) / 2 - bcy) < r]
                    if not prv:
                        continue
                    q = min(prv, key=lambda q: np.hypot((q[0] + q[2]) / 2 - bcx, (q[1] + q[3]) / 2 - bcy))
                    mx, my = bcx - (q[0] + q[2]) / 2, bcy - (q[1] + q[3]) / 2; ms = np.hypot(mx, my)
                    if vs < 2 and ms < 2:
                        ok_b = b; break
                    if vs >= 2 and ms >= 1 and 0.5 <= ms / vs <= 1.5 and (mx * vx + my * vy) / (ms * vs) >= np.cos(np.pi / 4):
                        ok_b = b; break
                if ok_b is not None:
                    cx, cy = (ok_b[0] + ok_b[2]) / 2, (ok_b[1] + ok_b[3]) / 2
                    dets.append((cx - pw / 2, cy - ph / 2, cx + pw / 2, cy + ph / 2, self.kconf))
                    self.krun[tid] = self.krun.get(tid, 0) + 1
                else:
                    self.krun[tid] = self.kmax  # stop keeping as soon as the blob stops matching
        self.prev_bl = cbl
        self.t["logic"].append(time.time() - t2)
        return dets
