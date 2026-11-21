"""Two-drone tracker: Kalman (constant velocity) + ByteTrack-style two-threshold matching,
max 2 ids, confirmation, short coasting, and re-identification of returning drones.
Works on any per-frame detection list, so the same code runs offline (on result files) and live on the Pi."""
import numpy as np
from scipy.optimize import linear_sum_assignment

REACQ_GAP = 30     # frames: how long after losing a drone we may take it back under its own id
REACQ_CONF = 0.20  # min box confidence for re-acquisition
REACQ_DIST = 1.5   # max distance from where it was last seen, in drone sizes (grows 15%/frame of gap)
HANDOVER = True    # a newly confirmed track takes over the number of a same-size track that is coasting on nothing
                   # (drone stopped dead / turned, prediction flew off): same drone keeps its number.
                   # Anti-UAV 67 videos: ID switches 63 -> 53; practice videos unchanged
INSIDE = 0.7       # a box lying this much inside a tracked drone's box can't start a new track (part of the same drone)
MERGE = True       # two tracks on the same drone (one box INSIDE the other): keep the one seen longer, drop the other
W_SIZE = 0.0       # weight of box size when deciding which lost drone came back (tested 0.5-1: no effect on practice)
HANDOVER_DIST = 3.0  # only if the new track is within this many drone sizes of where the coasting one was last SEEN


class KF:
    """state [cx, cy, w, h, vx, vy, vw, vh]"""
    def __init__(self, box):
        cx, cy, w, h = box
        self.x = np.array([cx, cy, w, h, 0, 0, 0, 0], float)
        s = max(w, h)
        self.P = np.diag([s, s, s, s, 10 * s, 10 * s, s, s]) ** 2 * 0.01 + np.eye(8)
        self.F = np.eye(8); self.F[:4, 4:] = np.eye(4)
        self.H = np.eye(4, 8)

    def predict(self):
        s = max(self.x[2], self.x[3], 4)
        Q = np.diag([.05 * s, .05 * s, .05 * s, .05 * s, .1 * s, .1 * s, .02 * s, .02 * s]) ** 2
        self.x = self.F @ self.x; self.P = self.F @ self.P @ self.F.T + Q
        self.x[2:4] = np.maximum(self.x[2:4], 2)
        return self.x[:4]

    def update(self, box):
        s = max(self.x[2], self.x[3], 4)
        R = np.diag([.05 * s, .05 * s, .1 * s, .1 * s]) ** 2
        y = np.asarray(box, float) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R; K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y; self.P = (np.eye(8) - K @ self.H) @ self.P


def xyxy_to_cxcywh(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1])


class Track:
    def __init__(self, box, conf, frame, feat=None):
        self.kf = KF(xyxy_to_cxcywh(box)); self.hits = 1; self.miss = 0
        self.conf = conf; self.last = frame; self.id = None; self.feat = feat
        self.hist = [(frame, tuple(box[:4]), conf)]  # kept while tentative, written out on confirmation
        self.aspect = (box[2] - box[0]) / max(box[3] - box[1], 1)
        self.last_obs = xyxy_to_cxcywh(box)

    def peek(self):
        """predicted box for the NEXT frame, without changing the filter"""
        cx, cy, w, h = (self.kf.F @ self.kf.x)[:4]
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    def box(self):
        cx, cy, w, h = self.kf.x[:4]
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def cost_matrix(tracks, dets, gate):
    """centre distance divided by object size; > gate = impossible"""
    C = np.full((len(tracks), len(dets)), 1e6)
    for i, t in enumerate(tracks):
        cx, cy, w, h = t.kf.x[:4]; s = max(w, h, 12) * (1 + 0.5 * min(t.miss, 10))
        for j, d in enumerate(dets):
            dx, dy, dw, dh = xyxy_to_cxcywh(d[:4])
            c = np.hypot(dx - cx, dy - cy) / s + 0.5 * abs(np.log(max(dw * dh, 1) / max(w * h, 1)))
            if c < gate:
                C[i, j] = c
    return C


def match(tracks, dets, gate):
    if not tracks or not dets:
        return [], list(range(len(tracks))), list(range(len(dets)))
    C = cost_matrix(tracks, dets, gate); r, c = linear_sum_assignment(C)
    m = [(i, j) for i, j in zip(r, c) if C[i, j] < 1e5]
    mi, mj = {i for i, _ in m}, {j for _, j in m}
    return m, [i for i in range(len(tracks)) if i not in mi], [j for j in range(len(dets)) if j not in mj]


class TwoDroneTracker:
    def __init__(self, high=0.6, low=0.05, new=0.3, confirm=3, coast=3, keep_lost=900, gate=2.5,
                 max_ids=2, w_feat=1.0, w_pos=1.0, new_id_cost=1.0, backfill=True, frame_size=(1920, 1080)):
        self.p = dict(high=high, low=low, new=new, confirm=confirm, coast=coast, keep_lost=keep_lost, gate=gate)
        self.max_ids, self.w_feat, self.w_pos, self.W, self.H = max_ids, w_feat, w_pos, *frame_size
        self.new_id_cost = new_id_cost
        self.backfill = backfill
        self.active, self.tentative = [], []
        self.lost = {}  # id -> Track (kept for re-id)

    def _free_ids(self):
        used = {t.id for t in self.active}
        return [i for i in range(1, self.max_ids + 1) if i not in used]

    def _assign_id(self, t, frame):
        free = self._free_ids()
        if not free:
            return None
        if len(free) == 1:
            return free[0]
        # several ids free: pick the lost track that best explains this new drone
        best, bid = 1e9, None
        for i in free:
            if i not in self.lost:
                c = self.new_id_cost  # never-seen id
            else:
                L = self.lost[i]; gap = max(frame - L.last, 1)
                lx, ly = L.kf.x[0], L.kf.x[1]  # last position (frozen when lost)
                nx, ny = t.kf.x[0], t.kf.x[1]
                c = self.w_pos * np.hypot(nx - lx, ny - ly) / np.hypot(self.W, self.H) * 4
                c += 0.5 * abs(np.log(max(t.aspect, .1) / max(L.aspect, .1)))
                # size: the two drones usually differ in size (distance / model), and size changes slowly
                c += W_SIZE * abs(np.log(max(t.kf.x[2] * t.kf.x[3], 1) / max(L.last_obs[2] * L.last_obs[3], 1)))
                if t.feat is not None and L.feat is not None:
                    c += self.w_feat * feat_dist(t.feat, L.feat) * 4
            if c < best:
                best, bid = c, i
        return bid

    def step(self, frame, dets, feats=None):
        """dets: list of (x1,y1,x2,y2,conf).
        returns list of (frame, id, (x1,y1,x2,y2), conf). Usually frame == current frame; when a track is
        confirmed, its first few (tentative) frames are returned too (backfill), so results files are complete."""
        P = self.p; self._frame = frame
        feats = feats if feats is not None else [None] * len(dets)
        for t in self.active + self.tentative:
            t.kf.predict()
        hi = [j for j, d in enumerate(dets) if d[4] >= P["high"]]
        lo = [j for j, d in enumerate(dets) if P["low"] <= d[4] < P["high"]]
        # 1) active tracks <- confident detections
        m, ut, ud = match(self.active, [dets[j] for j in hi], P["gate"])
        used = set()
        for i, k in m:
            self._hit(self.active[i], dets[hi[k]], feats[hi[k]], frame); used.add(hi[k])
        rest = [self.active[i] for i in ut]
        # 2) remaining active tracks <- weak detections
        m2, ut2, _ = match(rest, [dets[j] for j in lo], P["gate"])
        for i, k in m2:
            self._hit(rest[i], dets[lo[k]], feats[lo[k]], frame); used.add(lo[k])
        for i in ut2:
            rest[i].miss += 1
        # 2b) RE-ACQUIRE: a drone lost a moment ago (e.g. it stopped / turned and the prediction ran off)
        #     comes back under its own id straight away from a moderately confident box near where it was last seen,
        #     instead of waiting for 3 confident frames as a brand-new track
        if REACQ_GAP > 0 and self.lost:
            cand = [j for j, d in enumerate(dets) if j not in used and d[4] >= REACQ_CONF]
            L = [(i, t) for i, t in self.lost.items() if 0 < frame - t.last <= REACQ_GAP and i not in {a.id for a in self.active}]
            if cand and L:
                C = np.full((len(L), len(cand)), 1e6)
                for r, (i, t) in enumerate(L):
                    lx, ly, lw, lh = t.last_obs; s = max(lw, lh, 12); gap = frame - t.last
                    for c_, j in enumerate(cand):
                        dx, dy, dw, dh = xyxy_to_cxcywh(dets[j][:4])
                        dist = np.hypot(dx - lx, dy - ly) / (s * (1 + 0.15 * gap))
                        size = abs(np.log(max(dw * dh, 1) / max(lw * lh, 1)))
                        if dist < REACQ_DIST and size < 1.0:
                            C[r, c_] = dist + 0.5 * size
                rr, cc = linear_sum_assignment(C)
                for r, c_ in zip(rr, cc):
                    if C[r, c_] >= 1e5 or len(self.active) >= self.max_ids:
                        continue
                    i, t = L[r]; j = cand[c_]; d = dets[j]
                    t.kf = KF(xyxy_to_cxcywh(d[:4])); t.miss = 0; t.hits += 1; t.conf = d[4]; t.last = frame
                    t.last_obs = xyxy_to_cxcywh(d[:4]); self.lost.pop(i); self.active.append(t); used.add(j)
        # 3) tentative tracks <- leftover confident detections
        left = [j for j in hi if j not in used]
        m3, ut3, ud3 = match(self.tentative, [dets[j] for j in left], P["gate"])
        for i, k in m3:
            self._hit(self.tentative[i], dets[left[k]], feats[left[k]], frame)
        self.tentative = [t for i, t in enumerate(self.tentative) if i not in set(ut3)]
        for k in ud3:
            d = dets[left[k]]
            if INSIDE and any(max(_inside(d, a.box()), _inside(a.box(), d)) >= INSIDE for a in self.active):
                continue
            if d[4] >= P["new"]:
                self.tentative.append(Track(d[:4], d[4], frame, feats[left[k]]))
        # promote confirmed tentatives
        backfill = []
        for t in list(self.tentative):
            if t.hits >= P["confirm"]:
                if HANDOVER:
                    tw, th = t.kf.x[2], t.kf.x[3]
                    coast = [a for a in self.active if a.miss >= 1
                             and abs(np.log(max(tw * th, 1) / max(a.kf.x[2] * a.kf.x[3], 1))) < 1.0
                             and np.hypot(a.last_obs[0] - t.kf.x[0], a.last_obs[1] - t.kf.x[1])
                             < HANDOVER_DIST * max(a.last_obs[2], a.last_obs[3], 12) * (1 + 0.15 * a.miss)]
                    if coast:
                        a = min(coast, key=lambda a: np.hypot(a.last_obs[0] - t.kf.x[0], a.last_obs[1] - t.kf.x[1]))
                        self.active.remove(a); t.id = a.id; self.active.append(t); self.tentative.remove(t)
                        backfill += [(f, t.id, b, c) for f, b, c in t.hist if f != frame and self.backfill]
                        t.hist = []
                        continue
                i = self._assign_id(t, frame)
                if i is not None:
                    t.id = i; self.lost.pop(i, None); self.active.append(t); self.tentative.remove(t)
                    backfill += [(f, i, b, c) for f, b, c in t.hist if f != frame and self.backfill]
                    t.hist = []
        if MERGE and len(self.active) > 1:
            # two tracks on one drone (either box mostly inside the other): a track that is only coasting (its drone
            # not seen this frame) gives way to one that is seen; otherwise the one seen for fewer frames gives way
            a, b = self.active[0], self.active[1]
            if max(_inside(a.box(), b.box()), _inside(b.box(), a.box())) >= INSIDE:
                drop = a if (a.miss > 0) > (b.miss > 0) else b if (b.miss > 0) > (a.miss > 0) else min((a, b), key=lambda t: t.hits)
                self.active.remove(drop); self.lost[drop.id] = drop
        # retire long-missing actives to the lost list
        for t in list(self.active):
            if t.miss > P["coast"]:
                self.active.remove(t); self.lost[t.id] = t
        out = list(backfill)
        for t in self.active:
            if t.miss == 0:
                out.append((frame, t.id, t.box(), t.conf))
            else:  # coasting: predicted box, lower confidence
                out.append((frame, t.id, t.box(), t.conf * 0.6 ** t.miss))
        return [o for o in (self._clip(o) for o in out) if o is not None]

    def _clip(self, o):
        """keep boxes inside the picture (a coasting drone flying out would otherwise get negative coordinates);
        drop a box if less than a quarter of it is still in view"""
        f, i, (x1, y1, x2, y2), c = o
        cx1, cy1, cx2, cy2 = max(x1, 0), max(y1, 0), min(x2, self.W), min(y2, self.H)
        if cx2 - cx1 < 2 or cy2 - cy1 < 2 or (cx2 - cx1) * (cy2 - cy1) < 0.25 * (x2 - x1) * (y2 - y1):
            return None
        return (f, i, (cx1, cy1, cx2, cy2), min(max(c, 0.0), 1.0))

    def apply_camera(self, H):
        """camera moved between the last frame and this one: H (3x3, full-res pixels) maps old positions to new ones.
        Shift every track (active, tentative, recently lost) so its prediction follows the camera, and turn its
        velocity with the camera, so the Kalman filter only has to model the drone's own motion."""
        A = H[:2, :2] / H[2, 2]
        def warp(x, y):
            v = H @ np.array([x, y, 1.0]); return v[0] / v[2], v[1] / v[2]
        now = getattr(self, "_frame", 0)
        recent = [t for t in self.lost.values() if now - t.last <= REACQ_GAP]  # long-lost: chained camera estimates drift
        for t in self.active + self.tentative + recent:
            t.kf.x[0], t.kf.x[1] = warp(t.kf.x[0], t.kf.x[1])
            t.kf.x[4:6] = A @ t.kf.x[4:6]
            lx, ly = warp(t.last_obs[0], t.last_obs[1]); t.last_obs = (lx, ly, *t.last_obs[2:])

    def predictions(self, max_lost_age=30):
        """where we expect each drone in the next frame: [(id or None, box, frames_missed)]"""
        out = [(t.id, t.peek(), t.miss) for t in self.active]
        return out

    def future_paths(self, steps=15):
        """straight-line guess of where each confirmed drone goes next: [(id, [(x, y), ...])], from the Kalman velocity"""
        out = []
        for t in self.active:
            if t.id is None:
                continue
            cx, cy, _, _, vx, vy = t.kf.x[:6]
            out.append((t.id, [(cx + vx * k, cy + vy * k) for k in range(steps + 1)]))
        return out

    def _hit(self, t, d, feat, frame):
        t.last_obs = xyxy_to_cxcywh(d[:4]); t.kf.update(xyxy_to_cxcywh(d[:4])); t.hits += 1; t.miss = 0; t.conf = d[4]; t.last = frame
        if t.id is None:
            t.hist.append((frame, tuple(d[:4]), d[4]))
        a = (d[2] - d[0]) / max(d[3] - d[1], 1); t.aspect = 0.9 * t.aspect + 0.1 * a
        if feat is not None:
            t.feat = feat if t.feat is None else 0.9 * t.feat + 0.1 * feat


def _inside(a, b):
    """share of box a's area that lies inside box b"""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / max((a[2] - a[0]) * (a[3] - a[1]), 1e-9)


def dedupe(rows):
    """rows: (frame, id, x1, y1, x2, y2, conf, ...). Keep one box per (frame, id), the most confident one
    (after a handover the back-filled frames can repeat a number that was coasting in those frames)."""
    best = {}
    for r in rows:
        k = (r[0], r[1])
        if k not in best or r[6] > best[k][6]:
            best[k] = r
    return sorted(best.values(), key=lambda r: (r[0], r[1]))


def dedupe_lines(lines):
    """same for result-file lines 'frame,id,left,top,w,h,conf', sorted by frame then id"""
    best = {}
    for l in lines:
        v = l.split(","); k = (int(v[0]), int(v[1]))
        if k not in best or float(v[6]) > float(best[k].split(",")[6]):
            best[k] = l
    return [best[k] for k in sorted(best)]


def feat_dist(a, b):
    """Bhattacharyya-like distance between normalised histograms, 0 = same, 1 = different"""
    return float(np.sqrt(max(0.0, 1 - np.sum(np.sqrt(a * b)))))


def appearance(frame_bgr, box):
    """hue/sat histogram + brightness histogram + contrast to surroundings"""
    import cv2
    H, W = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    crop = frame_bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h1 = cv2.calcHist([hsv], [0, 1], None, [8, 4], [0, 180, 0, 256]).ravel()
    h2 = cv2.calcHist([hsv], [2], None, [8], [0, 256]).ravel()
    v = np.concatenate([h1 / max(h1.sum(), 1), h2 / max(h2.sum(), 1)]) / 2
    return v
