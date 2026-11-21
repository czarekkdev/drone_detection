"""Consensus shape layer: fuse the layers.py cues into a few accurate blobs.
cues (each 0/1 per pixel): motion = max(short, long) > t_motion, bg > t_bg, shape > t_shape
keep pixel if votes >= need  OR  (near a tracker prediction AND votes >= need_near)
then close/open, connected components, drop too small / too big / too elongated, require persistence
(a blob near one of the previous frame's blobs), rank by evidence, keep top_k.
All in layers.py work coordinates (default 960 wide); blobs() returns full-res boxes."""
import cv2
import numpy as np


class Consensus:
    def __init__(self, t_motion=128, t_bg=128, t_shape=150, need=2, need_near=1, near=60,
                 min_area=3, max_area_frac=0.004, max_aspect=6.0, persist=True, top_k=5):
        self.p = dict(t_motion=t_motion, t_bg=t_bg, t_shape=t_shape, need=need, need_near=need_near, near=near,
                      min_area=min_area, max_area_frac=max_area_frac, max_aspect=max_aspect, persist=persist, top_k=top_k)
        self.prev = []  # previous frame's blob centres (work coords)
        self.mask = None

    def step(self, L, scale):
        """L: dict from LayerMaker.step; scale: work/full (LayerMaker.s). returns [(x1,y1,x2,y2,score)] full-res"""
        p = self.p
        motion = np.maximum(L["short"], L["long"]) > p["t_motion"]
        bg = L["bg"] > p["t_bg"]; shape = L["shape"] > p["t_shape"]
        votes = motion.astype(np.uint8) + bg + shape
        near = L["track"] > 40
        m = (votes >= p["need"]) | (near & (votes >= p["need_near"]))
        m = m.astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        self.mask = m
        n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
        H, W = m.shape; evid = (L["short"].astype(np.float32) + L["long"] + L["bg"] + L["shape"]) / 4
        cand = []
        for i in range(1, n):
            x, y, w, h, area = st[i]
            if area < p["min_area"] or area > p["max_area_frac"] * H * W:
                continue
            if max(w, h) / max(min(w, h), 1) > p["max_aspect"]:
                continue
            cx, cy = cen[i]
            s = float(evid[lab == i].mean()) / 255 + (0.5 if near[int(cy), int(cx)] else 0)
            cand.append((s, x, y, w, h, cx, cy))
        if p["persist"]:
            keep = [c for c in cand if any(np.hypot(c[5] - u, c[6] - v) < p["near"] for u, v in self.prev)
                    or c[0] >= 1.0]  # very strong or near a prediction passes without history
            self.prev = [(c[5], c[6]) for c in cand]
            cand = keep
        cand.sort(key=lambda c: -c[0])
        out = []
        for s, x, y, w, h, *_ in cand[:p["top_k"]]:
            out.append((x / scale, y / scale, (x + w) / scale, (y + h) / scale, s))
        return out


def single(L, key, thr, scale, top_k=5, min_area=3, max_area_frac=0.004, max_aspect=6.0):
    """same blob extraction from ONE layer (for comparison). returns (kept boxes full-res, raw blob count)"""
    im = np.maximum(L["short"], L["long"]) if key == "motion" else L[key]
    m = ((im > thr).astype(np.uint8)) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
    H, W = m.shape; cand = []
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if area < min_area or area > max_area_frac * H * W or max(w, h) / max(min(w, h), 1) > max_aspect:
            continue
        cand.append((float(im[lab == i].mean()), x, y, w, h))
    raw = len(cand); cand.sort(key=lambda c: -c[0])
    return [(x / scale, y / scale, (x + w) / scale, (y + h) / scale, s) for s, x, y, w, h in cand[:top_k]], raw


def view(mask, blobs, scale, size):
    """demo picture: consensus mask (white) with kept blobs boxed (green)"""
    v = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    for x1, y1, x2, y2, s in blobs:
        cv2.rectangle(v, (int(x1 * scale) - 3, int(y1 * scale) - 3), (int(x2 * scale) + 3, int(y2 * scale) + 3), (0, 255, 0), 1)
    return cv2.resize(v, size, interpolation=cv2.INTER_NEAREST)
