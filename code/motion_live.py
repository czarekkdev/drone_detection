"""Live motion detector (one frame at a time) with camera-shake compensation.
md = MotionDetector(); blobs = md.step(frame_bgr)   # list of (x, y, w, h, strength, changed_px) in full-res pixels
md.diff -> last difference image (float, small size), md.mask -> last binary mask (small size)"""
import cv2
import numpy as np


class MotionDetector:
    def __init__(self, width=960, thr=4.0, min_area=6, max_area=6000, max_blobs=30):
        self.width, self.thr, self.min_area, self.max_area, self.max_blobs = width, thr, min_area, max_area, max_blobs
        self.hist = []; self.diff = None; self.mask = None; self.noise = 1.0; self.s = None
        self.prev_centers = [[], []]

    @staticmethod
    def _shift(src, dst):
        """fallback when there are too few corners (plain sky): pure translation by phase correlation"""
        (dx, dy), resp = cv2.phaseCorrelate(np.float32(src), np.float32(dst))
        return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], float)

    @classmethod
    def _align(cls, src, dst, pts):
        if pts is None or len(pts) < 20:
            return cls._shift(src, dst)
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(src, dst, pts, None, winSize=(21, 21), maxLevel=3)
        ok = st.ravel() == 1
        if ok.sum() < 20:
            return cls._shift(src, dst)
        H, inl = cv2.findHomography(pts[ok], nxt[ok], cv2.RANSAC, 2.0)
        return H if H is not None and inl.sum() >= 15 else cls._shift(src, dst)

    def step(self, frame):
        if self.s is None:
            self.s = self.width / frame.shape[1]
        g = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(frame, None, fx=self.s, fy=self.s, interpolation=cv2.INTER_AREA),
                                          cv2.COLOR_BGR2GRAY), (3, 3), 0)
        feats = cv2.goodFeaturesToTrack(g, 800, 0.01, 8)
        blobs = []
        if len(self.hist) == 2:
            h, w = g.shape; diffs = []
            for pg, pf in self.hist:
                H = self._align(pg, g, pf)
                if H is None:
                    break
                warped = cv2.warpPerspective(pg, H, (w, h), borderMode=cv2.BORDER_REPLICATE)
                valid = cv2.warpPerspective(np.ones_like(pg), H, (w, h)) > 0
                d = cv2.absdiff(g, warped).astype(np.float32); d[~valid] = 0; diffs.append(d)
            if len(diffs) == 2:
                d = np.minimum(diffs[0], diffs[1]); self.diff = d
                self.noise = noise = np.median(d) * 1.4826 + 1.0
                m = (d > self.thr * noise).astype(np.uint8)
                m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
                m = cv2.dilate(m, np.ones((5, 5), np.uint8)); self.mask = m
                k, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
                for i in range(1, k):
                    x, y, bw, bh, area = stats[i]
                    if self.min_area <= area <= self.max_area and bw < w * 0.3 and bh < h * 0.3:
                        roi = d[y:y + bh, x:x + bw]
                        npx = int((roi > self.thr * noise).sum()); strength = float(roi.max() / noise)
                        blobs.append((strength, x, y, bw, bh, npx))
                blobs.sort(reverse=True); blobs = blobs[: self.max_blobs]
        self.hist = (self.hist + [(g, feats)])[-2:]
        s = self.s
        out = [(x / s, y / s, bw / s, bh / s, st, npx) for st, x, y, bw, bh, npx in blobs]
        # steadiness: how many of the 2 previous frames had a spot nearby
        cur = [(x + w / 2, y + h / 2) for x, y, w, h, *_ in out]
        res = []
        for (x, y, w, h, st, npx), (cx, cy) in zip(out, cur):
            per = sum(any(abs(cx - u) < 60 and abs(cy - v) < 60 for u, v in pc) for pc in self.prev_centers)
            res.append((x, y, w, h, st, npx, per))
        self.prev_centers = [self.prev_centers[1], cur]
        return res

    def diff_view(self, size):
        """colour picture of the difference image (bright = changed), resized to (W, H)"""
        if self.diff is None:
            return np.zeros((size[1], size[0], 3), np.uint8)
        v = np.clip(self.diff / self.noise * 40, 0, 255).astype(np.uint8)
        return cv2.resize(cv2.applyColorMap(v, cv2.COLORMAP_INFERNO), size, interpolation=cv2.INTER_NEAREST)

    def mask_view(self, size):
        if self.mask is None:
            return np.zeros((size[1], size[0], 3), np.uint8)
        return cv2.resize(cv2.cvtColor(self.mask * 255, cv2.COLOR_GRAY2BGR), size, interpolation=cv2.INTER_NEAREST)
