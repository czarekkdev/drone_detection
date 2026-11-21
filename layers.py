"""Extra input layers for the 'B' model, computed frame by frame (same code for training data and live use).
lm = LayerMaker(); L = lm.step(frame_bgr, predicted_boxes)   # predicted_boxes: [(x1,y1,x2,y2), ...] full-res
L is a dict of 5 uint8 images (work size, default 960 wide):
  short  - change vs 2 frames ago (camera-aligned)          -> fast movers
  long   - change vs 10 frames ago (camera-aligned)         -> slow / hovering drones
  bg     - difference to a slowly updated 'empty scene'     -> anything that doesn't belong
  track  - soft spots where the tracker expects each drone  -> 'look here'
  prev   - previous frame, aligned to this one (gray)       -> lets the network judge motion itself
  shape  - brightness shape: how much each pixel stands out from its surroundings, divided by how busy
           those surroundings are -> a drone in plain sky becomes a clean bright silhouette even when it
           hovers (no motion needed), while tree/grass texture is pushed down
stack(L) -> HxWx5 array in a fixed order (ORDER)."""
import cv2
import numpy as np
import time

ORDER = ("short", "long", "bg", "track", "prev", "shape")


def sane(H, max_scale=0.05, max_persp=2e-4):
    """reject warps a hand-held camera can't do between frames (plain-sky noise corners give crazy ones)"""
    A = H[:2, :2] / H[2, 2]
    return (np.abs(A - np.eye(2)).max() < max_scale and np.abs(H[2, :2]).max() < max_persp
            and np.abs(H[:2, 2] / H[2, 2]).max() < 60)


class LayerMaker:
    def __init__(self, width=960, long_gap=10, bg_alpha=0.05, gain=32.0, fast_align=False):
        self.width, self.long_gap, self.alpha, self.gain = width, long_gap, bg_alpha, gain
        self.fast_align = fast_align
        self.grays, self.Hs = [], []  # Hs[i]: homography grays[i] -> grays[i+1]
        self.bg = None; self.s = None; self.prev_feats = None
        self.timing = {k: 0.0 for k in ("gray", "features", "alignment", "flow", "homography", "phase", "warp", "normalize", "shape", "track")}

    def _timed(self, key, started):
        self.timing[key] += time.perf_counter() - started

    def _h(self, src, dst, pts):
        if not self.fast_align and pts is not None and len(pts) >= 20:
            started = time.perf_counter()
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(src, dst, pts, None, winSize=(21, 21), maxLevel=3)
            self._timed("flow", started)
            ok = st.ravel() == 1
            if ok.sum() >= 20:
                started = time.perf_counter()
                H, inl = cv2.findHomography(pts[ok], nxt[ok], cv2.RANSAC, 2.0)
                self._timed("homography", started)
                if H is not None and inl.sum() >= 15 and sane(H):
                    return H
        started = time.perf_counter()
        (dx, dy), resp = cv2.phaseCorrelate(np.float32(src), np.float32(dst))  # plain-sky fallback
        self._timed("phase", started)
        if resp < 0.1 or max(abs(dx), abs(dy)) > 40:  # nothing to lock on to: assume the camera held still
            dx = dy = 0.0
        return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], float)

    def _warp_to_now(self, k):
        """gray frame k steps back, warped into the current frame's coordinates (+ valid mask)"""
        started = time.perf_counter()
        H = np.eye(3)
        for i in range(len(self.Hs) - k, len(self.Hs)):
            H = self.Hs[i] @ H
        src = self.grays[-1 - k]; h, w = src.shape
        result = (cv2.warpPerspective(src, H, (w, h), borderMode=cv2.BORDER_REPLICATE),
                  cv2.warpPerspective(np.ones_like(src), H, (w, h)) > 0)
        self._timed("warp", started)
        return result

    def _norm(self, d, valid):
        started = time.perf_counter()
        d = d.astype(np.float32); d[~valid] = 0
        noise = np.median(d) * 1.4826 + 1.0
        result = np.clip(d / noise * self.gain / 4, 0, 255).astype(np.uint8)
        self._timed("normalize", started)
        return result

    def step(self, frame, pred_boxes=(), layers=("motion", "bg", "shape")):
        """Compute requested consensus cues only; retain the temporal state each cue needs."""
        want_motion = "motion" in layers
        want_bg = "bg" in layers
        want_shape = "shape" in layers
        temporal = want_motion or want_bg
        if self.s is None:
            self.s = self.width / frame.shape[1]
        started = time.perf_counter()
        g = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(frame, None, fx=self.s, fy=self.s, interpolation=cv2.INTER_AREA),
                                          cv2.COLOR_BGR2GRAY), (3, 3), 0)
        self._timed("gray", started)
        h, w = g.shape
        L = {}
        if temporal:
            if self.grays:
                started = time.perf_counter()
                self.Hs.append(self._h(self.grays[-1], g, self.prev_feats))
                self._timed("alignment", started)
            self.grays.append(g)
            started = time.perf_counter()
            if self.fast_align:
                self.prev_feats = None
            else:
                self.prev_feats = cv2.goodFeaturesToTrack(g, 800, 0.01, 8)
            self._timed("features", started)
            keep = self.long_gap + 1 if want_motion else 2
            self.grays = self.grays[-keep:]
            self.Hs = self.Hs[-(keep - 1):] if keep > 1 else []
            n = len(self.grays) - 1  # how many past frames are available

            if want_motion:
                zero = np.zeros((h, w), np.uint8)
                if n >= 2:
                    a1, v1 = self._warp_to_now(1); a2, v2 = self._warp_to_now(2)
                    L["short"] = self._norm(np.minimum(cv2.absdiff(g, a1), cv2.absdiff(g, a2)), v1 & v2)
                else:
                    L["short"] = zero
                if n >= self.long_gap:
                    al, vl = self._warp_to_now(self.long_gap)
                    L["long"] = self._norm(cv2.absdiff(g, al), vl)
                else:
                    L["long"] = zero

            if want_bg:
                # Warp the running background with the camera, then blend in the new frame.
                if self.bg is None:
                    self.bg = g.astype(np.float32)
                else:
                    started = time.perf_counter()
                    bgw = cv2.warpPerspective(self.bg, self.Hs[-1], (w, h),
                                              borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
                    bgw[bgw < 0] = g[bgw < 0]
                    self.bg = (1 - self.alpha) * bgw + self.alpha * g
                    self._timed("warp", started)
                if n >= 1:
                    L["bg"] = self._norm(cv2.absdiff(g.astype(np.float32), self.bg), np.ones_like(g, bool))
                else:
                    L["bg"] = np.zeros((h, w), np.uint8)

        if want_shape:
            started = time.perf_counter()
            L["shape"] = brightness_shape(g)
            self._timed("shape", started)
        started = time.perf_counter()
        L["track"] = track_map((h, w), [tuple(v * self.s for v in b) for b in pred_boxes])
        self._timed("track", started)
        return L


def brightness_shape(g, win=121, gain=48.0):
    """|pixel - local mean| / (local spread + 4): stand-out-ness, scaled to 0-255"""
    f = g.astype(np.float32)
    m = cv2.boxFilter(f, -1, (win, win), borderType=cv2.BORDER_REFLECT)
    sd = np.sqrt(np.maximum(cv2.boxFilter(f * f, -1, (win, win), borderType=cv2.BORDER_REFLECT) - m * m, 0))
    return np.clip(np.abs(f - m) / (sd + 4.0) * gain, 0, 255).astype(np.uint8)


def track_map(shape, boxes):
    """soft gaussian spot (peak 255) per expected drone, sized like the drone"""
    h, w = shape; m = np.zeros((h, w), np.float32)
    if not boxes:
        return m.astype(np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    for x1, y1, x2, y2 in boxes:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2; sx, sy = max((x2 - x1) * 0.6, 4), max((y2 - y1) * 0.6, 4)
        m = np.maximum(m, np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2) / 2))
    return (m * 255).astype(np.uint8)


def stack(L, size=None):
    a = np.dstack([L[k] for k in ORDER])
    return cv2.resize(a, size, interpolation=cv2.INTER_LINEAR) if size else a


def view(L, key, size):
    """colour picture of one layer for the demo"""
    im = L[key]
    if key == "prev":
        return cv2.resize(cv2.cvtColor(im, cv2.COLOR_GRAY2BGR), size)
    cmap = cv2.COLORMAP_JET if key == "track" else cv2.COLORMAP_INFERNO
    return cv2.resize(cv2.applyColorMap(im, cmap), size, interpolation=cv2.INTER_NEAREST)


def grid(L, cam, size):
    """Camera plus five colorized maps; omit the two grayscale-only panels."""
    W, H = size
    keys = ("short", "long", "bg", "track", "shape")
    names = ("camera",) + keys
    tw = W // 3
    th = max(1, round(tw * H / W))  # keep every tile at the source aspect ratio
    top = max(0, (H - 2 * th) // 2)
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    for i, (name, key) in enumerate(zip(names, (None,) + keys)):
        x0, x1 = i % 3 * W // 3, (i % 3 + 1) * W // 3
        y0, y1 = top + (i // 3) * th, top + (i // 3 + 1) * th
        tile = cv2.resize(cam if key is None else view(L, key, (x1 - x0, y1 - y0)),
                          (x1 - x0, y1 - y0))
        cv2.putText(tile, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        canvas[y0:y1, x0:x1] = tile
    return canvas
