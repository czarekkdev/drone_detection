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

ORDER = ("short", "long", "bg", "track", "prev", "shape")


def sane(H, max_scale=0.05, max_persp=2e-4):
    """reject warps a hand-held camera can't do between frames (plain-sky noise corners give crazy ones)"""
    A = H[:2, :2] / H[2, 2]
    return (np.abs(A - np.eye(2)).max() < max_scale and np.abs(H[2, :2]).max() < max_persp
            and np.abs(H[:2, 2] / H[2, 2]).max() < 60)


class LayerMaker:
    def __init__(self, width=960, long_gap=10, bg_alpha=0.05, gain=32.0):
        self.width, self.long_gap, self.alpha, self.gain = width, long_gap, bg_alpha, gain
        self.grays, self.Hs = [], []  # Hs[i]: homography grays[i] -> grays[i+1]
        self.bg = None; self.s = None; self.prev_feats = None

    def _h(self, src, dst, pts):
        if pts is not None and len(pts) >= 20:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(src, dst, pts, None, winSize=(21, 21), maxLevel=3)
            ok = st.ravel() == 1
            if ok.sum() >= 20:
                H, inl = cv2.findHomography(pts[ok], nxt[ok], cv2.RANSAC, 2.0)
                if H is not None and inl.sum() >= 15 and sane(H):
                    return H
        (dx, dy), resp = cv2.phaseCorrelate(np.float32(src), np.float32(dst))  # plain-sky fallback
        if resp < 0.1 or max(abs(dx), abs(dy)) > 40:  # nothing to lock on to: assume the camera held still
            dx = dy = 0.0
        return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], float)

    def _warp_to_now(self, k):
        """gray frame k steps back, warped into the current frame's coordinates (+ valid mask)"""
        H = np.eye(3)
        for i in range(len(self.Hs) - k, len(self.Hs)):
            H = self.Hs[i] @ H
        src = self.grays[-1 - k]; h, w = src.shape
        return cv2.warpPerspective(src, H, (w, h), borderMode=cv2.BORDER_REPLICATE), \
            cv2.warpPerspective(np.ones_like(src), H, (w, h)) > 0

    def _norm(self, d, valid):
        d = d.astype(np.float32); d[~valid] = 0
        noise = np.median(d) * 1.4826 + 1.0
        return np.clip(d / noise * self.gain / 4, 0, 255).astype(np.uint8)

    def step(self, frame, pred_boxes=()):
        if self.s is None:
            self.s = self.width / frame.shape[1]
        g = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(frame, None, fx=self.s, fy=self.s, interpolation=cv2.INTER_AREA),
                                          cv2.COLOR_BGR2GRAY), (3, 3), 0)
        h, w = g.shape
        if self.grays:
            self.Hs.append(self._h(self.grays[-1], g, self.prev_feats))
        self.grays.append(g); self.prev_feats = cv2.goodFeaturesToTrack(g, 800, 0.01, 8)
        keep = self.long_gap + 1
        self.grays, self.Hs = self.grays[-keep:], self.Hs[-(keep - 1):] if keep > 1 else []
        zero = np.zeros((h, w), np.uint8); L = {}
        n = len(self.grays) - 1  # how many past frames we have
        if n >= 2:
            a1, v1 = self._warp_to_now(1); a2, v2 = self._warp_to_now(2)
            L["short"] = self._norm(np.minimum(cv2.absdiff(g, a1), cv2.absdiff(g, a2)), v1 & v2)
            L["prev"] = a1
        else:
            L["short"] = zero; L["prev"] = g.copy()
        if n >= self.long_gap:
            al, vl = self._warp_to_now(self.long_gap)
            L["long"] = self._norm(cv2.absdiff(g, al), vl)
        else:
            L["long"] = zero
        # background: carry the old background along with the camera, then blend in the new frame
        if self.bg is None:
            self.bg = g.astype(np.float32)
        else:
            Hl = self.Hs[-1]
            bgw = cv2.warpPerspective(self.bg, Hl, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
            bgw[bgw < 0] = g[bgw < 0]
            self.bg = (1 - self.alpha) * bgw + self.alpha * g
        L["bg"] = self._norm(cv2.absdiff(g.astype(np.float32), self.bg), np.ones_like(g, bool)) if n >= 1 else zero
        L["shape"] = brightness_shape(g)
        L["track"] = track_map((h, w), [tuple(v * self.s for v in b) for b in pred_boxes])
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
    """camera + all layers and camera-compensated motion blobs in a 4x2 grid"""
    W, H = size; tw, th = W // 4, H // 2
    tiles = [cv2.resize(cam, (tw, th))] + [view(L, k, (tw, th)) for k in ORDER]
    motion = np.maximum(L["short"], L["long"])
    blobs = (motion > 96).astype(np.uint8) * 255
    blobs = cv2.morphologyEx(blobs, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(blobs, 8)
    clean = np.zeros_like(blobs)
    for i in range(1, count):
        if 2 <= stats[i, cv2.CC_STAT_AREA] <= 0.004 * blobs.size:
            clean[labels == i] = 255
    tiles.append(cv2.resize(cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR), (tw, th), interpolation=cv2.INTER_NEAREST))
    for t, name in zip(tiles, ("camera",) + ORDER + ("motion blobs",)):
        cv2.putText(t, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    g = cv2.vconcat([cv2.hconcat(tiles[:4]), cv2.hconcat(tiles[4:])])
    return cv2.resize(g, size)
