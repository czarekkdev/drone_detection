"""The whole live detector, one frame at a time:  detector -> prediction crops -> motion crops -> tracker.
stack = DroneStack(YOLO(weights)); out = stack.step(frame)   # [(frame_no, id, (x1,y1,x2,y2), conf)]
stack.info holds what happened this frame (for drawing): dets, crops, motion spots, predictions."""
import numpy as np
from tracker import TwoDroneTracker
from motion_live import MotionDetector


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy; return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


class DroneStack:
    def __init__(self, model, device="cpu", imgsz=1280, crop=320, crop_imgsz=320, motion_topk=4, min_persist=1,
                 use_pred=True, use_motion=True, frame_size=(1920, 1080)):
        self.m, self.dev, self.imgsz, self.crop, self.cimgsz = model, device, imgsz, crop, crop_imgsz
        self.topk, self.min_persist, self.use_pred, self.use_motion = motion_topk, min_persist, use_pred, use_motion
        self.T = TwoDroneTracker(frame_size=frame_size); self.MD = MotionDetector(); self.n = 0; self.info = {}
        self.need_motion_view = False

    def _crops(self, img, centers):
        H, W = img.shape[:2]; tiles, offs, rects = [], [], []
        for cx, cy in centers:
            x0 = int(min(max(cx - self.crop / 2, 0), max(W - self.crop, 0))); y0 = int(min(max(cy - self.crop / 2, 0), max(H - self.crop, 0)))
            tiles.append(img[y0:y0 + self.crop, x0:x0 + self.crop]); offs.append((x0, y0)); rects.append((x0, y0, x0 + self.crop, y0 + self.crop))
        out = []
        if tiles:
            for r, (x0, y0) in zip(self.m.predict(tiles, imgsz=self.cimgsz, conf=0.05, device=self.dev, verbose=False), offs):
                for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                    out.append((x1 + x0, y1 + y0, x2 + x0, y2 + y0, c))
        return out, rects

    @staticmethod
    def _merge(dets, new):
        for b in new:
            same = [i for i, d in enumerate(dets) if iou(b, d) > 0.3]
            if same:
                if b[4] > dets[same[0]][4]:
                    dets[same[0]] = b
            else:
                dets.append(b)
        return dets

    def step(self, img, dets=None):
        self.n += 1; info = {"pred_crops": [], "motion_crops": [], "added": []}
        if dets is None:
            r = self.m.predict(img, imgsz=self.imgsz, conf=0.01, device=self.dev, verbose=False)[0]
            dets = [(*b, c) for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]
        dets = list(dets); info["dets"] = list(dets)
        # motion costs CPU: only computed when motion crops are on or someone is looking at the difference view
        spots = self.MD.step(img) if (self.use_motion or self.need_motion_view) else []
        info["spots"] = spots
        preds = self.T.predictions(); info["preds"] = preds
        if self.use_pred:
            centers = [((pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2) for _, pb, _ in preds
                       if not any(iou(pb, d) > 0.2 and d[4] >= self.T.p["low"] for d in dets)]
            if centers:
                new, info["pred_crops"] = self._crops(img, centers); info["added"] += new; dets = self._merge(dets, new)
        if self.use_motion and spots:
            pc = [((pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2) for _, pb, _ in preds]; cand = []
            for x, y, w, h, s, npx, per in spots:
                cx, cy = x + w / 2, y + h / 2
                if per < self.min_persist:
                    continue
                if any(d[0] - 10 <= cx <= d[2] + 10 and d[1] - 10 <= cy <= d[3] + 10 for d in dets if d[4] >= 0.05):
                    continue
                dist = min([np.hypot(cx - u, cy - v) for u, v in pc], default=1e9)
                cand.append((0 if dist < 300 else 1, -s, cx, cy))
            cand.sort(); centers = []
            for _, _, cx, cy in cand:
                if any(abs(cx - u) < self.crop / 3 and abs(cy - v) < self.crop / 3 for u, v in centers):
                    continue
                centers.append((cx, cy))
                if len(centers) >= self.topk:
                    break
            if centers:
                new, info["motion_crops"] = self._crops(img, centers); info["added"] += new; dets = self._merge(dets, new)
        out = self.T.step(self.n, dets); self.info = info
        return out
