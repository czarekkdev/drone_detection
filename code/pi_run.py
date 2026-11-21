"""Pi-style real-time pipeline: 3 threads that overlap (read -> model -> tracker), per-stage timing, result file.
Same code on the Mac (PyTorch backend) and on the Raspberry Pi + AI HAT+ (Hailo backend).

  Mac:  python pi_run.py ../test_sample_clean/test_6_hard_PXL_20261113_154422245_hard/video.mp4
  Pi:   python pi_run.py video.mp4 --backend hailo --hef skyguard.hef
  live: python pi_run.py 0 --backend hailo --hef skyguard.hef          (camera 0)
  all practice videos:  python pi_run.py ../test_sample_clean --out results

Prints: end-to-end fps, average ms per stage (read/model/track), and the slowest stage (= the fps limit)."""
import argparse, os, glob, queue, threading, time
import cv2
import numpy as np
from tracker import TwoDroneTracker


# ---------------- model backends ----------------
class TorchBackend:
    def __init__(self, weights, imgsz, device=None):
        import torch
        from ultralytics import YOLO
        self.m = YOLO(weights); self.imgsz = imgsz
        self.dev = device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    def infer(self, images, imgsz=None):
        """images: list of BGR arrays -> list of [(x1,y1,x2,y2,conf), ...] in each image's pixels"""
        rs = self.m.predict(images, imgsz=imgsz or self.imgsz, conf=0.01, device=self.dev, verbose=False)
        return [[(*b, c) for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())] for r in rs]


class HailoBackend:
    """HailoRT backend (Raspberry Pi AI HAT+). Expects a .hef compiled WITH the NMS post-process
    (standard for Hailo model-zoo YOLO builds). Not testable off the Pi: report errors and we fix them there."""
    def __init__(self, hef_path):
        from hailo_platform import (HEF, VDevice, HailoStreamInterface, ConfigureParams, InferVStreams,
                                    InputVStreamParams, OutputVStreamParams, FormatType)
        self.hef = HEF(hef_path); self.dev = VDevice()
        cp = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        self.ng = self.dev.configure(self.hef, cp)[0]; self.ngp = self.ng.create_params()
        self.inp = self.hef.get_input_vstream_infos()[0]; self.outname = self.hef.get_output_vstream_infos()[0].name
        self.h, self.w = self.inp.shape[0], self.inp.shape[1]
        ip = InputVStreamParams.make_from_network_group(self.ng, quantized=False, format_type=FormatType.UINT8)
        op = OutputVStreamParams.make_from_network_group(self.ng, quantized=False, format_type=FormatType.FLOAT32)
        self._ctx = self.ng.activate(self.ngp); self._ctx.__enter__()
        self.pipe = InferVStreams(self.ng, ip, op); self.pipe.__enter__()

    def _letterbox(self, img):
        H, W = img.shape[:2]; s = min(self.w / W, self.h / H)
        nw, nh = int(round(W * s)), int(round(H * s)); px, py = (self.w - nw) // 2, (self.h - nh) // 2
        canvas = np.full((self.h, self.w, 3), 114, np.uint8)
        canvas[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), s, px, py

    def infer(self, images, imgsz=None):
        batch, meta = [], []
        for im in images:
            x, s, px, py = self._letterbox(im); batch.append(x); meta.append((s, px, py))
        res = self.pipe.infer({self.inp.name: np.stack(batch)})[self.outname]
        out = []
        for r, (s, px, py) in zip(res, meta):
            dets = []
            for cls_dets in r:  # per class: array of [ymin, xmin, ymax, xmax, score] normalised to the input
                for y1, x1, y2, x2, sc in np.asarray(cls_dets).reshape(-1, 5):
                    dets.append(((x1 * self.w - px) / s, (y1 * self.h - py) / s, (x2 * self.w - px) / s, (y2 * self.h - py) / s, float(sc)))
            out.append(dets)
        return out


# ---------------- pipeline ----------------
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy; return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def run(source, backend, out_path, use_pred=True, crop=320, realtime=False):
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    W, H = int(cap.get(3)), int(cap.get(4)); src_fps = cap.get(5) or 30
    q1, q2 = queue.Queue(maxsize=4), queue.Queue(maxsize=4)
    T = TwoDroneTracker(frame_size=(W, H)); lock = threading.Lock()
    t = {"read": [], "model": [], "track": []}; lines = []; done_t = {}

    def reader():
        n = 0; t_start = time.time()
        while True:
            t0 = time.time(); ok, f = cap.read(); t["read"].append(time.time() - t0)
            if not ok:
                break
            n += 1
            if realtime:  # behave like a camera: frames arrive at the video's fps, drop if we fall behind
                wait = t_start + n / src_fps - time.time()
                if wait > 0:
                    time.sleep(wait)
            q1.put((n, f, time.time()))
        q1.put(None)

    def model():
        while True:
            item = q1.get()
            if item is None:
                q2.put(None); break
            n, f, t_in = item; t0 = time.time()
            dets = backend.infer([f])[0]
            if use_pred:  # crops at predicted spots of drones the full-frame pass missed
                with lock:
                    preds = T.predictions()
                low = T.p["low"]; centers = []
                for _, pb, _ in preds:
                    if not any(iou(pb, d) > 0.2 and d[4] >= low for d in dets):
                        centers.append(((pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2))
                if centers:
                    tiles, offs = [], []
                    for cx, cy in centers:
                        x0 = int(min(max(cx - crop / 2, 0), W - crop)); y0 = int(min(max(cy - crop / 2, 0), H - crop))
                        tiles.append(f[y0:y0 + crop, x0:x0 + crop]); offs.append((x0, y0))
                    for ds, (x0, y0) in zip(backend.infer(tiles, imgsz=crop), offs):
                        for x1, y1, x2, y2, c in ds:
                            b = (x1 + x0, y1 + y0, x2 + x0, y2 + y0, c)
                            if not any(iou(b, d) > 0.3 for d in dets):
                                dets.append(b)
            t["model"].append(time.time() - t0); q2.put((n, dets, t_in))

    def tracker():
        lat = []
        while True:
            item = q2.get()
            if item is None:
                break
            n, dets, t_in = item; t0 = time.time()
            with lock:
                out = T.step(n, dets)
            for f2, i, (x1, y1, x2, y2), c in out:
                lines.append(f"{f2},{i},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{c:.4f}")
            t["track"].append(time.time() - t0); lat.append(time.time() - t_in)
        done_t["lat"] = lat; done_t["n"] = len(lat)

    ths = [threading.Thread(target=x) for x in (reader, model, tracker)]
    t_start = time.time()
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    total = time.time() - t_start; n = done_t["n"]
    lines.sort(key=lambda l: (int(l.split(",")[0]), int(l.split(",")[1])))
    open(out_path, "w").write("\n".join(lines) + "\n")
    ms = {k: 1000 * np.mean(v) if v else 0 for k, v in t.items()}
    slow = max(ms, key=ms.get)
    print(f"{os.path.basename(out_path)}: {n} frames in {total:.1f}s = {n / total:.1f} fps end-to-end | "
          f"read {ms['read']:.1f} ms, model {ms['model']:.1f} ms, track {ms['track']:.2f} ms per frame | "
          f"limit = {slow} ({1000 / max(ms[slow], 1e-6):.0f} fps) | latency {1000 * np.median(done_t['lat']):.0f} ms")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source", help="video file, folder of video folders (practice-set layout), or camera number")
    p.add_argument("--backend", choices=["torch", "hailo"], default="torch")
    p.add_argument("--weights", default="../pretrained/drone-detection-yolov11/best.pt")
    p.add_argument("--hef", default="skyguard.hef")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--device", default=None)
    p.add_argument("--no-pred", action="store_true")
    p.add_argument("--realtime", action="store_true", help="feed frames at the video's own fps, like a camera")
    p.add_argument("--out", default="results")
    a = p.parse_args()
    be = HailoBackend(a.hef) if a.backend == "hailo" else TorchBackend(a.weights, a.imgsz, a.device)
    os.makedirs(a.out, exist_ok=True)
    if os.path.isdir(a.source):
        for v in sorted(glob.glob(os.path.join(a.source, "*", "video.mp4"))):
            run(v, be, os.path.join(a.out, os.path.basename(os.path.dirname(v)) + ".txt"), not a.no_pred)
    else:
        name = "camera" if a.source.isdigit() else (os.path.basename(os.path.dirname(os.path.abspath(a.source))) if os.path.basename(a.source) == "video.mp4" else os.path.splitext(os.path.basename(a.source))[0])
        run(a.source, be, os.path.join(a.out, name + ".txt"), not a.no_pred, realtime=a.realtime)
