"""Real-time pipeline for the Raspberry Pi + AI HAT+ (Hailo-8), running the FULL best setup (MOTA 0.883 on the practice videos):
whole-frame SkyGuard + 2x-enlarged trajectory crops with the fine-tuned crop model + shape layer (verify, agreement booster,
motion-checked keeper) + tracker with quick re-acquisition. Same stages and numbers as pipeline_boost.py's best flags (boost.py).
3 threads overlap: read -> whole-frame model -> finish (crops + shape layer + booster/keeper + tracker).
Same code on the Mac (PyTorch backend) and on the Pi (Hailo backend, two .hef files: whole frame and crop model).

  Mac, all practice videos (check the score):
    python pi_run.py ../test_sample_clean --out results
  Mac, same but reuse the cached whole-frame detections (must give exactly the pipeline_boost.py result):
    python pi_run.py ../test_sample_clean --out results --dets preds/hf_skyguard
  Pi:   python pi_run.py video.mp4 --backend hailo --hef skyguard_1280.hef --crop-hef crop_640.hef
  live: python pi_run.py 0 --backend hailo --hef skyguard_1280.hef --crop-hef crop_640.hef      (camera 0)
  old simple pipeline (no shape layer, no 2x fine-tuned crops): add --simple

Prints: end-to-end fps, average ms per stage (read / whole-frame model / crops / shape layer / tracker), and the slowest stage
(= the fps limit)."""
import argparse, os, glob, queue, threading, time
import cv2
import numpy as np
from tracker import TwoDroneTracker, dedupe_lines
from boost import Booster


# ---------------- model backends ----------------
class TorchBackend:
    def __init__(self, weights, imgsz, device=None):
        import torch
        from ultralytics import YOLO
        self.m = YOLO(weights); self.imgsz = imgsz
        self.dev = device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    def infer(self, images, imgsz=None, conf=0.01):
        """images: list of BGR arrays -> list of [(x1,y1,x2,y2,conf), ...] in each image's pixels"""
        rs = self.m.predict(images, imgsz=imgsz or self.imgsz, conf=conf, device=self.dev, verbose=False)
        return [[(*b, c) for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())] for r in rs]


class HailoBackend:
    """HailoRT backend (Raspberry Pi AI HAT+). Expects a .hef compiled WITH the NMS post-process
    (standard for Hailo model-zoo YOLO builds). Not testable off the Pi: report errors and we fix them there."""
    def __init__(self, hef_path, vdevice=None):
        from hailo_platform import (HEF, VDevice, HailoStreamInterface, ConfigureParams, InferVStreams,
                                    InputVStreamParams, OutputVStreamParams, FormatType)
        self.hef = HEF(hef_path); self.dev = vdevice or VDevice()  # share one VDevice for the 2 models
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

    def infer(self, images, imgsz=None, conf=0.01):
        batch, meta = [], []
        for im in images:
            x, s, px, py = self._letterbox(im); batch.append(x); meta.append((s, px, py))
        res = self.pipe.infer({self.inp.name: np.stack(batch)})[self.outname]
        out = []
        for r, (s, px, py) in zip(res, meta):
            dets = []
            for cls_dets in r:  # per class: array of [ymin, xmin, ymax, xmax, score] normalised to the input
                for y1, x1, y2, x2, sc in np.asarray(cls_dets).reshape(-1, 5):
                    if sc < conf:
                        continue
                    dets.append(((x1 * self.w - px) / s, (y1 * self.h - py) / s, (x2 * self.w - px) / s, (y2 * self.h - py) / s, float(sc)))
            out.append(dets)
        return out


# ---------------- pipeline ----------------
def read_dets(path):
    D = {}
    if path and os.path.exists(path):
        for l in open(path):
            v = l.strip().split(",")
            if len(v) > 5:
                f, _, x, y, w, h, c = [float(q) for q in v[:7]]
                D.setdefault(int(f), []).append((x, y, x + w, y + h, c))
    return D


def run(source, backend, booster, out_path, realtime=False, cached=None, output_format="mot"):
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    W, H = int(cap.get(3)), int(cap.get(4)); src_fps = cap.get(5) or 30
    q1, q2 = queue.Queue(maxsize=4), queue.Queue(maxsize=4)
    T = TwoDroneTracker(frame_size=(W, H)); booster.reset()
    t = {"read": [], "model": [], "finish": [], "track": []}; lines = []; done_t = {}
    got_times, answer_times, frame_boxes = {}, {}, {}

    def reader():
        n = 0; t_start = time.time()
        while True:
            t0 = time.time(); ok, f = cap.read(); t["read"].append(time.time() - t0)
            if not ok:
                break
            t_got = time.time()
            n += 1
            if realtime:  # behave like a camera: frames arrive at the video's fps
                wait = t_start + n / src_fps - time.time()
                if wait > 0:
                    time.sleep(wait)
            got_times[n] = t_got
            q1.put((n, f, t_got))
        q1.put(None)

    def model():  # whole-frame detector (or cached detections)
        while True:
            item = q1.get()
            if item is None:
                q2.put(None); break
            n, f, t_in = item; t0 = time.time()
            dets = list(cached.get(n, [])) if cached is not None else backend.infer([f])[0]
            t["model"].append(time.time() - t0); q2.put((n, f, dets, t_in))

    def finish():  # crops + shape layer + booster/keeper + tracker, in frame order (needs the up-to-date tracker)
        lat = []
        while True:
            item = q2.get()
            if item is None:
                break
            n, f, dets, t_in = item; t0 = time.time()
            dets = booster.step(f, dets, T)
            t1 = time.time()
            tracked = T.step(n, dets)
            t_answer = time.time()
            answer_times[n] = t_answer
            for f2, i, (x1, y1, x2, y2), c in tracked:
                lines.append(f"{f2},{i},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{c:.4f}")
                frame_boxes.setdefault(f2, []).append((i, (x1, y1, x2, y2), c, t_answer))
            t["track"].append(time.time() - t1); t["finish"].append(time.time() - t0); lat.append(time.time() - t_in)
        done_t["lat"] = lat; done_t["n"] = len(lat)

    ths = [threading.Thread(target=x) for x in (reader, model, finish)]
    t_start = time.time()
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    total = time.time() - t_start; n = done_t["n"]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if output_format == "competition":
        formatted = []
        for frame in range(1, n + 1):
            boxes = {}
            for tid, box, conf, _ in frame_boxes.get(frame, []):
                if tid not in boxes or conf > boxes[tid][1]:
                    boxes[tid] = (box, conf)
            t_got = got_times[frame]
            t_answer = max([answer_times.get(frame, t_got)] +
                           [item[3] for item in frame_boxes.get(frame, [])])
            if not boxes:
                formatted.append(f"{frame},-1,0,0,0,0,{t_got:.3f},{t_answer:.3f}")
                continue
            for tid, (box, _) in sorted(boxes.items()):
                x1, y1, x2, y2 = box
                formatted.append(f"{frame},{tid},{x1:.2f},{y1:.2f},{x2-x1:.2f},{y2-y1:.2f},"
                                 f"{t_got:.3f},{t_answer:.3f}")
        with open(out_path, "w") as f:
            f.write("\n".join(formatted) + "\n")
    else:
        lines = dedupe_lines(lines)
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")
    ms = {k: 1000 * np.mean(v) if v else 0 for k, v in t.items()}
    bt = {k: 1000 * np.mean(v) if v else 0 for k, v in booster.t.items()}
    stages = {"read": ms["read"], "model": ms["model"], "finish": ms["finish"]}  # the 3 threads
    slow = max(stages, key=stages.get)
    print(f"{os.path.basename(out_path)}: {n} frames in {total:.1f}s = {n / total:.1f} fps end-to-end | "
          f"read {ms['read']:.1f} | whole-frame model {ms['model']:.1f}{' (cached)' if cached is not None else ''} | "
          f"finish {ms['finish']:.1f} = crops {bt['crops']:.1f} + shape layer {bt['layers']:.1f} + booster/keeper {bt['logic']:.2f} "
          f"+ tracker {ms['track']:.2f} ms/frame | limit = {slow} thread ({1000 / max(stages[slow], 1e-6):.0f} fps) | "
          f"latency {1000 * np.median(done_t['lat']):.0f} ms")
    if booster.use_layers and n:
        lt = booster.LM.timing
        parts = " | ".join(f"{key} {1000 * value / n:.2f}" for key, value in lt.items())
        print(f"  layer CPU ms/input-frame (amortized; align total includes flow/homography/phase): "
              f"{parts}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source", help="video file, folder of video folders (practice-set layout), or camera number")
    p.add_argument("--backend", choices=["torch", "hailo"], default="torch")
    p.add_argument("--weights", default="../pretrained/drone-detection-yolov11/best.pt", help="whole-frame model (SkyGuard)")
    p.add_argument("--crop-weights", default="runs/ft_1280/weights/epoch0.pt", help="crop model (release v0.883 asset)")
    p.add_argument("--hef", default="skyguard.hef", help="Hailo: whole-frame model")
    p.add_argument("--crop-hef", default="crop_640.hef", help="Hailo: crop model, compiled at 640x640")
    p.add_argument("--imgsz", type=int, default=1280, help="whole-frame size (torch backend; on Hailo it is fixed by the .hef)")
    p.add_argument("--layer-width", type=int, default=480)
    p.add_argument("--layers", default="motion,bg,shape",
                   help="consensus cues: comma-separated motion,bg,shape (default: all three)")
    p.add_argument("--layer-every", type=int, default=1,
                   help="compute layer cues once every N frames; detections/tracking still run each frame")
    p.add_argument("--fast-align", action="store_true",
                   help="use translation-only phase correlation instead of CPU feature tracking and homography")
    p.add_argument("--device", default=None)
    p.add_argument("--dets", default="", help="folder of cached whole-frame detections (run_video.py output) instead of the model")
    p.add_argument("--simple", action="store_true", help="old pipeline: no shape layer, plain 1x crops with the whole-frame model")
    p.add_argument("--no-layers", action="store_true",
                   help="disable layer processing but keep the separate crop HEF and enlarged crops")
    p.add_argument("--realtime", action="store_true", help="feed frames at the video's own fps, like a camera")
    p.add_argument("--cmc", action="store_true", help="experimental: shift the tracker with the camera motion (off: 0.860 -> 0.863 MOTA "
                   "and 101 -> 87 wrong boxes in a test, but one re-entry after both drones left got the other number)")
    p.add_argument("--out", default="results")
    p.add_argument("--output-format", choices=("mot", "competition"), default="mot",
                   help="MOT text files in --out, or detection.txt beside each input video with timestamps")
    a = p.parse_args()
    layer_names = tuple(x.strip() for x in a.layers.split(",") if x.strip())
    invalid_layers = set(layer_names) - {"motion", "bg", "shape"}
    if invalid_layers or not layer_names:
        p.error("--layers must contain one or more of: motion,bg,shape")
    if a.layer_every < 1:
        p.error("--layer-every must be at least 1")
    if a.cmc and a.layer_every > 1:
        p.error("--cmc requires --layer-every 1")
    if a.backend == "hailo":
        from hailo_platform import VDevice
        params = VDevice.create_params()
        if not a.simple:  # scheduling is only useful when the separate crop HEF is active
            try:
                from hailo_platform import HailoSchedulingAlgorithm
                params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
            except Exception:
                pass
        vd = VDevice(params)
        be = HailoBackend(a.hef, vd)
        cb = be if a.simple else HailoBackend(a.crop_hef, vd)
    else:
        be = TorchBackend(a.weights, a.imgsz, a.device)
        cb = be if a.simple else TorchBackend(a.crop_weights, 640, a.device)
    if a.simple:
        booster = Booster(cb, crop_scale=1.0, crop_imgsz=320, use_layers=False)
    else:
        booster = Booster(cb, crop_scale=2.0, crop_imgsz=320, layer_width=a.layer_width, cmc=a.cmc,
                          use_layers=not a.no_layers, layers=layer_names, layer_every=a.layer_every,
                          fast_align=a.fast_align)
    if a.output_format == "mot":
        os.makedirs(a.out, exist_ok=True)
    if os.path.isdir(a.source):
        for v in sorted(glob.glob(os.path.join(a.source, "*", "video.mp4"))):
            name = os.path.basename(os.path.dirname(v))
            cached = read_dets(os.path.join(a.dets, name + ".txt")) if a.dets else None
            out_path = (os.path.join(os.path.dirname(v), "detection.txt") if a.output_format == "competition"
                        else os.path.join(a.out, name + ".txt"))
            run(v, be, booster, out_path, cached=cached, output_format=a.output_format)
    else:
        name = "camera" if a.source.isdigit() else (os.path.basename(os.path.dirname(os.path.abspath(a.source))) if os.path.basename(a.source) == "video.mp4" else os.path.splitext(os.path.basename(a.source))[0])
        cached = read_dets(os.path.join(a.dets, name + ".txt")) if a.dets else None
        if a.output_format == "competition":
            out_path = os.path.join(a.out, "detection.txt") if a.source.isdigit() else os.path.join(os.path.dirname(os.path.abspath(a.source)), "detection.txt")
        else:
            out_path = os.path.join(a.out, name + ".txt")
        run(a.source, be, booster, out_path, realtime=a.realtime, cached=cached,
            output_format=a.output_format)
