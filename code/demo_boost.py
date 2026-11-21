"""Live demo of the current best pipeline (MOTA 0.878 on the practice videos):
  original SkyGuard on the full frame (low conf)  +  ft_1280 model on trajectory crops (real frame)
  + consensus white-shape layer (480 px): verifier, agreement BOOST, motion-consistent KEEPER  +  tracker
  python demo_boost.py                                  (hard practice video, window on)
  python demo_boost.py --video path/to/any.mp4          (any video; --video 0 = camera)
Keys:  1 camera   c consensus / white-shape view   0 all layers   t trajectory on/off
       b booster on/off   k keeper on/off   space pause   q quit
Colours: orange/blue = drone 1/2 (final output), thin white = correct answer (practice videos only),
  YELLOW box = a faint detection that the booster raised (white blob at the same place),
  DASHED box = a frame where the keeper held the track (blob moving like the drone), green = trajectory crop checked.
Writes demo_out/boost_<name>.mp4 and .txt (skip the video with --no-save). This is a separate demo; demo.py is unchanged."""
import argparse, os, time
import cv2, numpy as np, torch
from ultralytics import YOLO
from tracker_reacq import TwoDroneTracker  # tracker.py + box clipping + quick re-acquisition after a drone is lost (sudden stops/turns)
import layers as LY
from consensus import Consensus, view as cview

HERE = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument("--video", default=os.path.join(HERE, "../test_sample_clean/test_6_hard_PXL_20261113_154422245_hard/video.mp4"))
p.add_argument("--full-weights", default=os.path.join(HERE, "../pretrained/drone-detection-yolov11/best.pt"))
p.add_argument("--crop-weights", default=os.path.join(HERE, "runs/ft_1280/weights/epoch0.pt"))
p.add_argument("--imgsz", type=int, default=1280)
p.add_argument("--no-show", action="store_true"); p.add_argument("--no-save", action="store_true")
p.add_argument("--out", default=os.path.join(HERE, "demo_out"))
p.add_argument("--layer-width", type=int, default=480)
p.add_argument("--boost-lo", type=float, default=0.10); p.add_argument("--boost-to", type=float, default=0.65)
p.add_argument("--still-window", type=int, default=30, help="frames before suppressing a track with no movement or pixel change (0 disables)")
p.add_argument("--still-motion", type=float, default=40.0, help="minimum mean aligned pixel-change score to keep a stationary track")
p.add_argument("--still-drift", type=float, default=6.0, help="minimum center movement in pixels across the stillness window")
a = p.parse_args()
dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
full, crop_m = YOLO(a.full_weights), YOLO(a.crop_weights)
src = int(a.video) if a.video.isdigit() else a.video
cap = cv2.VideoCapture(src)
W, H = int(cap.get(3)), int(cap.get(4)); fps_in = cap.get(5) or 30
vdir = "" if isinstance(src, int) else os.path.dirname(os.path.abspath(a.video))
name = f"camera{src}" if isinstance(src, int) else (os.path.basename(vdir) if os.path.basename(a.video) == "video.mp4" else os.path.splitext(os.path.basename(a.video))[0])
gt = {}
if vdir and os.path.exists(os.path.join(vdir, "gt", "gt.txt")):
    for l in open(os.path.join(vdir, "gt", "gt.txt")):
        v = l.split(","); gt.setdefault(int(v[0]), []).append(tuple(map(float, v[2:6])))
os.makedirs(a.out, exist_ok=True)
writer = None if a.no_save else cv2.VideoWriter(os.path.join(a.out, f"boost_{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (W, H))
T = TwoDroneTracker(frame_size=(W, H)); LM = LY.LayerMaker(width=a.layer_width); CS = Consensus(near=60 * a.layer_width / 960)
COL = {1: (0, 165, 255), 2: (255, 120, 0)}
booster, keeper, traj, paused, view = True, True, True, False, 1
krun, prev_bl, TRAIL, lines = {}, [], {}, []
still_history = {}  # per-ID recent center positions and camera-aligned pixel change


def iou(p_, q):
    ix = max(0, min(p_[2], q[2]) - max(p_[0], q[0])); iy = max(0, min(p_[3], q[3]) - max(p_[1], q[1]))
    i = ix * iy; return i / ((p_[2] - p_[0]) * (p_[3] - p_[1]) + (q[2] - q[0]) * (q[3] - q[1]) - i + 1e-9)


def same_place(d, b):
    dcx, dcy = (d[0] + d[2]) / 2, (d[1] + d[3]) / 2; bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    inside = (b[0] - 4 <= dcx <= b[2] + 4 and b[1] - 4 <= dcy <= b[3] + 4) or (d[0] <= bcx <= d[2] and d[1] <= bcy <= d[3])
    ds, bs = max(d[2] - d[0], d[3] - d[1], 1), max(b[2] - b[0], b[3] - b[1], 1)
    return inside and 0.5 <= ds / bs <= 2.0


def crops(img, centers, size=320, up=2):
    """zoomed check where a drone is expected; the crop is enlarged `up` times so the drone looks bigger to the model"""
    tiles, offs = [], []
    for cx, cy in centers:
        x0 = int(min(max(cx - size / 2, 0), max(W - size, 0))); y0 = int(min(max(cy - size / 2, 0), max(H - size, 0)))
        t = img[y0:y0 + size, x0:x0 + size]
        tiles.append(cv2.resize(t, (t.shape[1] * up, t.shape[0] * up), interpolation=cv2.INTER_CUBIC)); offs.append((x0, y0))
    out = []
    if tiles:
        for r, (x0, y0) in zip(crop_m.predict(tiles, imgsz=size * up, conf=0.05, device=dev, verbose=False), offs):
            for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                out.append((x1 / up + x0, y1 / up + y0, x2 / up + x0, y2 / up + y0, c))
    return out, offs


def merge(dets, new):
    for b in new:
        same = [i for i, d in enumerate(dets) if iou(b, d) > 0.3]
        if same:
            if b[4] > dets[same[0]][4]:
                dets[same[0]] = b
        else:
            dets.append(b)
    return dets


def dashed(img, b, col, step=10):
    x1, y1, x2, y2 = map(int, b[:4])
    for x in range(x1, x2, step * 2):
        cv2.line(img, (x, y1), (min(x + step, x2), y1), col, 2); cv2.line(img, (x, y2), (min(x + step, x2), y2), col, 2)
    for y in range(y1, y2, step * 2):
        cv2.line(img, (x1, y), (x1, min(y + step, y2)), col, 2); cv2.line(img, (x2, y), (x2, min(y + step, y2)), col, 2)


def motion_outline(img, box, motion, scale):
    """Draw the largest compact camera-compensated pixel-change region inside a track box."""
    x1, y1, x2, y2 = [int(v * scale) for v in box]
    x1, y1 = max(x1, 0), max(y1, 0); x2, y2 = min(x2, motion.shape[1]), min(y2, motion.shape[0])
    if x2 <= x1 or y2 <= y1:
        return
    roi = (motion[y1:y2, x1:x2] > 96).astype(np.uint8) * 255
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(roi, 8)
    if count <= 1:
        return
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[idx, cv2.CC_STAT_AREA] < 2:
        return
    component = (labels == idx).astype(np.uint8) * 255
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [np.round(c.astype(np.float32) / scale + np.array([[[x1 / scale, y1 / scale]]])).astype(np.int32) for c in contours]
    cv2.drawContours(img, contours, -1, (255, 255, 0), 1)


def step(img):
    global prev_bl
    info = {"boosted": [], "kept": [], "crops": []}
    r = full.predict(img, imgsz=a.imgsz, conf=0.01, device=dev, verbose=False)[0]
    dets = [(*b, c) for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]
    preds = T.predictions()
    centers = [((pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2) for _, pb, _ in preds
               if not any(iou(pb, d) > 0.2 and d[4] >= T.p["low"] for d in dets)]
    if centers:
        new, offs = crops(img, centers); dets = merge(dets, new); info["crops"] = offs
    L = LM.step(img, [pb for _, pb, _ in preds]); cbl = CS.step(L, LM.s); info["L"], info["blobs"] = L, cbl
    mk = CS.mask; Hm, Wm = mk.shape; nd = []
    for d in dets:  # verifier: weak boxes with no layer support count less
        if d[4] < 0.5:
            x1, y1, x2, y2 = [int(v * LM.s) for v in d[:4]]
            x1, y1 = max(x1 - 2, 0), max(y1 - 2, 0); x2, y2 = min(x2 + 3, Wm), min(y2 + 3, Hm)
            if x2 <= x1 or y2 <= y1 or not mk[y1:y2, x1:x2].any():
                d = (*d[:4], d[4] * 0.3)
        nd.append(d)
    dets = nd
    if booster:  # faint box + white blob at the same place & size -> trust it
        nd = []
        for d in dets:
            if a.boost_lo <= d[4] < 0.3 and any(same_place(d, b) for b in cbl):
                d = (*d[:4], max(d[4], a.boost_to)); info["boosted"].append(d)
            nd.append(d)
        dets = nd
    if keeper:  # keep a track through a gap only if a blob moves like the drone
        vel = {t.id: (t.kf.x[4], t.kf.x[5]) for t in T.active if t.id is not None}
        for tid, pb, miss in preds:
            if any(iou(pb, d) > 0.2 and d[4] >= T.p["low"] for d in dets):
                krun[tid] = 0; continue
            if krun.get(tid, 0) >= 5 or tid not in vel:
                continue
            pw, ph = pb[2] - pb[0], pb[3] - pb[1]; rr = max(pw, ph, 20)
            pcx, pcy = (pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2; vx, vy = vel[tid]; vs = np.hypot(vx, vy); ok_b = None
            for b in cbl:
                bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                if np.hypot(bcx - pcx, bcy - pcy) >= rr or not 0.5 <= max(b[2] - b[0], b[3] - b[1], 1) / max(pw, ph, 1) <= 2.0:
                    continue
                prv = [q for q in prev_bl if np.hypot((q[0] + q[2]) / 2 - bcx, (q[1] + q[3]) / 2 - bcy) < rr]
                if not prv:
                    continue
                q = min(prv, key=lambda q: np.hypot((q[0] + q[2]) / 2 - bcx, (q[1] + q[3]) / 2 - bcy))
                mx, my = bcx - (q[0] + q[2]) / 2, bcy - (q[1] + q[3]) / 2; ms = np.hypot(mx, my)
                if (vs < 2 and ms < 2) or (vs >= 2 and ms >= 1 and 0.5 <= ms / vs <= 1.5 and (mx * vx + my * vy) / (ms * vs) >= np.cos(np.pi / 4)):
                    ok_b = b; break
            if ok_b is not None:
                cx, cy = (ok_b[0] + ok_b[2]) / 2, (ok_b[1] + ok_b[3]) / 2
                kb = (cx - pw / 2, cy - ph / 2, cx + pw / 2, cy + ph / 2, 0.25); dets.append(kb); info["kept"].append(kb)
                krun[tid] = krun.get(tid, 0) + 1
            else:
                krun[tid] = 5
    prev_bl = cbl
    out = T.step(n, dets)
    if a.still_window > 0:
        motion = np.maximum(L["short"], L["long"])
        kept = []
        for f, tid, box, conf in out:
            if f == n and tid is not None:
                x1, y1, x2, y2 = [int(v * LM.s) for v in box]
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, motion.shape[1]), min(y2, motion.shape[0])
                roi = motion[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else np.zeros((0, 0), np.uint8)
                pixel_change = float(roi.mean()) if roi.size else 0.0
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                hist = still_history.setdefault(tid, [])
                hist.append((cx, cy, pixel_change))
                del hist[:-a.still_window]
                no_motion = (len(hist) == a.still_window
                             and np.mean([q[2] for q in hist]) < a.still_motion
                             and np.hypot(hist[-1][0] - hist[0][0], hist[-1][1] - hist[0][1]) < a.still_drift)
                if no_motion:
                    continue
            kept.append((f, tid, box, conf))
        out = kept
    return out, info


n, t0, frame, out, info = 0, time.time(), None, [], {}
while True:
    if not paused or frame is None:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        out, info = step(frame)
        seen = set()
        for f, i, (x1, y1, x2, y2), c in out:
            lines.append(f"{f},{i},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{c:.4f}")
            if f == n:
                seen.add(i); TRAIL.setdefault(i, []).append((int((x1 + x2) / 2), int((y1 + y2) / 2))); TRAIL[i] = TRAIL[i][-30:]
        for i in list(TRAIL):
            if i not in seen:
                TRAIL[i] = TRAIL[i][1:]
                if not TRAIL[i]:
                    del TRAIL[i]
    if view == 0:
        img = LY.grid(info["L"], frame, (W, H))
    elif view == 11:
        img = cview(CS.mask, info["blobs"], LM.s, (W, H))
    else:
        img = frame.copy()
    if view != 0:
        for (x, y, w, h) in gt.get(n, []):
            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (255, 255, 255), 1)
        for x0, y0 in info.get("crops", []):
            cv2.rectangle(img, (x0, y0), (x0 + 320, y0 + 320), (0, 255, 0), 1)
        for d in info.get("boosted", []):
            cv2.rectangle(img, (int(d[0]) - 4, int(d[1]) - 4), (int(d[2]) + 4, int(d[3]) + 4), (0, 255, 255), 2)
        for d in info.get("kept", []):
            dashed(img, [d[0] - 6, d[1] - 6, d[2] + 6, d[3] + 6], (255, 255, 255))
        if traj:
            for i, pts in TRAIL.items():
                for k in range(1, len(pts)):
                    cv2.line(img, pts[k - 1], pts[k], COL.get(i, (0, 255, 0)), 2)
        for f, i, (x1, y1, x2, y2), c in out:
            if f == n:
                motion_outline(img, (x1, y1, x2, y2), np.maximum(info["L"]["short"], info["L"]["long"]), LM.s)
                col = COL.get(i, (0, 255, 0))
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), col, 3)
                cv2.putText(img, f"drone {i} {c:.2f}", (int(x1), max(20, int(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
    fps = n / max(time.time() - t0, 1e-6)
    vname = {1: "camera", 11: "consensus (white shapes)", 0: "all layers"}[view]
    cv2.putText(img, f"frame {n}  {fps:.0f} fps  view: {vname}  booster {'ON' if booster else 'off'} (b)  keeper {'ON' if keeper else 'off'} (k)"
                f"  boosted {len(info.get('boosted', []))}  kept {len(info.get('kept', []))}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    if writer is not None and not paused:
        writer.write(img)
    if not a.no_show:
        cv2.imshow("drone demo - booster", cv2.resize(img, (W // 2, H // 2)))
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("1"):
            view = 1
        elif k == ord("0"):
            view = 0
        elif k == ord("c"):
            view = 11
        elif k == ord("b"):
            booster = not booster
        elif k == ord("k"):
            keeper = not keeper
        elif k == ord("t"):
            traj = not traj
        elif k == ord(" "):
            paused = not paused
if writer is not None:
    writer.release()
cap.release(); cv2.destroyAllWindows()
lines = sorted(set(lines), key=lambda l: (int(l.split(",")[0]), int(l.split(",")[1])))
open(os.path.join(a.out, f"boost_{name}.txt"), "w").write("\n".join(lines) + "\n")
print(f"{n} frames, {n / max(time.time() - t0, 1e-6):.1f} fps. Saved {a.out}/boost_{name}.mp4 and .txt")
