"""Live demo of the current best pipeline (MOTA 0.878 on the practice videos):
  original SkyGuard on the full frame (low conf)  +  ft_1280 model on trajectory crops (real frame)
  + consensus white-shape layer (480 px): verifier, agreement BOOST, motion-consistent KEEPER  +  tracker
  python demo_boost.py                                  (hard practice video, window on)
  python demo_boost.py --video path/to/any.mp4          (any video; --video 0 = camera)
Keys:  1 camera   0 next individual layer (cycles short/long/bg/track/prev/shape/consensus/motion blobs)
       m motion blobs   c consensus   o shape overlay   g colorized grid   b booster   k keeper   t trails   space pause   q quit
Colours: orange/blue = drone 1/2 (final output), thin white = correct answer (practice videos only),
  YELLOW box = a faint detection that the booster raised (white blob at the same place),
  DASHED box = a frame where the keeper held the track (blob moving like the drone), green = trajectory crop checked.
Writes demo_out/boost_<name>.mp4 and .txt (skip the video with --no-save). This is a separate demo; demo.py is unchanged."""
import argparse, os, time
import cv2, numpy as np, torch
from ultralytics import YOLO
from tracker import TwoDroneTracker
import tracker as TRK
CONFS, LAST, DRAWN = {}, {}, {}  # per drone number: confidences in its current appearance, last frame seen
import layers as LY
from consensus import Consensus, single, view as cview

HERE = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument("--video", default=os.path.join(HERE, "../test_sample_clean/test_6_hard_PXL_20261113_154422245_hard/video.mp4"))
p.add_argument("--full-weights", default=os.path.join(HERE, "../pretrained/drone-detection-yolov11/best.pt"))
p.add_argument("--crop-weights", default=os.path.join(HERE, "runs/ft_1280/weights/epoch0.pt"))
p.add_argument("--imgsz", type=int, default=1280)
p.add_argument("--no-show", action="store_true"); p.add_argument("--no-save", action="store_true")
p.add_argument("--out", default=os.path.join(HERE, "demo_out"))
p.add_argument("--layer-width", type=int, default=480)
p.add_argument("--one-model", action="store_true", help="drop the second AI (fine-tuned crop model); crops use the main model")
p.add_argument("--view", default="camera", choices=["camera", "overlay", "shapes", "layers"],
               help="start view (keys: 1 camera, o overlay, c consensus, m motion blobs, 0 cycle layer views, g grid)")
p.add_argument("--show-min", type=float, default=0.3, help="a NEW drone is drawn only at this confidence; once on screen it stays down to tracker.MIN_SHOW "
               "(0.05), so it doesn't flicker. Screen only, the saved .txt is not affected")
p.add_argument("--boost-lo", type=float, default=0.10); p.add_argument("--boost-to", type=float, default=0.65)
a = p.parse_args()
dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
full = YOLO(a.full_weights)
crop_m = full if a.one_model else YOLO(a.crop_weights)  # --one-model: no second AI, the main model also checks the crops
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
booster, keeper, traj, paused = True, True, True, False
view = {"camera": 1, "overlay": 12, "shapes": 11, "layers": 0}[a.view]
LAYER_KEYS = ("short", "long", "bg", "track", "prev", "shape")
LAYER_VIEW_START = 20
CONSENSUS_VIEW = LAYER_VIEW_START + len(LAYER_KEYS)
MOTION_VIEW = CONSENSUS_VIEW + 1
krun, prev_bl, TRAIL, lines = {}, [], {}, []


def iou(p_, q):
    ix = max(0, min(p_[2], q[2]) - max(p_[0], q[0])); iy = max(0, min(p_[3], q[3]) - max(p_[1], q[1]))
    i = ix * iy; return i / ((p_[2] - p_[0]) * (p_[3] - p_[1]) + (q[2] - q[0]) * (q[3] - q[1]) - i + 1e-9)


def same_place(d, b):
    dcx, dcy = (d[0] + d[2]) / 2, (d[1] + d[3]) / 2; bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    inside = (b[0] - 4 <= dcx <= b[2] + 4 and b[1] - 4 <= dcy <= b[3] + 4) or (d[0] <= bcx <= d[2] and d[1] <= bcy <= d[3])
    ds, bs = max(d[2] - d[0], d[3] - d[1], 1), max(b[2] - b[0], b[3] - b[1], 1)
    return inside and 0.5 <= ds / bs <= 2.0


def crops(img, centers, size=320):
    tiles, offs = [], []
    for cx, cy in centers:
        x0 = int(min(max(cx - size / 2, 0), max(W - size, 0))); y0 = int(min(max(cy - size / 2, 0), max(H - size, 0)))
        t = cv2.resize(img[y0:y0 + size, x0:x0 + size], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)  # crops enlarged 2x (0.880 -> 0.883)
        tiles.append(t); offs.append((x0, y0))
    out = []
    if tiles:
        for r, (x0, y0) in zip(crop_m.predict(tiles, imgsz=2 * size, conf=0.05, device=dev, verbose=False), offs):
            for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                out.append((x1 / 2 + x0, y1 / 2 + y0, x2 / 2 + x0, y2 / 2 + y0, c))
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
    if view == MOTION_VIEW:
        info["motion_blobs"] = single(L, "motion", 128, LM.s)[0]
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
    return T.step(n, dets), info


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
    elif LAYER_VIEW_START <= view < CONSENSUS_VIEW:
        img = LY.view(info["L"], LAYER_KEYS[view - LAYER_VIEW_START], (W, H))
    elif view == CONSENSUS_VIEW:
        img = cview(CS.mask, info["blobs"], LM.s, (W, H))
    elif view == MOTION_VIEW:
        motion_mask = (np.maximum(info["L"]["short"], info["L"]["long"]) > 128).astype(np.uint8) * 255
        img = cview(motion_mask, info.get("motion_blobs", []), LM.s, (W, H))
    elif view == 11:
        img = cview(CS.mask, info["blobs"], LM.s, (W, H))
    elif view == 12:  # camera with the combined shape layer painted on top (red = pixels at least 2 layers agree on)
        img = frame.copy(); m = cv2.resize(CS.mask, (W, H), interpolation=cv2.INTER_NEAREST) > 0
        img[m] = (0.45 * img[m] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
    else:
        img = frame.copy()
    solo_layer_view = LAYER_VIEW_START <= view <= MOTION_VIEW
    if view != 0 and not solo_layer_view:
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
        for f, i, (x1, y1, x2, y2), c in out:  # on screen: a drone is drawn once it has been there MIN_TRACK frames
            if f == n:
                if LAST.get(i, -99) < n - 5:
                    CONFS[i] = []; DRAWN[i] = False  # new appearance
                CONFS[i].append(c); LAST[i] = n
            seen = CONFS.get(i, [])
            bar = TRK.MIN_SHOW if DRAWN.get(i) else a.show_min  # new drone: --show-min; already on screen: much lower
            if f == n and c >= bar and len(seen) >= TRK.MIN_TRACK and np.median(seen[:60]) >= TRK.MIN_TRACK_CONF:
                DRAWN[i] = True
                col = COL.get(i, (0, 255, 0))
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), col, 3)
                cv2.putText(img, f"drone {i} {c:.2f}", (int(x1), max(20, int(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
    fps = n / max(time.time() - t0, 1e-6)
    layer_names = {LAYER_VIEW_START + i: key for i, key in enumerate(LAYER_KEYS)}
    layer_names[CONSENSUS_VIEW] = "consensus blobs"
    layer_names[MOTION_VIEW] = "motion blobs"
    vname = {1: "camera", 11: "consensus (white shapes)", 0: "colorized layer grid", 12: "camera + shapes", **layer_names}[view]
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
            view = (view + 1 if LAYER_VIEW_START <= view <= MOTION_VIEW else LAYER_VIEW_START)
            if view > MOTION_VIEW:
                view = LAYER_VIEW_START
        elif k == ord("g"):
            view = 0
        elif k == ord("m"):
            view = MOTION_VIEW
        elif k == ord("c"):
            view = CONSENSUS_VIEW
        elif k == ord("o"):
            view = 12
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
from tracker import dedupe_lines
lines = dedupe_lines(lines)
open(os.path.join(a.out, f"boost_{name}.txt"), "w").write("\n".join(lines) + "\n")
print(f"{n} frames, {n / max(time.time() - t0, 1e-6):.1f} fps. Saved {a.out}/boost_{name}.mp4 and .txt")
