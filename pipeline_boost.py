"""Experimental pipeline = pipeline.py + consensus-layer stages (best setup: MOTA 0.878 on the 3 practice videos).
Stages per frame: 1 whole-frame detector (or cached --dets)  2 prediction crops (--weights model, real frame)
  3 motion crops (off: --no-motion)  4 consensus layer (layers.py at --layer-width) ->
  --verify (weak boxes with no layer support x0.3)  --agree-boost LO --boost-to HI (faint box + same-place/size blob -> HI)
  --keeper2 (keep a track <=5 frames if a blob near the prediction moves like the track)  5 tracker
Other experiment flags (blobs crops, plain keeper, agree-filter, mask) were tested and are worse; kept for reference.
Best setup (whole-frame dets from the ORIGINAL SkyGuard cached with run_video.py --imgsz 1280 --conf 0.01, crops with ft_1280 ep1):
python pipeline_boost.py --videos ../test_sample_clean --dets preds/hf_skyguard --out preds/boost --weights runs/ft_1280/weights/epoch0.pt \
  --no-motion --layer-width 480 --verify --keeper2 --agree-boost 0.1 --boost-to 0.65 --crop-scale 2
Note: without --dets the --weights model is also used for the whole frame."""
import argparse, os, glob, time
import cv2, numpy as np, torch
from ultralytics import YOLO
from tracker import TwoDroneTracker, dedupe_lines
import layers as LY
from consensus import Consensus, single

p = argparse.ArgumentParser()
p.add_argument("--videos", required=True); p.add_argument("--out", required=True)
p.add_argument("--dets", default=""); p.add_argument("--motion", default="")
p.add_argument("--weights", default="../pretrained/drone-detection-yolov11/best.pt")
p.add_argument("--imgsz", type=int, default=1280, help="whole-frame size (live mode)")
p.add_argument("--crop", type=int, default=320); p.add_argument("--crop-imgsz", type=int, default=320)
p.add_argument("--crop-scale", type=float, default=1.0, help="enlarge each crop with cubic resize before the crop model (2 = best, 0.883)")
p.add_argument("--motion-topk", type=int, default=4)
p.add_argument("--min-persist", type=int, default=1, help="motion spot must be seen in this many of the 2 previous frames")
p.add_argument("--no-pred", action="store_true"); p.add_argument("--no-motion", action="store_true")
p.add_argument("--device", default="")
p.add_argument("--blobs", default="none", choices=["none", "consensus", "motion"])
p.add_argument("--blob-topk", type=int, default=3)
p.add_argument("--verify", action="store_true")
p.add_argument("--keeper", action="store_true")
p.add_argument("--layer-width", type=int, default=960)
p.add_argument("--verify-below", type=float, default=0.5)
p.add_argument("--verify-mult", type=float, default=0.3)
p.add_argument("--keeper-conf", type=float, default=0.25)
p.add_argument("--keeper-max", type=int, default=1000, help="max consecutive kept frames per track")
p.add_argument("--agree-boost", type=float, default=0.0, help="boost dets with conf in [this, 0.3) when a size-matching blob sits at the same place")
p.add_argument("--boost-to", type=float, default=0.4)
p.add_argument("--agree-filter", action="store_true", help="drop dets < 0.6 with no blob at the same place")
p.add_argument("--keeper2", action="store_true", help="motion-consistent keeper")
a = p.parse_args()
dev = a.device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
model = YOLO(a.weights); os.makedirs(a.out, exist_ok=True)


def read(path):
    D = {}
    if path and os.path.exists(path):
        for l in open(path):
            v = l.strip().split(",")
            if len(v) > 5:
                D.setdefault(int(float(v[0])), []).append([float(x) for x in v[1:]])
    return D


def iou(a_, b):
    ix = max(0, min(a_[2], b[2]) - max(a_[0], b[0])); iy = max(0, min(a_[3], b[3]) - max(a_[1], b[1]))
    i = ix * iy; return i / ((a_[2] - a_[0]) * (a_[3] - a_[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def run_crops(img, centers, size):
    H, W = img.shape[:2]; tiles, offs = [], []
    for cx, cy in centers:
        x0 = int(min(max(cx - size / 2, 0), max(W - size, 0))); y0 = int(min(max(cy - size / 2, 0), max(H - size, 0)))
        t = img[y0:y0 + size, x0:x0 + size]
        if a.crop_scale != 1:
            t = cv2.resize(t, None, fx=a.crop_scale, fy=a.crop_scale, interpolation=cv2.INTER_CUBIC)
        tiles.append(t); offs.append((x0, y0))
    out = []; k = a.crop_scale
    if tiles:
        isz = int(round(a.crop_imgsz * k))
        for r, (x0, y0) in zip(model.predict(tiles, imgsz=isz, conf=0.05, device=dev, verbose=False), offs):
            for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                out.append((x1 / k + x0, y1 / k + y0, x2 / k + x0, y2 / k + y0, c))
    return out


def merge(dets, new):
    for b in new:
        same = [i for i, d in enumerate(dets) if iou(b, d) > 0.3]
        if same:
            i = same[0]
            if b[4] > dets[i][4]:
                dets[i] = b
        else:
            dets.append(b)
    return dets


stats = {}
for vdir in sorted(glob.glob(os.path.join(a.videos, "*/"))):
    name = os.path.basename(os.path.normpath(vdir))
    D = {k: [(x, y, x + w, y + h, c) for _, x, y, w, h, c in v] for k, v in read(os.path.join(a.dets, name + ".txt") if a.dets else "").items()}
    M = read(os.path.join(a.motion, name + ".txt") if a.motion else "")
    cap = cv2.VideoCapture(os.path.join(vdir, "video.mp4"))
    T = TwoDroneTracker(); lines = []; n = 0; npred = nmot = nblob = 0; t0 = time.time(); LM = LY.LayerMaker(width=a.layer_width); CS = Consensus(near=60 * a.layer_width / 960); tlay = 0.0; nver = nkeep = 0; krun = {}; nboost = nfilt = 0; prev_bl = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        n += 1
        dets = list(D.get(n, [])) if a.dets else [(*b, c) for b, c in zip(*[(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()) for r in [model.predict(img, imgsz=a.imgsz, conf=0.01, device=dev, verbose=False)[0]]][0])]
        preds = T.predictions()
        # 2) prediction crops
        if not a.no_pred:
            centers = []
            for tid, pb, miss in preds:
                if any(iou(pb, d) > 0.2 and d[4] >= T.p["low"] for d in dets):
                    continue  # detector already sees it
                centers.append(((pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2))
            if centers:
                npred += len(centers); dets = merge(dets, run_crops(img, centers, a.crop))
        # 3) motion crops
        if not a.no_motion and M:
            pc = [((pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2) for _, pb, _ in preds]
            cand = []
            def seen(f, cx, cy):
                return any(abs(cx - (b[0] + b[2] / 2)) < 60 and abs(cy - (b[1] + b[3] / 2)) < 60 for b in M.get(f, []))
            for x, y, w, h, s, *_ in M.get(n, []):
                cx, cy = x + w / 2, y + h / 2
                if a.min_persist and (seen(n - 1, cx, cy) + seen(n - 2, cx, cy)) < a.min_persist:
                    continue  # not moving steadily: flicker / leaves
                if any(d[0] - 10 <= cx <= d[2] + 10 and d[1] - 10 <= cy <= d[3] + 10 for d in dets if d[4] >= 0.05):
                    continue
                dist = min([np.hypot(cx - u, cy - v) for u, v in pc], default=1e9)
                cand.append((0 if dist < 300 else 1, -s, cx, cy))
            cand.sort(); centers = []
            for _, _, cx, cy in cand:
                if any(abs(cx - u) < a.crop / 3 and abs(cy - v) < a.crop / 3 for u, v in centers):
                    continue
                centers.append((cx, cy))
                if len(centers) >= a.motion_topk:
                    break
            if centers:
                nmot += len(centers); dets = merge(dets, run_crops(img, centers, a.crop))
        # 4) consensus / motion blob crops (blobs not already explained by a detection)
        need_layers = a.blobs != "none" or a.verify or a.keeper or a.agree_boost or a.agree_filter or a.keeper2
        if need_layers:
            tb = time.time()
            L = LM.step(img, [pb for _, pb, _ in preds])
            cbl = CS.step(L, LM.s)
            bl = cbl if a.blobs == "consensus" else (single(L, "motion", 128, LM.s)[0] if a.blobs == "motion" else [])
            tlay += time.time() - tb
        if a.verify:  # down-weight weak detections that no layer supports
            mk = CS.mask; Hm, Wm = mk.shape; nd = []
            for d in dets:
                if d[4] < a.verify_below:
                    x1, y1, x2, y2 = [int(v * LM.s) for v in d[:4]]
                    x1, y1 = max(x1 - 2, 0), max(y1 - 2, 0); x2, y2 = min(x2 + 3, Wm), min(y2 + 3, Hm)
                    if x2 <= x1 or y2 <= y1 or not mk[y1:y2, x1:x2].any():
                        d = (*d[:4], d[4] * a.verify_mult); nver += 1
                nd.append(d)
            dets = nd
        if a.keeper:  # keep a track alive when a consensus blob sits at its predicted spot
            for tid, pb, miss in preds:
                if any(iou(pb, d) > 0.2 and d[4] >= T.p["low"] for d in dets):
                    krun[tid] = 0; continue
                if krun.get(tid, 0) >= a.keeper_max:
                    continue
                pw, ph = pb[2] - pb[0], pb[3] - pb[1]; r = max(pw, ph, 20)
                pcx, pcy = (pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2
                near = [b for b in cbl if np.hypot((b[0] + b[2]) / 2 - pcx, (b[1] + b[3]) / 2 - pcy) < r]
                if near:
                    b = min(near, key=lambda b: np.hypot((b[0] + b[2]) / 2 - pcx, (b[1] + b[3]) / 2 - pcy))
                    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                    dets.append((cx - pw / 2, cy - ph / 2, cx + pw / 2, cy + ph / 2, a.keeper_conf)); nkeep += 1; krun[tid] = krun.get(tid, 0) + 1
        def same_place(d, b):
            dcx, dcy = (d[0] + d[2]) / 2, (d[1] + d[3]) / 2; bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            inside = (b[0] - 4 <= dcx <= b[2] + 4 and b[1] - 4 <= dcy <= b[3] + 4) or (d[0] <= bcx <= d[2] and d[1] <= bcy <= d[3])
            ds, bs = max(d[2] - d[0], d[3] - d[1], 1), max(b[2] - b[0], b[3] - b[1], 1)
            return inside and 0.5 <= ds / bs <= 2.0
        if a.agree_boost:
            nd = []
            for d in dets:
                if a.agree_boost <= d[4] < 0.3 and any(same_place(d, b) for b in cbl):
                    d = (*d[:4], max(d[4], a.boost_to)); nboost += 1
                nd.append(d)
            dets = nd
        if a.agree_filter:
            nd = []
            for d in dets:
                if d[4] < 0.6 and not any(same_place(d, b) for b in cbl):
                    nfilt += 1; continue
                nd.append(d)
            dets = nd
        if a.keeper2:
            vel = {t.id: (t.kf.x[4], t.kf.x[5]) for t in T.active if t.id is not None}
            for tid, pb, miss in preds:
                if any(iou(pb, d) > 0.2 and d[4] >= T.p["low"] for d in dets):
                    krun[tid] = 0; continue
                if krun.get(tid, 0) >= 5 or tid not in vel:
                    continue
                pw, ph = pb[2] - pb[0], pb[3] - pb[1]; r = max(pw, ph, 20)
                pcx, pcy = (pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2; vx, vy = vel[tid]; vs = np.hypot(vx, vy)
                ok_b = None
                for b in cbl:
                    bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                    if np.hypot(bcx - pcx, bcy - pcy) >= r:
                        continue
                    bsz = max(b[2] - b[0], b[3] - b[1], 1)
                    if not 0.5 <= bsz / max(pw, ph, 1) <= 2.0:
                        continue
                    prv = [q for q in prev_bl if np.hypot((q[0] + q[2]) / 2 - bcx, (q[1] + q[3]) / 2 - bcy) < r]
                    if not prv:
                        continue
                    q = min(prv, key=lambda q: np.hypot((q[0] + q[2]) / 2 - bcx, (q[1] + q[3]) / 2 - bcy))
                    mx, my = bcx - (q[0] + q[2]) / 2, bcy - (q[1] + q[3]) / 2; ms = np.hypot(mx, my)
                    if vs < 2 and ms < 2:
                        ok_b = b; break
                    if vs >= 2 and ms >= 1 and 0.5 <= ms / vs <= 1.5 and (mx * vx + my * vy) / (ms * vs) >= np.cos(np.pi / 4):
                        ok_b = b; break
                if ok_b is not None:
                    cx, cy = (ok_b[0] + ok_b[2]) / 2, (ok_b[1] + ok_b[3]) / 2
                    dets.append((cx - pw / 2, cy - ph / 2, cx + pw / 2, cy + ph / 2, a.keeper_conf)); nkeep += 1; krun[tid] = krun.get(tid, 0) + 1
                else:
                    krun[tid] = 5  # stop keeping as soon as the blob stops matching
        if need_layers:
            prev_bl = cbl
        if a.blobs != "none":
            centers = []
            for x1, y1, x2, y2, s in bl:
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                if any(d[0] - 10 <= cx <= d[2] + 10 and d[1] - 10 <= cy <= d[3] + 10 for d in dets if d[4] >= 0.05):
                    continue
                if any(abs(cx - u) < a.crop / 3 and abs(cy - v) < a.crop / 3 for u, v in centers):
                    continue
                centers.append((cx, cy))
                if len(centers) >= a.blob_topk:
                    break
            if centers:
                nblob += len(centers); dets = merge(dets, run_crops(img, centers, a.crop))
        for f2, i, (x1, y1, x2, y2), c in T.step(n, dets):
            lines.append(f"{f2},{i},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{c:.4f}")
    lines = dedupe_lines(lines)
    open(os.path.join(a.out, name + ".txt"), "w").write("\n".join(lines) + "\n")
    print(f"{name}: {n} frames, prediction crops/frame {npred / n:.2f}, motion crops/frame {nmot / n:.2f}, blob crops/frame {nblob / n:.2f}, verified-down {nver}, kept {nkeep}, boosted {nboost}, filtered {nfilt}, layer+blob CPU {1000 * tlay / n:.1f} ms/frame, {n / (time.time() - t0):.1f} fps")
