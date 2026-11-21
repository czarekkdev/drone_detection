"""Learned rescoring of candidate boxes (replaces the hand-made booster / verifier rules of pipeline_boost.py).
Every candidate (whole-frame box conf >= 0.01, trajectory-crop box) gets features from the detector, the layers,
the consensus blobs and the tracker; a small model turns them into one confidence that is fed to the same tracker.

dump  (features + labels, tracker driven by the baseline rules so candidates look like the real pipeline's):
  python pipeline_rs.py --mode dump --videos ../test_sample_clean --dets preds/hf_skyguard --dump preds/rs/practice.npz
  python pipeline_rs.py --mode dump --auav "../Anti-UAV-RGBT training data/val" --dets preds/auav_val_sky --dump preds/rs/auav.npz
apply (conf = model probability):
  python pipeline_rs.py --mode apply --videos ../test_sample_clean --dets preds/hf_skyguard --model preds/rs/model.pkl --out preds/rs/x
base  (= pipeline_boost.py best setup, for checking):  --mode base"""
import argparse, os, glob, time, json, pickle, zlib
import cv2, numpy as np, torch
from ultralytics import YOLO
import tracker as TR
from tracker import TwoDroneTracker
import layers as LY
from consensus import Consensus
import rs_util  # noqa: F401  (model class; the .pkl files are our own, made by fit_rs.py)

p = argparse.ArgumentParser()
p.add_argument("--mode", default="apply", choices=["dump", "apply", "base"])
p.add_argument("--videos", default=""); p.add_argument("--auav", default=""); p.add_argument("--names", default="")
p.add_argument("--max-frames", type=int, default=500); p.add_argument("--every", type=int, default=1, help="dump: record every n-th frame")
p.add_argument("--dets", required=True); p.add_argument("--out", default=""); p.add_argument("--dump", default="")
p.add_argument("--model", default=""); p.add_argument("--only", default="", help="comma list of substrings: videos to run")
p.add_argument("--weights", default="runs/ft_1280/weights/epoch0.pt")
p.add_argument("--layer-width", type=int, default=480)
p.add_argument("--no-keeper", action="store_true")
p.add_argument("--blend", type=float, default=1.0, help="conf = blend*p + (1-blend)*raw conf")
p.add_argument("--geo", action="store_true", help="conf = sqrt(raw conf * p)")
p.add_argument("--pscale", type=float, default=1.0, help="conf = min(1, p*pscale) (the one allowed threshold knob)")
p.add_argument("--device", default="")
p.add_argument("--static-thr", type=float, default=0.0, help="base: a box with max(short,long) motion below this inside it counts as not moving")
p.add_argument("--static-mult", type=float, default=0.3)
p.add_argument("--static-below", type=float, default=1.01, help="only boxes with conf below this are checked")
p.add_argument("--nms", type=float, default=0.0, help="apply: drop a box overlapping (IoU > this) a higher-scored one after rescoring")
p.add_argument("--shake", type=float, default=0.0, help="dump: synthetic hand-held camera (random pan, px std of the offset; ~40 gives 10-15 px/frame)")
a = p.parse_args()
dev = a.device or ("mps" if torch.backends.mps.is_available() else "cpu")
model = YOLO(a.weights)
CLF = pickle.load(open(a.model, "rb")) if a.model else None
KEYS = ("short", "long", "bg", "shape", "track")


def read(path):
    D = {}
    if os.path.exists(path):
        for l in open(path):
            v = l.strip().split(",")
            if len(v) > 5:
                fr, _, x, y, w, h, c = [float(t) for t in v[:7]]
                D.setdefault(int(fr), []).append((x, y, x + w, y + h, c))
    return D


def iou(a_, b):
    ix = max(0, min(a_[2], b[2]) - max(a_[0], b[0])); iy = max(0, min(a_[3], b[3]) - max(a_[1], b[1]))
    i = ix * iy; return i / ((a_[2] - a_[0]) * (a_[3] - a_[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def run_crops(img, centers, size=320, k=2.0, isz=320):
    H, W = img.shape[:2]; tiles, offs = [], []
    for cx, cy in centers:
        x0 = int(min(max(cx - size / 2, 0), max(W - size, 0))); y0 = int(min(max(cy - size / 2, 0), max(H - size, 0)))
        tiles.append(cv2.resize(img[y0:y0 + size, x0:x0 + size], None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)); offs.append((x0, y0))
    out = []
    for r, (x0, y0) in zip(model.predict(tiles, imgsz=int(round(isz * k)), conf=0.05, device=dev, verbose=False), offs):
        for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
            out.append((x1 / k + x0, y1 / k + y0, x2 / k + x0, y2 / k + y0, c))
    return out


def merge(cands, new):
    """cands: list of [x1,y1,x2,y2,conf,full_conf,crop_conf]; same rule as pipeline_boost (IoU>0.3, keep higher box)"""
    for b in new:
        same = [i for i, d in enumerate(cands) if iou(b, d) > 0.3]
        if same:
            d = cands[same[0]]; d[6] = max(d[6], b[4])
            if b[4] > d[4]:
                d[:5] = list(b[:5])
        else:
            cands.append([*b[:5], 0.0, b[4]])
    return cands


def same_place(d, b):
    dcx, dcy = (d[0] + d[2]) / 2, (d[1] + d[3]) / 2; bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    inside = (b[0] - 4 <= dcx <= b[2] + 4 and b[1] - 4 <= dcy <= b[3] + 4) or (d[0] <= bcx <= d[2] and d[1] <= bcy <= d[3])
    ds, bs = max(d[2] - d[0], d[3] - d[1], 1), max(b[2] - b[0], b[3] - b[1], 1)
    return inside and 0.5 <= ds / bs <= 2.0


FEATS = (["conf", "full_conf", "crop_conf", "has_full", "has_crop", "logw", "logh", "log_aspect", "rank", "n_strong", "max_other"]
         + [f"{k}_{s}" for k in KEYS for s in ("in_mean", "in_max", "ring_mean", "contrast")]
         + ["mask_frac", "nblobs", "blob_dist", "blob_logsize", "blob_iou", "blob_same", "blob_score",
            "trk_dist", "trk_logsize", "trk_miss", "trk_hits", "trk_confirmed", "trk_speed", "lost_dist", "pan"])


def box_sum(I, x1, y1, x2, y2):
    return I[y2, x2] - I[y1, x2] - I[y2, x1] + I[y1, x1]


def features(cands, L, mask, cbl, T, pan, s, frame):
    Hm, Wm = mask.shape
    ints = {k: cv2.integral(L[k]) for k in KEYS}; mint = cv2.integral((mask > 0).astype(np.uint8))
    confs = [c[4] for c in cands]; order = np.argsort(-np.array(confs)) if confs else []
    rank = np.empty(len(cands)); rank[order] = np.arange(len(cands))
    n_strong = sum(c >= 0.3 for c in confs)
    trks = [t for t in T.active + T.tentative]
    X = []
    for j, c in enumerate(cands):
        x1, y1, x2, y2 = c[:4]; w, h = max(x2 - x1, 1), max(y2 - y1, 1); sz = max(w, h)
        f = [c[4], c[5], c[6], float(c[5] > 0), float(c[6] > 0), np.log(w), np.log(h), np.log(w / h), rank[j], n_strong,
             max([confs[i] for i in range(len(cands)) if i != j], default=0.0)]
        # box in layer coords (at least 1 px), ring = box grown by half its size on each side
        bx1, by1 = int(np.clip(np.floor(x1 * s), 0, Wm - 1)), int(np.clip(np.floor(y1 * s), 0, Hm - 1))
        bx2, by2 = int(np.clip(np.ceil(x2 * s), bx1 + 1, Wm)), int(np.clip(np.ceil(y2 * s), by1 + 1, Hm))
        gw, gh = max((bx2 - bx1) // 2, 2), max((by2 - by1) // 2, 2)
        rx1, ry1, rx2, ry2 = max(bx1 - gw, 0), max(by1 - gh, 0), min(bx2 + gw, Wm), min(by2 + gh, Hm)
        ain = (bx2 - bx1) * (by2 - by1); aout = (rx2 - rx1) * (ry2 - ry1)
        for k in KEYS:
            si = box_sum(ints[k], bx1, by1, bx2, by2); so = box_sum(ints[k], rx1, ry1, rx2, ry2)
            mi = si / ain; mr = (so - si) / max(aout - ain, 1)
            f += [mi, float(L[k][by1:by2, bx1:bx2].max()), mr, mi - mr]
        f.append(box_sum(mint, bx1, by1, bx2, by2) / ain)
        # nearest consensus blob
        if cbl:
            dd = [np.hypot((b[0] + b[2]) / 2 - (x1 + x2) / 2, (b[1] + b[3]) / 2 - (y1 + y2) / 2) for b in cbl]
            b = cbl[int(np.argmin(dd))]; bsz = max(b[2] - b[0], b[3] - b[1], 1)
            f += [len(cbl), min(min(dd) / sz, 10), np.log(bsz / sz), iou(c, b), float(any(same_place(c, q) for q in cbl)), b[4]]
        else:
            f += [0, 10, 0, 0, 0, 0]
        # nearest track (active or tentative), from the tracker's prediction for this frame
        if trks:
            best = None
            for t in trks:
                pb = t.peek(); pcx, pcy = (pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2; ps = max(pb[2] - pb[0], pb[3] - pb[1], 4)
                dn = np.hypot(pcx - (x1 + x2) / 2, pcy - (y1 + y2) / 2) / max(ps, 12)
                if best is None or dn < best[0]:
                    best = (dn, np.log(sz / ps), t)
            dn, ls, t = best
            f += [min(dn, 10), ls, min(t.miss, 10), min(t.hits, 50), float(t.id is not None), np.hypot(t.kf.x[4], t.kf.x[5]) / max(t.kf.x[2], t.kf.x[3], 4)]
        else:
            f += [10, 0, 10, 0, 0, 0]
        lost = [np.hypot(t.last_obs[0] - (x1 + x2) / 2, t.last_obs[1] - (y1 + y2) / 2) / max(t.last_obs[2], t.last_obs[3], 12)
                for i, t in T.lost.items() if 0 < frame - t.last <= 60]
        f += [min(min(lost, default=10), 10), pan]
        X.append(f)
    return np.array(X, np.float32).reshape(-1, len(FEATS))


def keeper2(dets, preds, T, cbl, prev_bl, krun):
    added = []
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
            added.append((cx - pw / 2, cy - ph / 2, cx + pw / 2, cy + ph / 2, 0.25)); krun[tid] = krun.get(tid, 0) + 1
        else:
            krun[tid] = 5
    return added


def sources():
    if a.videos:
        for vdir in sorted(glob.glob(os.path.join(a.videos, "*/"))):
            name = os.path.basename(os.path.normpath(vdir)); G = {}
            gp = os.path.join(vdir, "gt", "gt.txt")
            if os.path.exists(gp):
                for l in open(gp):
                    v = [float(t) for t in l.strip().split(",")[:6]]
                    G.setdefault(int(v[0]), []).append((v[2], v[3], v[2] + v[4], v[3] + v[5]))
            yield name, os.path.join(vdir, "video.mp4"), G, 10 ** 9
    if a.auav:
        for vp in sorted(glob.glob(os.path.join(a.auav, "*/visible.mp4"))):
            name = vp.split(os.sep)[-2]
            if a.names and name not in a.names.split(","):
                continue
            if not os.path.exists(os.path.join(a.dets, name + ".txt")):
                continue
            lab = json.load(open(vp.replace(".mp4", ".json"))); G = {}
            for i, (e, r) in enumerate(zip(lab["exist"], lab["gt_rect"])):
                if e and len(r) == 4 and r[2] > 0 and r[3] > 0:
                    G[i + 1] = [(r[0], r[1], r[0] + r[2], r[1] + r[3])]
            yield name, vp, G, a.max_frames


ROWS = []; tfeat = []; tall = []
for name, vp, G, maxf in sources():
    if a.only and not any(s in name for s in a.only.split(",")):
        continue
    D = read(os.path.join(a.dets, name + ".txt"))
    cap = cv2.VideoCapture(vp); W, H = int(cap.get(3)), int(cap.get(4))
    T = TwoDroneTracker(frame_size=(W, H)); LM = LY.LayerMaker(width=a.layer_width); CS = Consensus(near=60 * a.layer_width / 960)
    lines = []; n = 0; prev_bl = []; krun = {}
    while n < maxf:
        ok, img = cap.read()
        if not ok:
            break
        n += 1; t0 = time.time()
        ox = oy = 0.0
        if a.shake:  # Ornstein-Uhlenbeck offset: smooth random pan, frame shifted, boxes shifted with it
            if n == 1:
                rng = np.random.default_rng(zlib.crc32(name.encode())); off = np.zeros(2)
            off = off * (1 - 1 / 20) + rng.normal(0, a.shake * np.sqrt(2 / 20), 2); ox, oy = off
            img = cv2.warpAffine(img, np.float32([[1, 0, ox], [0, 1, oy]]), (W, H), borderMode=cv2.BORDER_REFLECT)
        def sh(b):
            return (b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy, *b[4:])
        cands = [[*sh(d), d[4], 0.0] for d in D.get(n, []) if 0 <= (d[0] + d[2]) / 2 + ox < W and 0 <= (d[1] + d[3]) / 2 + oy < H]
        preds = T.predictions()
        centers = [((pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2) for tid, pb, miss in preds
                   if not any(iou(pb, d) > 0.2 and d[4] >= T.p["low"] for d in cands)]
        if centers:
            cands = merge(cands, run_crops(img, centers))
        L = LM.step(img, [pb for _, pb, _ in preds]); cbl = CS.step(L, LM.s)
        Hl = LM.Hs[-1] if LM.Hs else np.eye(3); pan = float(np.hypot(Hl[0, 2], Hl[1, 2]) / LM.s)
        t1 = time.time()
        if a.mode == "base" or a.mode == "dump":
            dets = []
            for c in cands:
                d = tuple(c[:5])
                if d[4] < 0.5:  # verifier
                    mk = CS.mask; Hm, Wm = mk.shape; x1, y1, x2, y2 = [int(v * LM.s) for v in d[:4]]
                    x1, y1 = max(x1 - 2, 0), max(y1 - 2, 0); x2, y2 = min(x2 + 3, Wm), min(y2 + 3, Hm)
                    if x2 <= x1 or y2 <= y1 or not mk[y1:y2, x1:x2].any():
                        d = (*d[:4], d[4] * 0.3)
                if 0.1 <= d[4] < 0.3 and any(same_place(d, b) for b in cbl):  # booster
                    d = (*d[:4], max(d[4], 0.65))
                if a.static_thr and len(LM.grays) > LM.long_gap and d[4] < a.static_below:  # no movement at all -> probably not a drone
                    Hm, Wm = L["short"].shape; x1, y1, x2, y2 = [int(v * LM.s) for v in d[:4]]
                    x1, y1 = max(x1 - 1, 0), max(y1 - 1, 0); x2, y2 = min(max(x2 + 2, x1 + 1), Wm), min(max(y2 + 2, y1 + 1), Hm)
                    if x2 > x1 and y2 > y1 and max(L["short"][y1:y2, x1:x2].max(), L["long"][y1:y2, x1:x2].max()) < a.static_thr:
                        d = (*d[:4], d[4] * a.static_mult)
                dets.append(d)
        if a.mode == "dump" or a.mode == "apply":
            X = features(cands, L, CS.mask, cbl, T, pan, LM.s, n)
        if a.mode == "dump" and (n % a.every == 0) and len(cands):
            g = [sh(b) for b in G.get(n, [])]
            for c, x in zip(cands, X):
                mi = max([iou(c, b) for b in g], default=0.0)
                ROWS.append((name, n, mi, x))
        if a.mode == "apply":
            pr = CLF.predict_proba(X)[:, 1] if len(cands) else []
            dets = [(*c[:4], min(1.0, a.blend * q * a.pscale + (1 - a.blend) * c[4])) for c, q in zip(cands, pr)]
            if a.geo:
                dets = [(*c[:4], float(np.sqrt(c[4] * q))) for c, q in zip(cands, pr)]
            if a.nms:
                keep = []
                for d in sorted(dets, key=lambda d: -d[4]):
                    if all(iou(d, k) <= a.nms for k in keep):
                        keep.append(d)
                dets = keep
        tfeat.append(time.time() - t1)
        if not a.no_keeper:
            dets += keeper2(dets, preds, T, cbl, prev_bl, krun)
        prev_bl = cbl
        for f2, i, (x1, y1, x2, y2), c in T.step(n, dets):
            lines.append(f"{f2},{i},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{c:.4f}")
        tall.append(time.time() - t0)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        best = {}  # one box per (frame, id): a backfilled real box beats an old coasting box with the same id
        for l in lines:
            k = tuple(l.split(",")[:2])
            if k not in best or float(l.split(",")[6]) > float(best[k].split(",")[6]):
                best[k] = l
        lines = sorted(best.values(), key=lambda l: (int(l.split(",")[0]), int(l.split(",")[1])))
        open(os.path.join(a.out, name + ".txt"), "w").write("\n".join(lines) + "\n")
    print(f"{name}: {n} frames, rows {len(ROWS)}", flush=True)
print(f"rescoring stage (features+model / rules) {1000 * np.mean(tfeat):.2f} ms/frame, whole loop {1000 * np.mean(tall):.1f} ms/frame (cached whole-frame dets)")
if a.dump:
    np.savez_compressed(a.dump, video=np.array([r[0] for r in ROWS]), frame=np.array([r[1] for r in ROWS]),
                        iou=np.array([r[2] for r in ROWS], np.float32), X=np.stack([r[3] for r in ROWS]), feats=np.array(FEATS))
