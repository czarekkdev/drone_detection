"""Convert Anti-UAV-RGBT (visible/colour half) to YOLO format.
python convert_antiuav.py --src "../Anti-UAV-RGBT training data" --dst drone_data/antiuav
- train = train + test splits, val = val split
- every --step-th frame (train) / --val-step-th frame (val); frames with exist=0 get an empty label (negative)
- frames are resized to --width (keeps aspect); labels are normalised so size doesn't matter
- videos whose frame count != number of labels are reported and skipped
- writes a few preview images with boxes drawn to <dst>/_check/"""
import argparse, os, json, glob, random
import cv2

p = argparse.ArgumentParser()
p.add_argument("--src", required=True)
p.add_argument("--dst", default="drone_data/antiuav")
p.add_argument("--step", type=int, default=10)
p.add_argument("--val-step", type=int, default=30)
p.add_argument("--width", type=int, default=1280)
a = p.parse_args()
random.seed(0)
os.makedirs(os.path.join(a.dst, "_check"), exist_ok=True)
stats = {}
for out_split, in_splits, step in (("train", ("train", "test"), a.step), ("val", ("val",), a.val_step)):
    idir = os.path.join(a.dst, out_split, "images"); ldir = os.path.join(a.dst, out_split, "labels")
    os.makedirs(idir, exist_ok=True); os.makedirs(ldir, exist_ok=True)
    n_img = n_pos = n_skip = 0
    for s in in_splits:
        for vdir in sorted(glob.glob(os.path.join(a.src, s, "*/"))):
            name = os.path.basename(os.path.normpath(vdir))
            jp, vp = os.path.join(vdir, "visible.json"), os.path.join(vdir, "visible.mp4")
            if not (os.path.exists(jp) and os.path.exists(vp)):
                print("missing files:", vdir); n_skip += 1; continue
            lab = json.load(open(jp)); ex, rect = lab.get("exist", []), lab.get("gt_rect", [])
            cap = cv2.VideoCapture(vp); frames = []; i = 0
            while True:
                if i % step == 0:
                    ok, fr = cap.read()
                    if not ok: break
                    frames.append((i, fr))
                else:
                    if not cap.grab(): break
                i += 1
            cap.release()
            if i != len(ex) or len(rect) != len(ex):
                print(f"SKIP {s}/{name}: {i} frames vs {len(ex)} exist / {len(rect)} boxes"); n_skip += 1; continue
            check = random.random() < 0.05
            for fi, fr in frames:
                H, W = fr.shape[:2]
                b = rect[fi] if fi < len(rect) else []
                line = ""
                if ex[fi] and b and len(b) == 4 and b[2] > 1 and b[3] > 1:
                    x, y, w, h = b
                    x0, y0 = max(0, x), max(0, y); x1, y1 = min(W, x + w), min(H, y + h)
                    if x1 > x0 and y1 > y0:
                        line = f"0 {(x0+x1)/2/W:.6f} {(y0+y1)/2/H:.6f} {(x1-x0)/W:.6f} {(y1-y0)/H:.6f}\n"
                stem = f"auav_{s}_{name}_{fi:05d}"
                if check and line:
                    vis = fr.copy(); cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 3)
                    cv2.imwrite(os.path.join(a.dst, "_check", stem + ".jpg"), vis); check = False
                if a.width and W > a.width:
                    fr = cv2.resize(fr, (a.width, round(H * a.width / W)), interpolation=cv2.INTER_AREA)
                cv2.imwrite(os.path.join(idir, stem + ".jpg"), fr, [cv2.IMWRITE_JPEG_QUALITY, 92])
                open(os.path.join(ldir, stem + ".txt"), "w").write(line)
                n_img += 1; n_pos += bool(line)
            print(f"{out_split} <- {s}/{name}: {len(frames)} frames")
    stats[out_split] = (n_img, n_pos, n_skip)
for k, (n, pos, sk) in stats.items():
    print(f"{k}: {n} images, {pos} with drone, {n-pos} negatives, {sk} videos skipped")
