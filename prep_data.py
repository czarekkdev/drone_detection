"""Build a clean single-class copy of hackathon_dataset without touching the original.
Images are symlinked; labels are rewritten with class 0. Prints what it found.
python prep_data.py --src "../hackathon_dataset" --dst drone_data"""
import argparse, os, glob, collections

p = argparse.ArgumentParser()
p.add_argument("--src", required=True)
p.add_argument("--dst", default="drone_data")
a = p.parse_args()
src, dst = os.path.abspath(a.src), os.path.abspath(a.dst)
for split in ("train", "val"):
    os.makedirs(os.path.join(dst, split, "labels"), exist_ok=True)
    # real folder of hard links (a symlinked dir gets resolved by Ultralytics -> wrong labels)
    link = os.path.join(dst, split, "images"); os.makedirs(link, exist_ok=True)
    srcimg = os.path.join(src, split, "images")
    imgs = [f for f in os.listdir(srcimg) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    for im in imgs:
        t = os.path.join(link, im)
        if not os.path.exists(t):
            os.link(os.path.join(srcimg, im), t)
    cls, nbox, empty, bad = collections.Counter(), 0, 0, 0
    for im in imgs:
        stem = os.path.splitext(im)[0]
        lp = os.path.join(src, split, "labels", stem + ".txt")
        out = []
        if os.path.exists(lp):
            for line in open(lp):
                v = line.split()
                if len(v) != 5:
                    if v: bad += 1
                    continue
                cls[v[0]] += 1
                out.append("0 " + " ".join(v[1:]))
        if not out:
            empty += 1
        nbox += len(out)
        with open(os.path.join(dst, split, "labels", stem + ".txt"), "w") as f:
            f.write("\n".join(out) + ("\n" if out else ""))
    print(f"{split}: {len(imgs)} images, {nbox} boxes, {empty} without drone, bad lines {bad}, original classes {dict(cls)}")
with open("drone.yaml", "w") as f:
    f.write(f"path: {dst}\ntrain: train/images\nval: val/images\nnames:\n  0: drone\n")
print("wrote drone.yaml")
