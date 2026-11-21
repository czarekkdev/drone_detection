"""Score result files against gt.  python score.py --pred preds/rgb --gt <test_sample_clean>
Reports per video: AP@0.5, AP@0.3 and precision/recall/F1 at the best confidence."""
import argparse, os, glob
import numpy as np


def load(path, has_conf):
    d = {}
    if not os.path.exists(path):
        return d
    for line in open(path):
        v = line.strip().split(",")
        if len(v) < 6:
            continue
        fr = int(float(v[0]))
        x, y, w, h = map(float, v[2:6])
        c = float(v[6]) if has_conf and len(v) > 6 else 1.0
        d.setdefault(fr, []).append((x, y, x + w, y + h, c))
    return d


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def evaluate(gt, pr, thr):
    npos = sum(len(v) for v in gt.values())
    dets = []  # (conf, tp)
    for fr, ps in pr.items():
        g = gt.get(fr, []); used = [False] * len(g)
        for p in sorted(ps, key=lambda p: -p[4]):
            best, bj = thr, -1
            for j, gb in enumerate(g):
                if not used[j]:
                    o = iou(p, gb)
                    if o >= best:
                        best, bj = o, j
            if bj >= 0:
                used[bj] = True
            dets.append((p[4], bj >= 0))
    if not dets:
        return 0.0, 0, 0, 0, 0
    dets.sort(key=lambda d: -d[0])
    tp = np.cumsum([d[1] for d in dets]); fp = np.cumsum([not d[1] for d in dets])
    rec = tp / max(npos, 1); prec = tp / (tp + fp)
    mprec = np.maximum.accumulate(prec[::-1])[::-1]
    ap = np.trapezoid(np.r_[mprec[:1], mprec], np.r_[0, rec]) if len(rec) else 0
    f1 = 2 * prec * rec / (prec + rec + 1e-9); k = int(np.argmax(f1))
    return ap, prec[k], rec[k], f1[k], dets[k][0]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True)
    p.add_argument("--gt", required=True)
    a = p.parse_args()
    rows = []
    for gdir in sorted(glob.glob(os.path.join(a.gt, "*/gt/gt.txt"))):
        name = gdir.split(os.sep)[-3]
        gt = load(gdir, False); pr = load(os.path.join(a.pred, name + ".txt"), True)
        ap5, P, R, F, c = evaluate(gt, pr, 0.5); ap3 = evaluate(gt, pr, 0.3)[0]
        rows.append((name.split("_")[-1], ap5, ap3, P, R, F, c))
    print(f"{'video':8} {'AP50':>6} {'AP30':>6} {'P':>6} {'R':>6} {'F1':>6} {'conf':>6}")
    for r in rows:
        print(f"{r[0]:8} " + " ".join(f"{v:6.3f}" for v in r[1:]))
    m = np.mean([r[1:6] for r in rows], 0)
    print(f"{'MEAN':8} " + " ".join(f"{v:6.3f}" for v in m))
