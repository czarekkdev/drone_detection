"""Tracking + detection score for result files.  python eval_track.py --pred DIR --gt test_sample_clean [--min-conf 0.3]
MOTA / IDF1 / ID switches (IoU>=0.5, motmetrics) + precision/recall of the boxes written, + AP50 (score.py)."""
import argparse, os, glob, sys
import numpy as np
import motmetrics as mm
from score import load, evaluate  # noqa


def iou_dist(g, p):
    d = np.full((len(g), len(p)), np.nan)
    for i, (gx, gy, gw, gh) in enumerate(g):
        for j, (px, py, pw, ph) in enumerate(p):
            ix = max(0, min(gx + gw, px + pw) - max(gx, px)); iy = max(0, min(gy + gh, py + ph) - max(gy, py))
            inter = ix * iy; u = gw * gh + pw * ph - inter
            o = inter / u if u > 0 else 0
            if o >= 0.5:
                d[i, j] = 1 - o
    return d


def mot(gt, pr, min_conf):
    acc = mm.MOTAccumulator(auto_id=True)
    for fr in range(1, max(list(gt) + list(pr) + [1]) + 1):
        g = [(x[0], x[1], x[2] - x[0], x[3] - x[1]) for x in gt.get(fr, [])]
        gi = [x[5] for x in gt.get(fr, [])]
        ps = [x for x in pr.get(fr, []) if x[4] >= min_conf]
        p = [(x[0], x[1], x[2] - x[0], x[3] - x[1]) for x in ps]
        pi = [x[5] for x in ps]
        d = iou_dist(g, p)
        acc.update(gi, pi, d)
    return acc


def load_ids(path, has_conf):
    d = {}
    if not os.path.exists(path):
        return d
    for line in open(path):
        v = line.strip().split(",")
        if len(v) < 6:
            continue
        fr, i = int(float(v[0])), int(float(v[1]))
        x, y, w, h = map(float, v[2:6]); c = float(v[6]) if has_conf and len(v) > 6 else 1.0
        d.setdefault(fr, []).append((x, y, x + w, y + h, c, i))
    return d


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True); p.add_argument("--gt", required=True)
    p.add_argument("--min-conf", type=float, default=0.0)
    a = p.parse_args()
    accs, names, aps = [], [], []
    for gp in sorted(glob.glob(os.path.join(a.gt, "*/gt/gt.txt"))):
        name = gp.split(os.sep)[-3]
        gt, pr = load_ids(gp, False), load_ids(os.path.join(a.pred, name + ".txt"), True)
        accs.append(mot(gt, pr, a.min_conf)); names.append(name.split("_")[-1])
        aps.append(evaluate({k: [b[:5] for b in v] for k, v in gt.items()}, {k: [b[:5] for b in v] for k, v in pr.items()}, 0.5)[0])
    mh = mm.metrics.create()
    s = mh.compute_many(accs, names=names, generate_overall=True,
                        metrics=["mota", "idf1", "num_switches", "precision", "recall", "num_false_positives", "num_misses"])
    s["AP50"] = aps + [float(np.mean(aps))]
    print(s.round(3).to_string())
