"""Fit the candidate rescorer.  python fit_rs.py --train preds/rs/auav_0.npz,... --out preds/rs/model_gb.pkl --kind gb"""
import argparse, pickle, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.inspection import permutation_importance
p = argparse.ArgumentParser(); p.add_argument("--train", required=True); p.add_argument("--test", default="")
p.add_argument("--out", default=""); p.add_argument("--kind", default="gb"); p.add_argument("--drop", default="")
p.add_argument("--test-only", default="", help="keep only test rows whose video contains this"); p.add_argument("--train-skip", default="")
p.add_argument("--ignore-band", action="store_true", help="train: leave out boxes that partly overlap a drone (0.05 < IoU < 0.5)"); p.add_argument("--imp", action="store_true"); p.add_argument("--mono", action="store_true")
a = p.parse_args()
def load(fs, only="", skip="", band=False):
    Z = [np.load(f) for f in fs.split(",")]
    X = np.concatenate([z["X"] for z in Z]); y = np.concatenate([z["iou"] for z in Z]) >= 0.5; v = np.concatenate([z["video"] for z in Z])
    m = np.ones(len(y), bool)
    if only: m &= np.array([only in s for s in v])
    if skip: m &= np.array([skip not in s for s in v])
    if band: io = np.concatenate([z["iou"] for z in Z]); m &= ~((io > 0.05) & (io < 0.5))
    return X[m], y[m], v[m], list(Z[0]["feats"])
X, y, v, F = load(a.train, skip=a.train_skip, band=a.ignore_band)
keep = [i for i, f in enumerate(F) if f not in a.drop.split(",")]
print(f"train rows {len(y)} pos {y.mean():.3f} videos {len(set(v))}")
if a.kind == "gb":
    def sign(f):
        if f in ("blob_dist", "trk_dist", "lost_dist"): return -1
        if f in ("conf", "full_conf", "crop_conf", "mask_frac", "blob_iou", "blob_same", "blob_score") or f.endswith(("_in_mean", "_in_max", "_contrast")): return 1
        return 0
    mono = [sign(F[i]) if a.mono else 0 for i in keep]
    clf = HistGradientBoostingClassifier(monotonic_cst=mono, max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0, random_state=0)
else:
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
from rs_util import Sel
clf.fit(X[:, keep], y); M = Sel(clf, keep)
if a.test:
    Xt, yt, vt, _ = load(a.test, only=a.test_only)
    pt = M.predict_proba(Xt)[:, 1]; c = Xt[:, F.index("conf")]
    print(f"test rows {len(yt)} pos {yt.mean():.3f} | AUC model {roc_auc_score(yt, pt):.3f} raw conf {roc_auc_score(yt, c):.3f} | AP model {average_precision_score(yt, pt):.3f} raw {average_precision_score(yt, c):.3f}")
    if a.imp:
        r = permutation_importance(clf, Xt[:, keep], yt, scoring="average_precision", n_repeats=3, random_state=0)
        for i in np.argsort(-r.importances_mean)[:15]:
            print(f"  {F[keep[i]]:16s} {r.importances_mean[i]:+.4f}")
if a.out:
    pickle.dump(M, open(a.out, "wb"))
