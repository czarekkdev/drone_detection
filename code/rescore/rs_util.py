class Sel:
    """classifier that uses a subset of the feature columns (so feature lists can change without re-dumping)"""
    def __init__(s, clf, keep): s.clf, s.keep = clf, keep
    def predict_proba(s, X): return s.clf.predict_proba(X[:, s.keep])


class RawConf:
    """sanity model: probability = the detector's own confidence (no rescoring)"""
    def predict_proba(s, X):
        import numpy as np
        return np.stack([1 - X[:, 0], X[:, 0]], 1)
