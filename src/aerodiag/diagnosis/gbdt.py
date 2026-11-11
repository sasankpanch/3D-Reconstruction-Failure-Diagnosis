"""Small gradient-boosted decision trees in numpy (no extra dependency): logistic loss, histogram splits on per-feature
quantile bins (decided on the training data), depth-limited trees, shrinkage, row subsampling, class-balanced weights,
L2 on leaf values. Missing values get their own bin. Interface mirrors classifier.fit_logreg / predict."""
from __future__ import annotations
import numpy as np

def make_bins(X, n_bins=32):
    """Per-feature bin edges from training quantiles (NaN -> bin n_bins)."""
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    return [np.unique(np.nanquantile(X[:, f], qs)) if np.isfinite(X[:, f]).any() else np.array([]) for f in range(X.shape[1])]

def binarize(X, edges, n_bins=32):
    B = np.empty(X.shape, np.uint8)
    for f, e in enumerate(edges):
        b = np.searchsorted(e, X[:, f], side="right"); b[~np.isfinite(X[:, f])] = n_bins; B[:, f] = b
    return B

def _tree(B, g, h, idx, depth, lam, min_leaf, nb):
    """Greedy tree on gradients g / hessians h; returns nested tuples (feature, threshold_bin, left, right) or leaf value."""
    G, H = g[idx].sum(), h[idx].sum()
    if depth == 0 or len(idx) < 2 * min_leaf:
        return -G / (H + lam)
    best = (0.0, None); base = G * G / (H + lam); Bi = B[idx]
    for f in range(B.shape[1]):
        hg = np.bincount(Bi[:, f], g[idx], nb + 1); hh = np.bincount(Bi[:, f], h[idx], nb + 1); hn = np.bincount(Bi[:, f], minlength=nb + 1)
        # missing bin (nb) is sent left or right, whichever is better: try both orderings
        for miss_left in (False, True):
            cg = np.cumsum(hg[:nb]); ch = np.cumsum(hh[:nb]); cn = np.cumsum(hn[:nb])
            if miss_left:
                cg = cg + hg[nb]; ch = ch + hh[nb]; cn = cn + hn[nb]
            ok = (cn >= min_leaf) & (len(idx) - cn >= min_leaf)
            if not ok.any():
                continue
            gain = cg ** 2 / (ch + lam) + (G - cg) ** 2 / (H - ch + lam) - base; gain[~ok] = -np.inf
            k = int(np.argmax(gain))
            if gain[k] > best[0]:
                best = (gain[k], (f, k, miss_left))
    if best[1] is None:
        return -G / (H + lam)
    f, k, ml = best[1]; b = Bi[:, f]; left = (b <= k) & (b != nb) | ((b == nb) & ml)
    return (f, k, ml, _tree(B, g, h, idx[left], depth - 1, lam, min_leaf, nb), _tree(B, g, h, idx[~left], depth - 1, lam, min_leaf, nb))

def _apply(t, B, nb):
    out = np.empty(len(B)); stack = [(t, np.arange(len(B)))]
    while stack:
        node, ix = stack.pop()
        if not isinstance(node, tuple):
            out[ix] = node; continue
        f, k, ml, l, r = node; b = B[ix, f]; left = (b <= k) & (b != nb) | ((b == nb) & ml)
        stack += [(l, ix[left]), (r, ix[~left])]
    return out

def fit_gbdt(X, y, rounds=150, depth=3, lr=0.1, lam=10.0, min_leaf=50, subsample=0.5, max_rows=60000, n_bins=32, seed=0, loss="logistic"):
    """loss='logistic' (binary y, class-balanced) or 'l2' (regression, squared error)."""
    rng = np.random.default_rng(seed); edges = make_bins(X, n_bins); B = binarize(X, edges, n_bins)
    if loss == "l2":
        w = np.ones(len(y)); f0 = float(np.mean(y))
    else:
        pos = y.mean(); w = np.where(y == 1, 0.5 / max(pos, 1e-6), 0.5 / max(1 - pos, 1e-6)); w /= w.mean(); f0 = 0.0
    f = np.full(len(y), f0); trees = []
    for _ in range(rounds):
        if loss == "l2":
            g = f - y; h = w
        else:
            p = 1 / (1 + np.exp(-f)); g = w * (p - y); h = w * p * (1 - p)
        n = min(int(len(y) * subsample), max_rows); idx = rng.choice(len(y), n, replace=False)
        t = _tree(B, g, h, idx, depth, lam, min_leaf, n_bins); trees.append(t); f += lr * _apply(t, B, n_bins)
    return dict(edges=edges, trees=trees, lr=lr, n_bins=n_bins, f0=f0, loss=loss)

def predict_gbdt(model, X):
    B = binarize(X, model["edges"], model["n_bins"]); f = np.full(len(X), model.get("f0", 0.0))
    for t in model["trees"]:
        f += model["lr"] * _apply(t, B, model["n_bins"])
    return f if model.get("loss") == "l2" else 1 / (1 + np.exp(-f))
