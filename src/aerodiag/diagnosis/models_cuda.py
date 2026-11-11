"""CUDA versions of the step-3 models (PyTorch). Same algorithms and hyper-parameters as the numpy reference
(classifier.fit_logreg, gbdt.fit_gbdt); accumulations in float64 so results match to rounding. Models are returned in
the reference format (numpy arrays / nested tuples), so a model fitted on the GPU can be applied by the CPU code and
vice versa."""
from __future__ import annotations
import numpy as np
import torch
from .gbdt import make_bins

DEV = torch.device("cuda")
f64 = lambda a: torch.as_tensor(np.asarray(a, np.float64), device=DEV)

# ---------------------------------------------------------------- logistic regression (full-batch gradient descent)
def fit_logreg(X, y, l2=1.0, iters=300, lr=0.5):
    X = f64(X); y = f64(y); n = len(y); w = torch.zeros(X.shape[1], dtype=torch.float64, device=DEV); b = torch.zeros((), dtype=torch.float64, device=DEV)
    pos = float(y.mean()); sw = torch.where(y == 1, 0.5 / max(pos, 1e-6), 0.5 / max(1 - pos, 1e-6))
    for _ in range(iters):
        z = X @ w + b; p = 1 / (1 + torch.exp(-z.clamp(-30, 30))); g = sw * (p - y)
        w = w - lr * (X.T @ g / n + l2 * w / n); b = b - lr * g.mean()
    return w.cpu().numpy(), float(b)

def predict_logreg(model, X):
    w, b = model; z = f64(X) @ f64(w) + b
    return (1 / (1 + torch.exp(-z.clamp(-30, 30)))).cpu().numpy()

# ---------------------------------------------------------------- gradient-boosted trees (histogram splits)
def binarize(X, edges, n_bins=32):
    """GPU binning, identical to gbdt.binarize: bin = number of edges <= x (searchsorted right); NaN -> n_bins."""
    Xt = f64(X); F = Xt.shape[1]; E = max(1, max(len(e) for e in edges))
    Et = torch.full((F, E), float("inf"), dtype=torch.float64, device=DEV)
    for f, e in enumerate(edges):
        if len(e):
            Et[f, :len(e)] = f64(e)
    B = torch.searchsorted(Et, Xt.T.contiguous(), right=True).T
    return torch.where(torch.isfinite(Xt), B, torch.full_like(B, n_bins)).to(torch.int64)

def _tree(B, g, h, idx, depth, lam, min_leaf, nb):
    G, H = float(g[idx].sum()), float(h[idx].sum())
    if depth == 0 or len(idx) < 2 * min_leaf:
        return -G / (H + lam)
    Bi = B[idx]; n, F = Bi.shape; key = (Bi + torch.arange(F, device=DEV) * (nb + 1)).reshape(-1)
    hist = lambda wt: torch.bincount(key, weights=None if wt is None else wt[idx][:, None].expand(n, F).reshape(-1), minlength=F * (nb + 1)).reshape(F, nb + 1).to(torch.float64)
    hg, hh, hn = hist(g), hist(h), hist(None)
    cg = torch.cumsum(hg[:, :nb], 1); ch = torch.cumsum(hh[:, :nb], 1); cn = torch.cumsum(hn[:, :nb], 1)
    base = G * G / (H + lam); gains = []
    for ml in (False, True):                                   # order (feature, miss_left, bin) = reference loop order
        g_, h_, n_ = (cg + hg[:, nb:], ch + hh[:, nb:], cn + hn[:, nb:]) if ml else (cg, ch, cn)
        ok = (n_ >= min_leaf) & (n - n_ >= min_leaf)
        gain = g_ ** 2 / (h_ + lam) + (G - g_) ** 2 / (H - h_ + lam) - base
        gains.append(torch.where(ok, gain, torch.full_like(gain, -float("inf"))))
    A = torch.stack(gains, 1).reshape(-1); j = int(torch.argmax(A)); best = float(A[j])
    if not best > 0.0:
        return -G / (H + lam)
    f, rem = divmod(j, 2 * nb); ml, k = divmod(rem, nb); ml = bool(ml)
    b = Bi[:, f]; left = ((b <= k) & (b != nb)) | ((b == nb) & ml)
    return (f, k, ml, _tree(B, g, h, idx[left], depth - 1, lam, min_leaf, nb), _tree(B, g, h, idx[~left], depth - 1, lam, min_leaf, nb))

def _apply(t, B, nb):
    out = torch.empty(len(B), dtype=torch.float64, device=DEV); stack = [(t, torch.arange(len(B), device=DEV))]
    while stack:
        node, ix = stack.pop()
        if not isinstance(node, tuple):
            out[ix] = node; continue
        f, k, ml, l, r = node; b = B[ix, f]; left = ((b <= k) & (b != nb)) | ((b == nb) & ml)
        stack += [(l, ix[left]), (r, ix[~left])]
    return out

def fit_gbdt(X, y, rounds=150, depth=3, lr=0.1, lam=10.0, min_leaf=50, subsample=0.5, max_rows=60000, n_bins=32, seed=0, loss="logistic"):
    rng = np.random.default_rng(seed); edges = make_bins(np.asarray(X, float), n_bins); B = binarize(X, edges, n_bins); yt = f64(y); N = len(y)
    if loss == "l2":
        w = torch.ones(N, dtype=torch.float64, device=DEV); f0 = float(np.mean(y))
    else:
        pos = float(np.mean(y)); w = torch.where(yt == 1, 0.5 / max(pos, 1e-6), 0.5 / max(1 - pos, 1e-6)); w = w / w.mean(); f0 = 0.0
    f = torch.full((N,), f0, dtype=torch.float64, device=DEV); trees = []
    for _ in range(rounds):
        if loss == "l2":
            g = f - yt; h = w
        else:
            p = 1 / (1 + torch.exp(-f)); g = w * (p - yt); h = w * p * (1 - p)
        n = min(int(N * subsample), max_rows); idx = torch.as_tensor(rng.choice(N, n, replace=False), device=DEV)
        t = _tree(B, g, h, idx, depth, lam, min_leaf, n_bins); trees.append(t); f = f + lr * _apply(t, B, n_bins)
    return dict(edges=edges, trees=trees, lr=lr, n_bins=n_bins, f0=f0, loss=loss)

def predict_gbdt(model, X):
    B = binarize(X, model["edges"], model["n_bins"]); f = torch.full((len(B),), model.get("f0", 0.0), dtype=torch.float64, device=DEV)
    for t in model["trees"]:
        f = f + model["lr"] * _apply(t, B, model["n_bins"])
    return (f if model.get("loss") == "l2" else 1 / (1 + torch.exp(-f))).cpu().numpy()

# ---------------------------------------------------------------- block bootstrap of AUROC / AP (all replicates at once)
def block_bootstrap(y, s, blocks, metric, reps=200, seed=0, rep_chunk=50):
    """Same resampling as classifier.block_bootstrap (identical rng draws: blocks drawn with replacement), evaluated as
    integer per-sample weights on the GPU. AUROC is exact (ties count 1/2, as with rank averaging); AP is the tie-aware
    step definition, identical to classifier.average_precision."""
    y = np.asarray(y); s = np.asarray(s, float); ub, inv = np.unique(blocks, return_inverse=True); rng = np.random.default_rng(seed)
    if len(ub) < 5:
        return (np.nan, np.nan)
    picks = np.stack([np.bincount(rng.integers(0, len(ub), len(ub)), minlength=len(ub)) for _ in range(reps)])   # (R, n_blocks)
    W_all = torch.as_tensor(picks, dtype=torch.float64, device=DEV)[:, torch.as_tensor(inv, device=DEV)]          # (R, n)
    pos = torch.as_tensor(y == 1, device=DEV); neg = torch.as_tensor(y == 0, device=DEV); vals = []
    sign = 1.0 if metric == "auroc" else -1.0                            # AUROC: ascending groups; AP: descending
    order = np.argsort(sign * s, kind="stable"); su = s[order]; grp = np.r_[0, np.cumsum(su[1:] != su[:-1])]; G = int(grp[-1]) + 1
    o = torch.as_tensor(order, device=DEV); gi = torch.as_tensor(grp, device=DEV); pz, nz = pos[o], neg[o]
    for r0 in range(0, reps, rep_chunk):
        W = W_all[r0:r0 + rep_chunk][:, o]
        if metric == "auroc":
            W1 = torch.zeros(len(W), G, dtype=torch.float64, device=DEV).index_add_(1, gi, W * pz)
            W0 = torch.zeros(len(W), G, dtype=torch.float64, device=DEV).index_add_(1, gi, W * nz)
            n1, n0 = W1.sum(1), W0.sum(1); below = torch.cumsum(W0, 1) - W0
            v = ((W1 * below).sum(1) + 0.5 * (W1 * W0).sum(1)) / (n1 * n0)
            v = torch.where((n1 >= 5) & (n0 >= 5), v, torch.full_like(v, float("nan")))
        else:                                                             # tie-aware AP over score groups
            W1 = torch.zeros(len(W), G, dtype=torch.float64, device=DEV).index_add_(1, gi, W * pz)
            Wn = torch.zeros(len(W), G, dtype=torch.float64, device=DEV).index_add_(1, gi, W)
            tp = torch.cumsum(W1, 1); nn = torch.cumsum(Wn, 1); n1 = tp[:, -1]
            v = (W1 * torch.where(nn > 0, tp / nn.clamp(min=1e-12), torch.zeros_like(nn))).sum(1) / n1
            v = torch.where(n1 > 0, v, torch.full_like(v, float("nan")))
        vals.append(v)
    v = torch.cat(vals).cpu().numpy(); v = v[np.isfinite(v)]
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) > 20 else (np.nan, np.nan)
