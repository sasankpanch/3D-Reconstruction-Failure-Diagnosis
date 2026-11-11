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
    """float32 matrix products (consumer GPUs run float64 ~64x slower), float64 parameters; predictions agree with
    the float64 numpy reference to ~1e-5 (checked in tests)."""
    X = torch.as_tensor(np.asarray(X, np.float32), device=DEV); y = torch.as_tensor(np.asarray(y, np.float32), device=DEV); n = len(y)
    w = torch.zeros(X.shape[1], dtype=torch.float64, device=DEV); b = 0.0
    pos = float(y.double().mean()); sw = torch.where(y == 1, 0.5 / max(pos, 1e-6), 0.5 / max(1 - pos, 1e-6)).float()
    torch.backends.cuda.matmul.allow_tf32 = False
    for _ in range(iters):
        z = (X @ w.float()).double() + b; p = 1 / (1 + torch.exp(-z.clamp(-30, 30))); g = sw.double() * (p - y.double())
        w = w - lr * ((X.T @ g.float()).double() / n + l2 * w / n); b = b - lr * float(g.mean())
    return w.cpu().numpy(), float(b)

def predict_logreg(model, X):
    w, b = model; z = (torch.as_tensor(np.asarray(X, np.float32), device=DEV) @ torch.as_tensor(np.asarray(w, np.float32), device=DEV)).double() + b
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

def _tree_levelwise(B, g, h, idx, depth, lam, min_leaf, nb):
    """Grow one tree breadth-first: every node of a level gets its histograms from ONE bincount and its best split
    from one vectorised argmax (same split rule, tie order and stopping rules as gbdt._tree). Heap layout: node k has
    children 2k+1, 2k+2. Returns the reference nested-tuple tree."""
    F = B.shape[1]; Bi = B[idx]; gi = g[idx]; hi = h[idx]; n_all = len(idx)
    node = torch.zeros(n_all, dtype=torch.long, device=DEV); splits = {}; leaves = {}; active = [0]
    for d in range(depth + 1):
        if not active:
            break
        nodes = torch.as_tensor(active, device=DEV); A = len(active); slot = torch.full((2 ** (depth + 1),), -1, dtype=torch.long, device=DEV)
        slot[nodes] = torch.arange(A, device=DEV); sl = slot[node]; inn = sl >= 0
        Gs = torch.zeros(A, dtype=torch.float64, device=DEV).index_add_(0, sl[inn], gi[inn]); Hs = torch.zeros(A, dtype=torch.float64, device=DEV).index_add_(0, sl[inn], hi[inn])
        Ns = torch.bincount(sl[inn], minlength=A)
        Gc, Hc, Nc = Gs.cpu().numpy(), Hs.cpu().numpy(), Ns.cpu().numpy(); nxt = []
        can = (d < depth) & (Nc >= 2 * min_leaf)
        if can.any():
            rows = inn & torch.as_tensor(can, device=DEV)[sl.clamp(min=0)]; r = torch.nonzero(rows).squeeze(1); sr = sl[r]
            key = (sr[:, None] * (F * (nb + 1)) + torch.arange(F, device=DEV)[None, :] * (nb + 1) + Bi[r]).reshape(-1); L = A * F * (nb + 1)
            hg = torch.bincount(key, weights=gi[r][:, None].expand(-1, F).reshape(-1), minlength=L).reshape(A, F, nb + 1)
            hh = torch.bincount(key, weights=hi[r][:, None].expand(-1, F).reshape(-1), minlength=L).reshape(A, F, nb + 1)
            hn = torch.bincount(key, minlength=L).reshape(A, F, nb + 1).double()
            cg = torch.cumsum(hg[..., :nb], -1); ch = torch.cumsum(hh[..., :nb], -1); cn = torch.cumsum(hn[..., :nb], -1)
            G_ = Gs[:, None, None]; H_ = Hs[:, None, None]; N_ = Ns.double()[:, None, None]; gains = []
            for ml in (False, True):
                g_, h_, n_ = (cg + hg[..., nb:], ch + hh[..., nb:], cn + hn[..., nb:]) if ml else (cg, ch, cn)
                ok = (n_ >= min_leaf) & (N_ - n_ >= min_leaf)
                gain = g_ ** 2 / (h_ + lam) + (G_ - g_) ** 2 / (H_ - h_ + lam) - G_ ** 2 / (H_ + lam)
                gains.append(torch.where(ok, gain, torch.full_like(gain, -float("inf"))))
            Gn = torch.stack(gains, 2).reshape(A, -1); best, j = Gn.max(1)          # max returns the first maximal index
            best, j = best.cpu().numpy(), j.cpu().numpy()
        for a_, k in enumerate(active):
            if can[a_] and best[a_] > 0.0:
                f, rem = divmod(int(j[a_]), 2 * nb); ml, kb = divmod(rem, nb); splits[k] = (f, kb, bool(ml)); nxt += [2 * k + 1, 2 * k + 2]
            else:
                leaves[k] = -Gc[a_] / (Hc[a_] + lam)
        if splits:                                                        # route rows of nodes split at this level
            sp = [k for k in active if k in splits]
            if sp:
                fT = torch.full((2 ** (depth + 1),), 0, dtype=torch.long, device=DEV); kT = fT.clone(); mT = torch.zeros_like(fT, dtype=torch.bool); isS = mT.clone()
                for k in sp:
                    fT[k], kT[k], mT[k], isS[k] = splits[k][0], splits[k][1], splits[k][2], True
                m = isS[node]; b = Bi[torch.arange(n_all, device=DEV), fT[node]]
                left = ((b <= kT[node]) & (b != nb)) | ((b == nb) & mT[node])
                node = torch.where(m, torch.where(left, 2 * node + 1, 2 * node + 2), node)
        active = nxt
    def build(k):
        if k in splits:
            f, kb, ml = splits[k]; return (f, kb, ml, build(2 * k + 1), build(2 * k + 2))
        return float(leaves[k])
    return build(0)

def _heap(t, depth):
    """nested tree -> heap arrays (feature, threshold bin, missing-left, is-split, leaf value) for vectorised apply."""
    M = 2 ** (depth + 2); f = np.zeros(M, np.int64); k = np.zeros(M, np.int64); ml = np.zeros(M, bool); sp = np.zeros(M, bool); v = np.zeros(M)
    stack = [(t, 0)]
    while stack:
        node, i = stack.pop()
        if isinstance(node, tuple):
            f[i], k[i], ml[i], sp[i] = node[0], node[1], node[2], True; stack += [(node[3], 2 * i + 1), (node[4], 2 * i + 2)]
        else:
            v[i] = node
    return [torch.as_tensor(a, device=DEV) for a in (f, k, ml, sp, v)]

def _apply(t, B, nb, depth=8):
    f, k, ml, sp, v = _heap(t, depth); node = torch.zeros(len(B), dtype=torch.long, device=DEV); r = torch.arange(len(B), device=DEV)
    for _ in range(depth + 1):
        m = sp[node]
        if not bool(m.any()):
            break
        b = B[r, f[node]]; left = ((b <= k[node]) & (b != nb)) | ((b == nb) & ml[node])
        node = torch.where(m, torch.where(left, 2 * node + 1, 2 * node + 2), node)
    return v[node].double()

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
        t = _tree_levelwise(B, g, h, idx, depth, lam, min_leaf, n_bins); trees.append(t); f = f + lr * _apply(t, B, n_bins, depth)
    return dict(edges=edges, trees=trees, lr=lr, n_bins=n_bins, f0=f0, loss=loss, depth=depth)

def predict_gbdt(model, X):
    B = binarize(X, model["edges"], model["n_bins"]); f = torch.full((len(B),), model.get("f0", 0.0), dtype=torch.float64, device=DEV)
    for t in model["trees"]:
        f = f + model["lr"] * _apply(t, B, model["n_bins"], model.get("depth", 8))
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
