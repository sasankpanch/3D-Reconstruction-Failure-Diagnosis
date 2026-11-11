"""'How much will it fail': per-cell height-error magnitude with distribution-free upper bounds.

Target t = log(|Terra - LiDAR| + eps) on reliable-ground-truth cells (comparison side, training/evaluation only).
Point model: gradient-boosted regression trees, out-of-fold over the same 25 m spatial blocks as the heads.
Upper bound: split-conformal on the out-of-fold residuals r = t - t_hat, cross-fitted over blocks (the quantile used
for a fold comes from the other folds only). Mondrian variant: the quantile is taken within the same decile of t_hat,
so confident cells get tight bounds and doubtful cells wide ones. Target coverage alpha (default 0.9)."""
from __future__ import annotations
import numpy as np
from .gbdt import fit_gbdt, predict_gbdt

def _fold_of(blocks, folds, seed):
    ub = np.unique(blocks); rng = np.random.default_rng(seed); f = dict(zip(ub, rng.integers(0, folds, len(ub))))
    return np.array([f[b] for b in blocks])

def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])

def magnitude_cv(X, err_abs, blocks, folds=5, seed=0, alpha=0.9, eps=0.1, n_bins=10, gbdt_kw=None):
    """Returns per-cell oof point prediction (m), upper bounds (global and Mondrian, m) and metrics."""
    gbdt_kw = dict(gbdt_kw or {}); gbdt_kw.pop("loss", None)
    t = np.log(err_abs + eps); fold = _fold_of(blocks, folds, seed); th = np.full(len(t), np.nan)
    for k in range(folds):
        tr, te = fold != k, fold == k
        th[te] = predict_gbdt(fit_gbdt(X[tr], t[tr], loss="l2", **gbdt_kw), X[te])
    r = t - th; fold2 = _fold_of(blocks, folds, seed + 7)                     # independent split for the conformal step
    edges = np.quantile(th, np.linspace(0, 1, n_bins + 1)[1:-1]); dec = np.searchsorted(edges, th)
    ub_g = np.full(len(t), np.nan); ub_m = np.full(len(t), np.nan)
    for k in range(folds):
        cal, te = fold2 != k, fold2 == k
        ub_g[te] = th[te] + np.quantile(r[cal], alpha)
        for b in range(n_bins):
            cb, tb = cal & (dec == b), te & (dec == b)
            if tb.any():
                ub_m[tb] = th[tb] + np.quantile(r[cb] if cb.sum() >= 30 else r[cal], alpha)
    back = lambda z: np.maximum(np.exp(z) - eps, 0)
    pred, U_g, U_m = back(th), back(ub_g), back(ub_m)
    met = dict(n=int(len(t)), spearman=spearman(th, t), mae_m=float(np.median(np.abs(pred - err_abs))),
               median_abs_error_m=float(np.median(err_abs)), alpha=alpha)
    for nm, U in [("global", U_g), ("mondrian", U_m)]:
        cov = err_abs <= U; tight = U < 0.5
        met[nm] = dict(coverage=float(cov.mean()), median_bound_m=float(np.median(U)),
                       coverage_by_fold=[float(cov[fold2 == k].mean()) for k in range(folds)],
                       frac_certified_below_0p5m=float(tight.mean()),
                       true_below_0p5m_among_certified=float((err_abs[tight] < 0.5).mean()) if tight.any() else np.nan,
                       coverage_on_failures=float(cov[err_abs >= 0.5].mean()))
    return pred, U_g, U_m, met
