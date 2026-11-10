"""Step 3c: small multi-label classifier (numpy one-vs-rest logistic regression, L2, class-balanced), with
spatially blocked cross-validation (no neighbouring-cell leakage). Features are standardised with medians imputed and
missing-value indicators added."""
from __future__ import annotations
import numpy as np, pandas as pd

def prepare(F: pd.DataFrame, cols=None, stats=None):
    X = (F[cols] if cols is not None else F.select_dtypes("number")).astype(float).replace([np.inf, -np.inf], np.nan)
    if stats is None:                                    # drop all-missing / constant columns (decided on the data given)
        X = X.loc[:, X.notna().any() & (X.std() > 0)]
    else:
        X = X[list(stats[0].index)]
    cols = list(X.columns)
    if stats is None:
        med = X.median(); Xf = X.fillna(med); mu = Xf.mean(); sd = Xf.std().replace(0, 1); stats = (med, mu, sd)
    med, mu, sd = stats
    miss = X.isna().astype(float).add_suffix("_missing"); Xs = ((X.fillna(med) - mu) / sd).clip(-6, 6)
    return np.c_[Xs.values, miss.values], cols, stats

def fit_logreg(X, y, l2=1.0, iters=300, lr=0.5):
    w = np.zeros(X.shape[1]); b = 0.0; pos = y.mean(); sw = np.where(y == 1, 0.5 / max(pos, 1e-6), 0.5 / max(1 - pos, 1e-6))
    for _ in range(iters):                               # full-batch gradient descent (convex; small problems)
        z = X @ w + b; p = 1 / (1 + np.exp(-np.clip(z, -30, 30))); g = sw * (p - y)
        w -= lr * (X.T @ g / len(y) + l2 * w / len(y)); b -= lr * g.mean()
    return w, b

def predict(model, X):
    w, b = model; return 1 / (1 + np.exp(-np.clip(X @ w + b, -30, 30)))

def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s); n1 = (y == 1).sum(); n0 = (y == 0).sum()
    if n1 < 5 or n0 < 5:
        return np.nan
    r = pd.Series(s).rank().values; return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def spatial_cv(F, X, labels: pd.DataFrame, xy, block_m=25.0, folds=5, seed=0, **kw):
    """Blocked K-fold CV. Returns out-of-fold probabilities (cells x labels) and per-label metrics."""
    blk = (np.floor(xy[:, 0] / block_m).astype(int) * 100003 + np.floor(xy[:, 1] / block_m).astype(int))
    ub = np.unique(blk); rng = np.random.default_rng(seed); fold_of = dict(zip(ub, rng.integers(0, folds, len(ub)))); fold = np.array([fold_of[b] for b in blk])
    oof = pd.DataFrame(np.nan, index=F.index, columns=labels.columns); metrics = {}
    for lab in labels.columns:
        y = labels[lab].values; m = np.isfinite(y)
        for k in range(folds):
            tr = m & (fold != k); te = m & (fold == k)
            if tr.sum() < 20 or (y[tr] == 1).sum() < 5:
                continue
            oof.loc[te, lab] = predict(fit_logreg(X[tr], y[tr], **kw), X[te])
        mm = m & oof[lab].notna().values
        metrics[lab] = dict(n=int(mm.sum()), positives=int((y[mm] == 1).sum()), auroc=auroc(y[mm], oof[lab].values[mm]),
                            base_rate=float(np.mean(y[mm] == 1)) if mm.any() else np.nan)
    return oof, metrics


def average_precision(y, s):
    y = np.asarray(y); s = np.asarray(s); o = np.argsort(-s); y = y[o]; tp = np.cumsum(y == 1)
    prec = tp / np.arange(1, len(y) + 1); n1 = (y == 1).sum()
    return float((prec * (y == 1)).sum() / n1) if n1 else np.nan

def recall_at_fpr(y, s, fpr=0.05):
    y = np.asarray(y); s = np.asarray(s); neg = s[y == 0]
    if len(neg) < 5 or (y == 1).sum() < 5:
        return np.nan
    thr = np.quantile(neg, 1 - fpr); return float((s[y == 1] > thr).mean())

def block_bootstrap(y, s, blocks, fn, reps=200, seed=0):
    """95% CI of fn(y, s) by resampling spatial blocks with replacement."""
    y = np.asarray(y); s = np.asarray(s); ub, inv = np.unique(blocks, return_inverse=True); rng = np.random.default_rng(seed)
    if len(ub) < 5:
        return (np.nan, np.nan)
    idx_by_block = [np.where(inv == k)[0] for k in range(len(ub))]; vals = []
    for _ in range(reps):
        pick = rng.integers(0, len(ub), len(ub)); ii = np.concatenate([idx_by_block[k] for k in pick])
        v = fn(y[ii], s[ii])
        if np.isfinite(v):
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if len(vals) > 20 else (np.nan, np.nan)

def full_metrics(y, s, blocks, seed=0):
    """Paper metrics for one binary head: AUROC [95% CI], AP [95% CI], recall@5%FPR, Brier, base rate."""
    m = np.isfinite(y) & np.isfinite(s); y, s, b = y[m], s[m], blocks[m]
    if len(y) == 0:
        return dict(n=0, positives=0, base_rate=np.nan, auroc=np.nan, auroc_ci95=(np.nan, np.nan), ap=np.nan, ap_ci95=(np.nan, np.nan),
                    recall_at_5pct_fpr=np.nan, brier=np.nan)
    return dict(n=int(m.sum()), positives=int((y == 1).sum()), base_rate=float((y == 1).mean()) if len(y) else np.nan,
                auroc=auroc(y, s), auroc_ci95=block_bootstrap(y, s, b, auroc, seed=seed),
                ap=average_precision(y, s), ap_ci95=block_bootstrap(y, s, b, average_precision, seed=seed),
                recall_at_5pct_fpr=recall_at_fpr(y, s), brier=float(np.mean((s - y) ** 2)) if 0 <= np.nanmin(s) and np.nanmax(s) <= 1 else np.nan)

def blocks_of(xy, block_m=25.0):
    return np.floor(xy[:, 0] / block_m).astype(np.int64) * 100003 + np.floor(xy[:, 1] / block_m).astype(np.int64)
