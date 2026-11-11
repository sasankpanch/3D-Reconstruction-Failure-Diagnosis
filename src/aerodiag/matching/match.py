"""Descriptor matching + POSE-INDEPENDENT outlier filtering.

The epipolar geometry is deliberately NOT used to filter matches: doing so would truncate the across-line
residual we want to measure. Filters: Lowe ratio test, mutual nearest neighbour, and local displacement
consistency (a match's displacement must agree with the median displacement of its k nearest neighbours in
image 1 within max(tol_px, tol_mad * neighbour MAD)). Moving objects survive if their neighbours move with them."""
from __future__ import annotations
import numpy as np, cv2
from scipy.spatial import cKDTree

def match_descriptors(d1, d2, ratio=0.8, mutual=True):
    from ..backend import use_cuda
    if use_cuda():
        from .match_cuda import match_descriptors as m
        return m(d1, d2, ratio, mutual)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    if len(d1) < 2 or len(d2) < 2:
        return np.zeros((0, 2), int)
    k12 = bf.knnMatch(d1, d2, k=2)
    a = np.array([[m[0].queryIdx, m[0].trainIdx, m[0].distance, m[1].distance] for m in k12 if len(m) == 2])
    a = a[a[:, 2] < ratio * a[:, 3]]
    pairs = a[:, :2].astype(int)
    if mutual and len(pairs):
        k21 = bf.match(d2, d1); back = np.full(len(d2), -1); back[[m.queryIdx for m in k21]] = [m.trainIdx for m in k21]
        pairs = pairs[back[pairs[:, 1]] == pairs[:, 0]]
    return pairs

def local_consistency(uv1, uv2, k=12, tol_px=12.0, tol_mad=4.0):
    from ..backend import use_cuda
    if use_cuda():
        from .match_cuda import local_consistency as lc
        return lc(uv1, uv2, k, tol_px, tol_mad)
    if len(uv1) <= k:
        return np.ones(len(uv1), bool)
    d = uv2 - uv1; _, nn = cKDTree(uv1).query(uv1, k + 1); nn = nn[:, 1:]
    med = np.median(d[nn], axis=1); mad = 1.4826 * np.median(np.abs(d[nn] - med[:, None, :]), axis=1)
    tol = np.maximum(tol_px, tol_mad * np.linalg.norm(mad, axis=1))
    return np.linalg.norm(d - med, axis=1) < tol

def match_pair(f1: dict, f2: dict, cfg: dict, rng=None):
    pr = match_descriptors(f1["des"], f2["des"], cfg["ratio"], cfg["mutual"])
    if len(pr) == 0:
        return pr, dict(n_raw=0, n_kept=0)
    ok = local_consistency(f1["uv"][pr[:, 0]], f2["uv"][pr[:, 1]], cfg["local_k"], cfg["local_tol_px"], cfg["local_tol_mad"])
    n_raw = len(pr); pr = pr[ok]
    if rng is not None and len(pr) > cfg["max_matches_per_pair"]:
        pr = pr[np.sort(rng.choice(len(pr), cfg["max_matches_per_pair"], replace=False))]
    return pr, dict(n_raw=int(n_raw), n_consistent=int(ok.sum()), n_kept=int(len(pr)))

def homography_filter(uv1, uv2, thr_px, iters=20000):
    """Loose global filter against gross mismatches (pose-independent). thr_px should exceed the expected terrain
    parallax so that genuine 3D structure is kept."""
    import cv2
    if len(uv1) < 8:
        return np.zeros(len(uv1), bool)
    H, inl = cv2.findHomography(uv1.astype(np.float64), uv2.astype(np.float64), cv2.USAC_MAGSAC, thr_px, maxIters=iters, confidence=0.9999)
    return inl.ravel().astype(bool) if inl is not None else np.zeros(len(uv1), bool)

def match_pair_full(f1, f2, cfg, homog_thr_px, rng=None):
    pr = match_descriptors(f1["des"], f2["des"], cfg["ratio"], cfg["mutual"])
    info = dict(n_raw=int(len(pr)))
    if len(pr) == 0:
        return pr, dict(info, n_consistent=0, n_homog=0, n_kept=0)
    ok = local_consistency(f1["uv"][pr[:, 0]], f2["uv"][pr[:, 1]], cfg["local_k"], cfg["local_tol_px"], cfg["local_tol_mad"])
    pr = pr[ok]; info["n_consistent"] = int(len(pr))
    ok = homography_filter(f1["uv"][pr[:, 0]], f2["uv"][pr[:, 1]], homog_thr_px); pr = pr[ok]; info["n_homog"] = int(len(pr))
    if rng is not None and len(pr) > cfg["max_matches_per_pair"]:
        pr = pr[np.sort(rng.choice(len(pr), cfg["max_matches_per_pair"], replace=False))]
    info["n_kept"] = int(len(pr)); return pr, info
