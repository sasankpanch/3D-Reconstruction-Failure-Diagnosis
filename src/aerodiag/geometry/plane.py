"""Robust plane fitting (iteratively trimmed SVD)."""
from __future__ import annotations
import numpy as np

def fit_plane(P: np.ndarray, iters: int = 4, k: float = 3.0):
    """Returns (unit normal with n_z >= 0, centroid, inlier mask, residual std)."""
    m = np.ones(len(P), bool)
    for _ in range(iters):
        c = P[m].mean(0); _, _, vt = np.linalg.svd(P[m] - c, full_matrices=False)
        n = vt[2] * (1 if vt[2][2] >= 0 else -1); r = (P - c) @ n
        s = 1.4826 * np.median(np.abs(r[m])) + 1e-9; m = np.abs(r) < k * s
    return n, c, m, float(np.std(r[m]))

def tilt_deg(n: np.ndarray, up=(0, 0, 1)) -> float:
    return float(np.degrees(np.arccos(np.clip(abs(n @ np.asarray(up, float)), -1, 1))))
