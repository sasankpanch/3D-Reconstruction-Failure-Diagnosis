"""B2: rotation-only pose refinement. Translation is kept from the prior; a small rotation correction w (rad, rotation
vector in camera-2 frame, R_wc2' = R_wc2 Exp(w)) is estimated from matches by minimising the robust Sampson error.
A full essential matrix is NOT fitted (degenerate on near-planar terrain).

With p_C2 = R21 p_C1 + t21: R21' = Exp(-w) R21, t21' = Exp(-w) t21.
Fallbacks (logged separately): too few matches -> prior; either image mostly water (LiDAR labels) -> prior."""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot
from ..geometry.epipolar import essential, sampson

def apply_correction(w, R21, t21):
    Rm = Rot.from_rotvec(-np.asarray(w)).as_matrix(); return Rm @ R21, Rm @ t21

def refine_rotation(x1, x2, R21, t21, f, loss="soft_l1", f_scale_px=1.0, max_fit=1500, rng=None):
    """Returns (w, info). x1, x2 undistorted normalised coordinates of matches used for fitting."""
    if rng is not None and len(x1) > max_fit:
        s = rng.choice(len(x1), max_fit, replace=False); x1, x2 = x1[s], x2[s]
    def res(w):
        R, t = apply_correction(w, R21, t21); return sampson(essential(R, t), x1, x2) * f
    r0 = res(np.zeros(3))
    sol = least_squares(res, np.zeros(3), loss=loss, f_scale=f_scale_px, x_scale=1e-3, max_nfev=50)
    return sol.x, dict(cost0=float(np.median(np.abs(r0))), cost1=float(np.median(np.abs(sol.fun))), nfev=int(sol.nfev),
                       w_deg=float(np.degrees(np.linalg.norm(sol.x))), n_fit=int(len(x1)))

def refine_cv(x1, x2, R21, t21, f, folds=2, rng=None, **kw):
    """Cross-validated refinement: each match gets the correction fitted on the OTHER folds.
    Returns (per-match R21', t21' as fold assignment, list of w, infos)."""
    n = len(x1); fold = (rng.permutation(n) if rng is not None else np.arange(n)) % folds
    ws, infos = [], []
    for k in range(folds):
        tr = fold != k
        w, info = refine_rotation(x1[tr], x2[tr], R21, t21, f, rng=rng, **kw); ws.append(w); infos.append(info)
    return fold, ws, infos
