"""Epipolar geometry from known poses and decomposition of correspondence residuals.

For a match x1 (image 1) <-> x2 (image 2) in UNDISTORTED NORMALISED coordinates and relative pose
p_C2 = R21 p_C1 + t21:
* epipolar line in image 2: l = E x1h, E = [t21]_x R21
* across  = signed distance of x2 to l (normalised units; x f -> pixels). ~0 for a static, opaque, correctly
            posed scene; non-zero = violated assumption (motion, view-dependent appearance, occlusion) or
            pose/calibration/matching error (the "noise floor").
* along   = signed position of x2 along l, measured from x_inf (image of x1's ray at infinite depth), positive
            toward finite depths. Encodes depth -> terrain height.
* depth1  = depth (camera-1 z) of the triangulated point along x1's ray (least squares on x2 x (l R x1 + t)=0).
Pixels: residuals in normalised units are multiplied by the mean focal length f (exact at the image centre;
the local scale of the distortion model differs by < 1% across this image).
"""
from __future__ import annotations
import numpy as np

def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

def relative_pose(T_wc1: np.ndarray, T_wc2: np.ndarray):
    """(R21, t21) with p_C2 = R21 p_C1 + t21."""
    R1, c1 = T_wc1[:3, :3], T_wc1[:3, 3]; R2, c2 = T_wc2[:3, :3], T_wc2[:3, 3]
    R21 = R2.T @ R1; t21 = R2.T @ (c1 - c2)
    return R21, t21

def essential(R21: np.ndarray, t21: np.ndarray) -> np.ndarray:
    return skew(t21) @ R21

def _h(x):
    return np.c_[x, np.ones(len(x))]

def sampson(E: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Sampson distance (normalised units, first-order geometric error; signed via x2^T E x1)."""
    a, b = _h(x1), _h(x2)
    Ex1 = a @ E.T; Etx2 = b @ E; num = (b * Ex1).sum(1)
    den = Ex1[:, 0] ** 2 + Ex1[:, 1] ** 2 + Etx2[:, 0] ** 2 + Etx2[:, 1] ** 2
    return num / np.sqrt(den + 1e-300)

def decompose(x1: np.ndarray, x2: np.ndarray, R21: np.ndarray, t21: np.ndarray, f: float = 1.0) -> dict:
    """Along/across decomposition (see module doc). Returns arrays in pixels (x f) plus depth1 (metres if t in m)."""
    a, b = _h(x1), _h(x2)
    E = essential(R21, t21)
    l = a @ E.T                                    # lines in image 2 (a,b,c)
    nrm = np.hypot(l[:, 0], l[:, 1]) + 1e-300
    across = (b * l).sum(1) / nrm
    d = np.stack([l[:, 1], -l[:, 0]], 1) / nrm[:, None]     # line direction
    ri = a @ R21.T; x_inf = ri[:, :2] / ri[:, 2:3]          # image of the ray at infinite depth
    near = ri + t21; x_near = near[:, :2] / near[:, 2:3]    # image of the ray at depth 1 (orientation only)
    sgn = np.sign(((x_near - x_inf) * d).sum(1)); sgn[sgn == 0] = 1; d *= sgn[:, None]
    along = ((x2 - x_inf) * d).sum(1)
    c1 = np.cross(b, ri); c2 = np.cross(b, np.broadcast_to(t21, b.shape))
    depth1 = -(c1 * c2).sum(1) / ((c1 * c1).sum(1) + 1e-300)
    return dict(across_px=across * f, along_px=along * f, depth1=depth1,
                sampson_px=sampson(E, x1, x2) * f, line_dir=d)

def triangulate_world(x1: np.ndarray, depth1: np.ndarray, T_wc1: np.ndarray) -> np.ndarray:
    Xc = _h(x1) * depth1[:, None]
    return Xc @ T_wc1[:3, :3].T + T_wc1[:3, 3]
