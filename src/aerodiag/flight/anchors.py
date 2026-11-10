"""Identify independently registered ("anchor") poses vs interpolated ones, and the reference set.

Phase 0 (F1): sampleinfos_interpolated.json holds Terra-registered poses only on the interval-5 grid
(orig_idx % 5 == 0, orig_idx = round((t - t0)/100 ms) counting dropped frames); other frames are
linear-position / slerp-rotation interpolations. A further set of off-grid non-interpolated frames (turns)
is unvalidated and excluded from the reference set by default (Q3)."""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation as Rot

def original_index(t_ns: np.ndarray, period_ns: int = 100_000_000) -> np.ndarray:
    return np.round((t_ns - t_ns[0]) / period_ns).astype(np.int64)

def detect_anchors(t_ns: np.ndarray, T_wc: np.ndarray, tol_m: float = 1e-3, tol_deg: float = 1e-3) -> np.ndarray:
    """Greedy removal of frames that equal the interpolation of their neighbouring anchors."""
    n = len(t_ns); C = T_wc[:, :3, 3]; R = Rot.from_matrix(T_wc[:, :3, :3])
    def dev(a, i, b):
        w = (t_ns[i] - t_ns[a]) / (t_ns[b] - t_ns[a])
        dp = np.linalg.norm(C[i] - ((1 - w) * C[a] + w * C[b]))
        Ri = R[a] * Rot.from_rotvec(w * (R[a].inv() * R[b]).as_rotvec())
        return dp, np.degrees((Ri.inv() * R[i]).magnitude())
    anch = list(range(n)); changed = True
    while changed:
        changed = False; keep = [anch[0]]
        for k in range(1, len(anch) - 1):
            dp, dr = dev(keep[-1], anch[k], anch[k + 1])
            if dp < tol_m and dr < tol_deg:
                changed = True
            else:
                keep.append(anch[k])
        keep.append(anch[-1]); anch = keep
    out = np.zeros(n, bool); out[anch] = True
    return out

def reference_mask(t_ns, T_wc, modulo=5, period_ns=100_000_000, tol_m=1e-3, tol_deg=1e-3, exclude_off_grid=True):
    orig = original_index(t_ns, period_ns); anchor = detect_anchors(t_ns, T_wc, tol_m, tol_deg)
    on_grid = orig % modulo == 0
    ref = anchor & on_grid if exclude_off_grid else anchor
    return dict(orig_idx=orig, is_anchor=anchor, on_grid=on_grid, is_reference=ref)
