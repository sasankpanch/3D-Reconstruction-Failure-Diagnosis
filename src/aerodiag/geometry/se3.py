"""Rigid / similarity transforms and pose interpolation.

T_wc: 4x4 camera->world (p_W = R p_C + t). Batched arrays have shape (N,4,4).
Sim3: p_dst = s * R @ p_src + t.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp

def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    R = np.asarray(R); t = np.asarray(t)
    T = np.zeros(R.shape[:-2] + (4, 4)); T[..., :3, :3] = R; T[..., :3, 3] = t; T[..., 3, 3] = 1
    return T

def invert_T(T: np.ndarray) -> np.ndarray:
    R = T[..., :3, :3]; t = T[..., :3, 3]; Rt = np.swapaxes(R, -1, -2)
    return make_T(Rt, -np.einsum("...ij,...j->...i", Rt, t))

def apply_T(T: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Apply a single 4x4 transform to points (N,3)."""
    return X @ T[:3, :3].T + T[:3, 3]

def world_to_camera(T_wc: np.ndarray, X_w: np.ndarray) -> np.ndarray:
    return (X_w - T_wc[:3, 3]) @ T_wc[:3, :3]

def camera_to_world(T_wc: np.ndarray, X_c: np.ndarray) -> np.ndarray:
    return X_c @ T_wc[:3, :3].T + T_wc[:3, 3]

def rot_angle_deg(R: np.ndarray) -> np.ndarray:
    c = (np.trace(R, axis1=-2, axis2=-1) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))

def small_rotation(rotvec: np.ndarray) -> np.ndarray:
    return Rot.from_rotvec(rotvec).as_matrix()

def interpolate_poses(t_query_ns: np.ndarray, t_knots_ns: np.ndarray, T_knots: np.ndarray) -> np.ndarray:
    """Linear position + slerp rotation between knots (clamped at the ends). Times int64 ns."""
    t0 = t_knots_ns[0]
    tk = (t_knots_ns - t0).astype(np.float64); tq = np.clip((t_query_ns - t0).astype(np.float64), tk[0], tk[-1])
    pos = np.stack([np.interp(tq, tk, T_knots[:, k, 3]) for k in range(3)], 1)
    rot = Slerp(tk, Rot.from_matrix(T_knots[:, :3, :3]))(tq).as_matrix()
    return make_T(rot, pos)

@dataclass
class Sim3:
    s: float
    R: np.ndarray
    t: np.ndarray

    def apply(self, X: np.ndarray) -> np.ndarray:
        return self.s * (np.asarray(X) @ self.R.T) + self.t

    def apply_rotation(self, R: np.ndarray) -> np.ndarray:
        return self.R @ R

    def inverse(self) -> "Sim3":
        Ri = self.R.T; return Sim3(1 / self.s, Ri, -(Ri @ self.t) / self.s)

    def to_dict(self) -> dict:
        return dict(s=float(self.s), R=self.R.tolist(), t=self.t.tolist())

    @staticmethod
    def from_dict(d) -> "Sim3":
        return Sim3(float(d["s"]), np.array(d["R"], float), np.array(d["t"], float))

    def describe(self) -> dict:
        """Rotation magnitude, tilt of the source +Z axis in the target frame, and yaw."""
        z = self.R[:, 2]
        return dict(scale=float(self.s), rotation_deg=float(rot_angle_deg(self.R)),
                    tilt_deg=float(np.degrees(np.arccos(np.clip(z[2], -1, 1)))),
                    yaw_deg=float(np.degrees(np.arctan2(self.R[1, 0], self.R[0, 0]))))

def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> Sim3:
    """Least-squares similarity dst ~ s R src + t (Umeyama 1991)."""
    mx, my = src.mean(0), dst.mean(0); Xc, Yc = src - mx, dst - my
    U, S, Vt = np.linalg.svd(Yc.T @ Xc / len(src)); D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / Xc.var(0).sum() if with_scale else 1.0
    return Sim3(float(s), R, my - s * R @ mx)
