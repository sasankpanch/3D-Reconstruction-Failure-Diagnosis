"""'rtk' source (PRIMARY onboard prior): RTK antenna position interpolated to image timestamps and mapped into
the working frame by the Phase-0 similarity (RTK-ENU -> Terra). Attitude: reference-keyframe slerp of Terra
rotations (the IMU stand-in until the rosbag is available) + configurable attitude noise.

Known limitations (logged in meta): no antenna->camera lever arm (not identifiable, Phase 0 §7); the RTK-Terra
mismatch is line-dependent (0.5-2.4 m, Phase 0 F5); attitude on non-keyframes inherits interpolation error
(5-20 px equivalent) until refined (B2)."""
from __future__ import annotations
import numpy as np, pandas as pd
from .base import PoseSource, PoseSet
from ..geometry.se3 import Sim3, umeyama, make_T, small_rotation
from ..io.rtk import interpolate_rtk

def fit_rtk_to_working(rtk: pd.DataFrame, t_ns: np.ndarray, C_ref: np.ndarray, mask: np.ndarray,
                       time_offset_ns: int = 0) -> tuple[Sim3, np.ndarray]:
    """Similarity RTK-ENU -> working frame on frames `mask`. Returns (sim3, residual norms for all frames)."""
    P = interpolate_rtk(rtk, t_ns + time_offset_ns)
    m = mask & np.isfinite(P).all(1)
    sim = umeyama(P[m], C_ref[m])
    return sim, np.linalg.norm(sim.apply(P) - C_ref, axis=1)

class RtkPoseSource(PoseSource):
    name = "rtk"

    def __init__(self, rtk: pd.DataFrame, sim: Sim3, attitude_T_wc: np.ndarray, attitude_t_ns: np.ndarray,
                 attitude_noise_deg: float = 0.0, time_offset_ns: int = 0, seed: int = 0):
        self.rtk, self.sim, self.att, self.att_t = rtk, sim, attitude_T_wc, attitude_t_ns
        self.noise, self.dt, self.seed = attitude_noise_deg, time_offset_ns, seed

    def get(self, t_ns: np.ndarray) -> PoseSet:
        P = interpolate_rtk(self.rtk, np.asarray(t_ns) + self.dt)
        C = self.sim.apply(P)
        idx = np.searchsorted(self.att_t, t_ns); idx = np.clip(idx, 0, len(self.att_t) - 1)
        if not np.array_equal(self.att_t[idx], t_ns):
            raise KeyError("attitude available only at image timestamps")
        R = self.att[idx, :3, :3].copy()
        if self.noise > 0:
            rng = np.random.default_rng(self.seed)
            dR = np.array([small_rotation(v) for v in rng.normal(0, np.radians(self.noise), (len(R), 3))])
            R = R @ dR                          # perturbation in the camera frame
        valid = np.isfinite(C).all(1)
        return PoseSet(self.name, np.asarray(t_ns), make_T(R, np.nan_to_num(C)), valid,
                       meta=dict(sim3=self.sim.to_dict(), attitude_noise_deg=self.noise, time_offset_ns=self.dt,
                                 seed=self.seed, attitude_source="terra reference-keyframe slerp",
                                 lever_arm="none (not identifiable)"))
