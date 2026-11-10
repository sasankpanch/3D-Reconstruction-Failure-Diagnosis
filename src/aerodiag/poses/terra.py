"""'terra' source: Terra T4x4 (offline-refined REFERENCE poses; NOT onboard). By default only the
Terra-registered keyframes are valid (Phase 0 F1, Q1); interpolated frames are flagged invalid."""
from __future__ import annotations
import numpy as np
from .base import PoseSource, PoseSet

class TerraPoseSource(PoseSource):
    name = "terra"

    def __init__(self, t_ns: np.ndarray, T_wc: np.ndarray, is_reference: np.ndarray, reference_only: bool = True):
        self.t_ns, self.T, self.ref, self.reference_only = t_ns, T_wc, is_reference, reference_only

    def get(self, t_ns: np.ndarray) -> PoseSet:
        idx = np.searchsorted(self.t_ns, t_ns)
        idx = np.clip(idx, 0, len(self.t_ns) - 1)
        if not np.array_equal(self.t_ns[idx], t_ns):
            raise KeyError("terra poses exist only at image timestamps")
        valid = self.ref[idx] if self.reference_only else np.ones(len(idx), bool)
        return PoseSet(self.name, np.asarray(t_ns), self.T[idx].copy(), valid,
                       meta=dict(reference_only=self.reference_only), flags=dict(is_reference=self.ref[idx].copy()))

    def reference_slerp(self) -> "TerraPoseSource":
        """Poses at all frames from slerp/linear interpolation of reference keyframes only (equals the JSON on
        on-grid-interpolated frames; replaces unvalidated off-grid anchors)."""
        from ..geometry.se3 import interpolate_poses
        T = interpolate_poses(self.t_ns, self.t_ns[self.ref], self.T[self.ref])
        return TerraPoseSource(self.t_ns, T, self.ref, reference_only=False)
