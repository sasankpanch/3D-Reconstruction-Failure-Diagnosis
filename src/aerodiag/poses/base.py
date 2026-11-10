"""Pose-source interface. A PoseSource produces a PoseSet for a set of frames (matched by int64-ns timestamp,
never by index). All poses: T_wc camera->world in the working (Terra) frame.

Sources: 'terra' (reference; comparison only), 'rtk' (PRIMARY onboard prior), 'noisy_prior' (controlled
sweeps), 'rosbag' (placeholder). Later modules (signals, diagnosis, evaluation) consume PoseSet only."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np

@dataclass
class PoseSet:
    name: str
    t_ns: np.ndarray           # (N,) int64
    T_wc: np.ndarray           # (N,4,4)
    valid: np.ndarray          # (N,) bool - pose usable at all
    meta: dict = field(default_factory=dict)     # every setting used (noise, seeds, transforms)
    flags: dict = field(default_factory=dict)    # per-frame arrays, e.g. is_reference, refined, fallback_*

    def __len__(self):
        return len(self.t_ns)

    def index_of(self, t_ns: np.ndarray, tol_ns: int = 1_000_000) -> np.ndarray:
        """Indices of frames matching timestamps (|dt| <= tol), -1 if none."""
        j = np.clip(np.searchsorted(self.t_ns, t_ns), 1, len(self.t_ns) - 1)
        j = np.where(np.abs(self.t_ns[j - 1] - t_ns) <= np.abs(self.t_ns[j] - t_ns), j - 1, j)
        return np.where(np.abs(self.t_ns[j] - t_ns) <= tol_ns, j, -1)

    def subset(self, mask: np.ndarray, name: str | None = None) -> "PoseSet":
        return PoseSet(name or self.name, self.t_ns[mask], self.T_wc[mask], self.valid[mask], dict(self.meta),
                       {k: v[mask] for k, v in self.flags.items()})

class PoseSource(ABC):
    name: str = "base"

    @abstractmethod
    def get(self, t_ns: np.ndarray) -> PoseSet:
        """Poses at the requested frame timestamps."""
