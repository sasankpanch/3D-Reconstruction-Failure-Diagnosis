"""'noisy_prior' source: a base PoseSet (normally Terra reference) + injected noise, for controlled sweeps.

Noise model per run (seeded): constant bias (drawn once) + per-frame white noise, for position (horizontal,
vertical separately, metres, world frame) and attitude (degrees, rotation vector in the camera frame).
Presets (configurable): 'rtk' = cm-level, 'gnss' = metre-level."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
from .base import PoseSource, PoseSet
from ..geometry.se3 import small_rotation

@dataclass
class NoiseSpec:
    pos_h_sigma: float = 0.0       # m, white
    pos_v_sigma: float = 0.0       # m, white
    pos_h_bias: float = 0.0        # m, std of the per-run constant bias
    pos_v_bias: float = 0.0
    att_sigma_deg: float = 0.0     # deg, white (per axis)
    att_bias_deg: float = 0.0      # deg, std of per-run constant bias (per axis)

PRESETS = {
    "rtk":  NoiseSpec(pos_h_sigma=0.02, pos_v_sigma=0.03, pos_h_bias=0.01, pos_v_bias=0.02, att_sigma_deg=0.05, att_bias_deg=0.02),
    "gnss": NoiseSpec(pos_h_sigma=0.5, pos_v_sigma=1.0, pos_h_bias=1.5, pos_v_bias=3.0, att_sigma_deg=0.3, att_bias_deg=0.2),
}

def inject_noise(T_wc: np.ndarray, spec: NoiseSpec, seed: int) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed); n = len(T_wc); T = T_wc.copy()
    bias = np.r_[rng.normal(0, spec.pos_h_bias, 2), rng.normal(0, spec.pos_v_bias, 1)]
    white = np.c_[rng.normal(0, spec.pos_h_sigma, (n, 2)), rng.normal(0, spec.pos_v_sigma, (n, 1))]
    dpos = bias + white
    abias = rng.normal(0, np.radians(spec.att_bias_deg), 3)
    drot = abias + rng.normal(0, np.radians(spec.att_sigma_deg), (n, 3))
    T[:, :3, 3] += dpos
    T[:, :3, :3] = np.einsum("nij,njk->nik", T[:, :3, :3], np.array([small_rotation(v) for v in drot]))
    return T, dict(dpos=dpos, drot_rad=drot, bias_pos=bias, bias_rot_rad=abias)

class NoisyPriorSource(PoseSource):
    name = "noisy_prior"

    def __init__(self, base: PoseSource, spec: NoiseSpec | str, seed: int = 0):
        self.base = base; self.preset = spec if isinstance(spec, str) else "custom"
        self.spec = PRESETS[spec] if isinstance(spec, str) else spec; self.seed = seed

    def get(self, t_ns: np.ndarray) -> PoseSet:
        b = self.base.get(t_ns); T, info = inject_noise(b.T_wc, self.spec, self.seed)
        return PoseSet(f"{self.name}:{self.preset}", b.t_ns, T, b.valid.copy(),
                       meta=dict(base=b.name, base_meta=b.meta, preset=self.preset, spec=asdict(self.spec), seed=self.seed),
                       flags=dict(b.flags, noise_pos=info["dpos"], noise_rot_rad=info["drot_rad"]))
