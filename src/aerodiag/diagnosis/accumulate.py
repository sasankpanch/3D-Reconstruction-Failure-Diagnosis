"""Step 3a: accumulate (cell x pair) signals into per-cell features, stratified by baseline class.

Baseline classes: short (within-line gap <= 5 frames: motion-sensitive, height-insensitive), mid (10-20),
long (40-80: occlusion / layering / height), cross (cross-line + crossing: large heading change -> view dependence).
Per class: robust statistics of each signal; overall: coverage, flags, glint.
All inputs are onboard step-2 outputs (no labels)."""
from __future__ import annotations
import numpy as np, pandas as pd

def baseline_class(pair_type, gap):
    return np.select([pair_type.ne("within_line"), gap <= 5, gap <= 20], ["cross", "short", "mid"], "long")

ABS = {"across_px", "gain", "dz_sigma"}
PER_CLASS = ["across_px", "cost_min", "sharpness", "flatness", "gain", "contrast_ratio", "dz_sigma"]

def accumulate(S: pd.DataFrame) -> pd.DataFrame:
    """Dispatch: CUDA port (accumulate_cuda) when available, else the pandas reference accumulate_np."""
    from ..backend import use_cuda
    if use_cuda():
        from .accumulate_cuda import accumulate as acc
        return acc(S)
    return accumulate_np(S)

def accumulate_np(S: pd.DataFrame) -> pd.DataFrame:
    S = S.copy(); S["bcls"] = baseline_class(S.pair_type, S.gap_frames)
    for k in ABS:
        S[k] = S[k].abs()
    valid = S.cost_min.notna()
    out = []
    g_all = S.groupby("cell_id")
    base = pd.DataFrame(dict(n_pairs=g_all.size(), textureless_frac=g_all.cost_min.apply(lambda s: s.isna().mean()),
                             glint_max=g_all.glint_risk.max(), max_baseline_m=g_all.baseline_m.max(),
                             max_heading_diff=g_all.heading_diff_deg.max()))
    V = S[valid]; gv = V.groupby("cell_id")
    base["n_valid"] = gv.size(); base["periodic_frac"] = gv.periodic.mean(); base["multi_min_frac"] = gv.n_minima.apply(lambda s: (s >= 2).mean())
    base["below_frac"] = gv.below_terrain.mean(); base["boundary_frac"] = gv.at_boundary.mean(); base["ref_std_med"] = gv.ref_std.median()
    # glint confirmation: geometric glint risk alone fires on any slope facing the half-vector; a real specular
    # reflection must also change brightness between the near-specular view and the other view. Excess |gain| of
    # high-risk pairs over the cell's own low-risk pairs (onboard photometry only).
    hi = V[V.glint_risk >= 0.3].groupby("cell_id").gain.median(); lo = V[V.glint_risk < 0.05].groupby("cell_id").gain.median()
    base["glint_frac"] = gv.glint_risk.apply(lambda s: (s >= 0.3).mean())
    base["glint_gain_excess"] = (hi - lo).reindex(base.index)
    out.append(base)
    for b in ["short", "mid", "long", "cross"]:
        gb = V[V.bcls == b].groupby("cell_id")
        f = pd.DataFrame({f"{b}_n": gb.size()})
        for k in PER_CLASS:
            f[f"{b}_{k}_p50"] = gb[k].median(); f[f"{b}_{k}_p90"] = gb[k].quantile(0.9)
        out.append(f)
    F = pd.concat(out, axis=1)
    F["n_valid"] = F.n_valid.fillna(0)
    return F
