"""Spatial-context features: failures are spatially coherent, so each cell also sees its neighbourhood.
For every feature f and window size w (cells): local mean over the w x w window (NaN-aware, the cell included) and
the cell's deviation from it (f - mean). Deviations also cancel slow scene-wide shifts. Onboard features only."""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy.ndimage import uniform_filter

def add_context(F: pd.DataFrame, grid, cols, windows=(3, 7)) -> pd.DataFrame:
    iy, ix = np.divmod(F.index.values, grid.nx); out = {}
    for c in cols:
        v = F[c].astype(float).replace([np.inf, -np.inf], np.nan).values
        img = np.zeros((grid.ny, grid.nx)); msk = np.zeros_like(img); ok = np.isfinite(v)
        img[iy[ok], ix[ok]] = v[ok]; msk[iy[ok], ix[ok]] = 1
        for w in windows:
            s = uniform_filter(img, w, mode="constant"); n = uniform_filter(msk, w, mode="constant")
            m = np.where(n > 1e-9, s / np.maximum(n, 1e-9), np.nan)[iy, ix]
            out[f"{c}_ctx{w}"] = m; out[f"{c}_dev{w}"] = v - m
    return pd.concat([F, pd.DataFrame(out, index=F.index)], axis=1)
