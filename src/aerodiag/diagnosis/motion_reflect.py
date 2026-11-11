"""Step 3b: separate moving from reflective (view-dependent) surfaces by fitting appearance change against time gap
and viewing-angle change, per cell, with neighbourhood pooling for sparsely observed cells.

Per (cell, pair) observation: dt (s), dtheta (deg, angle between the two viewing rays at the cell),
y_geo = |across_px| (geometric violation) and y_pho = 1 - NCC (photometric change), y_gain = |log brightness ratio|.
Model per cell (least squares): y = a + b_t * dt + b_th * dtheta, fitted for each y.
Moving surfaces: violation grows with dt (b_t > 0) at small dtheta. View-dependent (reflective) surfaces: photometric
change grows with dtheta (b_th > 0) without a time trend. Cells with < min_obs observations borrow observations from
the 3x3 (then 5x5) neighbourhood (pooled flag recorded). Scores are slope / standard error (t-statistics)."""
from __future__ import annotations
import numpy as np, pandas as pd

def _cell_fit(cell, X, y, n_cells):
    """Per-cell OLS y ~ X (X includes intercept). Returns coef (n_cells,p), t-stats, n. CUDA when available (same
    normal equations in float64)."""
    from ..backend import use_cuda
    if use_cuda():
        return _cell_fit_cuda(cell, X, y, n_cells)
    p = X.shape[1]
    XtX = np.zeros((n_cells, p, p)); Xty = np.zeros((n_cells, p)); yy = np.zeros(n_cells); n = np.bincount(cell, minlength=n_cells)
    np.add.at(XtX, cell, X[:, :, None] * X[:, None, :]); np.add.at(Xty, cell, X * y[:, None]); np.add.at(yy, cell, y * y)
    reg = 1e-6 * np.eye(p); coef = np.full((n_cells, p), np.nan); tst = np.full((n_cells, p), np.nan)
    ok = n > p + 1
    A = XtX[ok] + reg; b = Xty[ok]; c = np.linalg.solve(A, b[..., None])[..., 0]
    rss = np.maximum(yy[ok] - np.einsum("np,np->n", c, b), 0); s2 = rss / np.maximum(n[ok] - p, 1)
    cov = np.linalg.inv(A) * s2[:, None, None]; se = np.sqrt(np.maximum(np.einsum("npp->np", cov), 1e-12))
    coef[ok] = c; tst[ok] = c / se
    return coef, tst, n

def fit_motion_reflectance(S: pd.DataFrame, grid, min_obs=6, pool_radii=(1, 2)):
    """S: cell x pair rows with columns cell_id, dt_s, dtheta_deg, across_px, cost_min, gain.
    Returns per-cell fit table indexed by cell_id (only cells with own observations)."""
    S = S.dropna(subset=["across_px", "cost_min", "gain", "dtheta_deg"])
    own = S.cell_id.values.astype(np.int64); cells, own_n = np.unique(own, return_counts=True)
    iy, ix = np.divmod(own, grid.nx)
    borrow_cell, borrow_src = [own], [np.arange(len(S))]
    radius = pd.Series(0, index=cells); have = pd.Series(own_n, index=cells)
    for r in pool_radii:
        needy = have.index[have.values < min_obs].values
        if len(needy) == 0:
            break
        radius.loc[needy] = r
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if max(abs(dx), abs(dy)) != r:          # only the new ring at this radius
                    continue
                tx, ty = ix - dx, iy - dy                # cell that borrows an observation located at offset (dx, dy)
                inside = (tx >= 0) & (tx < grid.nx) & (ty >= 0) & (ty < grid.ny)
                tgt = ty * grid.nx + tx; m = inside & np.isin(tgt, needy)
                borrow_cell.append(tgt[m]); borrow_src.append(np.where(m)[0])
        allc = np.concatenate(borrow_cell)
        have = pd.Series(allc).value_counts().reindex(cells).fillna(0)
    cell_all = np.concatenate(borrow_cell); src = np.concatenate(borrow_src)
    keep = np.isin(cell_all, cells); cell_all, src = cell_all[keep], src[keep]
    cid, cinv = np.unique(cell_all, return_inverse=True)
    D = S.iloc[src]; X = np.c_[np.ones(len(D)), D.dt_s.values, D.dtheta_deg.values]
    out = pd.DataFrame(index=pd.Index(cid, name="cell_id"))
    for name, y in [("geo", D.across_px.abs().values), ("pho", D.cost_min.values), ("gain", D.gain.abs().values)]:
        coef, tst, n = _cell_fit(cinv, X, y, len(cid))
        out[f"{name}_a"] = coef[:, 0]; out[f"{name}_bt"] = coef[:, 1]; out[f"{name}_bth"] = coef[:, 2]
        out[f"{name}_t_dt"] = tst[:, 1]; out[f"{name}_t_dth"] = tst[:, 2]
    out["n_obs_fit"] = np.bincount(cinv, minlength=len(cid)); out["pool_radius"] = radius.reindex(cid).values
    out["moving_score"] = np.clip(out.geo_t_dt, 0, None)
    out["reflective_score"] = np.clip(np.fmax(out.pho_t_dth, out.gain_t_dth), 0, None)
    return out

def _cell_fit_cuda(cell, X, y, n_cells):
    import torch
    d = torch.device("cuda"); c = torch.as_tensor(cell, device=d); Xt = torch.as_tensor(X, dtype=torch.float64, device=d)
    yt = torch.as_tensor(y, dtype=torch.float64, device=d); p = X.shape[1]
    XtX = torch.zeros(n_cells, p * p, dtype=torch.float64, device=d).index_add_(0, c, (Xt[:, :, None] * Xt[:, None, :]).reshape(-1, p * p)).reshape(n_cells, p, p)
    Xty = torch.zeros(n_cells, p, dtype=torch.float64, device=d).index_add_(0, c, Xt * yt[:, None])
    yy = torch.zeros(n_cells, dtype=torch.float64, device=d).index_add_(0, c, yt * yt); n = torch.bincount(c, minlength=n_cells)
    ok = n > p + 1; A = XtX[ok] + 1e-6 * torch.eye(p, dtype=torch.float64, device=d); b = Xty[ok]
    co = torch.linalg.solve(A, b[..., None])[..., 0]; rss = torch.clamp(yy[ok] - (co * b).sum(1), min=0); s2 = rss / torch.clamp(n[ok] - p, min=1)
    cov = torch.linalg.inv(A) * s2[:, None, None]; se = torch.sqrt(torch.clamp(torch.diagonal(cov, dim1=1, dim2=2), min=1e-12))
    coef = torch.full((n_cells, p), float("nan"), dtype=torch.float64, device=d); tst = coef.clone(); coef[ok] = co; tst[ok] = co / se
    return coef.cpu().numpy(), tst.cpu().numpy(), n.cpu().numpy()
