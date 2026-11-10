"""Track A consistency checks: Q7 global LiDAR time offset, Q5 line-to-line consistency, Q4 LiDAR bias."""
from __future__ import annotations
import numpy as np, pandas as pd
from ..grid.ground_grid import GroundGrid
from ..grid.rasterize import grouped_quantiles

def _raster_median(X, grid: GroundGrid, min_pts=3):
    ids = grid.cell_id(X[:, :2].astype(np.float64)); cnt, q = grouped_quantiles(ids, X[:, 2].astype(np.float64), grid.n, (0.5,))
    z = q[0]; z[cnt < min_pts] = np.nan; return z

def estimate_time_offset(X, pt_frame, v_frame, group_a, group_b, deltas_s, cell=0.5):
    """Opposite-heading groups (per point boolean masks). Misplacement differs by 2*v*delta between groups, so the
    best delta minimises the height disagreement of the two groups' rasters, measured on high-gradient cells."""
    Xa, Xb = X[group_a].astype(np.float64), X[group_b].astype(np.float64)
    lo = np.minimum(Xa.min(0), Xb.min(0)); hi = np.maximum(Xa.max(0), Xb.max(0))
    grid = GroundGrid.from_bounds(lo[0], lo[1], hi[0], hi[1], cell)
    z0 = _raster_median(Xa, grid); img = grid.to_image(z0)
    gy, gx = np.gradient(img); grad = np.hypot(gx, gy).ravel() / cell
    rows = []
    for d in deltas_s:
        za = _raster_median(Xa + v_frame[pt_frame[group_a]] * d, grid); zb = _raster_median(Xb + v_frame[pt_frame[group_b]] * d, grid)
        m = np.isfinite(za) & np.isfinite(zb) & (grad > 0.5)
        rows.append((d, int(m.sum()), float(np.median(np.abs(za[m] - zb[m])))))
    df = pd.DataFrame(rows, columns=["delta_s", "n_cells", "median_abs_dz"])
    # parabolic refinement around the minimum
    k = int(df.median_abs_dz.idxmin()); best = df.delta_s[k]
    if 0 < k < len(df) - 1:
        y0, y1, y2 = df.median_abs_dz[k - 1], df.median_abs_dz[k], df.median_abs_dz[k + 1]; h = df.delta_s[k + 1] - df.delta_s[k]
        den = y0 - 2 * y1 + y2
        if den > 0:
            best = df.delta_s[k] + 0.5 * h * (y0 - y2) / den
    return float(best), df

def pairwise_line_alignment(X, line_of_point, pairs, voxel=0.3, max_dist=1.0):
    """Q5: point-to-plane ICP of line A's points onto line B's in their overlap. Returns translation/rotation."""
    import open3d as o3d
    rows = []
    for a, b in pairs:
        A = X[line_of_point == a].astype(np.float64); B = X[line_of_point == b].astype(np.float64)
        lo = np.maximum(A[:, :2].min(0), B[:, :2].min(0)); hi = np.minimum(A[:, :2].max(0), B[:, :2].max(0))
        def crop(P):
            m = np.all((P[:, :2] > lo) & (P[:, :2] < hi), 1); return P[m]
        A, B = crop(A), crop(B)
        # restrict to the true overlap: cells (2 m) populated by both
        g = GroundGrid.from_bounds(lo[0], lo[1], hi[0], hi[1], 2.0)
        ca, cb = g.cell_id(A[:, :2]), g.cell_id(B[:, :2]); both = np.intersect1d(np.unique(ca), np.unique(cb))
        A, B = A[np.isin(ca, both)], B[np.isin(cb, both)]
        if len(A) < 1000 or len(B) < 1000:
            rows.append(dict(a=a, b=b, n_overlap_cells=len(both))); continue
        pa = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(A)).voxel_down_sample(voxel)
        pb = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(B)).voxel_down_sample(voxel)
        pb.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
        reg = o3d.pipelines.registration
        r = reg.registration_icp(pa, pb, max_dist, np.eye(4), reg.TransformationEstimationPointToPlane(reg.TukeyLoss(k=0.2)),
                                 reg.ICPConvergenceCriteria(max_iteration=60))
        T = r.transformation; ang = np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))
        rows.append(dict(a=a, b=b, n_overlap_cells=len(both), area_m2=4 * len(both), dx=T[0, 3], dy=T[1, 3], dz=T[2, 3],
                         rot_deg=ang, fitness=r.fitness, rmse=r.inlier_rmse))
    return pd.DataFrame(rows)

def lidar_geometry_regression(r, xn, yn, rng):
    """Q4(a): regress per-point height residual (vs the cell median over all frames) on scan geometry.
    A LiDAR-camera rotation error of eps shows as dz ~ eps * range * (x or y component); a range-scale error
    k shows as dz ~ -k * range * cos(off-nadir)."""
    A = np.c_[np.ones_like(r), xn * rng, yn * rng, rng / np.sqrt(1 + xn ** 2 + yn ** 2)]
    coef, *_ = np.linalg.lstsq(A, r, rcond=None)
    return dict(offset_m=coef[0], tilt_x_mrad=coef[1] * 1e3, tilt_y_mrad=coef[2] * 1e3, range_scale_ppm=coef[3] * 1e6,
                resid_std_before=float(np.std(r)), resid_std_after=float(np.std(r - A @ coef)))
