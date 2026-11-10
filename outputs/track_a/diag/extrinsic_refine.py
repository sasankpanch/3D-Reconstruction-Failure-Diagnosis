"""Diagnostic (Q4 follow-up): can a small LiDAR->camera rotation correction dR (and range scale k) make the
LiDAR map self-consistent across flight lines and altitudes? Objective: mean over group pairs of the median
|z_a - z_b| on 1 m cells observed by both groups (>=3 pts each). Groups: L0..L5 + low-altitude (hover/climb).
p_C' = dR @ (R_CL @ ((1+k) p_L)) + t_CL. Terra keyframe poses fixed."""
import numpy as np, pandas as pd, json, time
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot
from aerodiag.config import load_config
from aerodiag.io.dataset import Dataset
from aerodiag.grid.ground_grid import GroundGrid
from aerodiag.grid.rasterize import grouped_quantiles
ds = Dataset(load_config("configs/data.yaml")); f = ds.frames
fidx = np.where(f.is_reference)[0]
PL, FR = [], []
for k, i in enumerate(fidx):
    x = ds.load_lidar(i); lab = ds.load_labels(i); m = (~np.isin(lab, [20, 24])) & (np.linalg.norm(x, axis=1) > 5)
    x = x[m][::3]; PL.append(x); FR.append(np.full(len(x), k))
PL = np.concatenate(PL); FR = np.concatenate(FR); T = ds.T_wc_json[fidx]
seg = f.segment.values[fidx]; line = f.line_id.values[fidx]
grp_f = np.where(np.isin(seg, ["cruise_line", "diagonal"]), line, np.where(np.isin(seg, ["hover", "climb"]), 10, -1))
grp = grp_f[FR]; G = sorted(set(grp_f) - {-1})
print("points", len(PL), "groups", G, flush=True)
grid = GroundGrid.from_bounds(-370, -85, 165, 285, 1.0)
def world(p):
    dR = Rot.from_rotvec(np.asarray(p[:3]) * 1e-3).as_matrix(); k = p[3] * 1e-6
    pc = (PL * (1 + k)) @ (dR @ ds.calib.R_CL).T + ds.calib.t_CL
    return np.einsum("nij,nj->ni", T[FR, :3, :3], pc) + T[FR, :3, 3]
def rasters(X):
    out = {}
    ids = grid.cell_id(X[:, :2])
    for g in G:
        ii = np.where(grp == g, ids, -1); c, q = grouped_quantiles(ii, X[:, 2], grid.n, (0.5,)); z = q[0]; z[c < 3] = np.nan; out[g] = z
    return out
def objective(p, detail=False):
    R = rasters(world(p)); vals = {}
    for i, a in enumerate(G):
        for b in G[i + 1:]:
            m = np.isfinite(R[a]) & np.isfinite(R[b])
            if m.sum() > 300: vals[(a, b)] = (np.median(np.abs(R[a][m] - R[b][m])), np.median(R[a][m] - R[b][m]), int(m.sum()))
    obj = np.mean([v[0] for v in vals.values()])
    return (obj, vals) if detail else obj
t0 = time.time(); o0, v0 = objective(np.zeros(4), True); print("baseline", round(o0, 4), f"{time.time()-t0:.1f}s", flush=True)
res = minimize(objective, np.zeros(4), method="Nelder-Mead", options=dict(xatol=0.05, fatol=1e-4, maxfev=160, initial_simplex=np.array(
    [[0, 0, 0, 0], [5, 0, 0, 0], [0, 5, 0, 0], [0, 0, 5, 0], [0, 0, 0, 3000]], float)))
o1, v1 = objective(res.x, True)
print("refined params (mrad x,y,z ; range ppm):", res.x.round(3), "obj", round(o1, 4), "nfev", res.nfev)
rows = [dict(a=a, b=b, n=v0[(a, b)][2], med_abs_before=v0[(a, b)][0], med_before=v0[(a, b)][1],
             med_abs_after=v1[(a, b)][0], med_after=v1[(a, b)][1]) for (a, b) in v0 if (a, b) in v1]
df = pd.DataFrame(rows).round(3); print(df.to_string())
json.dump(dict(params_mrad_ppm=res.x.tolist(), obj_before=o0, obj_after=o1, rot_deg=float(np.linalg.norm(res.x[:3]) * 1e-3 * 180 / np.pi)),
          open("outputs/track_a/diag/extrinsic_refine.json", "w"), indent=1)
df.to_csv("outputs/track_a/diag/extrinsic_refine_pairs.csv", index=False)
