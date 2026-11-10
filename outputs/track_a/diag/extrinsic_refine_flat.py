"""Q4 follow-up v2: extrinsic rotation (and range scale) refinement scored ONLY on flat paved cells.
Objective: robust spread of per-group flat-cell height offsets around the cell consensus (median over groups),
groups = flight lines L0-L5, climb, hover-at-altitude, descent. Terra keyframe poses fixed."""
import numpy as np, pandas as pd, json
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot
from aerodiag.cli.track_a import TrackA
from aerodiag.config import load_config
from aerodiag.grid.rasterize import grouped_quantiles
t = TrackA(load_config("configs/data.yaml"), load_config("configs/track_a.yaml")); t.stage_map(); t.stage_offset(); t.stage_grid()
C = t.C; ds = t.ds; f = ds.frames.iloc[t.fidx]
flat = (C.dominant_class.isin([10, 3, 18]) & (C.lidar_spread < 0.15) & (C.lidar_slope < 0.1)).values
keep_groups = [0, 1, 2, 3, 4, 5, 102, 103, 105, 127, 129, 130]
m = np.isin(t.pt_group, keep_groups) & (t.ids >= 0); m[m] = flat[t.ids[m]]
# camera-frame points for the selected points: recompute from world with the current calibration
fr = t.M["pt_frame"][m]; T = t.T; Xw = t.X[m].astype(float)
pc = np.einsum("nji,nj->ni", T[fr, :3, :3], Xw - T[fr, 3 - 3:3, 3] if False else Xw - T[fr, :3, 3])   # world -> camera
pl = (pc - ds.calib.t_CL) @ ds.calib.R_CL                                                            # camera -> LiDAR
grp = t.pt_group[m]; ug, gi = np.unique(grp, return_inverse=True); G = len(ug); cid = t.ids[m]; n = t.grid.n
print("points", len(pl), "groups", ug.tolist(), flush=True)
def offsets(p):
    dR = Rot.from_rotvec(np.asarray(p[:3]) * 1e-3).as_matrix(); k = p[3] * 1e-6
    pcn = (pl * (1 + k)) @ (dR @ ds.calib.R_CL).T + ds.calib.t_CL
    X = np.einsum("nij,nj->ni", T[fr, :3, :3], pcn) + T[fr, :3, 3]
    cg = cid * G + gi; c, q = grouped_quantiles(cg, X[:, 2], n * G, (0.5,)); ok = c >= 5
    cell = np.arange(n * G) // G; cc, cq = grouped_quantiles(np.where(ok, cell, -1), q[0], n, (0.5,))
    good = ok & (cc[cell] >= 2); d = q[0][good] - cq[0][cell[good]]; g = (np.arange(n * G) % G)[good]
    return d, g
def objective(p):
    d, g = offsets(p); return float(np.mean([np.abs(np.median(d[g == k])) for k in range(G) if (g == k).sum() > 30]) + 0.5 * np.median(np.abs(d)))
def table(p):
    d, g = offsets(p); return pd.DataFrame([dict(group=int(ug[k]), cells=int((g == k).sum()), dz_median=float(np.median(d[g == k])),
                                                 dz_mad=float(1.4826 * np.median(np.abs(d[g == k] - np.median(d[g == k]))))) for k in range(G) if (g == k).sum() > 30])
b0 = table(np.zeros(4)); o0 = objective(np.zeros(4)); print("before", round(o0, 4))
res = minimize(objective, np.zeros(4), method="Nelder-Mead", options=dict(xatol=0.02, fatol=1e-5, maxfev=300,
               initial_simplex=np.array([[0, 0, 0, 0], [4, 0, 0, 0], [0, 4, 0, 0], [0, 0, 4, 0], [0, 0, 0, 3000]], float)))
b1 = table(res.x); print("params mrad (x,y,z), range ppm:", res.x.round(3), "obj", round(res.fun, 4), "nfev", res.nfev)
out = b0.merge(b1, on="group", suffixes=("_before", "_after")).round(3); print(out.to_string())
out.to_csv("outputs/track_a/diag/extrinsic_refine_flat.csv", index=False)
json.dump(dict(params_mrad_ppm=res.x.tolist(), obj_before=o0, obj_after=res.fun), open("outputs/track_a/diag/extrinsic_refine_flat.json", "w"), indent=1)
