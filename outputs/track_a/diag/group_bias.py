"""Which viewing groups (flight lines / segment runs) disagree with the balanced LiDAR Z on flat paved cells?"""
import numpy as np, pandas as pd
from aerodiag.cli.track_a import TrackA
from aerodiag.config import load_config
from aerodiag.grid.rasterize import grouped_quantiles
t = TrackA(load_config("configs/data.yaml"), load_config("configs/track_a.yaml"))
t.stage_map(); t.stage_offset(); t.stage_grid()
C = t.C; X = t.X.astype(float); ids = t.ids
flat = (C.dominant_class.isin([10, 3, 18]) & (C.lidar_spread < 0.15) & (C.lidar_slope < 0.1)).values
f = t.ds.frames.iloc[t.fidx]; rows = []
for g in np.unique(t.pt_group):
    m = (t.pt_group == g) & (ids >= 0); m[m] = flat[ids[m]]
    if m.sum() < 500: continue
    c, q = grouped_quantiles(ids[m], X[m, 2], t.grid.n, (0.5,)); ok = c >= 5
    d = q[0][ok] - C.lidar_z.values[ok]; fr = t.frame_group == g
    rows.append(dict(group=g, segment=f.segment.values[fr][0], n_frames=int(fr.sum()), first=int(t.fidx[fr][0]), last=int(t.fidx[fr][-1]),
                     cells=int(ok.sum()), dz_median=np.median(d), dz_mad=1.4826 * np.median(np.abs(d - np.median(d))),
                     range_median=float(np.median(t.M["range"][m])), off_nadir_median=float(np.median(t.M["off_nadir"][m]))))
df = pd.DataFrame(rows).round(3); print(df.to_string())
df.to_csv("outputs/track_a/diag/group_bias.csv", index=False)
