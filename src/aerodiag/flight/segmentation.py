"""Flight segmentation from camera centres. Segments: hover, climb, descent, cruise_line (with line_id),
diagonal (line_id), turn. Rules documented in configs/data.yaml:segmentation."""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy.ndimage import uniform_filter1d
from ..io.timestamps import rel_seconds

def segment_flight(t_ns: np.ndarray, C: np.ndarray, hover_speed=1.5, vertical_speed=1.0, heading_window_s=3.0,
                   heading_stable_deg=8.0, line_min_duration_s=6.0, line_min_speed=5.0, parallel_tol_deg=15.0,
                   smooth_frames=11) -> pd.DataFrame:
    t = rel_seconds(t_ns, t_ns[0]); n = len(t)
    Cs = uniform_filter1d(C, smooth_frames, axis=0, mode="nearest")
    v = np.gradient(Cs, t, axis=0); v[np.diff(t, prepend=-1.0) > 1.0] = np.nan
    v = pd.DataFrame(v).ffill().bfill().to_numpy()
    sp = np.linalg.norm(v[:, :2], axis=1); hd = np.degrees(np.arctan2(v[:, 1], v[:, 0]))
    seg = np.full(n, "turn", dtype=object)
    seg[sp < hover_speed] = "hover"
    vert = np.abs(v[:, 2]) > vertical_speed
    seg[vert] = np.where(v[vert, 2] > 0, "climb", "descent")
    uh = np.unwrap(np.radians(hd)); lo = np.searchsorted(t, t - heading_window_s); hi = np.searchsorted(t, t + heading_window_s, "right")
    stab = np.array([np.degrees(np.ptp(uh[a:b])) < heading_stable_deg for a, b in zip(lo, hi)])
    cand = (seg == "turn") & stab & (sp > line_min_speed)
    runs, i = [], 0
    while i < n:
        if cand[i]:
            j = i
            while j + 1 < n and cand[j + 1] and t[j + 1] - t[j] < 1.0:
                j += 1
            if t[j] - t[i] >= line_min_duration_s:
                runs.append((i, j))
            i = j + 1
        else:
            i += 1
    line_id = np.full(n, -1)
    rh = np.array([np.degrees(np.angle(np.mean(np.exp(1j * np.radians(hd[a:b + 1]))))) for a, b in runs])
    dd = lambda x, y: np.abs(np.mod(x - y + 90, 180) - 90)                  # axial angle difference (deg)
    # dominant line direction = the heading shared by the MOST runs (a survey grid), not by the most frames:
    # on HKisland one long transit leg outweighs four shorter survey lines by frame count
    if len(runs):
        nlen = np.array([b - a + 1 for a, b in runs])
        votes = [((dd(rh, h) < parallel_tol_deg).sum(), nlen[dd(rh, h) < parallel_tol_deg].sum()) for h in rh]
        dom = rh[max(range(len(rh)), key=lambda k: votes[k])]
    for k, (a, b) in enumerate(runs):
        dlt = dd(rh[k], dom)
        seg[a:b + 1] = "cruise_line" if dlt < parallel_tol_deg else "diagonal"; line_id[a:b + 1] = k
    return pd.DataFrame(dict(segment=seg, line_id=line_id, speed=sp, heading_deg=hd, vz=v[:, 2]))
