"""Pair selection. Each pair is tagged: pair_type (within_line / cross_line / crossing / climb / turn), gap_frames,
dt_s, heading difference (deg), baseline (m), segments and line ids, and whether both frames are reference keyframes.
Positions/headings from Terra camera centres (selection only; residuals use the pose source under test)."""
from __future__ import annotations
import numpy as np, pandas as pd

def _hd(a, b):
    return float(abs((a - b + 180) % 360 - 180))

def select_pairs(frames: pd.DataFrame, C: np.ndarray, cfg: dict) -> pd.DataFrame:
    f = frames; ref = f.is_reference.values; seg = f.segment.values; lid = f.line_id.values; t = f.t_ns.values
    hd = f.heading_deg.values; rows = []
    def add(i, j, typ, gap):
        rows.append(dict(i=int(i), j=int(j), pair_type=typ, gap_frames=int(gap), dt_s=(t[j] - t[i]) * 1e-9,
                         heading_diff_deg=_hd(hd[i], hd[j]), baseline_m=float(np.linalg.norm(C[j] - C[i])),
                         seg_i=seg[i], seg_j=seg[j], line_i=int(lid[i]), line_j=int(lid[j]), both_reference=bool(ref[i] and ref[j])))
    in_seg = np.isin(seg, cfg["within_segments"])
    for i in np.where(ref & in_seg)[0]:
        for g in cfg["within_gaps"]:
            j = i + g
            if j < len(f) and lid[j] == lid[i] and seg[j] == seg[i] and (t[j] - t[i]) < (g + 0.5) * 1e8:
                add(i, j, "within_line", g)
    # cross-line: parallel lines, nearest keyframe on another parallel line within distance
    par = np.where(ref & (seg == "cruise_line"))[0]
    for i in par[:: cfg["cross_line_stride"]]:
        for L in sorted(set(lid[par]) - {lid[i]}):
            cand = par[lid[par] == L]; d = np.linalg.norm(C[cand, :2] - C[i, :2], axis=1); k = np.argmin(d)
            if d[k] < cfg["cross_line_max_center_dist"] and lid[i] < L:
                add(i, cand[k], "cross_line", abs(cand[k] - i))
    dia = np.where(ref & (seg == "diagonal"))[0]
    for i in dia:
        for L in sorted(set(lid[par])):
            cand = par[lid[par] == L]; d = np.linalg.norm(C[cand, :2] - C[i, :2], axis=1); k = np.argmin(d)
            if d[k] < cfg["crossing_max_center_dist"]:
                add(cand[k], i, "crossing", abs(cand[k] - i))
    for s, gaps in cfg.get("extra_segments", {}).items():
        for i in np.where(ref & (seg == s))[0]:
            for g in gaps:
                j = i + g
                if j < len(f) and seg[j] == s and (t[j] - t[i]) < (g + 0.5) * 1e8:
                    add(i, j, s, g)
    df = pd.DataFrame(rows).drop_duplicates(["i", "j"]).reset_index(drop=True); df.index.name = "pair_id"
    return df
