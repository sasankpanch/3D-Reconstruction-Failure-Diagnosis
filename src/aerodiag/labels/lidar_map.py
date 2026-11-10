"""A1: accumulate LiDAR scans into one working-frame map (GROUND TRUTH), using Terra REFERENCE keyframe poses
and the verified camera-LiDAR calibration. Dynamic objects removed first.

Per point we keep: X (working frame, float32), label, frame index, range (m), camera-frame direction
(xn = x_c/z_c, yn = y_c/z_c) and off-nadir angle (deg, vs working -Z).
Deskew (Q7): per-point times are not in the data; we apply one global time offset delta to every point,
X += v_frame * delta (v from the reference trajectory). Within-scan skew remains: uniform over 0..100 ms,
i.e. +-0.45 m along-track around the corrected position (std ~0.26 m at 9 m/s); vertical effect ~= skew * slope."""
from __future__ import annotations
from multiprocessing import Pool
import numpy as np, pandas as pd

def _load_one(args):
    lidar_path, label_path, T, R_CL, t_CL, min_range, dynamic, stride = args
    pl = pd.read_csv(lidar_path, sep=r"\s+", header=None, dtype=np.float64, engine="c").to_numpy()[::stride]
    lab = pd.read_csv(label_path, header=None, dtype=np.int32, engine="c").to_numpy()[::stride, 0].astype(np.uint8)
    rng = np.linalg.norm(pl, axis=1); keep = (rng >= min_range) & ~np.isin(lab, dynamic)
    n_dyn = int(np.isin(lab, dynamic).sum())
    pc = pl[keep] @ R_CL.T + t_CL; Xw = pc @ T[:3, :3].T + T[:3, 3]
    ray = Xw - T[:3, 3]; off = np.degrees(np.arccos(np.clip(-ray[:, 2] / np.linalg.norm(ray, axis=1), -1, 1)))
    return (Xw.astype(np.float32), lab[keep], rng[keep].astype(np.float32), (pc[:, 0] / pc[:, 2]).astype(np.float32),
            (pc[:, 1] / pc[:, 2]).astype(np.float32), off.astype(np.float32), n_dyn)

def frame_velocity(t_ns: np.ndarray, C: np.ndarray, max_gap_s: float = 1.0) -> np.ndarray:
    """Central-difference velocity (m/s) on a sparse (reference) trajectory; one-sided across gaps."""
    t = (t_ns - t_ns[0]) * 1e-9; n = len(t); v = np.zeros((n, 3))
    for i in range(n):
        a = i - 1 if i > 0 and t[i] - t[i - 1] <= max_gap_s else i
        b = i + 1 if i < n - 1 and t[i + 1] - t[i] <= max_gap_s else i
        if b > a:
            v[i] = (C[b] - C[a]) / (t[b] - t[a])
    return v

def build_map(ds, frame_idx: np.ndarray, T_wc: np.ndarray, min_range=5.0, dynamic=(20, 24), stride=1, workers=20) -> dict:
    """T_wc: poses for frame_idx (N,4,4). Returns dict of per-point arrays + per-frame arrays."""
    args = [(ds.p["lidar_dir"] / ds.lidar_name(i), ds.p["label_id_dir"] / ds.lidar_name(i), T_wc[k], ds.calib.R_CL,
             ds.calib.t_CL, min_range, list(dynamic), stride) for k, i in enumerate(frame_idx)]
    with Pool(workers) as p:
        res = p.map(_load_one, args, chunksize=4)
    n = np.array([len(r[0]) for r in res])
    out = dict(X=np.concatenate([r[0] for r in res]), label=np.concatenate([r[1] for r in res]),
               range=np.concatenate([r[2] for r in res]), xn=np.concatenate([r[3] for r in res]),
               yn=np.concatenate([r[4] for r in res]), off_nadir=np.concatenate([r[5] for r in res]),
               pt_frame=np.repeat(np.arange(len(frame_idx), dtype=np.int32), n),
               frame_idx=np.asarray(frame_idx), frame_npts=n, frame_ndynamic=np.array([r[6] for r in res]))
    return out

def apply_time_offset(X: np.ndarray, pt_frame: np.ndarray, v_frame: np.ndarray, delta_s: float) -> np.ndarray:
    return (X.astype(np.float64) + v_frame[pt_frame] * delta_s).astype(np.float32)
