"""Onboard-only geometry: everything here uses images, timestamps and RTK, never LiDAR or Terra products.

Frame of reference: local ENU (metres) at the dataset's nominal origin; heights = RTK ellipsoidal heights (assumed).
Frames are selected by TIME (every 5th 10-Hz frame = 0.5 s), flight segments are derived from RTK positions.
The only non-onboard input is the attitude used to INITIALISE rotations (stand-in for the IMU until the rosbag is
available); the bundle re-estimates every rotation from the images, and `selfcal` checks insensitivity to it."""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
from ..io.rtk import load_rtk, interpolate_rtk
from ..flight.segmentation import segment_flight
from ..matching.features import FeatureStore
from ..matching.match import match_pair_full
from ..calib.tracks import build_tracks
from ..calib.bundle import BundleProblem

def rtk_enu(ds) -> np.ndarray:
    rtk = load_rtk(ds.p["rtk_csv"], ds.terra_origin)
    return interpolate_rtk(rtk, ds.frames.t_ns.values)

def rtk_segments(ds, enu) -> pd.DataFrame:
    seg = segment_flight(ds.frames.t_ns.values, enu, **ds.cfg["segmentation"])
    t = ds.frames.t_ns.values; s = seg.segment.values
    new = np.r_[True, (s[1:] != s[:-1]) | (np.diff(t) > 1_000_000_000)]
    seg["run"] = np.cumsum(new) - 1
    seg["time_grid"] = ds.frames.orig_idx.values % ds.cfg["reference_poses"]["keyframe_modulo"] == 0
    return seg

def init_rotations_enu(ds, sim_R: np.ndarray) -> np.ndarray:
    """Stand-in IMU attitude (initial guess only): slerp of keyframe attitudes, rotated working->ENU."""
    from ..poses.terra import TerraPoseSource
    f = ds.frames; att = TerraPoseSource(f.t_ns.values, ds.T_wc_json, f.is_reference.values).reference_slerp().T
    return np.einsum("ij,fjk->fik", sim_R.T, att[:, :3, :3])

def match_pairs(ds, pairs: np.ndarray, cfg: dict, cache_dir, workers=20):
    """Match (i,j) pairs with per-pair caching. Returns (pair_frames, k1, k2, pair_of_match, info DataFrame)."""
    from multiprocessing import Pool
    cdir = Path(cache_dir) / "matches_pairs"; cdir.mkdir(parents=True, exist_ok=True)
    fs = FeatureStore(ds, cache_dir, cfg["features"]["max_features"], cfg["features"]["texture_window"], cfg["features"]["workers"])
    fs.ensure(np.unique(pairs))
    todo = [(int(i), int(j)) for i, j in pairs if not (cdir / f"{i}_{j}.npz").exists()]
    if todo:
        global _MP
        _MP = (ds, fs, cfg, cdir)
        with Pool(workers) as p:
            p.map(_match_one, todo, chunksize=4)
    k1, k2, pom, info = [], [], [], []
    for n, (i, j) in enumerate(pairs):
        z = np.load(cdir / f"{int(i)}_{int(j)}.npz"); k1.append(z["k1"]); k2.append(z["k2"]); pom.append(np.full(len(z["k1"]), n))
        info.append(dict(i=int(i), j=int(j), n=int(len(z["k1"]))))
    return np.asarray(pairs), np.concatenate(k1), np.concatenate(k2), np.concatenate(pom), pd.DataFrame(info)

_MP = None
def _match_one(ij):
    ds, fs, cfg, cdir = _MP; i, j = ij
    a = fs.load(i); b = fs.load(j)
    rng = np.random.default_rng(cfg["seed"] * 100003 + i * 7 + j)
    pr, info = match_pair_full(a, b, cfg["matching"], cfg["_homog_thr"].get((i, j), 300.0), rng)
    np.savez(cdir / f"{i}_{j}.npz", k1=pr[:, 0].astype(np.int32) if len(pr) else np.zeros(0, np.int32),
             k2=pr[:, 1].astype(np.int32) if len(pr) else np.zeros(0, np.int32))

def homography_thresholds(pairs, enu, f_px, agl, relief=15.0):
    """Loose homography tolerance per pair from RTK baseline length and height above ground (no direction used)."""
    out = {}
    for i, j in pairs:
        B = float(np.linalg.norm(enu[j] - enu[i])); H = float(max(min(agl[i], agl[j]), 20.0))
        out[(int(i), int(j))] = max(40.0, 1.5 * f_px * B * relief / H ** 2)
    return out

def bundle_from_pairs(ds, frames: np.ndarray, pairs, k1, k2, pom, enu, R0, seg_of_frame, cam, estimate=("f", "k1", "k2", "k3"),
                      estimate_lever=True, fixed_segment=None, lever0=None, min_views=3, max_tracks=60000, seed=0):
    """Build tracks over `frames` and set up the bundle. Returns (BundleProblem, frame list, track info)."""
    fsel = np.asarray(sorted(set(int(x) for x in frames))); fpos = {f: k for k, f in enumerate(fsel)}
    keep = np.isin(pairs[pom, 0], fsel) & np.isin(pairs[pom, 1], fsel)
    T = build_tracks(pairs, k1[keep], k2[keep], pom[keep], min_views=min_views)
    rng = np.random.default_rng(seed)
    if T["n_tracks"] > max_tracks:
        chosen = np.sort(rng.choice(T["n_tracks"], max_tracks, replace=False)); m = np.isin(T["track"], chosen)
        remap = -np.ones(T["n_tracks"], int); remap[chosen] = np.arange(len(chosen))
        T = dict(frame=T["frame"][m], kp=T["kp"][m], track=remap[T["track"][m]], n_tracks=len(chosen), n_dropped_inconsistent=T["n_dropped_inconsistent"])
    fs = FeatureStore(ds, Path(ds.out) / "cache")
    uv = np.zeros((len(T["frame"]), 2))
    for f in np.unique(T["frame"]):
        s = T["frame"] == f; uv[s] = fs.load(int(f))["uv"][T["kp"][s]]
    segs = seg_of_frame[fsel]; useg, segi = np.unique(segs, return_inverse=True)
    fixed = 0 if fixed_segment is None else int(np.where(useg == fixed_segment)[0][0])
    bp = BundleProblem(enu[fsel], R0[fsel], segi, np.array([fpos[int(f)] for f in T["frame"]]), T["track"], uv, cam,
                       estimate=tuple(estimate), estimate_lever=estimate_lever, fixed_segment=fixed,
                       lever0=np.zeros(3) if lever0 is None else lever0)
    return bp, fsel, dict(n_tracks=int(T["n_tracks"]), n_obs=int(len(T["frame"])), dropped_inconsistent=int(T["n_dropped_inconsistent"]),
                          segments=[int(s) for s in useg])
