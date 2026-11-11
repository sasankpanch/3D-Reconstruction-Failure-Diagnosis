"""Step 2 CLI (fast path): per-cell plane sweep along epipolar lines and the five step-2 signals.

aerodiag-step2 [--data configs/data.yaml] [--cfg configs/step2.yaml] [--stages poses,terrain,sweep,evaluate]

ONBOARD inputs: images, Track B matches, the 'rtk' pose source (RTK positions mapped to the working frame; attitude =
stand-in IMU + 0.1 deg noise) with per-pair rotation-only refinement fitted on ALL matches of the pair, lab intrinsics.
Coarse terrain: triangulated from the same matches with the refined pair poses (no LiDAR, no Terra surfaces).
Cells: the Track A grid geometry (1 m, working frame) - geometry only; labels are joined in `evaluate` (comparison).
Known compromises (time-boxed): attitude of camera 1 comes from the stand-in IMU; intrinsics = lab calibration
(self-calibration postponed); the RTK->working similarity places onboard results in the comparison frame."""
from __future__ import annotations
import argparse, json, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd

from ..config import load_config, write_params
from ..io.dataset import Dataset
from ..grid.ground_grid import GroundGrid
from ..matching.features import FeatureStore
from ..geometry.epipolar import relative_pose, decompose, triangulate_world
from ..poses.refine import refine_rotation
from ..signals.terrain import CoarseTerrain
from ..signals.sweep import sweep_pair, _project
from ..signals.sun import sun_vector_enu
from scipy.spatial.transform import Rotation as Rot

def cuda_available() -> bool:
    """True if PyTorch with a CUDA device is installed; checked in a subprocess so the parent never initialises CUDA
    (forked sweep workers create their own CUDA contexts)."""
    import subprocess, sys
    try:
        return subprocess.run([sys.executable, "-c", "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"],
                              capture_output=True, timeout=120).returncode == 0
    except Exception:
        return False

def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)

_W = {}

def _refine_worker(pid):
    M, P, T, cam, cfg = _W["M"], _W["P"], _W["T"], _W["cam"], _W["cfg"]
    i, j = int(P.i[pid]), int(P.j[pid]); s = M["pair_id"] == pid
    if s.sum() < cfg["matching"]["min_matches"]:
        return pid, None, None
    x1 = cam.pixel_to_normalized(M["uv1"][s]); x2 = cam.pixel_to_normalized(M["uv2"][s])
    R, t = relative_pose(T[i], T[j]); rng = np.random.default_rng(cfg["seed"] + pid)
    w, info = refine_rotation(x1, x2, R, t, cam.f, rng=rng)
    R2 = T[j][:3, :3] @ Rot.from_rotvec(w).as_matrix()                 # R_wc2' = R_wc2 Exp(w)
    Tj = T[j].copy(); Tj[:3, :3] = R2
    pts = None
    if P.pair_type[pid] == "within_line" and P.gap_frames[pid] >= cfg["terrain"]["min_gap_frames"]:
        R2r, t2r = relative_pose(T[i], Tj); d = decompose(x1, x2, R2r, t2r, cam.f)
        ok = (d["depth1"] > 10) & (d["depth1"] < 300) & (np.abs(d["across_px"]) < 2.0)
        pts = triangulate_world(x1[ok], d["depth1"][ok], T[i])
    return pid, (Tj, info["w_deg"]), pts

def _sweep_worker(pid):
    np.seterr(all="ignore"); import warnings; warnings.filterwarnings("ignore")
    ds, cam, Tp, P, cells, sun, cfg = (_W[k] for k in ("ds", "cam", "Tp", "P", "cells", "sun", "cfg"))
    i, j = int(P.i[pid]), int(P.j[pid]); T1 = _W["T"][i]; T2 = Tp[pid]
    X = np.c_[cells["xy"], cells["z0"]]; m = np.ones(len(X), bool); mg = cfg["sweep"]["margin_px"]
    for T in (T1, T2):
        uv = _project(cam, T[:3, :3], T[:3, 3], X); m &= (uv[:, 0] > mg) & (uv[:, 0] < cam.width - mg) & (uv[:, 1] > mg) & (uv[:, 1] < cam.height - mg)
    idx = np.where(m)[0][:: cfg["sweep"]["cell_stride"]]
    if len(idx) == 0:
        return pid, None
    img1 = ds.load_image(i, gray=True).astype(np.float32); img2 = ds.load_image(j, gray=True).astype(np.float32)
    gpu = None
    if _W.get("use_cuda"):
        from ..signals.sweep_gpu import GpuSweep
        gpu = GpuSweep(cam, img1, img2, precision=cfg["sweep"].get("cuda_precision", "float32"))
    outs = []; ch = cfg["sweep"].get("cuda_chunk_cells", 4000) if gpu is not None else cfg["sweep"]["chunk_cells"]
    for s0 in range(0, len(idx), ch):
        k = idx[s0:s0 + ch]
        o = sweep_pair(img1, img2, cam, T1[:3, :3], T1[:3, 3], T2[:3, :3], T2[:3, 3], cells["xy"][k], cells["z0"][k], cells["n"][k],
                       (sun[i], sun[j]), cfg["sweep"], gpu=gpu)
        o["cell_id"] = cells["id"][k]; outs.append(pd.DataFrame(o))
    df = pd.concat(outs, ignore_index=True); df["pair_id"] = np.int32(pid)
    return pid, df

class Step2:
    def __init__(self, dcfg, cfg):
        self.dcfg, self.cfg = dcfg, cfg; self.ds = Dataset(dcfg); ds = self.ds
        self.out = Path(ds.out) / "step2"; self.out.mkdir(parents=True, exist_ok=True); self.res = {}
        self.cam = ds.camera                                                # lab calibration (configs/data.yaml camera.model)
        tb = Path(ds.out) / "track_b"
        P = pd.read_csv(tb / "pairs_selected.csv", index_col=0); P = P[P.pair_type.isin(cfg["pairs"]["types"])]
        wl = P.pair_type == "within_line"; base_rank = P.i.rank(method="dense").astype(int)
        P = P[~wl | (base_rank % cfg["pairs"]["within_base_stride"] == 0)]
        self.P = P; P.to_csv(self.out / "pairs.csv")
        from .track_b import TrackB
        tbo = TrackB.__new__(TrackB); tbo.ds = ds; tbo.cfg = load_config(Path(dcfg["root"]) / "configs" / "track_b.yaml"); tbo.dcfg = dcfg; tbo.results = {}
        self.T = tbo.build_sources()["rtk"].T_wc; self.sim_R = None
        lo = dcfg.get("rtk_line_offsets")                                   # onboard per-line RTK correction (calib/line_offsets.py)
        if lo:
            off = np.load(Path(dcfg["root"]) / lo); self.T = self.T.copy(); self.T[:, :3, 3] += off
            log("rtk line offsets applied:", lo, "max |offset| %.2f m" % np.abs(off).max())
        from ..io.rtk import load_rtk
        from ..poses.rtk import fit_rtk_to_working
        rtk = load_rtk(ds.p["rtk_csv"], ds.terra_origin); f = ds.frames
        fitmask = f.is_reference.values & ~f.segment.isin(dcfg["rtk"]["fit_segments_exclude"]).values
        self.sim, _ = fit_rtk_to_working(rtk, f.t_ns.values, ds.T_wc_json[:, :3, 3], fitmask)

    def stage_poses(self):
        z = np.load(Path(self.ds.out) / "cache" / "track_b_matches.npz"); fs = FeatureStore(self.ds, Path(self.ds.out) / "cache")
        keep = np.isin(z["pair_id"], self.P.index.values); pid = z["pair_id"][keep]; k1 = z["k1"][keep]; k2 = z["k2"][keep]
        uv1 = np.zeros((len(pid), 2), np.float32); uv2 = uv1.copy()
        for p in np.unique(pid):
            s = pid == p; uv1[s] = fs.load(int(self.P.i[p]))["uv"][k1[s]]; uv2[s] = fs.load(int(self.P.j[p]))["uv"][k2[s]]
        _W.update(M=dict(pair_id=pid, uv1=uv1.astype(float), uv2=uv2.astype(float)), P=self.P, T=self.T, cam=self.cam, cfg=self.cfg)
        from ..backend import use_cuda
        if use_cuda():
            res = self._refine_cuda(pid, uv1.astype(float), uv2.astype(float))
        else:
            with Pool(20) as p:
                res = p.map(_refine_worker, self.P.index.tolist(), chunksize=8)
        self.Tp = {r[0]: r[1][0] for r in res if r[1] is not None}
        self.pts = np.concatenate([r[2] for r in res if r[2] is not None])
        w = pd.Series({r[0]: r[1][1] for r in res if r[1] is not None})
        self.res["poses"] = dict(pairs=int(len(self.P)), refined=int(len(self.Tp)), dropped_few_matches=int(len(self.P) - len(self.Tp)),
                                 correction_deg_median=float(w.median()), terrain_points=int(len(self.pts)))
        log("poses:", self.res["poses"])

    def _refine_cuda(self, pid, uv1, uv2):
        """GPU path of _refine_worker for all pairs at once (batched LM, same rng draws); terrain points as before."""
        from ..poses.refine_cuda import refine_rotation_batch, pixel_to_normalized
        cam, cfg, P, T = self.cam, self.cfg, self.P, self.T; x1a = pixel_to_normalized(cam, uv1); x2a = pixel_to_normalized(cam, uv2)
        order = np.argsort(pid, kind="stable"); ps, starts = np.unique(pid[order], return_index=True); ends = np.r_[starts[1:], len(order)]
        rows = {int(p): order[a:b] for p, a, b in zip(ps, starts, ends)}; items, keep = [], []
        for p in P.index.tolist():
            r = rows.get(int(p), np.zeros(0, int))
            if len(r) >= cfg["matching"]["min_matches"]:
                i, j = int(P.i[p]), int(P.j[p]); R, t = relative_pose(T[i], T[j]); items.append((x1a[r], x2a[r], R, t)); keep.append(p)
        t0 = time.time(); sol = refine_rotation_batch(items, cam.f, rngs=[np.random.default_rng(cfg["seed"] + p) for p in keep])
        log(f"poses: {len(keep)} pairs refined on cuda in {time.time() - t0:.1f}s")
        res = [(p, None, None) for p in P.index if p not in set(keep)]
        for (x1, x2, _, _), p, (w, info) in zip(items, keep, sol):
            i, j = int(P.i[p]), int(P.j[p]); Tj = T[j].copy(); Tj[:3, :3] = T[j][:3, :3] @ Rot.from_rotvec(w).as_matrix(); pts = None
            if P.pair_type[p] == "within_line" and P.gap_frames[p] >= cfg["terrain"]["min_gap_frames"]:
                R2r, t2r = relative_pose(T[i], Tj); d = decompose(x1, x2, R2r, t2r, cam.f)
                ok = (d["depth1"] > 10) & (d["depth1"] < 300) & (np.abs(d["across_px"]) < 2.0); pts = triangulate_world(x1[ok], d["depth1"][ok], T[i])
            res.append((p, (Tj, info["w_deg"]), pts))
        return res

    def stage_terrain(self):
        tc = self.cfg["terrain"]; self.terr = CoarseTerrain(self.pts, tc["coarse_cell_m"], tc["min_points"], tc["smooth_cells"])
        g = GroundGrid.from_json((Path(self.ds.out) / "track_a" / "grid.json").read_text()); self.grid = g
        xy = g.centers(); keep = self.terr.data_distance(xy) <= tc["max_extrapolation_m"]
        self.cells = dict(id=np.where(keep)[0], xy=xy[keep], z0=self.terr.height(xy[keep]), n=self.terr.normal(xy[keep]))
        np.savez(self.out / "coarse_terrain.npz", cell_id=self.cells["id"], z0=self.cells["z0"], normal=self.cells["n"], points=self.pts)
        self.res["terrain"] = dict(points=int(len(self.pts)), cells=int(len(self.cells["id"])))
        log("terrain:", self.res["terrain"])

    def stage_sweep(self):
        from ..config import config_hash
        import hashlib
        pose_sig = hashlib.sha1(np.round(self.T[:, :3, 3], 4).tobytes()).hexdigest()[:12]      # poses enter the cache key
        sweep_cfg = {k: v for k, v in self.cfg["sweep"].items() if k not in ("workers", "backend", "cuda_workers", "cuda_chunk_cells", "chunk_cells", "cuda_precision")}
        key = config_hash(dict(sweep=sweep_cfg, pairs=self.cfg["pairs"], terrain=self.cfg["terrain"], n_pairs=len(self.Tp), poses=pose_sig))
        done = self.out / "cell_pair_signals.parquet"; tag = self.out / "cell_pair_signals.hash"
        if done.exists() and tag.exists() and tag.read_text() == key:
            log("sweep: cached"); return
        lat0, lon0, _ = self.ds.terra_origin; fr = np.unique(np.r_[self.P.i, self.P.j])
        s_enu = sun_vector_enu(self.ds.frames.t_ns.values[fr], lat0, lon0)
        s_w = s_enu @ self.sim.R.T                                          # ENU -> working frame (rotation of the nominal similarity)
        sun = {int(f): v for f, v in zip(fr, s_w)}
        P = self.P.loc[list(self.Tp.keys())]
        from ..backend import use_cuda as _uc
        use_cuda = _uc() if self.cfg["sweep"].get("backend", "auto") == "auto" else self.cfg["sweep"]["backend"] == "cuda"
        workers = self.cfg["sweep"].get("cuda_workers", 4) if use_cuda else self.cfg["sweep"]["workers"]
        _W.update(ds=self.ds, cam=self.cam, Tp=self.Tp, P=P, cells=self.cells, sun=sun, cfg=self.cfg, use_cuda=use_cuda)
        self.res["sweep_backend"] = dict(backend="cuda" if use_cuda else "cpu", workers=workers)
        log(f"sweeping {len(P)} pairs over {len(self.cells['id'])} candidate cells ({'cuda' if use_cuda else 'cpu'}, {workers} workers)")
        import pyarrow as pa, pyarrow.parquet as pq
        tmp = done.with_suffix(".parquet.tmp"); writer = None; t0 = time.time(); stats = dict(rows=0, valid=0); cells_seen = set(); pairs_seen = set()
        Pj = P[["pair_type", "gap_frames", "dt_s", "heading_diff_deg", "baseline_m", "i", "j"]]
        from multiprocessing.pool import ThreadPool                     # CUDA: threads share this process's GPU context
        with (ThreadPool(workers) if use_cuda else Pool(workers)) as p:
            for k, (pid, df) in enumerate(p.imap_unordered(_sweep_worker, P.index.tolist(), chunksize=1)):
                if df is not None:                                       # stream to disk (bounded memory on large flights)
                    df = df.join(Pj, on="pair_id"); tbl = pa.Table.from_pandas(df, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(tmp, tbl.schema)
                    writer.write_table(tbl.cast(writer.schema)); stats["rows"] += len(df); stats["valid"] += int(df.cost_min.notna().sum())
                    cells_seen.update(np.unique(df.cell_id).tolist()); pairs_seen.add(int(pid))
                if k % 50 == 0:
                    log(f"  {k}/{len(P)} pairs ({time.time() - t0:.0f}s)")
        writer.close(); tmp.replace(done); tag.write_text(key); self.res["sweep_seconds"] = float(time.time() - t0)
        self.res["sweep"] = dict(pairs=len(pairs_seen), rows=stats["rows"], valid_rows=stats["valid"], cells=len(cells_seen))
        log("sweep:", self.res["sweep"])

    def save(self):
        write_params(self.out, "step2", dict(data=self.dcfg, step2=self.cfg))
        rp = self.out / "results.json"; prev = json.loads(rp.read_text()) if rp.exists() else {}; prev.update(self.res)
        rp.write_text(json.dumps(prev, indent=2, default=float))

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="configs/data.yaml"); ap.add_argument("--cfg", default="configs/step2.yaml")
    ap.add_argument("--stages", default="poses,terrain,sweep,evaluate"); a = ap.parse_args(argv); st = a.stages.split(",")
    s = Step2(load_config(a.data), load_config(a.cfg))
    s.stage_poses(); s.stage_terrain()
    if "sweep" in st:
        s.stage_sweep()
    if "evaluate" in st:
        from ..analysis.step2_eval import evaluate
        s.res["evaluate"] = evaluate(s)
    s.save()

if __name__ == "__main__":
    main()
