"""Step 2 CLI: onboard poses (bundle with fixed intrinsics), coarse terrain from triangulated image points, per-cell
plane sweep and signals.  aerodiag-step2 [--data configs/data.yaml] [--cfg configs/step2.yaml] [--stages ...]

ONBOARD side (inputs: images, timestamps, RTK, self-calibrated intrinsics; attitude only as an initial guess):
  poses -> terrain -> sweep.  Frame: local ENU (m).  Cells: 1 m grid in ENU.
EVALUATION side (comparison only; LiDAR / Terra never feed back): `calib_eval` and `evaluate` map onboard results into
the working frame with the nominal RTK->working similarity, register them to the LiDAR map with a rigid ICP (as done for
Terra in Track A) and compare per cell."""
from __future__ import annotations
import argparse, json, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd

from ..config import load_config, write_params
from ..io.dataset import Dataset
from ..io.rtk import load_rtk
from ..geometry.camera import CameraModel
from ..grid.ground_grid import GroundGrid
from ..flight.pairs import select_pairs
from ..calib.onboard import rtk_enu, rtk_segments, init_rotations_enu, match_pairs, homography_thresholds, bundle_from_pairs
from ..poses.rtk import fit_rtk_to_working
from ..signals.terrain import CoarseTerrain
from ..signals.sweep import sweep_pair
from ..signals.sun import sun_vector_enu

def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)

_W = {}

def _sweep_worker(pid):
    ds, cam, poses, pairs, cells, terr, sun, cfg = (_W[k] for k in ("ds", "cam", "poses", "pairs", "cells", "terr", "sun", "cfg"))
    i, j = int(pairs.i[pid]), int(pairs.j[pid]); R1, c1 = poses[i]; R2, c2 = poses[j]
    xy, z0, nrm, cid = cells["xy"], cells["z0"], cells["n"], cells["id"]
    # cells whose centre projects inside both images (with margin)
    from ..signals.sweep import _project
    X = np.c_[xy, z0]; m = np.ones(len(xy), bool); mg = cfg["sweep"]["margin_px"]
    for R, c in ((R1, c1), (R2, c2)):
        uv = _project(cam, R, c, X); m &= (uv[:, 0] > mg) & (uv[:, 0] < cam.width - mg) & (uv[:, 1] > mg) & (uv[:, 1] < cam.height - mg)
    idx = np.where(m)[0]
    if len(idx) == 0:
        return pid, None
    img1 = ds.load_image(i, gray=True).astype(np.float32); img2 = ds.load_image(j, gray=True).astype(np.float32)
    outs = []; ch = cfg["sweep"]["chunk_cells"]
    for s0 in range(0, len(idx), ch):
        k = idx[s0:s0 + ch]
        o = sweep_pair(img1, img2, cam, R1, c1, R2, c2, xy[k], z0[k], nrm[k], (sun[i], sun[j]), cfg["sweep"])
        o["cell_id"] = cid[k]; outs.append(pd.DataFrame(o))
    df = pd.concat(outs, ignore_index=True); df["pair_id"] = pid
    return pid, df

class Step2:
    def __init__(self, dcfg, cfg):
        self.dcfg, self.cfg = dcfg, cfg; self.ds = Dataset(dcfg); ds = self.ds
        self.out = Path(ds.out) / "step2"; self.out.mkdir(parents=True, exist_ok=True); self.cache = Path(ds.out) / "cache"
        self.enu = rtk_enu(ds); self.seg = rtk_segments(ds, self.enu); self.res = {}
        rtk = load_rtk(ds.p["rtk_csv"], ds.terra_origin); self.ground = rtk.u.values[0]; self.agl = self.enu[:, 2] - self.ground
        f = ds.frames
        # nominal ENU->working similarity: used for (i) the attitude INITIAL GUESS (stand-in IMU) and (ii) evaluation only
        self.sim, _ = fit_rtk_to_working(rtk, f.t_ns.values, ds.T_wc_json[:, :3, 3], f.is_reference.values)
        self.R0 = init_rotations_enu(ds, self.sim.R)
        sc = json.loads((Path(ds.out) / "selfcal" / "intrinsics.json").read_text()); lab = ds.calib
        K = lab.K.copy(); K[0, 0] = K[1, 1] = sc["f"]
        self.cams = {"selfcal": CameraModel(K, np.array([sc["k1"], sc["k2"], sc["p1"], sc["p2"], sc["k3"]]), 2448, 2048),
                     "lab": CameraModel(lab.K, lab.dist, 2448, 2048)}
        self.lever = np.array(sc["lever_arm_camera_frame_m"])
        # pairs from RTK-derived segmentation and the time grid
        fr = pd.DataFrame(dict(t_ns=f.t_ns.values, is_reference=self.seg.time_grid.values, segment=self.seg.segment.values,
                               line_id=self.seg.line_id.values, heading_deg=self.seg.heading_deg.values))
        P = select_pairs(fr, self.enu, cfg["pairs"]); P = P[P.pair_type.isin(cfg["pairs"]["types"])].reset_index(drop=True)
        P.index.name = "pair_id"; self.P = P; P.to_csv(self.out / "pairs.csv")
        self.frames = np.unique(np.r_[P.i, P.j])

    # ------------------------------------------------------------------ onboard poses
    def stage_poses(self):
        mcfg = dict(features=self.cfg["features"], matching=self.cfg["matching"], seed=self.cfg["seed"],
                    _homog_thr=homography_thresholds(self.P[["i", "j"]].values, self.enu, 1444.0, self.agl))
        pairs = self.P[["i", "j"]].values
        Pf, k1, k2, pom, info = match_pairs(self.ds, pairs, mcfg, self.cache)
        self.match_info = info; info.to_csv(self.out / "pair_matches.csv", index=False)
        log("matches:", int(len(k1)), "pairs with < 40:", int((info.n < 40).sum()))
        runs = self.seg.run.values; line_runs = pd.Series(runs[self.frames]).value_counts()
        self.bundles = {}
        for name, cam in self.cams.items():
            path = self.out / f"poses_{name}.npz"
            if path.exists():
                self.bundles[name] = dict(np.load(path)); log("poses cached:", name); continue
            bp, fsel, tinfo = bundle_from_pairs(self.ds, self.frames, Pf, k1, k2, pom, self.enu, self.R0, runs, cam, estimate=(),
                                                estimate_lever=False, fixed_segment=int(line_runs.index[0]), lever0=self.lever,
                                                max_tracks=self.cfg["bundle"]["max_tracks"], seed=self.cfg["seed"])
            log("bundle", name, tinfo)
            hist = bp.solve(rounds=self.cfg["bundle"]["rounds"], log=lambda h: log(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in h.items()}))
            R, c = bp.poses(); e = bp.residuals_all(); m = bp.valid
            # per-observation radius for calibration evaluation
            r = np.hypot(bp.obs_uv[:, 0] - cam.K[0, 2], bp.obs_uv[:, 1] - cam.K[1, 2])
            nview = np.bincount(bp.obs_p[m], minlength=bp.N)
            rmean = np.bincount(bp.obs_p[m], r[m], minlength=bp.N) / np.maximum(nview, 1)
            d = dict(frames=fsel, R=R, c=c, X=bp.X, point_ok=bp.point_ok & (nview >= 3), point_rmean=rmean, nview=nview,
                     obs_err=np.linalg.norm(e, axis=1)[m], obs_r=r[m], offsets=bp.o, hist_rms=np.array([h["rms_px"] for h in hist]))
            np.savez(path, **d); self.bundles[name] = d
        self.res["poses"] = {n: dict(frames=int(len(b["frames"])), points=int(b["point_ok"].sum()), final_rms_px=float(b["hist_rms"][-1]),
                                     run_offsets_m=np.asarray(b["offsets"]).round(3).tolist()) for n, b in self.bundles.items()}
        log("poses:", self.res["poses"])

    # ------------------------------------------------------------------ evaluation of the calibration (held-out lines)
    def stage_calib_eval(self):
        rows = []; bins = [0, 400, 800, 1100, 1400, 1700]
        L = pd.read_parquet(Path(self.ds.out) / "track_a" / "labels.parquet"); grid = GroundGrid.from_json((Path(self.ds.out) / "track_a" / "grid.json").read_text())
        zl = np.full(grid.n, np.nan); zl[L.cell_id.values] = L.lidar_z.values; good = np.zeros(grid.n, bool)
        good[L.cell_id.values] = (L.dominant_class.isin([10, 3, 18]).values & (L.lidar_spread < 0.1).values & (L.lidar_slope < 0.1).values)
        for name, b in self.bundles.items():
            e, r = b["obs_err"], b["obs_r"]
            for a_, b_ in zip(bins[:-1], bins[1:]):
                s = (r >= a_) & (r < b_); rows.append(dict(intrinsics=name, metric="reprojection_px (images only)", r_bin=f"{a_}-{b_}",
                                                           median=float(np.median(e[s])), p90=float(np.percentile(e[s], 90)), n=int(s.sum())))
            # comparison only: sparse points vs LiDAR on flat paved cells, after mapping + vertical offset removal
            X = self.sim.apply(b["X"][b["point_ok"]]); rm = b["point_rmean"][b["point_ok"]]
            cid = grid.cell_id(X[:, :2]); ok = (cid >= 0) & good[np.clip(cid, 0, None)]
            dz = X[ok, 2] - zl[cid[ok]]; dz -= np.median(dz)
            for a_, b_ in zip(bins[:-1], bins[1:]):
                s = (rm[ok] >= a_) & (rm[ok] < b_)
                if s.sum() > 30:
                    rows.append(dict(intrinsics=name, metric="height_vs_lidar_m (comparison; offset removed)", r_bin=f"{a_}-{b_}",
                                     median=float(np.median(dz[s])), p90=float(np.percentile(np.abs(dz[s]), 90)), n=int(s.sum())))
        T = pd.DataFrame(rows); T.to_csv(self.out / "calibration_eval_heldout.csv", index=False); self.res["calib_eval"] = T.round(4).to_dict("records")
        log("calibration evaluation:\n" + T.round(3).to_string())

    # ------------------------------------------------------------------ coarse terrain + cells
    def stage_terrain(self):
        b = self.bundles[self.cfg["intrinsics"]]; X = b["X"][b["point_ok"]]; tc = self.cfg["terrain"]
        self.terr = CoarseTerrain(X, tc["coarse_cell_m"], tc["min_points"], tc["smooth_cells"])
        g = GroundGrid.from_bounds(*X[:, :2].min(0), *X[:, :2].max(0), self.cfg["grid"]["cell_m"]); self.grid = g
        xy = g.centers(); keep = self.terr.data_distance(xy) <= tc["max_extrapolation_m"]
        self.cells = dict(id=np.where(keep)[0], xy=xy[keep], z0=self.terr.height(xy[keep]), n=self.terr.normal(xy[keep]))
        (self.out / "grid_enu.json").write_text(g.to_json())
        np.savez(self.out / "coarse_terrain.npz", cell_id=self.cells["id"], xy=self.cells["xy"], z0=self.cells["z0"], normal=self.cells["n"], points=X)
        self.res["terrain"] = dict(points=int(len(X)), cells=int(len(self.cells["id"])), grid=json.loads(g.to_json()))
        log("terrain:", self.res["terrain"])

    # ------------------------------------------------------------------ sweep + signals
    def stage_sweep(self):
        b = self.bundles[self.cfg["intrinsics"]]; poses = {int(f): (b["R"][k], b["c"][k]) for k, f in enumerate(b["frames"])}
        lat0, lon0, _ = self.ds.terra_origin                           # nominal site coordinates (sun position only)
        sun = {int(f): v for f, v in zip(self.frames, sun_vector_enu(self.ds.frames.t_ns.values[self.frames], lat0, lon0))}
        P = self.P[self.P.i.isin(poses) & self.P.j.isin(poses)]
        mi = self.match_info.set_index(["i", "j"]).n
        P = P[[mi.get((int(i), int(j)), 0) >= self.cfg["matching"]["min_matches"] for i, j in zip(P.i, P.j)]]
        _W.update(ds=self.ds, cam=self.cams[self.cfg["intrinsics"]], poses=poses, pairs=P, cells=self.cells, terr=None, sun=sun, cfg=self.cfg)
        log(f"sweeping {len(P)} pairs over {len(self.cells['id'])} candidate cells")
        parts = []
        with Pool(self.cfg["sweep"]["workers"]) as p:
            for k, (pid, df) in enumerate(p.imap_unordered(_sweep_worker, P.index.tolist(), chunksize=1)):
                if df is not None:
                    parts.append(df)
                if k % 100 == 0:
                    log(f"  {k}/{len(P)} pairs")
        S = pd.concat(parts, ignore_index=True)
        S = S.join(P[["pair_type", "gap_frames", "dt_s", "heading_diff_deg", "baseline_m"]], on="pair_id")
        S.to_parquet(self.out / "cell_pair_signals.parquet", index=False)
        self.res["sweep"] = dict(pairs=int(S.pair_id.nunique()), rows=int(len(S)), valid_rows=int(S.cost_min.notna().sum()),
                                 cells=int(S.cell_id.nunique()))
        log("sweep:", self.res["sweep"])

    def save(self):
        write_params(self.out, "step2", dict(data=self.dcfg, step2=self.cfg))
        rp = self.out / "results.json"; prev = json.loads(rp.read_text()) if rp.exists() else {}; prev.update(self.res)
        rp.write_text(json.dumps(prev, indent=2, default=float))

STAGES = ["poses", "calib_eval", "terrain", "sweep", "evaluate"]

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="configs/data.yaml"); ap.add_argument("--cfg", default="configs/step2.yaml")
    ap.add_argument("--stages", default=",".join(STAGES)); a = ap.parse_args(argv); st = a.stages.split(",")
    s = Step2(load_config(a.data), load_config(a.cfg))
    s.stage_poses()
    if "calib_eval" in st:
        s.stage_calib_eval()
    if any(x in st for x in ("terrain", "sweep", "evaluate")):
        s.stage_terrain()
    if "sweep" in st:
        s.stage_sweep()
    if "evaluate" in st:
        from ..analysis.step2_bundle_eval import evaluate
        s.res["evaluate"] = evaluate(s)
    s.save()

if __name__ == "__main__":
    main()
