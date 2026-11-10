"""Track B CLI: onboard pose sources, rotation-only refinement, pair residuals, noise floor (go/no-go gate).

aerodiag-track-b [--data configs/data.yaml] [--cfg configs/track_b.yaml] [--stages match,geometry,residuals,analysis]

Residuals: for each match (pose-independent filtering), across-epipolar and along-epipolar components in pixels
(image 2, undistorted normalised coordinates x mean focal length) under each pose source. Refined sources ('+ref')
are cross-validated (correction fitted on the other half of the pair's matches). Each match is assigned to the
ground cell hit by its image-1 ray on the LiDAR height raster (Track A), using the Terra JSON pose of frame i."""
from __future__ import annotations
import argparse, json, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd

from ..config import load_config, write_params
from ..io.dataset import Dataset
from ..io.rtk import load_rtk
from ..grid.ground_grid import GroundGrid
from ..flight.pairs import select_pairs
from ..matching.features import FeatureStore
from ..matching.match import match_pair_full
from ..geometry.epipolar import relative_pose, decompose, triangulate_world
from ..poses.base import PoseSet
from ..poses.terra import TerraPoseSource
from ..poses.rtk import RtkPoseSource, fit_rtk_to_working
from ..poses.noisy import NoisyPriorSource
from ..poses.refine import refine_cv, apply_correction

def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)

# ---------------------------------------------------------------- worker globals (fork)
_G = {}

def _match_worker(pid):
    ds, fs, cfg, P = _G["ds"], _G["fs"], _G["cfg"], _G["P"]
    i, j, B = int(P.i[pid]), int(P.j[pid]), float(P.baseline_m[pid])
    H = float(np.median(np.abs(_G["agl"][[i, j]])))
    thr = max(40.0, 1.5 * ds.camera.f * B * 15.0 / H ** 2)            # tolerance > parallax of +-15 m relief
    rng = np.random.default_rng(cfg["seed"] * 100003 + pid)
    pr, info = match_pair_full(fs.load(i), fs.load(j), cfg["matching"], thr, rng)
    info.update(pair_id=int(pid), homog_thr_px=thr)
    return pid, pr, info

def _residual_worker(pid):
    """All sources for one pair. Returns dict of arrays (per match) and pair-level info."""
    M, P, S, cfg, f = _G["M"], _G["P"], _G["S"], _G["cfg"], _G["f"]
    sel = M["pair_id"] == pid; x1 = M["x1"][sel]; x2 = M["x2"][sel]; i, j = int(P.i[pid]), int(P.j[pid])
    out, info = {}, dict(pair_id=int(pid), n=int(sel.sum()))
    rc = cfg["refine"]; rng = np.random.default_rng(cfg["seed"] * 7919 + int(pid))
    for name, ps in S.items():
        a = np.full(len(x1), np.nan, np.float32); al = a.copy()
        if ps.valid[i] and ps.valid[j] and len(x1) > 0:
            R, t = relative_pose(ps.T_wc[i], ps.T_wc[j]); d = decompose(x1, x2, R, t, f)
            a[:] = d["across_px"]; al[:] = d["along_px"]
            if name == "terra":
                out["z_terra"] = triangulate_world(x1, d["depth1"], ps.T_wc[i])[:, 2].astype(np.float32)
            if name in rc["sources"] or (name.startswith("sweep_") and "sweep" in rc["sources"]):
                ar, alr = np.full(len(x1), np.nan, np.float32), np.full(len(x1), np.nan, np.float32)
                water = max(_G["water"][i], _G["water"][j]) > rc["water_max_fraction"]
                if water:
                    reason = "water"; ar[:], alr[:] = a, al
                elif len(x1) < rc["min_matches"]:
                    reason = "few_matches"; ar[:], alr[:] = a, al
                else:
                    reason = "refined"
                    fold, ws, infos = refine_cv(x1, x2, R, t, f, folds=rc["folds"], rng=rng, loss=rc["loss"],
                                                f_scale_px=rc["f_scale_px"], max_fit=rc["max_matches_fit"])
                    for k, w in enumerate(ws):
                        Rk, tk = apply_correction(w, R, t); m = fold == k
                        dk = decompose(x1[m], x2[m], Rk, tk, f); ar[m] = dk["across_px"]; alr[m] = dk["along_px"]
                    info[f"{name}+ref_w_deg"] = float(np.mean([np.degrees(np.linalg.norm(w)) for w in ws]))
                    info[f"{name}+ref_w_fold_diff_deg"] = float(np.degrees(np.linalg.norm(ws[0] - ws[-1])))
                info[f"{name}+ref_status"] = reason
                out[f"across_{name}+ref"] = ar; out[f"along_{name}+ref"] = alr
        out[f"across_{name}"] = a; out[f"along_{name}"] = al
    return pid, out, info

class TrackB:
    def __init__(self, dcfg, cfg):
        self.dcfg, self.cfg = dcfg, cfg; self.ds = Dataset(dcfg)
        cl = dcfg.get("classes", {})                                         # scene-specific class lists override the defaults
        for k_scene, k_tb in (("healthy_class_based", "class_based"), ("vegetation", "vegetation")):
            if k_scene in cl:
                cfg["healthy"][k_tb] = list(cl[k_scene])
        self.out = Path(self.ds.out) / "track_b"; self.out.mkdir(parents=True, exist_ok=True)
        self.cache = Path(self.ds.out) / "cache"; self.results = {}
        f = self.ds.frames; self.P = select_pairs(f, self.ds.T_wc_json[:, :3, 3], cfg["pairs"])
        self.P.to_csv(self.out / "pairs_selected.csv")
        tab = pd.read_parquet(Path(self.ds.out) / "track_a" / "labels.parquet")
        self.grid = GroundGrid.from_json((Path(self.ds.out) / "track_a" / "grid.json").read_text())
        self.L = tab.set_index("cell_id")
        z = np.full(self.grid.n, np.nan); z[tab.cell_id.values] = tab.lidar_z.values; self.zgrid = z
        self.agl = self.ds.T_wc_json[:, 2, 3] - np.nanmedian(z)

    # ------------------------------------------------------------------ B3 matching
    def stage_match(self):
        P = self.P; fs = FeatureStore(self.ds, self.cache, self.cfg["features"]["max_features"], self.cfg["features"]["texture_window"],
                                      self.cfg["features"]["workers"])
        n = fs.ensure(np.unique(np.r_[P.i, P.j])); log(f"features: {n} newly extracted")
        mpath = self.cache / "track_b_matches.npz"
        if mpath.exists():
            z = np.load(mpath, allow_pickle=False); self.matches = {k: z[k] for k in z.files}
            self.match_info = pd.read_csv(self.out / "match_info.csv"); log("matches: cached"); return
        _G.update(ds=self.ds, fs=fs, cfg=self.cfg, P=P, agl=self.agl)
        with Pool(20) as p:
            res = p.map(_match_worker, P.index.tolist(), chunksize=4)
        pid = np.concatenate([np.full(len(r[1]), r[0]) for r in res]); idx = np.concatenate([r[1] for r in res if len(r[1])] or [np.zeros((0, 2), int)])
        self.matches = dict(pair_id=pid.astype(np.int32), k1=idx[:, 0].astype(np.int32), k2=idx[:, 1].astype(np.int32))
        np.savez(mpath, **self.matches)
        self.match_info = pd.DataFrame([r[2] for r in res]); self.match_info.to_csv(self.out / "match_info.csv", index=False)
        log("matches:", len(pid), "total;", self.match_info.n_kept.describe().round(0).to_dict())

    # ------------------------------------------------------------------ B4 geometry per match
    def stage_geometry(self):
        P, ds, cam = self.P, self.ds, self.ds.camera; fs = FeatureStore(ds, self.cache, self.cfg["features"]["max_features"])
        M = self.matches; n = len(M["pair_id"])
        uv1 = np.zeros((n, 2), np.float32); uv2 = uv1.copy(); tex = np.zeros(n, np.float32)
        for pid in np.unique(M["pair_id"]):
            s = M["pair_id"] == pid; a = fs.load(int(P.i[pid])); b = fs.load(int(P.j[pid]))
            uv1[s] = a["uv"][M["k1"][s]]; uv2[s] = b["uv"][M["k2"][s]]; tex[s] = np.minimum(a["tex"][M["k1"][s]], b["tex"][M["k2"][s]])
        M.update(uv1=uv1, uv2=uv2, tex=tex, x1=cam.pixel_to_normalized(uv1.astype(float)), x2=cam.pixel_to_normalized(uv2.astype(float)),
                 r1=cam.radius(uv1).astype(np.float32), r2=cam.radius(uv2).astype(np.float32))
        # cell assignment: intersect image-1 ray (Terra JSON pose of frame i) with the LiDAR height raster
        fi = P.i.values[M["pair_id"]]; T = ds.T_wc_json[fi]; b1 = np.c_[M["x1"], np.ones(n)]
        dw = np.einsum("nij,nj->ni", T[:, :3, :3], b1); c = T[:, :3, 3]; z = np.full(n, np.nanmedian(self.zgrid))
        for _ in range(8):
            lam = (z - c[:, 2]) / dw[:, 2]; xy = c[:, :2] + lam[:, None] * dw[:, :2]; cid = self.grid.cell_id(xy)
            zz = np.where(cid >= 0, self.zgrid[np.clip(cid, 0, None)], np.nan); z = np.where(np.isfinite(zz), zz, z)
        M["cell_id"] = cid.astype(np.int64)
        # vehicle flag: within dilate px of projected sedan/truck LiDAR points in frame i (at uv1) or frame j (at uv2)
        from scipy.spatial import cKDTree
        veh = np.zeros(n, bool); hc = self.cfg["healthy"]; frames = np.unique(np.r_[P.i, P.j]); water = np.zeros(len(ds.frames))
        vp = {}
        for i in frames:
            lab = ds.load_labels(i); water[i] = np.isin(lab, self.cfg["refine"]["water_classes"]).mean()
            m = np.isin(lab, hc["vehicle_classes"])
            if m.any():
                pc = ds.calib.lidar_to_camera(ds.load_lidar(i)[m]); uv, ok = cam.project(pc); vp[i] = cKDTree(uv[ok]) if ok.any() else None
        for side, key, col in [(1, "uv1", P.i.values), (2, "uv2", P.j.values)]:
            fr = col[M["pair_id"]]
            for i in np.unique(fr):
                if vp.get(i) is None:
                    continue
                s = fr == i; d, _ = vp[i].query(M[key][s], distance_upper_bound=hc["vehicle_mask_dilate_px"]); veh[np.where(s)[0][np.isfinite(d)]] = True
        M["vehicle"] = veh; self.water = water
        self.results["geometry"] = dict(n_matches=int(n), frac_with_cell=float((cid >= 0).mean()), n_vehicle_matches=int(veh.sum()),
                                        max_water_fraction=float(water.max()))
        log("geometry:", self.results["geometry"])

    # ------------------------------------------------------------------ B1 pose sources
    def build_sources(self) -> dict:
        ds, f = self.ds, self.ds.frames; t = f.t_ns.values; sc = self.cfg["sources"]; seed = self.cfg["seed"]
        terra = TerraPoseSource(t, ds.T_wc_json, f.is_reference.values, reference_only=True)
        att = terra.reference_slerp()
        rtk = load_rtk(ds.p["rtk_csv"], ds.terra_origin)
        fitmask = f.is_reference.values & ~f.segment.isin(self.dcfg["rtk"]["fit_segments_exclude"]).values
        sim, _ = fit_rtk_to_working(rtk, t, ds.T_wc_json[:, :3, 3], fitmask)
        S = {"terra": terra.get(t), "terra_json": TerraPoseSource(t, ds.T_wc_json, f.is_reference.values, reference_only=False).get(t)}
        for name, c in sc.items():
            if name.startswith("rtk"):
                S[name] = RtkPoseSource(rtk, sim, att.T, t, c["attitude_noise_deg"], seed=seed).get(t)
            elif name.startswith("noisy"):
                S[name] = NoisyPriorSource(terra, c["preset"], seed=seed).get(t)
        from ..poses.noisy import NoiseSpec
        sw = self.cfg.get("sweep", {})
        for sig in sw.get("pos_sigma_m", []):
            spec = NoiseSpec(pos_h_sigma=sig, pos_v_sigma=sig, att_sigma_deg=sw["att_sigma_deg"])
            S[f"sweep_pos{sig:g}"] = NoisyPriorSource(terra, spec, seed=seed).get(t)
        self.results["sources"] = {k: dict(name=v.name, n_valid=int(v.valid.sum()), meta=v.meta) for k, v in S.items()}
        return S

    # ------------------------------------------------------------------ B2/B4 residuals
    def stage_residuals(self):
        S = self.build_sources(); M = self.matches
        _G.update(M=M, P=self.P, S=S, cfg=self.cfg, f=self.ds.camera.f, water=self.water)
        pids = np.unique(M["pair_id"]).tolist()
        with Pool(20) as p:
            res = p.map(_residual_worker, pids, chunksize=8)
        cols = sorted({k for r in res for k in r[1]})
        order = np.argsort(M["pair_id"], kind="stable"); assert np.all(np.diff(M["pair_id"][order]) >= 0)
        R = {c: np.full(len(M["pair_id"]), np.nan, np.float32) for c in cols}
        for pid, out, _ in res:
            s = np.where(M["pair_id"] == pid)[0]
            for c, v in out.items():
                R[c][s] = v
        df = pd.DataFrame(dict(pair_id=M["pair_id"], u1=M["uv1"][:, 0], v1=M["uv1"][:, 1], u2=M["uv2"][:, 0], v2=M["uv2"][:, 1],
                               r1=M["r1"], r2=M["r2"], tex=M["tex"], cell_id=M["cell_id"], vehicle=M["vehicle"], **R))
        df.to_parquet(self.out / "residuals.parquet", index=False)
        pinfo = pd.DataFrame([r[2] for r in res]); self.pairs = self.P.join(pinfo.set_index("pair_id"), how="left").join(
            self.match_info.set_index("pair_id")[["n_raw", "n_consistent", "n_homog", "n_kept", "homog_thr_px"]], how="left")
        self.pairs.to_csv(self.out / "pairs.csv")
        st = {c: self.pairs[c].value_counts().to_dict() for c in self.pairs.columns if c.endswith("_status")}
        self.results["refine_status"] = st; log("residuals written:", len(df), "rows; refine status:", st)

    def save(self):
        write_params(self.out, "track_b", dict(data=self.dcfg, track_b=self.cfg))
        rp = self.out / "results.json"; prev = json.loads(rp.read_text()) if rp.exists() else {}; prev.update(self.results)
        rp.write_text(json.dumps(prev, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x)))

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="configs/data.yaml"); ap.add_argument("--cfg", default="configs/track_b.yaml")
    ap.add_argument("--stages", default="match,geometry,residuals,analysis")
    a = ap.parse_args(argv); st = a.stages.split(",")
    tb = TrackB(load_config(a.data), load_config(a.cfg))
    if any(s in st for s in ("match", "geometry", "residuals")):
        tb.stage_match(); tb.stage_geometry()
    if "residuals" in st:
        tb.stage_residuals()
    if "analysis" in st:
        from ..analysis.noise_floor import run_analysis
        tb.results["analysis"] = run_analysis(tb)
    tb.save()

if __name__ == "__main__":
    main()
