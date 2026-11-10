"""Track A CLI: ground-truth error labels.  aerodiag-track-a [--data configs/data.yaml] [--cfg configs/track_a.yaml]
                                                           [--stages map,offset,sanity,grid,checks,terra,align,outputs]

Conventions: working frame (Terra local), metres. Label: error = Terra_Z_aligned - LiDAR_Z per cell
(positive = Terra above LiDAR). LiDAR = ground truth; Terra = reference reconstruction (never ground truth).
Intermediates cached under outputs/cache; every parameter logged to outputs/track_a/*.params.json."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd

from ..config import load_config, write_params, config_hash
from ..cache import Cache
from ..io.dataset import Dataset
from ..io.labels import CMAP, CRGB, name as cname
from ..io.terra import load_cloud, load_mesh
from ..grid.ground_grid import GroundGrid
from ..grid.rasterize import grouped_quantiles, grouped_mad, grouped_mode, grouped_nunique
from ..labels.lidar_map import build_map, frame_velocity, apply_time_offset
from ..labels.checks import estimate_time_offset, pairwise_line_alignment, lidar_geometry_regression
from ..labels.terra_raster import rasterize_points, mesh_top_z, mesh_dz_points
from ..labels.align import robust_icp, describe_rigid

def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)

class TrackA:
    def __init__(self, data_cfg, cfg):
        self.dcfg, self.cfg = data_cfg, cfg
        self.ds = Dataset(data_cfg); self.out = Path(self.ds.out) / "track_a"; self.out.mkdir(parents=True, exist_ok=True)
        self.cache = Cache(Path(self.ds.out) / "cache"); self.results = {}
        f = self.ds.frames; mc = cfg["map"]
        sel = f.is_reference & f.segment.isin(mc["segments"])
        self.fidx = np.where(sel)[0]
        self.T = self.ds.T_wc_json[self.fidx]
        self.v = frame_velocity(f.t_ns.values[self.fidx], self.T[:, :3, 3])

    # ------------------------------------------------------------------ A1
    def stage_map(self):
        mc = self.cfg["map"]; key = dict(stage="map", frames=self.fidx.tolist(), mc=mc, calib=self.ds.calib.entry_name)
        def compute():
            log(f"building LiDAR map from {len(self.fidx)} reference frames")
            return build_map(self.ds, self.fidx, self.T, mc["min_range"], self.dcfg["classes"]["dynamic"], mc["point_stride"])
        self.M = self.cache.get_or_compute("lidar_map", key, compute)
        # drop low-altitude frames (median LiDAR range below threshold): consistent -0.4 m offset at 33 m AGL (diag/group_bias.csv)
        _, fq = grouped_quantiles(self.M["pt_frame"].astype(np.int64), self.M["range"].astype(np.float64), len(self.fidx), (0.5,))
        keepf = fq[0] >= mc["min_frame_median_range"]
        if not keepf.all():
            newidx = np.cumsum(keepf) - 1; kp = keepf[self.M["pt_frame"]]
            M = {k: v[kp] for k, v in self.M.items() if k in ("X", "label", "range", "xn", "yn", "off_nadir", "pt_frame")}
            M["pt_frame"] = newidx[M["pt_frame"]].astype(np.int32)
            for k in ("frame_idx", "frame_npts", "frame_ndynamic"):
                M[k] = self.M[k][keepf]
            self.M = M; self.fidx = self.fidx[keepf]; self.T = self.T[keepf]; self.v = self.v[keepf]
        self.results["map_excluded_low_altitude_frames"] = int((~keepf).sum())
        n_dyn = int(self.M["frame_ndynamic"].sum())
        self.results["map"] = dict(n_frames=len(self.fidx), n_points=int(len(self.M["X"])), n_dynamic_removed=n_dyn,
                                   frames_by_segment=self.ds.frames.segment.iloc[self.fidx].value_counts().to_dict())
        f = self.ds.frames.iloc[self.fidx]
        self.pt_seg = f.segment.values[self.M["pt_frame"]]
        lid = np.where(f.segment.isin(["cruise_line", "diagonal"]), f.line_id.values, -1)
        self.pt_line = lid[self.M["pt_frame"]]
        # viewing groups: one per flight line; other segments split into contiguous runs (gap > 1 s or segment change)
        t = f.t_ns.values; segv = f.segment.values; new = np.r_[True, (segv[1:] != segv[:-1]) | (np.diff(t) > 1_000_000_000)]
        run = np.cumsum(new) - 1; grp = np.where(lid >= 0, lid, 100 + run)
        self.frame_group = grp; self.pt_group = grp[self.M["pt_frame"]]
        log("map:", self.results["map"], "viewing groups:", len(np.unique(grp)))

    # ------------------------------------------------------------------ Q7 time offset
    def stage_offset(self):
        mc = self.cfg["map"]; f = self.ds.frames.iloc[self.fidx]
        if mc["time_offset_s"] != "auto":
            self.delta = float(mc["time_offset_s"]); self.results["time_offset"] = dict(delta_s=self.delta, source="config")
        else:
            key = dict(stage="offset", frames=self.fidx.tolist(), search=mc["time_offset_search_ms"], cls=mc["time_offset_classes"])
            r = self.cache.get("time_offset", key)
            if r is None:
                hd = f.heading_deg.values[self.M["pt_frame"]]; cr = np.isin(self.pt_seg, ["cruise_line"])
                manmade = np.isin(self.M["label"], mc["time_offset_classes"])
                east = cr & manmade & (np.cos(np.radians(hd)) > 0.9); west = cr & manmade & (np.cos(np.radians(hd)) < -0.9)
                a, b, s = mc["time_offset_search_ms"]; deltas = np.arange(a, b + 1e-9, s) / 1000
                log(f"estimating global LiDAR time offset over {len(deltas)} candidates")
                best, df = estimate_time_offset(self.M["X"], self.M["pt_frame"], self.v, east, west, deltas)
                r = dict(best=np.array(best), table=df.to_numpy()); self.cache.put("time_offset", key, r)
            df = pd.DataFrame(r["table"], columns=["delta_s", "n_cells", "median_abs_dz"]); self.delta = float(r["best"])
            df.to_csv(self.out / "time_offset_search.csv", index=False)
            self.results["time_offset"] = dict(delta_s=self.delta, source="estimated (east- vs westbound lines)",
                                               median_abs_dz_at_0=float(df.median_abs_dz[np.argmin(np.abs(df.delta_s))]),
                                               median_abs_dz_at_best=float(df.median_abs_dz.min()))
        self.X = apply_time_offset(self.M["X"], self.M["pt_frame"], self.v, self.delta)
        # per-frame LiDAR-vs-image timestamp difference (0 on the HK sequences, ~-18 ms on AMtown): the pose is the
        # image pose, so shift each scan by v * (t_lidar - t_image)
        dtl = self._lidar_dt(self.fidx)
        if np.any(dtl != 0):
            self.X = apply_time_offset(self.X, self.M["pt_frame"], self.v * dtl[:, None], 1.0)
        self.results["lidar_image_dt_ms"] = dict(median=float(np.median(dtl) * 1e3), max_abs=float(np.abs(dtl).max() * 1e3))
        log("time offset:", self.results["time_offset"], "lidar-image dt (ms):", self.results["lidar_image_dt_ms"])

    def _lidar_dt(self, fr):
        t = self.ds.frames.t_ns.values
        return np.array([(self.ds.lidar_t_ns(int(i)) - int(t[i])) * 1e-9 for i in fr])

    # ------------------------------------------------------------------ A1 sanity
    def stage_sanity(self):
        import open3d as o3d
        f = self.ds.frames; n = self.cfg["sanity"]["n_consecutive"]
        # window centred on the cruise reference frame with the most paved (airstrip/road/car park) LiDAR points
        paved = np.bincount(self.M["pt_frame"], weights=np.isin(self.M["label"], self.dcfg["classes"]["paved"]), minlength=len(self.fidx))
        cr = f.segment.values[self.fidx] == "cruise_line"; best = self.fidx[np.argmax(np.where(cr, paved, -1))]
        line = f[f.segment.eq("cruise_line") & f.line_id.eq(f.line_id.iat[best])].index.values
        mid = int(np.clip(best, line[0] + n // 2, line[-1] - n // 2))
        all_frames = np.arange(mid - n // 2, mid + n // 2)                               # every frame (incl. interpolated poses)
        ref_line = line[f.is_reference.values[line]]; k = np.searchsorted(ref_line, mid)
        ref_frames = ref_line[max(0, k - n // 2): k + n // 2]                                   # 50 consecutive keyframes
        mesh = load_mesh(self.ds.p["terra_dir"]); rows = {}; rng = np.random.default_rng(self.cfg["seed"])
        for nm, fr in [("50_consecutive_frames_json_poses", all_frames), ("50_consecutive_keyframes", ref_frames)]:
            M = build_map(self.ds, fr, self.ds.T_wc_json[fr], self.cfg["map"]["min_range"], self.dcfg["classes"]["dynamic"])
            v = frame_velocity(f.t_ns.values[fr], self.ds.T_wc_json[fr][:, :3, 3]) if nm.endswith("keyframes") else \
                frame_velocity(f.t_ns.values[fr], self.ds.T_wc_json[fr][:, :3, 3])
            X = apply_time_offset(M["X"], M["pt_frame"], v, self.delta)
            X = apply_time_offset(X, M["pt_frame"], v * self._lidar_dt(fr)[:, None], 1.0).astype(np.float64)
            pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(X))
            col = np.array([CRGB.get(int(l), (128, 128, 128)) for l in range(256)], float)[M["label"]] / 255
            pc.colors = o3d.utility.Vector3dVector(col); o3d.io.write_point_cloud(str(self.out / f"sanity_{nm}.ply"), pc)
            # thickness on flat paved surfaces: robust spread in 0.5 m cells
            g = GroundGrid.from_bounds(*X[:, :2].min(0), *X[:, :2].max(0), 0.5); ids = g.cell_id(X[:, :2])
            flat = np.isin(M["label"], self.dcfg["classes"]["paved"]); idf = np.where(flat, ids, -1)
            cnt, q = grouped_quantiles(idf, X[:, 2], g.n, (0.5,)); sp = grouped_mad(idf, X[:, 2], g.n, q[0])
            nfr = grouped_nunique(idf, M["pt_frame"], g.n); ok = (cnt >= 20) & (nfr >= 5)
            s = rng.choice(len(X), min(400_000, len(X)), replace=False); dz = mesh_dz_points(mesh, X[s])
            lab = M["label"][s]
            per = {cname(c): dict(n=int((lab == c).sum()), dz_median=float(np.nanmedian(dz[lab == c])),
                                  dz_iqr=float(np.nanpercentile(dz[lab == c], 75) - np.nanpercentile(dz[lab == c], 25)))
                   for c in np.unique(lab) if (lab == c).sum() > 500}
            # between-frame consistency (doubling / smear): spread of per-frame medians in 1 m flat cells
            g1 = GroundGrid.from_bounds(*X[:, :2].min(0), *X[:, :2].max(0), 1.0); c1 = np.where(flat, g1.cell_id(X[:, :2]), -1)
            nfrm = len(fr); cf = np.where(c1 >= 0, c1 * nfrm + M["pt_frame"], -1)
            fc, fq = grouped_quantiles(cf, X[:, 2], g1.n * nfrm, (0.5,)); okf = fc >= 5; cell_of = np.arange(g1.n * nfrm) // nfrm
            zf = fq[0]; cc, cq = grouped_quantiles(np.where(okf, cell_of, -1), zf, g1.n, (0.5,))
            dev = np.abs(zf - cq[0][cell_of]); _, mq = grouped_quantiles(np.where(okf, cell_of, -1), dev, g1.n, (0.5,))
            okc = cc >= 5; between = 1.4826 * mq[0][okc]
            if not okc.any():                                               # no paved cells in the window (e.g. HKisland)
                between = np.array([np.nan])
            if not ok.any():
                ok = np.zeros_like(ok); sp = np.full_like(sp, np.nan); ok[0] = True
            rows.setdefault("_between", {})[nm] = dict(cells=int(okc.sum()), between_frame_spread_median_m=float(np.median(between)),
                                                      between_frame_spread_p90_m=float(np.percentile(between, 90)))
            rows[nm] = dict(frames=[int(fr[0]), int(fr[-1])], n_frames=len(fr), n_points=len(X),
                            between_frame_spread_median_m=float(np.median(between)), between_frame_spread_p90_m=float(np.percentile(between, 90)),
                            between_frame_cells=int(okc.sum()),
                            flat_cells=int(ok.sum()), flat_spread_median_m=float(np.nanmedian(sp[ok])),
                            flat_spread_p90_m=float(np.nanpercentile(sp[ok], 90)), lidar_minus_mesh_by_class=per)
            if nm.endswith("keyframes"):
                self._profile_plot(X, M["label"], mesh)
        rows.pop("_between", None); self.results["sanity"] = rows; log("sanity:", json.dumps(rows, indent=1, default=float)[:1500])

    def _profile_plot(self, X, lab, mesh):
        """Cross-track profile: LiDAR points within +-0.5 m of a north-south transect vs the Terra mesh surface."""
        import matplotlib.pyplot as plt
        from ..analysis.plots import CAT, _style, INK2
        cand = np.isin(lab, [1, 10, 3, 18]); xs = np.median(X[cand, 0]) if cand.any() else np.median(X[:, 0])
        m = np.abs(X[:, 0] - xs) < 0.5; P = X[m]; L = lab[m]; ys = np.linspace(P[:, 1].min(), P[:, 1].max(), 1500)
        zm = mesh_top_z(mesh, np.c_[np.full_like(ys, xs), ys])
        fig, ax = plt.subplots(figsize=(12, 4))
        for k, (grp, ids) in enumerate([("paved", list(self.dcfg["classes"]["paved"])), ("roof", [1]), ("vegetation/other", None)]):
            mm = np.isin(L, ids) if ids else ~np.isin(L, list(self.dcfg["classes"]["paved"]) + [1])
            ax.scatter(P[mm, 1], P[mm, 2], s=1, c=CAT[k], label=f"LiDAR: {grp}")
        ax.plot(ys, zm, color="#0b0b0b", lw=1.2, label="Terra mesh (top surface)")
        ax.set_xlabel("working-frame y (m)"); ax.set_ylabel("Z (m)"); ax.legend(frameon=False, fontsize=8, markerscale=6)
        ax.set_title(f"Transect x = {xs:.1f} m: sanity-stack LiDAR (keyframes) vs Terra mesh (before alignment)", loc="left", fontsize=11); _style(ax)
        fig.tight_layout(); fig.savefig(self.out / "sanity_profile.png", dpi=130); plt.close(fig)

    # ------------------------------------------------------------------ A2
    def stage_grid(self):
        gc = self.cfg["grid"]; X = self.X.astype(np.float64)
        lo = np.percentile(X[:, :2], 0.05, axis=0); hi = np.percentile(X[:, :2], 99.95, axis=0)
        self.grid = GroundGrid.from_bounds(lo[0], lo[1], hi[0], hi[1], gc["cell_size"])
        ids = self.grid.cell_id(X[:, :2]); self.ids = ids; n = self.grid.n
        cnt, q = grouped_quantiles(ids, X[:, 2], n, (0.1, 0.5, 0.9))
        # balanced Z: median over viewing groups of each group's cell median (groups with >= min_group_points)
        ug, ginv = np.unique(self.pt_group, return_inverse=True); G = len(ug)
        cg = np.where(ids >= 0, ids * G + ginv, -1)
        gcnt, gq = grouped_quantiles(cg, X[:, 2], n * G, (0.5, 0.9))
        okg = gcnt >= gc["min_group_points"]; cell_of = np.arange(n * G) // G
        bc, bq = grouped_quantiles(np.where(okg, cell_of, -1), gq[0], n, (0.0, 0.5, 1.0, 0.25, 0.75))
        _, bq90 = grouped_quantiles(np.where(okg, cell_of, -1), gq[1], n, (0.5,))
        sp = grouped_mad(ids, X[:, 2], n, q[1]); mode, frac = grouped_mode(ids, self.M["label"], n)
        nfr = grouped_nunique(ids, self.M["pt_frame"], n)
        lid = np.where(self.pt_line >= 0, self.pt_line, 99); nl = grouped_nunique(np.where(self.pt_line >= 0, ids, -1), lid, n)
        on = np.bincount(ids[ids >= 0], weights=self.M["off_nadir"][ids >= 0], minlength=n) / np.maximum(cnt, 1)
        has_b = bc >= 1
        self.C = pd.DataFrame(dict(cell_id=np.arange(n), x=self.grid.centers()[:, 0], y=self.grid.centers()[:, 1],
                                   lidar_z=np.where(has_b, bq[1], q[1]), lidar_z_top=np.where(has_b, bq90[0], q[2]),
                                   lidar_z_pooled=q[1], lidar_z_p10=q[0], lidar_z_p90=q[2],
                                   n_groups=bc, gt_group_range=np.where(bc >= 2, bq[2] - bq[0], np.nan), gt_group_iqr=np.where(bc >= 2, bq[4] - bq[3], np.nan),
                                   lidar_spread=sp, point_count=cnt,
                                   n_frames=nfr, n_lines=nl, mean_off_nadir_deg=np.where(cnt > 0, on, np.nan),
                                   dominant_class=mode, dominant_frac=frac))
        self.C["no_gt"] = self.C.point_count < gc["min_points"]
        # GT quality: >= 2 viewing groups whose medians agree within good_max_group_range (pose-driven GT inconsistency)
        self.C["gt_good"] = ~self.C.no_gt & (self.C.n_groups >= 2) & (self.C.gt_group_iqr <= gc["good_max_group_iqr"])
        z = self.grid.to_image(self.C.lidar_z.values)
        gy, gx = np.gradient(z); slope = np.hypot(gx, gy) / gc["cell_size"]
        self.C["lidar_slope"] = slope.ravel()
        sc = self.cfg["stable"]
        self.C["stable"] = (self.C.point_count >= sc["min_points"]) & (self.C.lidar_spread <= sc["max_spread"]) & \
                           ~(self.C.gt_group_range > sc["max_group_range"]) & \
                           (self.C.lidar_slope <= sc["max_neighbour_slope"]) & ~self.C.dominant_class.isin(self.dcfg["classes"]["dynamic"])
        cov = self.C[self.C.point_count > 0]
        self.results["grid"] = dict(grid=json.loads(self.grid.to_json()), cells_total=n, cells_with_points=int(len(cov)),
                                    cells_gt=int((~self.C.no_gt).sum()), cells_stable=int(self.C.stable.sum()),
                                    points_per_gt_cell_median=float(self.C.point_count[~self.C.no_gt].median()))
        log("grid:", self.results["grid"])

    # ------------------------------------------------------------------ Q4 / Q5
    def stage_checks(self):
        X = self.X.astype(np.float64); ids = self.ids; C = self.C
        # Q5: line-to-line consistency of LiDAR placed with Terra keyframe poses
        lines = sorted(set(self.pt_line[self.pt_line >= 0].tolist()))
        pairs = [(a, b) for i, a in enumerate(lines) for b in lines[i + 1:]]
        q5 = pairwise_line_alignment(X, self.pt_line, pairs)
        q5 = q5[q5.n_overlap_cells > 500] if "dx" in q5 else q5
        q5.to_csv(self.out / "q5_line_consistency.csv", index=False)
        # RTK residual per line for comparison
        from ..io.rtk import load_rtk
        from ..poses.rtk import fit_rtk_to_working
        f = self.ds.frames; rtk = load_rtk(self.ds.p["rtk_csv"], self.ds.terra_origin)
        fitmask = f.is_reference.values & ~f.segment.isin(self.dcfg["rtk"]["fit_segments_exclude"]).values
        sim, res = fit_rtk_to_working(rtk, f.t_ns.values, self.ds.T_wc_json[:, :3, 3], fitmask)
        from ..geometry.se3 import Sim3
        P = __import__("aerodiag.io.rtk", fromlist=["interpolate_rtk"]).interpolate_rtk(rtk, f.t_ns.values)
        dvec = sim.apply(P) - self.ds.T_wc_json[:, :3, 3]
        per_line = {}
        for lid_ in lines:
            m = f.is_reference.values & (f.line_id.values == lid_) & f.segment.isin(["cruise_line", "diagonal"]).values
            per_line[int(lid_)] = dict(rtk_minus_terra_median=np.median(dvec[m], 0).round(3).tolist(), norm_median=float(np.median(res[m])))
        (Path(self.ds.out) / "rtk_to_working_sim3.json").write_text(json.dumps(dict(sim3=sim.to_dict(), describe=sim.describe(),
            fit_frames=int(fitmask.sum()), fit_excluded_segments=self.dcfg["rtk"]["fit_segments_exclude"],
            source_frame="local ENU at Terra metadata origin (lat0, lon0, h0), RTK alt as ellipsoidal height (Q8, unverified)",
            target_frame="Terra working frame"), indent=2))
        # Q4(a): LiDAR internal geometry regression on stable flat cells
        st = C.stable.values[np.clip(ids, 0, None)] & (ids >= 0)
        r = X[st, 2] - C.lidar_z.values[ids[st]]
        q4a = lidar_geometry_regression(r, self.M["xn"][st].astype(float), self.M["yn"][st].astype(float), self.M["range"][st].astype(float))
        # Q4(b): short-range (hover/climb ~33-80 m) vs cruise (~80-95 m) LiDAR over the same stable cells
        def seg_med(mask):
            ii = np.where(mask & (ids >= 0), ids, -1); cnt, q = grouped_quantiles(ii, X[:, 2], self.grid.n, (0.5,)); z = q[0]; z[cnt < 10] = np.nan; return z
        z_low = seg_med(np.isin(self.pt_seg, ["hover", "climb"]) & (self.M["range"] < 60))
        z_high = seg_med(np.isin(self.pt_seg, ["cruise_line", "diagonal"]))
        both = np.isfinite(z_low) & np.isfinite(z_high) & C.stable.values
        q4b = dict(n_cells=int(both.sum()), median_dz_short_minus_long=float(np.median(z_low[both] - z_high[both])) if both.any() else None,
                   by_class={cname(c): float(np.median((z_low - z_high)[both & (C.dominant_class.values == c)]))
                             for c in np.unique(C.dominant_class.values[both]) if (both & (C.dominant_class.values == c)).sum() > 30})
        self.results["checks"] = dict(q5_line_pairs=q5.round(3).to_dict("records"), rtk_per_line=per_line,
                                      rtk_sim3=sim.describe(), q4a_geometry_regression=q4a, q4b_short_vs_long_range=q4b)
        log("checks:", json.dumps(self.results["checks"], indent=1, default=float)[:3000])

    # ------------------------------------------------------------------ A3
    def stage_terra(self):
        key = dict(stage="terra", grid=self.grid.to_json())
        def compute():
            P = load_cloud(self.ds.p["terra_dir"]); z, cnt = rasterize_points(P, self.grid)
            zm = mesh_top_z(load_mesh(self.ds.p["terra_dir"]), self.grid.centers())
            return dict(z=z, cnt=cnt, zm=zm)
        r = self.cache.get_or_compute("terra_raster", key, compute)
        self.C["terra_z_raw"] = r["z"]; self.C["terra_count"] = r["cnt"]; self.C["terra_mesh_z_raw"] = r["zm"]
        log("terra raster: cells with Terra", int(np.isfinite(r["z"]).sum()))

    # ------------------------------------------------------------------ A4 / A5
    def stage_align(self):
        C = self.C; st = C.stable.values & np.isfinite(C.terra_z_raw.values)
        dz0 = C.terra_z_raw.values - C.lidar_z.values
        glob = dict(median_terra_minus_lidar_stable=float(np.median(dz0[st])), n_stable_cells=int(st.sum()),
                    by_class={cname(c): float(np.median(dz0[st & (C.dominant_class.values == c)]))
                              for c in np.unique(C.dominant_class.values[st]) if (st & (C.dominant_class.values == c)).sum() > 50})
        key = dict(stage="align", grid=self.grid.to_json(), stable=self.cfg["stable"], align=self.cfg["align"], delta=self.delta)
        def compute():
            P = load_cloud(self.ds.p["terra_dir"]); cid = self.grid.cell_id(P[:, :2])
            src = P[(cid >= 0) & st[np.clip(cid, 0, None)]]
            X = self.X.astype(np.float64); dst = X[(self.ids >= 0) & st[np.clip(self.ids, 0, None)]]
            T0 = np.eye(4); T0[2, 3] = -glob["median_terra_minus_lidar_stable"]
            ac = self.cfg["align"]; T, logs = robust_icp(src, dst, ac["max_corr_dist"], ac["tukey_k"], ac["voxel"], init=T0)
            Pa = P @ T[:3, :3].T + T[:3, 3]; z, _ = rasterize_points(Pa, self.grid)
            mesh = load_mesh(self.ds.p["terra_dir"]); mesh.transform(T); zm = mesh_top_z(mesh, self.grid.centers())
            return dict(T=T, z=z, zm=zm, logs=np.array(json.dumps(logs)))
        r = self.cache.get_or_compute("terra_align", key, compute)
        T = r["T"]; self.C["terra_z"] = r["z"]; self.C["terra_mesh_z"] = r["zm"]
        self.C["error"] = self.C.terra_z - self.C.lidar_z
        self.C["error_mesh"] = self.C.terra_mesh_z - self.C.lidar_z
        self.C["error_top"] = self.C.terra_z - self.C.lidar_z_top       # visible-top-surface variant (vegetation)
        self.C.loc[self.C.no_gt, ["error", "error_mesh", "error_top"]] = np.nan
        np.savetxt(self.out / "terra_to_lidar_T.txt", T, header="4x4 rigid transform applied to Terra (X' = R X + t), working frame, metres")
        self.results["align"] = dict(global_offset_pre_alignment=glob, transform=describe_rigid(T), icp=json.loads(str(r["logs"])),
                                     residual_on_stable_after=float(np.nanmedian(self.C.error[st])),
                                     abs_residual_on_stable_after_median=float(np.nanmedian(np.abs(self.C.error[st]))))
        log("align:", json.dumps(self.results["align"], indent=1, default=float))

    # ------------------------------------------------------------------ A7
    def stage_outputs(self):
        from ..analysis.plots import heatmap, class_map, mask_map, SEQ, DIV
        C, g, o = self.C, self.grid, self.out
        cols = ["cell_id", "x", "y", "error", "error_top", "error_mesh", "lidar_z", "lidar_z_top", "lidar_z_pooled", "lidar_z_p10",
                "lidar_z_p90", "terra_z", "terra_mesh_z", "terra_z_raw", "lidar_spread", "gt_group_range", "gt_group_iqr", "n_groups",
                "lidar_slope", "point_count", "n_frames", "n_lines", "mean_off_nadir_deg",
                "no_gt", "gt_good", "stable", "dominant_class", "dominant_frac", "terra_count"]
        tab = C[cols].copy(); tab["dominant_class_name"] = tab.dominant_class.map(lambda c: CMAP.get(int(c), "none"))
        keep = (tab.point_count > 0) | np.isfinite(tab.terra_z)
        tab[keep].to_parquet(o / "labels.parquet", index=False)
        (o / "grid.json").write_text(g.to_json())
        gt = ~C.no_gt
        zlo, zhi = np.nanpercentile(C.lidar_z[gt], [1, 99])
        heatmap(g, np.where(gt, C.lidar_z, np.nan), o / "heat_lidar_z.png", "LiDAR Z (median per 1 m cell; ground truth)", "Z (m, working frame)", SEQ, zlo, zhi)
        heatmap(g, C.terra_z, o / "heat_terra_z.png", "Terra Z (aligned; reference reconstruction)", "Z (m, working frame)", SEQ, zlo, zhi)
        heatmap(g, C.error, o / "heat_error.png", "Label: error = Terra Z (aligned) - LiDAR Z", "error (m); red = Terra above LiDAR", DIV, -1.0, 1.0, center=0.0)
        heatmap(g, np.where(C.point_count > 0, np.log10(C.point_count.clip(lower=1)), np.nan), o / "heat_point_count.png",
                "LiDAR point count per cell", "log10(points)", SEQ)
        mask_map(g, C.gt_good.values, o / "heat_gt_good.png", f"GT quality: >= 2 viewing groups, IQR of group medians <= {self.cfg['grid']['good_max_group_iqr']} m", "gt_good", "not good / no GT")
        mask_map(g, gt.values, o / "heat_no_gt.png", f"Ground-truth coverage (no_gt = fewer than {self.cfg['grid']['min_points']} points)", "has GT", "no_gt")
        class_map(g, np.where(gt, C.dominant_class, 255).astype(np.uint8), o / "heat_class.png", "Dominant semantic class per cell (LiDAR labels)", CRGB, cname)
        mask_map(g, C.stable.values, o / "heat_stable.png", "Stable cells used for Terra-LiDAR alignment", "stable", "not stable")
        heatmap(g, np.where(gt, C.gt_group_iqr, np.nan), o / "heat_gt_uncertainty.png",
                "GT uncertainty: IQR of per-viewing-group LiDAR medians", "IQR (m)", SEQ, 0, 1.0)
        heatmap(g, C.error_top, o / "heat_error_top.png", "Label variant: Terra Z (aligned) - LiDAR top surface (p90)",
                "error (m); red = Terra above LiDAR", DIV, -1.0, 1.0, center=0.0)
        # statistics
        e = C[gt & np.isfinite(C.error)]
        def stats(s):
            a = s.abs(); return pd.Series(dict(cells=len(s), median=s.median(), mean=s.mean(), mad=1.4826 * (s - s.median()).abs().median(),
                                               abs_p50=a.median(), abs_p90=a.quantile(.9), abs_p99=a.quantile(.99),
                                               frac_abs_gt_0_25=(a > .25).mean(), frac_abs_gt_0_5=(a > .5).mean(), frac_abs_gt_1=(a > 1).mean()))
        cls = e.dominant_class.map(lambda c: CMAP.get(int(c), str(c)))
        overall = stats(e.error); per = e.groupby(cls).error.apply(stats).unstack().sort_values("cells", ascending=False)
        per_top = e.groupby(cls).error_top.apply(stats).unstack().loc[per.index]
        per.to_csv(o / "error_stats_by_class.csv"); per_top.to_csv(o / "error_top_stats_by_class.csv")
        overall_top = stats(e.error_top.dropna())
        eg = e[e.gt_good]; per_good = eg.groupby(eg.dominant_class.map(lambda c: CMAP.get(int(c), str(c)))).error.apply(stats).unstack()
        per_good = per_good.sort_values("cells", ascending=False); per_good.to_csv(o / "error_stats_by_class_gt_good.csv")
        self.results["stats_gt_good"] = dict(overall=stats(eg.error).round(4).to_dict(), by_class=per_good.round(4).to_dict("index"))
        pd.concat([overall.to_frame("error_median_label").T, overall_top.to_frame("error_top_label").T]).to_csv(o / "error_stats_overall.csv")
        unc = e.gt_group_iqr.dropna()
        self.results["stats"] = dict(overall=overall.round(4).to_dict(), overall_top=overall_top.round(4).to_dict(),
                                     by_class=per.round(4).to_dict("index"), by_class_top=per_top.round(4).to_dict("index"),
                                     gt_uncertainty=dict(cells_multi_group=int(len(unc)), median=float(unc.median()), p90=float(unc.quantile(.9)),
                                         by_class={k: float(v) for k, v in e.groupby(cls).gt_group_iqr.median().items()}),
                                     cells_no_gt=int(C.no_gt.sum()), cells_gt=int(gt.sum()), cells_gt_without_terra=int((gt & ~np.isfinite(C.terra_z)).sum()))
        # error vs geometry
        e2 = e.assign(onb=pd.cut(e.mean_off_nadir_deg, [0, 10, 20, 30, 40, 90]), slb=pd.cut(e.lidar_slope, [0, .1, .3, .6, 1.2, 100]))
        e2.groupby("onb", observed=True).error.describe(percentiles=[.5]).to_csv(o / "error_vs_off_nadir.csv")
        e2.groupby("slb", observed=True).error.describe(percentiles=[.5]).to_csv(o / "error_vs_slope.csv")
        self._hist(e)
        log("stats overall:", overall.round(3).to_dict())

    def _hist(self, e):
        import matplotlib.pyplot as plt
        from ..analysis.plots import CAT, _style
        fig, ax = plt.subplots(figsize=(9, 4.5)); bins = np.linspace(-2, 2, 161)
        groups = [("airstrip", [10]), ("roads/car park", [3, 18, 2]), ("roof", [1]), ("green_field (forest)", [13])]
        for k, (nm, ids) in enumerate(groups):
            s = e.error[e.dominant_class.isin(ids)]
            ax.hist(s, bins, histtype="step", lw=2, color=CAT[k], density=True, label=f"{nm} (n={len(s)})")
        ax.axvline(0, color="#52514e", lw=0.8); ax.set_xlabel("error = Terra - LiDAR (m)"); ax.set_ylabel("density")
        ax.legend(frameon=False, fontsize=8); ax.set_title("Per-cell label distribution by class", loc="left", fontsize=11); _style(ax)
        fig.tight_layout(); fig.savefig(self.out / "error_hist_by_class.png", dpi=130); plt.close(fig)

    def save(self):
        write_params(self.out, "track_a", dict(data=self.dcfg, track_a=self.cfg, n_reference_frames=len(self.fidx)))
        rp = self.out / "results.json"; prev = json.loads(rp.read_text()) if rp.exists() else {}
        prev.update(self.results)        # partial-stage runs update only their own sections
        rp.write_text(json.dumps(prev, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x)))

STAGES = ["map", "offset", "sanity", "grid", "checks", "terra", "align", "outputs"]

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="configs/data.yaml"); ap.add_argument("--cfg", default="configs/track_a.yaml")
    ap.add_argument("--stages", default=",".join(STAGES))
    a = ap.parse_args(argv)
    t = TrackA(load_config(a.data), load_config(a.cfg)); want = a.stages.split(",")
    for s in STAGES:
        needed = s in want or (s in ("map", "offset") ) or (s == "grid" and any(x in want for x in ("checks", "terra", "align", "outputs"))) \
                 or (s == "terra" and any(x in want for x in ("align", "outputs"))) or (s == "align" and "outputs" in want)
        if needed:
            getattr(t, f"stage_{s}")()
    t.save()

if __name__ == "__main__":
    main()
