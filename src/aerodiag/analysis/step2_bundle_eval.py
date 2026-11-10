"""Step-2 sanity evaluation (COMPARISON ONLY: LiDAR labels and the RTK->working similarity are used here and nowhere
upstream). Onboard per-cell results are mapped into the working frame, registered to the LiDAR raster with a rigid
robust ICP (as Terra was in Track A), and joined to the Track A label table."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ..grid.ground_grid import GroundGrid
from ..io.labels import CMAP
from .plots import heatmap, SEQ, DIV, CAT, _style

SIGNALS = ["across_px", "cost_min", "sharpness", "flatness", "n_minima", "periodic", "dz", "dz_sigma", "below_terrain", "gain",
           "contrast_ratio", "glint_risk"]

def _q(s):
    s = pd.Series(s).dropna(); return pd.Series(dict(n=len(s), p50=s.median() if len(s) else np.nan, p90=s.quantile(.9) if len(s) else np.nan))

def evaluate(st) -> dict:
    out = st.out; res = {}
    S = pd.read_parquet(out / "cell_pair_signals.parquet"); g = st.grid
    S["grp"] = np.where(S.pair_type == "within_line", "within:" + S.gap_frames.astype(str), S.pair_type)
    xy = g.centers()[S.cell_id.values]
    # ---- map to working frame and register (evaluation only)
    import open3d as o3d
    L = pd.read_parquet(Path(st.ds.out) / "track_a" / "labels.parquet"); ga = GroundGrid.from_json((Path(st.ds.out) / "track_a" / "grid.json").read_text())
    surf = S[S.grp.isin(["within:20", "within:40", "within:80"])].groupby("cell_id").z_best.median()
    Psrc = st.sim.apply(np.c_[g.centers()[surf.index.values], surf.values])
    tgt = L[(~L.no_gt) & L.lidar_z.notna()]; Ptgt = np.c_[tgt.x, tgt.y, tgt.lidar_z]
    reg = o3d.pipelines.registration
    ps = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Psrc)); pt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Ptgt))
    pt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3.0, max_nn=30))
    T = np.eye(4); T[2, 3] = np.median(Ptgt[:, 2]) - np.median(Psrc[:, 2])
    for d in (5.0, 2.0, 1.0, 0.5):
        r = reg.registration_icp(ps, pt, d, T, reg.TransformationEstimationPointToPlane(reg.TukeyLoss(k=0.3))); T = r.transformation
    res["eval_registration"] = dict(T=np.asarray(T).round(5).tolist(), fitness=r.fitness, rmse=r.inlier_rmse)
    def to_work(xy_, z_):
        P = st.sim.apply(np.c_[xy_, z_]); return P @ T[:3, :3].T + T[:3, 3]
    Pw = to_work(xy, S.z_best.fillna(S.z_best.median()).values); Pc = to_work(xy, np.zeros(len(xy)) + st.cells["z0"].mean())
    S["a_cell"] = ga.cell_id(Pw[:, :2])
    Lx = L.set_index("cell_id"); ok = S.a_cell >= 0; idx = S.a_cell.where(ok, Lx.index[0])
    for c in ["lidar_z", "dominant_class", "gt_good", "error", "no_gt", "lidar_spread", "lidar_slope"]:
        S[c] = np.where(ok, Lx[c].reindex(idx).values, np.nan)
    S["cls"] = S.dominant_class.map(lambda c: CMAP.get(int(c), "none") if np.isfinite(c) else "none")
    S["z_work"] = Pw[:, 2]; S["dz_lidar"] = S.z_work - S.lidar_z
    paved = S.dominant_class.isin([10, 3, 18]) & (S.lidar_spread < 0.15) & (S.lidar_slope < 0.15)
    # 1. height accuracy of the 1D search vs LiDAR on flat paved cells, by pair group (vs predicted sigma_z)
    H = S[paved & S.dz_lidar.notna()].groupby("grp").apply(lambda d: pd.Series(dict(
        n=len(d), dz_median=d.dz_lidar.median(), abs_dz_p50=(d.dz_lidar - d.dz_lidar.median()).abs().median(),
        abs_dz_p90=(d.dz_lidar - d.dz_lidar.median()).abs().quantile(.9), predicted_sigma_z_median=d.sigma_z.median(),
        px_per_m=d.px_per_m.median())), include_groups=False)
    H.to_csv(out / "eval_height_vs_lidar.csv"); res["height_vs_lidar"] = H.round(3).reset_index().to_dict("records")
    # 2-6. signals by class (within-line pairs, gaps 5-40)
    W = S[S.grp.isin(["within:5", "within:10", "within:20", "within:40"])]
    rows = []
    for c, d in W.groupby("cls"):
        if len(d) < 500:
            continue
        r = dict(cls=c, rows=len(d), textureless_frac=float(d.cost_min.isna().mean()))
        v = d.dropna(subset=["cost_min"])
        for k in ["across_px", "cost_min", "sharpness", "flatness", "gain", "contrast_ratio", "dz_sigma"]:
            a = v[k].abs() if k in ("across_px", "gain", "dz_sigma") else v[k]; r[f"{k}_p50"] = a.median(); r[f"{k}_p90"] = a.quantile(.9)
        r["multi_minima_frac"] = float((v.n_minima >= 2).mean()); r["periodic_frac"] = float(v.periodic.mean())
        r["below_terrain_frac"] = float(v.below_terrain.mean()); r["boundary_frac"] = float(v.at_boundary.mean())
        r["glint_p99"] = v.glint_risk.quantile(.99); rows.append(r)
    C = pd.DataFrame(rows).sort_values("rows", ascending=False); C.to_csv(out / "eval_signals_by_class.csv", index=False)
    res["signals_by_class"] = C.round(4).to_dict("records")
    # across: sweep vs pair group on flat paved (noise floor of the sweep-based across signal)
    A = S[paved].groupby("grp").across_px.apply(lambda s: _q(s.abs())).unstack(); A.to_csv(out / "eval_across_noise_floor.csv")
    res["across_noise_floor_paved"] = A.round(3).reset_index().to_dict("records")
    # below-terrain / periodic rates by pair group
    R = S.dropna(subset=["cost_min"]).groupby("grp")[["below_terrain", "periodic", "at_boundary"]].mean(); R.to_csv(out / "eval_flag_rates_by_pairs.csv")
    res["flag_rates_by_pairs"] = R.round(4).reset_index().to_dict("records")
    # 7. glint
    from ..signals.sun import solar_position
    lat0, lon0, _ = st.ds.terra_origin; az, el = solar_position(st.ds.frames.t_ns.values[st.frames], lat0, lon0)
    res["glint"] = dict(sun_azimuth_deg=[float(az.min()), float(az.max())], sun_elevation_deg=[float(el.min()), float(el.max())],
                        cells_glint_gt_0_5=int((S.groupby("cell_id").glint_risk.max() > 0.5).sum()),
                        specular_deg_min=float(S.specular_deg.min()), specular_deg_p01=float(S.specular_deg.quantile(.01)))
    # descriptive association with Terra |error| on gt_good cells (Spearman, per-cell medians over within-line pairs)
    G = W[W.gt_good == 1].groupby("a_cell").agg(err=("error", "first"), **{s: (s, "median") for s in ["across_px", "cost_min", "sharpness", "flatness", "dz_sigma", "gain"]})
    G["across_px"] = G.across_px.abs(); G["dz_sigma"] = G.dz_sigma.abs(); G["gain"] = G.gain.abs(); G["abs_err"] = G.err.abs()
    res["spearman_with_abs_terra_error"] = {s: float(G[[s, "abs_err"]].corr(method="spearman").iloc[0, 1]) for s in ["across_px", "cost_min", "sharpness", "flatness", "dz_sigma", "gain"]}
    # maps (ENU grid, per-cell medians over within-line gaps 5-40)
    M = W.groupby("cell_id").agg(across=("across_px", lambda s: s.abs().median()), cost=("cost_min", "median"), sharp=("sharpness", "median"),
                                 periodic=("periodic", "mean"), below=("below_terrain", "mean"), glint=("glint_risk", "max"),
                                 textureless=("cost_min", lambda s: s.isna().mean()))
    for col, title, unit, cm, lo, hi in [("across", "Plane-sweep |across| (per-cell median, within-line 5-40)", "px", SEQ, 0, 2),
                                         ("cost", "Photometric residual 1-NCC at best height", "1-NCC", SEQ, 0, 0.6),
                                         ("sharp", "Cost-curve sharpness at the minimum", "per px^2", SEQ, 0, 0.05),
                                         ("periodic", "Fraction of pairs with evenly spaced minima (repetitive)", "fraction", SEQ, 0, 0.5),
                                         ("below", "Fraction of pairs with best height below coarse terrain", "fraction", SEQ, 0, 0.5),
                                         ("glint", "Max glint risk over pairs", "risk", SEQ, 0, 1),
                                         ("textureless", "Fraction of pairs with textureless reference patch", "fraction", SEQ, 0, 1)]:
        v = np.full(g.n, np.nan); v[M.index.values] = M[col].values
        heatmap(g, v, out / f"map_{col}.png", title, unit, cm, lo, hi)
    # height-error histogram by gap
    fig, ax = plt.subplots(figsize=(8, 4))
    for k, gp in enumerate(["within:5", "within:20", "within:80", "cross_line"]):
        d = S[paved & (S.grp == gp)].dz_lidar.dropna(); d = d - d.median()
        if len(d):
            ax.hist(d.clip(-3, 3), np.linspace(-3, 3, 121), histtype="step", lw=2, color=CAT[k], density=True, label=f"{gp} (n={len(d)})")
    ax.set_xlabel("1D-search height - LiDAR (m, median removed), flat paved cells"); ax.set_ylabel("density"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("Height from the epipolar search vs ground truth", loc="left", fontsize=10); _style(ax); fig.tight_layout(); fig.savefig(out / "fig_height_vs_lidar.png", dpi=130); plt.close(fig)
    S.drop(columns=["z_work"]).to_parquet(out / "cell_pair_signals_eval.parquet", index=False)
    return res
