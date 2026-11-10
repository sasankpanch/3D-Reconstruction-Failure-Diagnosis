"""Step-2 sanity evaluation (COMPARISON ONLY: Track A LiDAR labels are joined here, never used upstream).
Cells share the Track A grid geometry, so the join is by cell_id."""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ..io.labels import CMAP
from .plots import heatmap, SEQ, CAT, _style

def evaluate(st) -> dict:
    out = st.out; res = {}; g = st.grid
    S = pd.read_parquet(out / "cell_pair_signals.parquet")
    S["grp"] = np.where(S.pair_type == "within_line", "within:" + S.gap_frames.astype(str), S.pair_type)
    L = pd.read_parquet(Path(st.ds.out) / "track_a" / "labels.parquet").set_index("cell_id")
    for c in ["lidar_z", "dominant_class", "gt_good", "error", "no_gt", "lidar_spread", "lidar_slope"]:
        S[c] = L[c].reindex(S.cell_id).values
    S["cls"] = S.dominant_class.map(lambda c: CMAP.get(int(c), "none") if np.isfinite(c) else "none")
    S["dz_lidar"] = S.z_best - S.lidar_z
    paved = S.dominant_class.isin(st.dcfg["classes"]["paved"]) & (S.lidar_spread < 0.15) & (S.lidar_slope < 0.15)
    # 1. height from the 1D search vs LiDAR on flat paved cells (global median offset reported, then removed)
    res["paved_rows"] = int(paved.sum())
    off = float(S.loc[paved, "dz_lidar"].median()) if paved.any() else 0.0; res["height_offset_vs_lidar_m"] = off
    def hstats(d):
        e = d.dz_lidar - off
        return pd.Series(dict(n=len(d), dz_median=e.median(), abs_dz_p50=e.abs().median(), abs_dz_p90=e.abs().quantile(.9),
                              predicted_sigma_z=d.sigma_z.median(), px_per_m=d.px_per_m.median()))
    H = S[paved & S.dz_lidar.notna()].groupby("grp").apply(hstats, include_groups=False); H.to_csv(out / "eval_height_vs_lidar.csv")
    res["height_vs_lidar"] = H.round(3).reset_index().to_dict("records")
    # 2. signals by class (within-line gaps 5-40)
    W = S[S.grp.isin(["within:5", "within:10", "within:20", "within:40"])]
    rows = []
    for c, d in W.groupby("cls"):
        if len(d) < 500:
            continue
        v = d.dropna(subset=["cost_min"]); r = dict(cls=c, rows=len(d), textureless_frac=float(d.cost_min.isna().mean()))
        for k in ["across_px", "cost_min", "sharpness", "flatness", "gain", "contrast_ratio", "dz_sigma"]:
            a = v[k].abs() if k in ("across_px", "gain", "dz_sigma") else v[k]; r[f"{k}_p50"] = a.median(); r[f"{k}_p90"] = a.quantile(.9)
        r.update(multi_minima_frac=float((v.n_minima >= 2).mean()), periodic_frac=float(v.periodic.mean()),
                 below_terrain_frac=float(v.below_terrain.mean()), boundary_frac=float(v.at_boundary.mean()), glint_max=float(v.glint_risk.max()))
        rows.append(r)
    C = pd.DataFrame(rows).sort_values("rows", ascending=False); C.to_csv(out / "eval_signals_by_class.csv", index=False)
    res["signals_by_class"] = C.round(4).to_dict("records")
    A = S[paved].groupby("grp").across_px.agg(n="count", p50=lambda s: s.abs().median(), p90=lambda s: s.abs().quantile(.9),
                                               p99=lambda s: s.abs().quantile(.99))
    A.to_csv(out / "eval_across_noise_floor_paved.csv"); res["across_noise_floor_paved"] = A.round(3).reset_index().to_dict("records")
    R = S.dropna(subset=["cost_min"]).groupby("grp")[["below_terrain", "periodic", "at_boundary"]].mean(); R.to_csv(out / "eval_flag_rates_by_pairs.csv")
    res["flag_rates_by_pairs"] = R.round(4).reset_index().to_dict("records")
    # 3. per-cell medians (within-line 5-40) and descriptive association with |Terra error| on gt_good cells
    agg = dict(across=("across_px", lambda s: s.abs().median()), cost=("cost_min", "median"), sharp=("sharpness", "median"),
               flat=("flatness", "median"), periodic=("periodic", "mean"), below=("below_terrain", "mean"), dzs=("dz_sigma", lambda s: s.abs().median()),
               gain=("gain", lambda s: s.abs().median()), glint=("glint_risk", "max"), textureless=("cost_min", lambda s: s.isna().mean()),
               npairs=("pair_id", "nunique"))
    M = W.groupby("cell_id").agg(**agg); M.to_parquet(out / "cell_signal_medians.parquet")
    Lg = L.reindex(M.index); good = (Lg.gt_good == True).values
    res["spearman_with_abs_terra_error_gt_good"] = {k: float(pd.Series(M[k].values[good]).corr(pd.Series(np.abs(Lg.error.values[good])), method="spearman"))
                                                     for k in ["across", "cost", "sharp", "flat", "periodic", "below", "dzs", "gain", "textureless"]}
    from ..signals.sun import solar_position
    lat0, lon0, _ = st.ds.terra_origin; az, el = solar_position(st.ds.frames.t_ns.values[np.unique(np.r_[st.P.i, st.P.j])], lat0, lon0)
    res["glint"] = dict(sun_azimuth_deg=[float(az.min()), float(az.max())], sun_elevation_deg=[float(el.min()), float(el.max())],
                        cells_glint_gt_0_5=int((M.glint > 0.5).sum()), specular_deg_min=float(S.specular_deg.min()))
    for col, title, unit, lo, hi in [("across", "Plane-sweep |across| (per-cell median, within-line gaps 5-40)", "px", 0, 2),
                                     ("cost", "Photometric residual 1-NCC at best height", "1-NCC", 0, 0.6),
                                     ("sharp", "Cost-curve sharpness at the minimum", "per px^2", 0, 0.3),
                                     ("periodic", "Fraction of pairs with evenly spaced minima (repetitive)", "fraction", 0, 0.5),
                                     ("below", "Fraction of pairs with best height below coarse terrain", "fraction", 0, 0.5),
                                     ("glint", "Max glint risk over pairs", "risk", 0, 1),
                                     ("textureless", "Fraction of pairs with textureless reference patch", "fraction", 0, 1)]:
        v = np.full(g.n, np.nan); v[M.index.values] = M[col].values; heatmap(g, v, out / f"map_{col}.png", title, unit, SEQ, lo, hi)
    fig, ax = plt.subplots(figsize=(8, 4))
    for k, gp in enumerate(["within:5", "within:20", "within:80", "cross_line"]):
        d = (S[paved & (S.grp == gp)].dz_lidar - off).dropna()
        if len(d):
            ax.hist(d.clip(-3, 3), np.linspace(-3, 3, 121), histtype="step", lw=2, color=CAT[k], density=True, label=f"{gp} (n={len(d)})")
    ax.set_xlabel("1D-search height - LiDAR (m, global offset removed), flat paved cells"); ax.set_ylabel("density"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("Height from the epipolar search vs ground truth", loc="left", fontsize=10); _style(ax); fig.tight_layout(); fig.savefig(out / "fig_height_vs_lidar.png", dpi=130); plt.close(fig)
    return res
