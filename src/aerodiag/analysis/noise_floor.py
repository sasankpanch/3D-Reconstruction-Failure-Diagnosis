"""B5/B6: healthy vs unhealthy ground and the across-epipolar noise-floor analysis.

Healthy (a) class-based: dominant class in {airstrip, paved road, car park, roof}, match not on a vehicle.
Healthy (b) error-based: gt_good & |error| <= error_max_abs & cell texture >= min_texture.
Unhealthy (b): gt_good & |error| >= unhealthy_error_min_abs (all classes; vegetation also reported separately).
Vehicles: matches on projected sedan/truck LiDAR points (moving OR parked).
Per-match statistics use |across| (px). Separability uses per-cell scores = median |across| of the cell's matches
(>= min_matches_cell) for one condition, compared with AUROC and the detection rate at the healthy 95th percentile."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .plots import CAT, _style, INK2, SEQ, DIV, heatmap
from ..io.labels import CMAP

MAIN = ["terra", "terra+ref", "rtk", "rtk+ref"]

def _q(a):
    a = np.abs(np.asarray(a, float)); a = a[np.isfinite(a)]
    if len(a) < 30:
        return dict(n=int(len(a)), p50=np.nan, p90=np.nan, p99=np.nan)
    p = np.percentile(a, [50, 90, 99]); return dict(n=int(len(a)), p50=p[0], p90=p[1], p99=p[2])

def auroc(pos, neg):
    """P(score_pos > score_neg) via ranks (ties = 0.5)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) < 5 or len(neg) < 5:
        return np.nan
    allv = np.r_[pos, neg]; r = pd.Series(allv).rank().values
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

def load(tb):
    R = pd.read_parquet(tb.out / "residuals.parquet"); P = pd.read_csv(tb.out / "pairs.csv", index_col=0)
    L = pd.read_parquet(Path(tb.ds.out) / "track_a" / "labels.parquet").set_index("cell_id"); hc = tb.cfg["healthy"]
    R = R.join(P[["pair_type", "gap_frames", "seg_i", "dt_s", "heading_diff_deg", "baseline_m"]], on="pair_id")
    R["grp"] = np.where(R.pair_type.isin(["within_line", "climb", "turn"]), R.pair_type + ":" + R.gap_frames.astype(str), R.pair_type)
    ok = R.cell_id >= 0; idx = R.cell_id.where(ok, L.index[0])
    for c in ["dominant_class", "gt_good", "error", "no_gt"]:
        v = L[c].reindex(idx).values; R[c] = np.where(ok, v, np.nan if c == "error" else (255 if c == "dominant_class" else False))
    R["dominant_class"] = R.dominant_class.fillna(255).astype(int); R["gt_good"] = R.gt_good.astype(bool)
    tex = R[ok].groupby("cell_id").tex.median(); R["cell_tex"] = tex.reindex(R.cell_id).values
    R["healthy_a"] = R.dominant_class.isin(hc["class_based"]) & ~R.vehicle
    R["healthy_b"] = R.gt_good & (R.error.abs() <= hc["error_max_abs"]) & (R.cell_tex >= hc["min_texture"]) & ~R.vehicle
    R["unhealthy_b"] = R.gt_good & (R.error.abs() >= hc["unhealthy_error_min_abs"]) & ~R.vehicle
    R["vegetation"] = R.dominant_class.isin(hc["vegetation"]) & ~R.vehicle
    R["rmax"] = np.maximum(R.r1, R.r2)
    b = tb.cfg["position_bins_px"]; R["pos"] = pd.cut(R.rmax, b, labels=["centre", "mid", "edge"])
    return R, P, L

def run_analysis(tb) -> dict:
    R, P, L = load(tb); out = tb.out; res = {}
    sources = [c[len("across_"):] for c in R.columns if c.startswith("across_")]
    groups = {"healthy_a": R.healthy_a, "healthy_b": R.healthy_b, "unhealthy_b": R.unhealthy_b, "vehicle": R.vehicle, "vegetation": R.vegetation}
    res["counts"] = {k: int(v.sum()) for k, v in groups.items()}
    res["cells"] = {k: int(R.loc[v & (R.cell_id >= 0), "cell_id"].nunique()) for k, v in groups.items() if k != "vehicle"}
    # ---- T1: per group x source x pair group
    rows = []
    for gname, gm in groups.items():
        sub = R[gm]
        for pg, d in sub.groupby("grp"):
            for s in sources:
                rows.append(dict(group=gname, pairs=pg, source=s, **_q(d[f"across_{s}"])))
    T1 = pd.DataFrame(rows); T1.to_csv(out / "noise_floor_by_group_source_pairs.csv", index=False)
    # ---- T2: image position (within-line gaps 5-40), T3: class, T4: segment
    W = R[(R.pair_type == "within_line") & R.gap_frames.between(5, 40)]
    T2 = pd.DataFrame([dict(group=g, pos=p, source=s, **_q(d[f"across_{s}"])) for g in ["healthy_a", "healthy_b"] for p, d in W[W[g]].groupby("pos", observed=True) for s in MAIN])
    T2.to_csv(out / "noise_floor_by_image_position.csv", index=False)
    Wa = R[R.pair_type == "within_line"]
    T3 = pd.DataFrame([dict(cls=CMAP.get(int(c), str(c)), source=s, **_q(d[f"across_{s}"])) for c, d in Wa[~Wa.vehicle].groupby("dominant_class") for s in ["terra", "terra+ref", "rtk+ref"]])
    T3.to_csv(out / "noise_floor_by_class.csv", index=False)
    seg = np.where(R.pair_type == "within_line", R.seg_i, R.pair_type)
    T4 = pd.DataFrame([dict(segment=sg, source=s, **_q(d[f"across_{s}"])) for sg, d in R[R.healthy_a].groupby(seg[R.healthy_a.values]) for s in MAIN])
    T4.to_csv(out / "noise_floor_by_segment.csv", index=False)
    # ---- separability per condition (per-cell median |across|)
    mc = 3; sep = []
    conds = [(pg, R.grp == pg) for pg in sorted(R.grp.unique())] + [("within_line:all", R.pair_type == "within_line"), ("all_pairs", R.grp.notna())]
    for pg, cm in conds:
        sub = R[cm & (R.cell_id >= 0)]
        for s in sources:
            col = f"across_{s}"; v = sub[["cell_id", col, "healthy_a", "healthy_b", "unhealthy_b", "vegetation"]].dropna(subset=[col])
            if len(v) == 0:
                continue
            v = v.assign(a=v[col].abs()); g = v.groupby("cell_id"); score = g.a.median(); n = g.a.size()
            flags = g[["healthy_a", "healthy_b", "unhealthy_b", "vegetation"]].max(); keep = n >= mc
            sc, fl = score[keep], flags[keep]
            hb, ub, ha, vg = sc[fl.healthy_b.astype(bool)], sc[fl.unhealthy_b.astype(bool)], sc[fl.healthy_a.astype(bool)], sc[fl.vegetation.astype(bool)]
            thr = np.percentile(hb, 95) if len(hb) > 20 else np.nan
            sep.append(dict(pairs=pg, source=s, cells_healthy_b=len(hb), cells_unhealthy_b=len(ub), cells_healthy_a=len(ha),
                            healthy_b_cell_p50=np.median(hb) if len(hb) else np.nan, healthy_b_cell_p95=thr,
                            unhealthy_b_cell_p50=np.median(ub) if len(ub) else np.nan,
                            auroc_b=auroc(ub, hb), auroc_unhealthy_vs_healthy_a=auroc(ub, ha),
                            detect_at_5pct_fpr=float((ub > thr).mean()) if len(ub) and np.isfinite(thr) else np.nan,
                            auroc_vegetation_vs_healthy_b=auroc(vg, hb)))
    SEP = pd.DataFrame(sep); SEP.to_csv(out / "separability.csv", index=False)
    # ---- vehicles (per match): fraction above healthy p99
    veh = []
    for s in MAIN:
        h = R.loc[R.healthy_a & (R.pair_type == "within_line"), f"across_{s}"].abs().dropna(); v = R.loc[R.vehicle & (R.pair_type == "within_line"), f"across_{s}"].abs().dropna()
        if len(h) and len(v):
            p99 = np.percentile(h, 99); veh.append(dict(source=s, healthy_p99=p99, vehicle_n=len(v), vehicle_p50=np.median(v), vehicle_p90=np.percentile(v, 90),
                                                         vehicle_frac_above_healthy_p99=float((v > p99).mean())))
    VEH = pd.DataFrame(veh); VEH.to_csv(out / "vehicles.csv", index=False)
    # ---- along check: Terra triangulated Z vs LiDAR Z on healthy_a, by gap
    alg = []
    lz = L.lidar_z.reindex(R.cell_id.where(R.cell_id >= 0, L.index[0])).values
    dz = R.z_terra.values - lz
    for pg in sorted(R[R.pair_type == "within_line"].grp.unique(), key=lambda s: int(s.split(":")[1])):
        m = (R.grp == pg).values & R.healthy_a.values & np.isfinite(dz)
        if m.sum() > 100:
            B = R.baseline_m[m].median(); H = 82.0
            alg.append(dict(pairs=pg, n=int(m.sum()), baseline_m=B, dz_median=np.median(dz[m]), dz_abs_p50=np.median(np.abs(dz[m] - np.median(dz[m]))),
                            dz_abs_p90=np.percentile(np.abs(dz[m] - np.median(dz[m])), 90), predicted_m_per_px=H * H / (B * tb.ds.camera.f)))
    ALG = pd.DataFrame(alg); ALG.to_csv(out / "along_height_check.csv", index=False)
    # ---- sweep
    sw = [s for s in sources if s.startswith("sweep_")]
    SW = []
    for s in sw:
        if s.endswith("+ref"):
            sig = float(s[len("sweep_pos"):-4])
            for pg in ["within_line:5", "within_line:10", "within_line:20", "within_line:40", "within_line:80", "cross_line", "crossing"]:
                d = R.loc[R.healthy_a & (R.grp == pg), f"across_{s}"]; SW.append(dict(pos_sigma_m=sig, pairs=pg, **_q(d)))
    SW = pd.DataFrame(SW); SW.to_csv(out / "sweep_position_noise.csv", index=False)
    _plots(tb, R, T1, T2, SEP, SW, L)
    res.update(T1=None)
    res["separability_key"] = SEP[SEP.pairs.isin(["within_line:all", "all_pairs", "within_line:1", "within_line:10", "within_line:40", "cross_line", "crossing"]) &
                                  SEP.source.isin(MAIN + ["noisy_gnss+ref"])].round(3).to_dict("records")
    res["vehicles"] = VEH.round(3).to_dict("records"); res["along"] = ALG.round(3).to_dict("records")
    res.pop("T1")
    return res

def _plots(tb, R, T1, T2, SEP, SW, L):
    out = tb.out
    # P1: healthy_a p50/p90 vs gap per main source
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    d = T1[(T1.group == "healthy_a") & T1.pairs.str.startswith("within_line")].copy(); d["gap"] = d.pairs.str.split(":").str[1].astype(int)
    for ax, q in zip(axes, ["p50", "p90"]):
        for k, s in enumerate(MAIN):
            e = d[d.source == s].sort_values("gap"); ax.plot(e.gap, e[q], "-o", color=CAT[k], lw=2, ms=5, label=s)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks([1, 5, 10, 20, 40, 80]); ax.set_xticklabels([1, 5, 10, 20, 40, 80])
        ax.set_xlabel("frame gap (within line)"); ax.set_title(f"|across| {q}, healthy ground (class-based)", loc="left", fontsize=10); _style(ax)
    axes[0].set_ylabel("px"); axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig_noise_floor_vs_gap.png", dpi=130); plt.close(fig)
    # P2: position
    fig, ax = plt.subplots(figsize=(7, 4))
    e = T2[T2.group == "healthy_a"]
    for k, s in enumerate(MAIN):
        f = e[e.source == s]; ax.plot(f.pos.astype(str), f.p90, "-o", color=CAT[k], lw=2, label=f"{s} (p90)")
        ax.plot(f.pos.astype(str), f.p50, "--", color=CAT[k], lw=1.2)
    ax.set_yscale("log"); ax.set_ylabel("|across| (px); solid p90, dashed p50"); ax.set_xlabel("image position (max radius of the two observations)")
    ax.set_title("Noise floor vs image position (within-line gaps 5-40)", loc="left", fontsize=10); ax.legend(frameon=False, fontsize=8); _style(ax)
    fig.tight_layout(); fig.savefig(out / "fig_noise_floor_vs_position.png", dpi=130); plt.close(fig)
    # P3: per-cell CDFs healthy_b vs unhealthy_b vs vegetation (rtk+ref, within_line:all) and terra+ref
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, s in zip(axes, ["terra+ref", "rtk+ref"]):
        sub = R[(R.pair_type == "within_line") & (R.cell_id >= 0)]; v = sub.assign(a=sub[f"across_{s}"].abs()).dropna(subset=["a"])
        g = v.groupby("cell_id"); sc = g.a.median()[g.a.size() >= 3]; fl = g[["healthy_b", "unhealthy_b", "vegetation", "healthy_a"]].max().loc[sc.index]
        for k, (nm, col) in enumerate([("healthy (error-based)", "healthy_b"), ("healthy (class-based)", "healthy_a"), ("unhealthy (|error| >= 0.5 m)", "unhealthy_b"), ("vegetation", "vegetation")]):
            x = np.sort(sc[fl[col].astype(bool)].values)
            if len(x):
                ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=CAT[k], lw=2, label=f"{nm} (n={len(x)})")
        ax.set_xscale("log"); ax.set_xlabel("per-cell median |across| (px)"); ax.set_title(f"Per-cell score CDF: {s}, within-line pairs", loc="left", fontsize=10); _style(ax)
    axes[0].set_ylabel("cumulative fraction of cells"); axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig_cell_score_cdf.png", dpi=130); plt.close(fig)
    # P4: sweep
    if len(SW):
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        for k, pg in enumerate(["within_line:5", "within_line:20", "within_line:80", "cross_line", "crossing"]):
            e = SW[SW.pairs == pg].sort_values("pos_sigma_m"); ax.plot(e.pos_sigma_m, e.p90, "-o", color=CAT[k], lw=2, label=f"{pg} p90")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("per-frame position noise sigma (m), attitude 0.1 deg, rotation refined")
        ax.set_ylabel("|across| p90 on healthy ground (px)"); ax.set_title("Required position quality", loc="left", fontsize=10); ax.legend(frameon=False, fontsize=8); _style(ax)
        fig.tight_layout(); fig.savefig(out / "fig_sweep_position_noise.png", dpi=130); plt.close(fig)
    # P5: map of per-cell median |across| for rtk+ref (within-line all)
    sub = R[(R.pair_type == "within_line") & (R.cell_id >= 0)]; v = sub.assign(a=sub["across_rtk+ref"].abs()).dropna(subset=["a"])
    g = v.groupby("cell_id").a; sc = g.median()[g.size() >= 3]; vals = np.full(tb.grid.n, np.nan); vals[sc.index.values] = sc.values
    heatmap(tb.grid, vals, out / "fig_map_cell_across_rtk_ref.png", "Per-cell median |across| (rtk+ref, within-line pairs)", "px", SEQ, 0, 2.0)
