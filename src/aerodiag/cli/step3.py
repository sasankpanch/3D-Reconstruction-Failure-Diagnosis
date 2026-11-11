"""Step 3 CLI: cause diagnosis. aerodiag-step3 [--data ...] [--cfg configs/step3.yaml]

3a accumulate step-2 (cell x pair) signals per cell (baseline-stratified);
3b moving vs reflective: per-cell fits of appearance change vs time gap and viewing-angle change (neighbour pooling);
3c physics scores for causes without labels (featureless, repetitive, weak geometry, moving, reflective) and a small
   multi-label classifier for causes/outcomes with independent proxy labels (failure, layered, vehicle), evaluated by
   spatially blocked cross-validation. Labels come from the comparison side (Track A LiDAR / Terra error) and are used
   ONLY as training targets and for evaluation, never as inputs.
Output: per-cell prediction file outputs/step3/cell_diagnosis.parquet."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd
from ..config import load_config, write_params
from ..io.dataset import Dataset
from ..grid.ground_grid import GroundGrid
from ..diagnosis.accumulate import accumulate
from ..diagnosis.motion_reflect import fit_motion_reflectance
from ..diagnosis.classifier import prepare, spatial_cv, fit_logreg, predict, auroc, full_metrics, blocks_of
from ..diagnosis.gbdt import fit_gbdt, predict_gbdt
from ..diagnosis.context import add_context
from ..diagnosis.magnitude import magnitude_cv

def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)

def read_columns(path, cols):
    """Low-memory parquet read: one column at a time into numpy (peak ~ final frame + one column); strings become
    categoricals; Arrow's pool is released after each column."""
    import pyarrow as pa, pyarrow.parquet as pq
    out = {}
    for c in cols:
        t = pq.read_table(path, columns=[c])[c]
        out[c] = pd.Categorical(t.to_pandas()) if pa.types.is_string(t.type) or pa.types.is_large_string(t.type) else t.to_numpy()
        del t; pa.default_memory_pool().release_unused()
    return pd.DataFrame(out)

def vehicle_cells(ds, grid, frames):
    """Count sedan/truck LiDAR points per cell (Terra keyframe poses) - label construction only (comparison side)."""
    cnt = np.zeros(grid.n)
    for i in frames:
        lab = ds.load_labels(i); m = np.isin(lab, ds.cfg["classes"].get("vehicle", [20, 24]))
        if m.any():
            X = ds.lidar_world(i)[m]; c = grid.cell_id(X[:, :2]); np.add.at(cnt, c[c >= 0], 1)
    return cnt

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="configs/data.yaml"); ap.add_argument("--cfg", default="configs/step3.yaml"); a = ap.parse_args(argv)
    dcfg, cfg = load_config(a.data), load_config(a.cfg); ds = Dataset(dcfg); out = Path(ds.out) / "step3"; out.mkdir(parents=True, exist_ok=True)
    grid = GroundGrid.from_json((Path(ds.out) / "track_a" / "grid.json").read_text()); res = {}
    S = read_columns(Path(ds.out) / "step2" / "cell_pair_signals.parquet",          # only the columns step 3 uses, one at a time
                     ["cell_id", "i", "j", "pair_type", "gap_frames", "dt_s", "z_best", "across_px", "cost_min", "sharpness",
                      "flatness", "gain", "contrast_ratio", "dz_sigma", "glint_risk", "baseline_m", "heading_diff_deg", "periodic",
                      "n_minima", "below_terrain", "at_boundary", "ref_std"])
    S["i"] = S.i.astype(np.int32); S["j"] = S.j.astype(np.int32)                     # pair_type is categorical (not 50-100M strings)
    # viewing-angle change per (cell, pair): rays from both camera centres (onboard rtk poses) to the cell point
    from .step2 import Step2
    st = Step2.__new__(Step2); st.ds = ds; st.dcfg = dcfg; st.cfg = load_config(Path(dcfg["root"]) / "configs" / "step2.yaml")
    from .track_b import TrackB
    tbo = TrackB.__new__(TrackB); tbo.ds = ds; tbo.cfg = load_config(Path(dcfg["root"]) / "configs" / "track_b.yaml"); tbo.dcfg = dcfg; tbo.results = {}
    T = tbo.build_sources()["rtk"].T_wc; C = T[:, :3, 3]
    centers = grid.centers(); zmed = float(np.nanmedian(S.z_best.values)); dth = np.empty(len(S), np.float32)
    for a0 in range(0, len(S), 5_000_000):                                         # chunked: bounded memory on large flights
        sl = slice(a0, a0 + 5_000_000); X = np.c_[centers[S.cell_id.values[sl]], np.nan_to_num(S.z_best.values[sl], nan=zmed)]
        v1 = C[S.i.values[sl]] - X; v2 = C[S.j.values[sl]] - X
        dth[sl] = np.degrees(np.arccos(np.clip((v1 * v2).sum(1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)), -1, 1)))
    S["dtheta_deg"] = dth; del X, v1, v2
    log("accumulating", len(S), "rows")
    F = accumulate(S)
    MR = fit_motion_reflectance(S[["cell_id", "dt_s", "dtheta_deg", "across_px", "cost_min", "gain"]], grid, cfg["motion"]["min_obs"], tuple(cfg["motion"]["pool_radii"]))
    del S                                                                    # free the cell x pair table before modelling
    F = F.join(MR[["geo_bt", "geo_bth", "pho_bt", "pho_bth", "gain_bth", "moving_score", "reflective_score", "n_obs_fit", "pool_radius"]], how="left")
    F = F[F.n_pairs >= cfg["min_pairs"]]
    log("cells:", len(F))
    # ---- physics / rule scores (0..1-ish, documented thresholds in configs/step3.yaml)
    pc = cfg["physics"]
    sc = pd.DataFrame(index=F.index)
    sc["featureless"] = np.clip(F.textureless_frac.fillna(1) + (F.ref_std_med.fillna(0) < pc["low_texture_std"]).astype(float) * 0.5, 0, 1)
    sc["repetitive"] = np.clip(F.periodic_frac.fillna(0) / pc["periodic_frac_full"], 0, 1)
    sc["moving"] = np.clip(F.moving_score.fillna(0) / pc["t_full"], 0, 1)
    # glint counts only when confirmed photometrically (brightness jump in the near-specular view), not on geometry alone
    glint_conf = F.glint_max.fillna(0) * np.clip(F.glint_gain_excess.fillna(0) / pc.get("glint_gain_full", 0.2), 0, 1)
    sc["reflective"] = np.clip(np.fmax(F.reflective_score.fillna(0) / pc["t_full"], glint_conf), 0, 1)
    sc["glint_geometric_only"] = F.glint_max.fillna(0)
    sc["weak_geometry"] = np.clip(1 - np.minimum(F.max_baseline_m.fillna(0) / pc["good_baseline_m"], 1) * np.minimum(F.n_valid / pc["good_n_pairs"], 1), 0, 1)
    sc["hidden_layered_physics"] = np.clip(np.fmax(F.long_cost_min_p50.fillna(0) - F.short_cost_min_p50.fillna(0), 0) / pc["layer_cost_jump"], 0, 1)
    # ---- proxy labels (comparison side) for trained heads
    L = pd.read_parquet(Path(ds.out) / "track_a" / "labels.parquet").set_index("cell_id").reindex(F.index); lc = cfg["labels"]
    lab = pd.DataFrame(index=F.index)
    good = L.gt_good == True; ae = L.error.abs()
    lab["failure"] = np.where(good & (ae >= lc["failure_min_abs_error"]), 1.0, np.where(good & (ae <= lc["healthy_max_abs_error"]), 0.0, np.nan))
    vspread = L.lidar_z_p90 - L.lidar_z_p10
    lab["layered"] = np.where(~L.no_gt.fillna(True).astype(bool) & (vspread >= lc["layered_min_spread"]), 1.0,
                              np.where(~L.no_gt.fillna(True).astype(bool) & (vspread <= lc["flat_max_spread"]), 0.0, np.nan))
    vc = vehicle_cells(ds, grid, np.where(ds.frames.is_reference.values)[0]); vcount = pd.Series(vc[F.index.values], index=F.index)
    lab["vehicle"] = np.where(vcount >= lc["vehicle_min_points"], 1.0, np.where(vcount == 0, 0.0, np.nan))
    # water: LiDAR semantic class only (independent of the Terra alignment); cells with a mixed class are left out
    wc = dcfg["classes"]["water"]; dom = L.dominant_class; frac = L.dominant_frac
    lab["water"] = np.where(dom.isin(wc) & (frac >= lc.get("water_min_frac", 0.6)), 1.0,
                            np.where(dom.notna() & ~dom.isin(wc), 0.0, np.nan))
    for k in dcfg.get("heads_exclude", []):                                  # labels unreliable on this scene (see config)
        if k in lab.columns:
            log(f"head {k}: excluded by scene config"); lab = lab.drop(columns=k)
    for k in list(lab.columns):                                              # a head needs positives in this scene
        if (lab[k] == 1).sum() < lc.get("min_positives", 30):
            log(f"head {k}: too few positives in this scene ({int((lab[k] == 1).sum())}), skipped"); lab = lab.drop(columns=k)
    res["label_counts"] = {k: dict(pos=int((lab[k] == 1).sum()), neg=int((lab[k] == 0).sum())) for k in lab}
    log("labels:", res["label_counts"])
    # ---- features & classifier (+ neighbourhood context: local means and deviations of every per-cell feature)
    if cfg.get("context", {}).get("windows"):
        F = add_context(F, grid, [c for c in F.columns if c not in ("n_obs_fit", "pool_radius")], tuple(cfg["context"]["windows"]))
        log("context features added:", F.shape[1])
    feat_cols = [c for c in F.columns if c not in ("n_obs_fit", "pool_radius")]
    Xm, cols, stats = prepare(F, feat_cols)
    oof, metrics = spatial_cv(F, Xm, lab, grid.centers()[F.index.values], cfg["cv"]["block_m"], cfg["cv"]["folds"], cfg["seed"], l2=cfg["cv"]["l2"])
    blocks = blocks_of(grid.centers()[F.index.values], cfg["cv"]["block_m"])
    res["cv_metrics"] = {l: full_metrics(lab[l].values, oof[l].values, blocks, cfg["seed"]) for l in lab.columns}
    log("cv:", res["cv_metrics"])
    # second model: gradient-boosted trees on the raw features (same folds; trees need no scaling, NaN has its own bin)
    Xraw = F[cols].astype(float).replace([np.inf, -np.inf], np.nan).values
    oof_gb, _ = spatial_cv(F, Xraw, lab, grid.centers()[F.index.values], cfg["cv"]["block_m"], cfg["cv"]["folds"], cfg["seed"],
                           fit=lambda X, y: fit_gbdt(X, y, **cfg.get("gbdt", {})), pred=predict_gbdt)
    res["cv_metrics_gbdt"] = {l: full_metrics(lab[l].values, oof_gb[l].values, blocks, cfg["seed"]) for l in lab.columns}
    log("cv (gbdt):", {l: round(v["auroc"], 3) for l, v in res["cv_metrics_gbdt"].items()})
    # confound check: is the failure head more than a vegetation-vs-pavement detector? metrics WITHIN class strata
    dc = L.dominant_class.reindex(F.index)
    veg, hard = dcfg["classes"]["vegetation"], dcfg["classes"]["healthy_class_based"]
    strata = {"vegetation": dc.isin(veg).values, "non-vegetation": (~dc.isin(veg)).values & dc.notna().values,
              "hard surfaces": dc.isin(hard).values, "background": dc.eq(0).values}
    if "failure" in lab.columns:
        res["failure_by_stratum"] = {k: full_metrics(np.where(m, lab.failure.values, np.nan), oof.failure.values, blocks, cfg["seed"])
                                     for k, m in strata.items()}
        if "failure" in oof_gb:
            res["failure_by_stratum_gbdt"] = {k: full_metrics(np.where(m, lab.failure.values, np.nan), oof_gb.failure.values, blocks, cfg["seed"])
                                              for k, m in strata.items()}
        res["failure_class_only_baseline"] = full_metrics(lab.failure.values, dc.isin(veg).astype(float).values, blocks, cfg["seed"])
        log("failure by stratum:", {k: (v["n"], v["positives"], v["auroc"]) for k, v in res["failure_by_stratum"].items()})
        log("class-only baseline (vegetation indicator; uses GT class, upper bound for a class detector):", res["failure_class_only_baseline"]["auroc"])
    # baselines: single signals, each oriented a priori by physics (higher value = more suspicious), never flipped
    # after looking at the labels
    single = {"low_texture (textureless_frac)": F.textureless_frac, "photometric (long 1-NCC p50)": F.long_cost_min_p50,
              "across (long |across| p50)": F.long_across_px_p50, "across (short |across| p50)": F.short_across_px_p50,
              "flat cost curve (-sharpness, long)": -F.long_sharpness_p50, "below-terrain fraction": F.below_frac,
              "coverage (-n_valid)": -F.n_valid.astype(float), "repetitive (periodic_frac)": F.periodic_frac}
    res["baselines"] = {l: {name: full_metrics(lab[l].values, v.values.astype(float), blocks, cfg["seed"]) for name, v in single.items()}
                        for l in lab.columns}
    res["physics_scores"] = {l: {s_: full_metrics(lab[l].values, sc[s_].values.astype(float), blocks, cfg["seed"]) for s_ in sc.columns}
                             for l in lab.columns}
        # signal-group ablation for the failure head (drop one group at a time)
    groups = {"across": [c for c in cols if "across" in c] + ["geo_bt", "geo_bth", "moving_score"],
              "curve": [c for c in cols if any(k in c for k in ("sharpness", "flatness", "periodic", "multi_min", "boundary"))],
              "photometric": [c for c in cols if any(k in c for k in ("cost_min", "gain", "contrast", "pho_", "reflective"))],
              "height": [c for c in cols if any(k in c for k in ("dz_sigma", "below"))],
              "glint": [c for c in cols if c.startswith("glint")],
              "coverage/texture": [c for c in cols if c.split("_ctx")[0].split("_dev")[0] in ("n_pairs", "n_valid", "textureless_frac", "ref_std_med", "max_baseline_m", "max_heading_diff")]}
    abl = {}
    for gname, gcols in groups.items():
        keep = [c for c in feat_cols if c not in gcols]
        Xg, _, _ = prepare(F, keep); _, mg = spatial_cv(F, Xg, lab[[c for c in ("failure", "layered", "water") if c in lab.columns]], grid.centers()[F.index.values], cfg["cv"]["block_m"], cfg["cv"]["folds"], cfg["seed"], l2=cfg["cv"]["l2"])
        abl[f"without_{gname}"] = {k: v["auroc"] for k, v in mg.items()}
    res["ablation_auroc"] = abl; log("ablation:", abl)
    # ---- how much: error magnitude + conformal upper bounds on reliable-GT cells (labels: comparison side only)
    mag_cols = {}
    if "failure" in lab.columns:
        gm = ((L.gt_good == True) & L.error.notna()).values
        if gm.sum() >= 500:
            pm, Ug, Um, res["magnitude"] = magnitude_cv(Xraw[gm], np.abs(L.error.values[gm]), blocks[gm], cfg["cv"]["folds"], cfg["seed"],
                                                        cfg.get("magnitude", {}).get("alpha", 0.9), gbdt_kw=cfg.get("gbdt", {}))
            for nm, v in [("mag_pred_m", pm), ("mag_upper_m", Um)]:
                col = np.full(len(F), np.nan); col[gm] = v; mag_cols[nm] = col
            log("magnitude:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res["magnitude"].items() if k != "global"})
    # ---- final models on all labelled cells -> predictions for every cell
    pred = pd.DataFrame(index=F.index)
    for l in lab.columns:
        y = lab[l].values; m = np.isfinite(y)
        pred[f"p_{l}"] = predict(fit_logreg(Xm[m], y[m], l2=cfg["cv"]["l2"]), Xm) if (y[m] == 1).sum() >= 5 else np.nan
    D = pd.concat([sc.add_prefix("cause_"), pred, oof.add_prefix("oof_"), oof_gb.add_prefix("oofgb_"), F[["n_pairs", "n_valid", "moving_score", "reflective_score", "pool_radius"]]], axis=1)
    for nm, v in mag_cols.items():
        D[nm] = v
    D["x"] = grid.centers()[D.index.values, 0]; D["y"] = grid.centers()[D.index.values, 1]
    D.to_parquet(out / "cell_diagnosis.parquet"); F.to_parquet(out / "cell_features.parquet"); lab.to_parquet(out / "proxy_labels.parquet")
    # maps
    from ..analysis.plots import heatmap, SEQ
    for c in [c for c in D.columns if c.startswith("cause_") or c.startswith("p_")]:
        v = np.full(grid.n, np.nan); v[D.index.values] = D[c].values; heatmap(grid, v, out / f"map_{c}.png", c, "score / probability", SEQ, 0, 1)
    (out / "results.json").write_text(json.dumps(res, indent=2, default=float)); write_params(out, "step3", dict(data=dcfg, step3=cfg))
    log("done")

if __name__ == "__main__":
    main()
