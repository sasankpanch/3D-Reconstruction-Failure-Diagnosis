"""Cross-flight summary: one row per flight (label quality, noise floor, detection, magnitude, calibration, timing).

aerodiag-flights --configs configs/data.yaml configs/data_airport_01.yaml ...   ->  outputs/flights/FLIGHTS.md, flights.csv
Label-quality columns come from Track A (Terra <-> LiDAR alignment of THIS flight's LiDAR), so they show whether the
shared per-site Terra model is a valid failure reference for each flight."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from ..config import load_config

def _j(p):
    p = Path(p); return json.loads(p.read_text()) if p.exists() else {}

def row(cfg_path):
    d = load_config(cfg_path); o = Path(d["root"]) / d["paths"]["outputs"]; r = dict(flight=d["scene"], config=cfg_path)
    A = _j(o / "track_a" / "results.json"); al = A.get("align", {}); g = A.get("grid", {}); st = A.get("stats", {}); gg = A.get("stats_gt_good", {}).get("overall", {})
    icp = al.get("icp", [{}])[-1] if al.get("icp") else {}
    r.update(frames=None, gt_cells=g.get("cells_gt"), stable_cells=g.get("cells_stable"), terra_coverage=1 - st.get("cells_gt_without_terra", np.nan) / max(st.get("cells_gt", 1), 1),
             icp_fitness_25cm=icp.get("fitness"), icp_rmse_m=icp.get("rmse"), gt_good_cells=gg.get("cells"), terra_abs_err_p50_m=gg.get("abs_p50"),
             frac_err_gt_0_5m=gg.get("frac_abs_gt_0_5"))
    fr = o / "frames.parquet"
    if fr.exists():
        r["frames"] = len(pd.read_parquet(fr, columns=["t_ns"]))
    N = o / "step2" / "eval_across_noise_floor_paved.csv"
    if N.exists():
        N = pd.read_csv(N).set_index("grp")
        for k in ("within:20", "within:40", "cross_line"):
            if k in N.index:
                r[f"across_p90_{k}"] = N.p90[k]
    E = _j(o / "step2" / "results.json"); r["sweep_pairs"] = E.get("sweep", {}).get("pairs"); r["sweep_rows_M"] = (E.get("sweep", {}).get("rows") or np.nan) / 1e6
    S3 = _j(o / "step3" / "results.json")
    for h, v in S3.get("cv_metrics_gbdt", {}).items():
        if v.get("auroc") == v.get("auroc"):
            r[f"{h}_auroc"] = v["auroc"]; r[f"{h}_ci"] = v["auroc_ci95"]; r[f"{h}_ap"] = v["ap"]; r[f"{h}_base"] = v["base_rate"]; r[f"{h}_pos"] = v["positives"]
    if "failure_class_only_baseline" in S3:
        r["class_only_auroc"] = S3["failure_class_only_baseline"]["auroc"]
    m = S3.get("magnitude", {}).get("mondrian", {})
    if m:
        r.update(mag_coverage=m["coverage"], mag_bound_m=m["median_bound_m"], mag_certified=m["frac_certified_below_0p5m"], mag_cov_fail=m["coverage_on_failures"])
    C = _j(o / "step3" / "calibration_gbdt.json")
    if "failure" in C:
        r["failure_ece_cal"] = C["failure"]["ece_cal"]
    T = o / "timings.csv"
    if T.exists():
        T = pd.read_csv(T); r.update({f"t_{s}_s": float(t) for s, t in zip(T.stage, T.seconds)}); r["t_total_min"] = float(T.seconds.sum() / 60)
        r["onboard_min"] = float(T[T.stage.isin(["track_b", "step2", "step3"])].seconds.sum() / 60)
    return r

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--configs", nargs="+", required=True); ap.add_argument("--out", default="outputs/flights")
    a = ap.parse_args(argv); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    D = pd.DataFrame([row(c) for c in a.configs]); D.to_csv(out / "flights.csv", index=False)
    f = lambda v: (f"{v:.3f}" if isinstance(v, float) and abs(v) < 100 else (f"{v:.0f}" if isinstance(v, float) else (f"[{v[0]:.2f}, {v[1]:.2f}]" if isinstance(v, (list, tuple)) else ("" if v is None else str(v)))))
    md = ["# All flights", ""]
    for title, cols in [("Labels (Terra <-> this flight's LiDAR)", ["flight", "frames", "gt_cells", "stable_cells", "terra_coverage", "icp_fitness_25cm", "icp_rmse_m", "gt_good_cells", "terra_abs_err_p50_m", "frac_err_gt_0_5m"]),
                        ("Detection (spatial CV, boosted trees)", ["flight", "failure_auroc", "failure_ci", "failure_ap", "failure_base", "class_only_auroc", "layered_auroc", "vehicle_auroc", "vehicle_ap", "water_auroc", "water_ap"]),
                        ("How much / calibration / noise floor", ["flight", "mag_coverage", "mag_bound_m", "mag_certified", "mag_cov_fail", "failure_ece_cal", "across_p90_within:20", "across_p90_within:40", "across_p90_cross_line"]),
                        ("Timing (fresh run)", ["flight", "frames", "sweep_pairs", "sweep_rows_M", "t_track_a_s", "t_track_b_s", "t_step2_s", "t_step3_s", "onboard_min", "t_total_min"])]:
        cs = [c for c in cols if c in D.columns]; md += [f"## {title}", "", "| " + " | ".join(cs) + " |", "|" + "---|" * len(cs)]
        md += ["| " + " | ".join(f(r[c]) if not (isinstance(r[c], float) and np.isnan(r[c])) else "–" for c in cs) + " |" for _, r in D.iterrows()]; md.append("")
    (out / "FLIGHTS.md").write_text("\n".join(md)); print("\n".join(md))

if __name__ == "__main__":
    main()
