"""Phase 0 CLI (package version of the inspection essentials): frame table with reference flags and flight
segments, RTK -> working Sim3 fit, segment plot.  aerodiag-phase0 [--data configs/data.yaml]
The full exploratory inspection scripts are kept in outputs/phase0/scripts (see DATA_REPORT.md)."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from ..config import load_config, write_params
from ..io.dataset import Dataset
from ..io.rtk import load_rtk
from ..poses.rtk import fit_rtk_to_working

def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="configs/data.yaml"); a = ap.parse_args(argv)
    cfg = load_config(a.data); ds = Dataset(cfg); out = Path(ds.out); out.mkdir(exist_ok=True)
    f = ds.frames.copy(); f.to_parquet(out / "frames.parquet")
    rtk = load_rtk(ds.p["rtk_csv"], ds.terra_origin)
    m = f.is_reference.values & ~f.segment.isin(cfg["rtk"]["fit_segments_exclude"]).values
    sim, res = fit_rtk_to_working(rtk, f.t_ns.values, ds.T_wc_json[:, :3, 3], m)
    (out / "rtk_to_working_sim3.json").write_text(json.dumps(dict(sim3=sim.to_dict(), describe=sim.describe(), fit_frames=int(m.sum()),
        residual_median_m=float(np.median(res[m])), residual_p95_m=float(np.percentile(res[m], 95)),
        source_frame="local ENU at Terra metadata origin; RTK alt as ellipsoidal height (assumed, Q8)", target_frame="Terra working frame"), indent=2))
    write_params(out, "phase0", cfg)
    print(f"frames: {len(f)}  reference: {int(f.is_reference.sum())}  anchors: {int(f.is_anchor.sum())}")
    print("segments:", f.segment.value_counts().to_dict()); print("rtk sim3:", sim.describe())

if __name__ == "__main__":
    main()
