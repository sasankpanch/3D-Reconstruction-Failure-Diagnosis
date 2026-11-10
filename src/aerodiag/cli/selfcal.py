"""Camera self-calibration from onboard data only (images + RTK); LiDAR and Terra are never inputs.

aerodiag-selfcal [--data configs/data.yaml] [--cfg configs/selfcal.yaml]

Calibration frames: time grid (every 0.5 s) in RTK-derived segments climb / hover / descent / diagonal. All parallel
cruise lines are HELD OUT (used later for evaluation). Estimated: f, k1, k2, k3 (+ antenna-camera lever arm, per-run RTK
offsets, per-frame rotations, points). cx, cy, p1, p2 stay at the lab calibration. Robustness: re-solve from a perturbed
start (rotations +-2 deg, f +-3 %, distortion zeroed) and report the spread."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial.transform import Rotation as Rot
from ..config import load_config, write_params
from ..io.dataset import Dataset
from ..geometry.camera import CameraModel
from ..calib.onboard import rtk_enu, rtk_segments, init_rotations_enu, match_pairs, homography_thresholds, bundle_from_pairs
from ..poses.rtk import fit_rtk_to_working
from ..io.rtk import load_rtk

def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)

def calibration_pairs(seg: pd.DataFrame, enu, agl, cfg):
    sc = cfg["selection"]; grid = seg.time_grid.values
    cal = np.where(grid & seg.segment.isin(sc["segments"]).values)[0]
    # thin long hovers (near-identical views)
    keep = []
    for r in np.unique(seg.run.values[cal]):
        fr = cal[seg.run.values[cal] == r]
        step = sc["hover_stride"] if seg.segment.values[fr[0]] == "hover" else 1
        keep.append(fr[::step])
    cal = np.concatenate(keep); pairs = set()
    runs = seg.run.values
    for r in np.unique(runs[cal]):
        fr = cal[runs[cal] == r]
        for k, i in enumerate(fr):
            for g in sc["within_run_steps"]:
                if k + g < len(fr):
                    pairs.add((int(i), int(fr[k + g])))
    for i in cal:                                            # cross-run links (overlapping footprints, any altitude)
        for r in np.unique(runs[cal]):
            if r == runs[i]:
                continue
            fr = cal[runs[cal] == r]; d = np.linalg.norm(enu[fr, :2] - enu[i, :2], axis=1)
            lim = sc["cross_run_overlap"] * min(agl[i], 80.0)
            for j in fr[np.argsort(d)[: sc["cross_run_links"]]]:
                if np.linalg.norm(enu[j, :2] - enu[i, :2]) < lim:
                    pairs.add((int(min(i, j)), int(max(i, j))))
    return cal, np.array(sorted(pairs))

def run(dcfg, cfg):
    ds = Dataset(dcfg); out = Path(ds.out) / "selfcal"; out.mkdir(parents=True, exist_ok=True); res = {}
    enu = rtk_enu(ds); seg = rtk_segments(ds, enu)
    rtk = load_rtk(ds.p["rtk_csv"], ds.terra_origin)
    ground = rtk.u.values[0]                                  # home point (on the ground before take-off)
    agl = enu[:, 2] - ground
    # attitude initialisation only (stand-in IMU); rotate working->ENU with the nominal frame transform
    f = ds.frames; sim, _ = fit_rtk_to_working(rtk, f.t_ns.values, ds.T_wc_json[:, :3, 3], f.is_reference.values)
    R0 = init_rotations_enu(ds, sim.R)
    cal, pairs = calibration_pairs(seg, enu, agl, cfg)
    res["selection"] = dict(n_frames=int(len(cal)), segments=seg.segment.values[cal].tolist() and pd.Series(seg.segment.values[cal]).value_counts().to_dict(),
                            n_pairs=int(len(pairs)), agl_range_m=[float(agl[cal].min()), float(agl[cal].max())])
    log("selection:", res["selection"])
    mcfg = dict(features=cfg["features"], matching=cfg["matching"], seed=cfg["seed"], _homog_thr=homography_thresholds(pairs, enu, 1444.0, agl))
    P, k1, k2, pom, info = match_pairs(ds, pairs, mcfg, Path(ds.out) / "cache"); info.to_csv(out / "calibration_pairs.csv", index=False)
    log("matches:", int(len(k1)), "pairs with >=40:", int((info.n >= 40).sum()))
    lab = ds.calib; cam_lab = CameraModel(lab.K, lab.dist, 2448, 2048)
    climb_run = int(pd.Series(seg.run.values[cal][seg.segment.values[cal] == "climb"]).mode()[0])
    def solve(R_init, cam_init, tag):
        bp, fsel, tinfo = bundle_from_pairs(ds, cal, P, k1, k2, pom, enu, R_init, seg.run.values, cam_init,
                                            estimate=cfg["estimate"], estimate_lever=True, fixed_segment=climb_run,
                                            max_tracks=cfg["max_tracks"], seed=cfg["seed"])
        log(tag, "tracks:", tinfo)
        hist = bp.solve(rounds=cfg["rounds"], log=lambda h: log(tag, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in h.items()}))
        return bp, fsel, tinfo, hist
    bp, fsel, tinfo, hist = solve(R0, cam_lab, "main")
    rng = np.random.default_rng(cfg["seed"] + 1)
    Rp = np.einsum("fij,fjk->fik", R0, Rot.from_rotvec(rng.normal(0, np.radians(cfg["robustness"]["rot_deg"]), (len(R0), 3))).as_matrix())
    Kp = lab.K.copy(); Kp[0, 0] = Kp[1, 1] = lab.K[0, 0] * (1 + cfg["robustness"]["f_rel"])
    cam_p = CameraModel(Kp, np.r_[0.0, 0.0, lab.dist[2], lab.dist[3], 0.0], 2448, 2048)
    bp2, _, _, hist2 = solve(Rp, cam_p, "perturbed")
    # compare distortion fields (px) between lab, main and perturbed solutions
    xn = np.stack(np.meshgrid(np.linspace(-0.85, 0.85, 31), np.linspace(-0.72, 0.72, 27)), -1).reshape(-1, 2)
    cm, cp = bp.camera(), bp2.camera()
    uv_m, uv_p, uv_l = cm.normalized_to_pixel(xn), cp.normalized_to_pixel(xn), cam_lab.normalized_to_pixel(xn)
    inside = (uv_l[:, 0] > 0) & (uv_l[:, 0] < 2448) & (uv_l[:, 1] > 0) & (uv_l[:, 1] < 2048)
    r = np.hypot(uv_l[:, 0] - lab.K[0, 2], uv_l[:, 1] - lab.K[1, 2])
    res["intrinsics"] = dict(model="pinhole + Brown-Conrady (OpenCV k1,k2,p1,p2,k3)", f=bp.intr["f"], cx=float(lab.K[0, 2]), cy=float(lab.K[1, 2]),
                             k1=bp.intr["k1"], k2=bp.intr["k2"], k3=bp.intr["k3"], p1=float(lab.dist[2]), p2=float(lab.dist[3]),
                             lever_arm_camera_frame_m=bp.l.tolist(), estimated=list(cfg["estimate"]), fixed_from_lab=["cx", "cy", "p1", "p2"])
    res["lab"] = dict(f=float(lab.K[0, 0]), k1=float(lab.dist[0]), k2=float(lab.dist[1]), k3=float(lab.dist[4]))
    res["robustness"] = dict(perturbation=cfg["robustness"], f_main=bp.intr["f"], f_perturbed=bp2.intr["f"],
                             max_field_diff_main_vs_perturbed_px=float(np.abs(uv_m - uv_p)[inside].max()),
                             lever_perturbed=bp2.l.tolist())
    res["field_vs_lab_px"] = {f"r<{b}": float(np.abs(uv_m - uv_l)[inside & (r < b) & (r >= a)].max()) for a, b in [(0, 400), (400, 800), (800, 1200), (1200, 1700)]}
    res["solve"] = dict(tracks=tinfo, final=hist[-1], rounds=len(hist), segment_offsets_m=bp.o.round(3).tolist())
    (out / "intrinsics.json").write_text(json.dumps(res["intrinsics"], indent=2))
    (out / "results.json").write_text(json.dumps(res, indent=2, default=float))
    pd.DataFrame(hist).to_csv(out / "history_main.csv", index=False); pd.DataFrame(hist2).to_csv(out / "history_perturbed.csv", index=False)
    write_params(out, "selfcal", dict(data=dcfg, selfcal=cfg))
    log("intrinsics:", res["intrinsics"]); log("robustness:", res["robustness"]); log("field vs lab:", res["field_vs_lab_px"])
    return res

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="configs/data.yaml"); ap.add_argument("--cfg", default="configs/selfcal.yaml")
    a = ap.parse_args(argv); run(load_config(a.data), load_config(a.cfg))

if __name__ == "__main__":
    main()
