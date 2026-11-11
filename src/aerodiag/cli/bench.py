"""Onboard-path benchmark: time every step the aircraft would run, per image and per pair, CPU vs GPU.

aerodiag-bench --data configs/data.yaml [--pairs 40] [--images 40]

Steps (onboard only; training, Track A and all evaluation are excluded):
  decode (JPEG -> grey), features (OpenCV SIFT, one CPU thread), matching (2-NN + filters + homography),
  rotation refinement, plane sweep (all candidate cells of the pair), accumulation (per-cell statistics),
  inference (boosted trees on per-cell features).
Every GPU timing is synchronised; the first call of each GPU step is a warm-up and is not timed.
Per-image totals: dataset spacing uses this scene's pairs-per-keyframe; survey spacing assumes 5 pairs per image
(3-4 in-line partners + ~2 cross-line). Output: <outputs>/bench/bench_onboard.json, BENCH.md"""
from __future__ import annotations
import argparse, json, time, platform, subprocess
from pathlib import Path
import numpy as np, pandas as pd, cv2

from ..config import load_config
from .. import backend

def _sync():
    if backend.use_cuda():
        import torch
        torch.cuda.synchronize()

def timed(fn, reps=1):
    _sync(); t = time.perf_counter()
    for _ in range(reps):
        out = fn()
    _sync(); return (time.perf_counter() - t) / reps, out

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="configs/data.yaml"); ap.add_argument("--pairs", type=int, default=40); ap.add_argument("--images", type=int, default=40)
    a = ap.parse_args(argv); dcfg = load_config(a.data)
    from .step2 import Step2
    from ..signals.sweep import sweep_pair, _project
    from ..matching.features import FeatureStore
    from ..matching.match import match_pair_full
    from ..geometry.epipolar import relative_pose
    from ..poses.refine import refine_rotation
    from ..diagnosis.gbdt import fit_gbdt, predict_gbdt
    s = Step2(dcfg, load_config(Path(dcfg["root"]) / "configs" / "step2.yaml")); ds, cam, cfg = s.ds, s.cam, s.cfg
    out = Path(ds.out) / "bench"; out.mkdir(parents=True, exist_ok=True); R = {}
    rng = np.random.default_rng(0)
    P = s.P[s.P.pair_type.isin(["within_line", "cross_line", "crossing"])]; P = P.loc[rng.choice(P.index, min(a.pairs, len(P)), replace=False)]
    imgs = np.unique(np.r_[P.i, P.j])[: a.images]
    # ---- decode + features (CPU, one thread; OpenCV SIFT is the default extractor)
    paths = [ds.p["cam_dir"] / f"{ds.frames.name.iat[int(i)]}.jpg" for i in imgs]
    cv2.setNumThreads(1); td = [timed(lambda p=p: cv2.imread(str(p), cv2.IMREAD_GRAYSCALE))[0] for p in paths]
    sift = cv2.SIFT_create(nfeatures=8000); G = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in paths[:10]]
    tf = [timed(lambda g=g: sift.detectAndCompute(g, None))[0] for g in G]; cv2.setNumThreads(-1)
    R["decode_s_per_image"] = float(np.median(td)); R["sift_cpu_1thread_s_per_image"] = float(np.median(tf))
    fs = FeatureStore(ds, Path(ds.out) / "cache", cfg["features"]["max_features"]); fs.ensure(np.r_[P.i, P.j])
    tb = load_config(Path(dcfg["root"]) / "configs" / "track_b.yaml")
    F = {int(i): fs.load(int(i)) for i in np.unique(np.r_[P.i, P.j])}
    # poses + terrain once (needed for the sweep inputs)
    s.stage_poses(); s.stage_terrain(); cells = s.cells
    for mode in ("cpu", "cuda"):
        backend.set_backend(mode)
        if mode == "cuda" and not backend.use_cuda():
            continue
        r = {}
        # ---- matching
        pr0 = list(P.index[:2]); [match_pair_full(F[int(P.i[p])], F[int(P.j[p])], tb["matching"], 300.0, np.random.default_rng(p)) for p in pr0]
        tm = [timed(lambda p=p: match_pair_full(F[int(P.i[p])], F[int(P.j[p])], tb["matching"], 300.0, np.random.default_rng(p)))[0] for p in P.index]
        r["matching_s_per_pair"] = float(np.median(tm))
        # ---- rotation refinement (per pair on CPU; batches of 5 on GPU = one survey image)
        items = []
        for p in P.index:
            pr, _ = match_pair_full(F[int(P.i[p])], F[int(P.j[p])], tb["matching"], 300.0, np.random.default_rng(p))
            if len(pr) >= 40:
                i, j = int(P.i[p]), int(P.j[p]); x1 = cam.pixel_to_normalized(F[i]["uv"][pr[:, 0]].astype(float)); x2 = cam.pixel_to_normalized(F[j]["uv"][pr[:, 1]].astype(float))
                Rr, t = relative_pose(s.T[i], s.T[j]); items.append((x1, x2, Rr, t))
        if mode == "cpu":
            tr = [timed(lambda it=it: refine_rotation(*it, cam.f, rng=np.random.default_rng(0)))[0] for it in items]; r["refine_s_per_pair"] = float(np.median(tr))
        else:
            from ..poses.refine_cuda import refine_rotation_batch
            refine_rotation_batch(items[:5], cam.f, rngs=[np.random.default_rng(0)] * 5)
            tb5 = [timed(lambda k=k: refine_rotation_batch(items[k:k + 5], cam.f, rngs=[np.random.default_rng(0)] * len(items[k:k + 5])))[0] / len(items[k:k + 5]) for k in range(0, len(items) - 4, 5)]
            r["refine_s_per_pair"] = float(np.median(tb5))
        # ---- sweep (all candidate cells of each pair, as step 2 does)
        ts, nc = [], []
        for n, p in enumerate(P.index[: min(len(P), 15)]):
            i, j = int(P.i[p]), int(P.j[p])
            if p not in s.Tp:
                continue
            T1, T2 = s.T[i], s.Tp[p]; X = np.c_[cells["xy"], cells["z0"]]; m = np.ones(len(X), bool); mg = cfg["sweep"]["margin_px"]
            for T in (T1, T2):
                uv = _project(cam, T[:3, :3], T[:3, 3], X); m &= (uv[:, 0] > mg) & (uv[:, 0] < cam.width - mg) & (uv[:, 1] > mg) & (uv[:, 1] < cam.height - mg)
            idx = np.where(m)[0]; im1 = ds.load_image(i, gray=True).astype(np.float32); im2 = ds.load_image(j, gray=True).astype(np.float32)
            sun = (np.array([0.3, 0.3, 0.9]),) * 2
            def run():
                g = None
                if mode == "cuda":
                    from ..signals.sweep_gpu import GpuSweep
                    g = GpuSweep(cam, im1, im2, precision=cfg["sweep"].get("cuda_precision", "float32"))
                ch = cfg["sweep"].get("cuda_chunk_cells", 4000) if g is not None else cfg["sweep"]["chunk_cells"]
                for s0 in range(0, len(idx), ch):
                    k = idx[s0:s0 + ch]
                    sweep_pair(im1, im2, cam, T1[:3, :3], T1[:3, 3], T2[:3, :3], T2[:3, 3], cells["xy"][k], cells["z0"][k], cells["n"][k], sun, cfg["sweep"], gpu=g)
            if n == 0:
                run()                                                    # warm-up
            ts.append(timed(run)[0]); nc.append(len(idx))
        r["sweep_s_per_pair"] = float(np.median(ts)); r["sweep_cells_per_pair"] = float(np.median(nc))
        r["sweep_s_per_1k_cells"] = float(np.median(np.array(ts) / np.array(nc) * 1000))
        # ---- accumulation (1M cell-pair rows) and inference (10k cells)
        S = pd.read_parquet(Path(ds.out) / "step2" / "cell_pair_signals.parquet").iloc[:1_000_000]
        from ..diagnosis.accumulate import accumulate
        accumulate(S.iloc[:2000]); r["accumulate_s_per_1M_rows"] = timed(lambda: accumulate(S))[0]
        Fe = pd.read_parquet(Path(ds.out) / "step3" / "cell_features.parquet"); X = Fe.astype(float).replace([np.inf, -np.inf], np.nan).values[:10000]
        y = (np.random.default_rng(0).uniform(size=len(X)) < 0.2).astype(float); mdl = fit_gbdt(X, y, rounds=150)
        predict_gbdt(mdl, X[:100]); r["inference_s_per_10k_cells"] = timed(lambda: predict_gbdt(mdl, X))[0]
        R[mode] = r; print(mode, {k: round(v, 4) for k, v in r.items()}, flush=True)
    # ---- per-image totals
    kf = len(np.unique(np.r_[s.P.i, s.P.j])); ppi_data = len(s.P) / max(kf, 1)
    for mode in [m for m in ("cpu", "cuda") if m in R]:
        r = R[mode]; per_pair = r["matching_s_per_pair"] + r["refine_s_per_pair"] + r["sweep_s_per_pair"]
        per_img_fixed = R["decode_s_per_image"] + r["inference_s_per_10k_cells"] * 1.5          # ~15k new cells per image
        r["per_image_dataset_spacing_s"] = per_img_fixed + ppi_data * per_pair; r["pairs_per_image_dataset"] = ppi_data
        r["per_image_survey_spacing_s"] = per_img_fixed + 5 * per_pair
        r["sift_runs_in_parallel_on_cpu_s"] = R["sift_cpu_1thread_s_per_image"]
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip() or "none"
    R["hardware"] = dict(gpu=gpu, cpu=platform.processor() or subprocess.run(["bash", "-c", "lscpu | sed -n 's/^Model name: *//p'"], capture_output=True, text=True).stdout.strip())
    (out / "bench_onboard.json").write_text(json.dumps(R, indent=2))
    rows = ["decode_s_per_image", "sift_cpu_1thread_s_per_image"]
    md = [f"# Onboard benchmark — {dcfg['scene']}", "", f"Hardware: {R['hardware']}", "", "| step | CPU | GPU |", "|---|---|---|"]
    md += [f"| {k} | {R[k]:.4f} | (CPU) |" for k in rows]
    for k in R.get("cpu", {}):
        md.append(f"| {k} | {R['cpu'][k]:.4f} | {R.get('cuda', {}).get(k, float('nan')):.4f} |")
    (out / "BENCH.md").write_text("\n".join(md)); print("\n".join(md))

if __name__ == "__main__":
    main()
