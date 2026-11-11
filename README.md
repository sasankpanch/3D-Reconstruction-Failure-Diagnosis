# 3D Reconstruction Failure Diagnosis

Pre-reconstruction failure diagnosis for aerial (UAV) mapping: for every 1 m ground cell, predict **before**
reconstruction whether photogrammetry will fail there and **why**, using two-view epipolar geometry.

For a static, solid scene a correspondence may only move *along* its epipolar line (that component encodes height).
Motion *across* the line signals a violated assumption: moving or reflective surfaces, occlusion, layered structure.
Pairing views by time gap and viewing angle separates "it moved" from "it looks different from another direction";
sun position anticipates glare.

**Inputs (onboard only):** images, timestamps, RTK positions, camera model. LiDAR is used **only** as ground truth
for evaluation; the DJI Terra reconstruction is the model under test and is never used as ground truth or as an input.

## Results (UAVScenes)

Spatial 5-fold cross-validation over 25 m blocks, 95% block-bootstrap intervals. Failure = |Terra − LiDAR| ≥ 0.5 m.
Model: gradient-boosted trees on per-cell signals plus their 3×3 / 7×7 neighbourhood context.

| Prediction | HKairport_GNSS03 | AMtown03 | HKisland_GNSS03 |
|---|---|---|---|
| Failure location (AUROC) | 0.94 [0.91, 0.96] | 0.92 [0.91, 0.94] | – ¹ |
| Failure AP (base rate 0.22) | 0.76 | 0.80 | – |
| Failure inside / outside vegetation | 0.92 / 0.92 | 0.92 / 0.92 | – |
| Land-cover class alone (failure) | 0.66 | 0.46 | – |
| Layered structure | 0.98 | 0.95 | – |
| Vehicles (AP, base rate 0.002) | 0.93 (0.26) | 0.89 (0.04) | – |
| Water (AP, base rate 0.07) | – (none) | – (none) | 0.95 (0.69) |

**How much:** per-cell error magnitude with a spatially cross-fitted conformal 90% upper bound. Coverage 0.90 on both
scenes, median bound 0.32 m (airport) / 0.38 m (town); 70% / 59% of cells are certified below 0.5 m, 97% of them
correctly. Coverage on cells that do fail is lower (0.65).

**Calibration:** cross-fitted Platt scaling brings the expected calibration error of the failure / layered / water heads
to ≤ 0.02. Vehicle probabilities stay near the base rate; use them as a ranking.

**Cross-scene transfer** (train on one scene, test on the other, no retraining), failure AUROC:
town → airport 0.90, airport → town 0.79 (`outputs/transfer/TRANSFER.md`).

¹ HKisland is almost entirely vegetation and sea; too few hard surfaces to align the Terra model to LiDAR reliably,
so only the water head (label from LiDAR semantic classes) is trained there (`heads_exclude` in its config).

Across-epipolar noise floor on healthy ground with RTK + per-pair rotation refinement: p90 below 1 px within a flight
line (1 px ≈ 5.5 cm at 80 m). Across lines RTK carries a camera–RTK time offset (−0.10 s airport, +0.08 s town) and a
per-line bias; `src/aerodiag/calib/line_offsets.py` estimates both from images + RTK only (cross-line p90 on airport
3.4 → 1.4 px on held-out pairs). Detailed reports: `outputs/DATA_REPORT.md`, `outputs/track_a/LABELS_REPORT.md`,
`outputs/track_b/NOISE_FLOOR_REPORT.md`, `outputs/PIPELINE_RESULTS.md`, `outputs/transfer/TRANSFER.md`.

## Pipeline

| Stage | CLI | What it does |
|---|---|---|
| Phase 0 | `aerodiag-phase0` | Frame table, registered keyframes, flight segmentation, RTK → working-frame fit |
| Track A | `aerodiag-track-a` | LiDAR ground-truth map, per-cell reliability, Terra ↔ LiDAR robust ICP, error labels |
| Track B | `aerodiag-track-b` | Feature matching, rotation-only pose refinement, across-epipolar noise floor (go/no-go gate) |
| Step 2 | `aerodiag-step2` | Coarse terrain triangulated from images; per-cell plane sweep along epipolar lines; five signals (across residual, cost-curve shape, photometric, height plausibility, sun glint) |
| Step 3 | `aerodiag-step3` | Per-cell accumulation by baseline class, neighbourhood context, motion / view-dependence fits, logistic + boosted-tree heads, error magnitude with conformal bounds, spatial CV, baselines, ablations |
| Calibrate | `aerodiag-calibrate` | Cross-fitted Platt scaling, reliability diagrams |
| Summary | `aerodiag-summary` | Paper metrics → `outputs/PIPELINE_RESULTS.md` |
| Transfer | `aerodiag-transfer` | Cross-scene evaluation (both models, training-scene or within-scene rank normalisation) |
| Line offsets | `python -m aerodiag.calib.line_offsets <cfg>` | Camera–RTK time offset + per-line RTK bias from per-line terrain alignment (optional; `configs/data_lo.yaml`) |

## Setup

```bash
python -m venv .venv && .venv/bin/pip install -e .
# optional, GPU plane sweep (~40x faster per pair; identical signals, CPU path stays the reference):
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pytest -q tests
```

**Compute backend.** `AERODIAG_BACKEND=auto` (default) runs every onboard stage and the step-3 training on the GPU
through PyTorch when one is available; `cpu` forces the numpy/OpenCV reference code. Each CUDA path is an exact port of
its CPU reference and is checked against it in `tests/` (agreement to float rounding):

| Stage | GPU (CUDA) | CPU only (reason) |
|---|---|---|
| Track B / step 2 matching | descriptor 2-NN + ratio + mutual, local-consistency filter | homography outlier filter (one small MAGSAC call per pair) |
| Step 2 poses | batched rotation-only refinement (all pairs in one solve), undistortion | — |
| Step 2 sweep | projection, sampling, NCC, across search | cost-curve shape (small per-cell arrays) |
| Step 3 | accumulation, motion / view fits, logistic regression, boosted trees, conformal head, bootstrap CIs | pandas bookkeeping, plots |
| Features | optional Kornia SIFT (`features_type: sift_cuda`; 5x faster, equal results on airport, but half the matches on opposite-heading cross-line pairs, so not yet the default) | OpenCV SIFT (default, reference) |
| Track A | — | LiDAR / Terra ground truth (evaluation only, never onboard) |

Airport, RTX 5080 vs 12-core CPU: step 2 45 min → 5.3 min, step 3 45 min → 49 s; Track B + step 2 + step 3 end to end in under 8 min.

Data: download the UAVScenes sequences (images, LiDAR, LiDAR labels, `sampleinfos_interpolated.json`,
`rtk_positions_raw.csv`, DJI Terra model) and `calibration_results.py`, and place them as

```
data/airport/{interval1_HKairport_GNSS03, interval1_HKairport_GNSS03_Labels, HKairport_GNSS_Terra}
data/island/{interval1_HKisland_GNSS03,  interval1_HKisland_GNSS03_Labels,  HKisland_GNSS}
data/town/{interval1_AMtown03,           interval1_AMtown03_Labels,          AMtown}
```

Set `root:` in `configs/data*.yaml` to your checkout, then run one scene end to end:

```bash
./scripts_run_scene.sh configs/data.yaml          # airport
./scripts_run_scene.sh configs/data_town.yaml
./scripts_run_scene.sh configs/data_island.yaml
.venv/bin/python -m aerodiag.cli.transfer --scenes configs/data.yaml configs/data_town.yaml
```

Large intermediates (LiDAR maps, matches, per-pair signals) are cached under `outputs/**/cache` and are not tracked.

**AMtown03 camera:** the distortion coefficients in `calibration_results.py` do not fit these images (across-epipolar
p90 5.0 px on held-out cruise lines). `configs/data_town.yaml` uses distortion re-estimated from onboard data only
(images + RTK, cruise lines held out; p90 0.84 px): `src/aerodiag/calib/distortion_fit.py`.

## Limitations

Moving-object causes remain untested (almost all vehicles are parked); hand-built "reflective"/"moving" rule scores and
the glint score do not separate water or failures on their own (the learned heads do); only two scenes carry failure
labels, and airport → town transfer loses 0.13 AUROC; the error bound under-covers cells that actually fail; the
line-offset correction improves geometry but not the detection heads.
