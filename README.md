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

| Prediction | HKairport_GNSS03 | AMtown03 | HKisland_GNSS03 |
|---|---|---|---|
| Failure location (AUROC) | 0.93 [0.90, 0.95] | 0.86 [0.84, 0.88] | – ¹ |
| Layered structure | 0.95 | 0.91 | – |
| Vehicles (AP, base rate 0.002) | 0.94 (0.06) | 0.87 (0.03) | – |
| Water (AP, base rate 0.07) | – (none) | – (none) | 0.93 (0.48) |
| Land-cover class alone (failure) | 0.66 | 0.46 | – |

Cross-scene transfer (train on one scene, test on the other, no retraining), failure AUROC:
airport → town 0.73 [0.69, 0.76], town → airport 0.79 [0.73, 0.85].

¹ HKisland is almost entirely vegetation and sea; too few hard surfaces to align the Terra model to LiDAR reliably,
so it is used only for the water head (label from LiDAR semantic classes).

Across-epipolar noise floor on healthy ground with RTK + per-pair rotation refinement: p90 below 1 px within a flight
line (1 px ≈ 5.5 cm at 80 m). Detailed reports: `outputs/DATA_REPORT.md`, `outputs/track_a/LABELS_REPORT.md`,
`outputs/track_b/NOISE_FLOOR_REPORT.md`, `outputs/PIPELINE_RESULTS.md`, `outputs/transfer/TRANSFER.md`.

## Pipeline

| Stage | CLI | What it does |
|---|---|---|
| Phase 0 | `aerodiag-phase0` | Frame table, registered keyframes, flight segmentation, RTK → working-frame fit |
| Track A | `aerodiag-track-a` | LiDAR ground-truth map, per-cell reliability, Terra ↔ LiDAR robust ICP, error labels |
| Track B | `aerodiag-track-b` | Feature matching, rotation-only pose refinement, across-epipolar noise floor (go/no-go gate) |
| Step 2 | `aerodiag-step2` | Coarse terrain triangulated from images; per-cell plane sweep along epipolar lines; five signals (across residual, cost-curve shape, photometric, height plausibility, sun glint) |
| Step 3 | `aerodiag-step3` | Per-cell accumulation by baseline class, motion / view-dependence fits, logistic heads, spatial CV, baselines, ablations |
| Summary | `aerodiag-summary` | Paper metrics → `outputs/PIPELINE_RESULTS.md` |
| Transfer | `aerodiag-transfer` | Cross-scene evaluation |

## Setup

```bash
python -m venv .venv && .venv/bin/pip install -e .
.venv/bin/python -m pytest -q tests
```

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

Moving-object causes remain untested (almost all vehicles are parked); hand-built "reflective"/"moving" rule scores do
not separate water on their own (the learned heads do); two-view height is biased by 0.4–0.6 m (RTK line offsets);
probabilities are uncalibrated rankings; cross-scene transfer loses 0.07–0.20 AUROC.
