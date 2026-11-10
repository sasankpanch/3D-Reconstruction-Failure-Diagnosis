# NOISE_FLOOR_REPORT — Track B: onboard pose + across-epipolar noise floor (go/no-go gate)

Run: `aerodiag-track-b` (`configs/track_b.yaml`). All settings, including every noise setting, are in `track_b.params.json`
and `results.json → sources`. Tables are the `*.csv` files here; per-match data is in `residuals.parquet` (3.3 M rows).
Units: pixels in image 2 (1 px ≈ 5.5 cm on the ground at cruise, ~82 m AGL). |across| is the absolute across-epipolar residual.

## Verdict

**1. Noise floor: GO.** With rotation-refined poses, the across-line residual on healthy ground is
**0.10–0.30 px median and 0.4–1.1 px p90 for within-line pairs at every frame gap (1–80)**.
That holds with the real onboard-grade prior (RTK position + 0.1° attitude noise + refinement) just as well as with the Terra reference.
It's far below the size of physically meaningful violations: an object moving 0.5 m/s gives ≈ 1 px at a 1-frame gap and ≈ 10 px at a 10-frame gap.

**2. Separability from Terra-error "unhealthy" cells: NOT demonstrated on this sequence.**
Per-cell median |across| separates unhealthy cells (|error| ≥ 0.5 m, reliable GT) from healthy ones only weakly:
AUROC 0.52–0.75, and 5–30 % detection at 5 % false alarms. The best case is long within-line baselines (gap 40–80, rtk+ref: AUROC 0.71–0.75).
This is mostly a mismatch between the signal and the labels, not a failure of the signal:
- Large Terra errors here are canopy-surface definition, slopes and edges, and smoothing. Those are height (along-line) errors that don't break the epipolar constraint.
- The causes the across channel targets (moving, reflective, hidden/layered) are rare here: no water, and nearly all vehicles are parked.
- Where 3D layering does exist (container stacks, roof and building edges), |across| is the highest in the scene. Roof p90 is 1.95 px vs 0.43 px on airstrip.

**→ Overall: conditional GO.** The across-line channel is precise enough to be a useful signal, so the physical premise holds.
On its own it does not predict reconstruction height error on this sequence. Steps 2–3 must combine it with along-line, texture
and match-density signals, and its value for *moving/reflective* causes still needs data that contains them.

**Pose quality required:**
- Rotation must be refined from image matches. Any realistic attitude prior (≤ 0.1–0.3°) is fine because refinement absorbs it.
- The relative position between the two frames needs to be ≤ ~5 cm for short gaps (5–20 frames), ≤ ~10 cm at 40 frames and ≤ ~20 cm at 80 frames to keep p90 ≲ 1 px.
- RTK meets this within a line, because its line-dependent bias cancels. It doesn't across lines (bias 0.5–2.4 m), where p90 rises to 3.5–3.8 px.
- Standard-GNSS-grade per-frame noise (σ 0.5–1 m) fails: p90 5–12 px.

## 1. Setup

**Pose sources (B1)**, all via `aerodiag.poses`:

| source | definition | valid frames |
|---|---|---|
| terra | Terra registered keyframes (reference; comparison only) | 629 |
| terra_json | JSON poses including interpolated ones (diagnostic) | 3147 |
| rtk_att0 / **rtk** | RTK interpolated to image times → working frame via flight-only Sim3 (scale 0.99866, rot 1.87°); attitude = keyframe slerp + 0° / **0.1°** white noise; no lever arm | 3147 |
| noisy_rtk | terra + bias(0.01 h, 0.02 v m) + white(0.02 h, 0.03 v m) + attitude 0.05° (+0.02° bias) | 629 |
| noisy_gnss | terra + bias(1.5 h, 3.0 v m) + white(0.5 h, 1.0 v m) + attitude 0.3° (+0.2° bias) | 629 |
| sweep_pos{σ} | terra + white position σ ∈ {0.02 … 1.0} m (h and v) + 0.1° attitude | 629 |
| rosbag | placeholder (`rosbags` reader, timestamp matching); not exercised, since no bag is available | — |

**Rotation-only refinement (B2):** translation is kept from the prior, and a rotation correction on camera 2 is fitted by minimising robust Sampson error (soft-L1, 1 px scale).
No essential matrix is fitted. It's **2-fold cross-validated**: every reported refined residual comes from a correction fitted on the *other* half of that pair's matches.
- Fallbacks: 110 pairs with < 40 matches kept the prior. Water fallback: 0 pairs (the maximum per-frame water fraction is 0, since this sequence has no water class).
- Median correction size: terra 0.09° within a line but **0.53° on cross-line pairs** (Terra's inter-line pose inconsistency, matching Track A).
  rtk 0.26° within a line and 1.25° across lines. noisy_gnss 1.3°.
- On synthetic data the refinement recovers a 0.27° error and leaves exact poses unchanged (unit tests).
- **Caveat for later height work:** rotation components that only move points *along* epipolar lines are weakly constrained by Sampson error.
  Refinement fixes across residuals, not necessarily height.

**Pairs (B3):** 1,834 pairs over 684 frames (median matches per pair in brackets).
- **Within-line, cruise lines + diagonal:** 1,360 pairs at gaps 1/5/10/20/40/80 = 256/252/246/234/210/162 (2,289).
  Gap 1 has no reference pose (Q1), so it's evaluated only with rtk / rtk_att0 / terra_json.
- **Cross-line:** 107 pairs, median 71 m baseline, mostly opposite headings (49). 41 of these pairs have < 40 matches; opposite views over forest match poorly.
- **Crossing:** 72 pairs, diagonal vs parallel lines, median heading difference 137° (615).
- **Separately reported:** climb 61 pairs and turn 234 pairs (gaps 5/10).

**Matching:** RootSIFT (8k per image). Filtering is **pose-independent** so the residual being measured isn't clipped:
ratio test 0.8, mutual nearest neighbour, local displacement consistency, and a loose homography RANSAC whose tolerance exceeds the parallax of ±15 m relief
(it uses baseline length and altitude only, never direction). Up to 3,000 matches are kept per pair.

**Residuals and cells (B4):** along and across components come from the decomposition in `aerodiag.geometry.epipolar`,
in undistorted normalised coordinates × mean focal length (calibration-file camera model).
Each match is assigned to the ground cell where its image-1 ray meets the LiDAR height raster (96.2 % assigned).
Vehicle flag: within 9 px of projected sedan/truck LiDAR points in either image (17,503 matches).

**Healthy / unhealthy (B5):**
- (a) **class-based healthy** = airstrip, paved road, car park or roof cells (4,053 cells, 416k matches).
- (b) **error-based healthy** = `gt_good` & |error| ≤ 0.15 m & cell texture ≥ 6 grey levels (4,824 cells).
- **Unhealthy** = `gt_good` & |error| ≥ 0.5 m (1,912 cells).
- Vegetation (green_field, 52k cells) and vehicles are reported separately.

## 2. Noise floor on healthy ground (B6)

**|across| p50 / p90 (px), class-based healthy, by pose source and pair type** (`fig_noise_floor_vs_gap.png`):

| pairs | terra | terra+ref | terra_json | rtk | **rtk+ref** | noisy_rtk+ref | noisy_gnss+ref |
|---|---|---|---|---|---|---|---|
| within, gap 1 | — | — | 2.02 / 6.03 | 3.58 / 10.0 | **0.10 / 0.40** | — | — |
| within, gap 5 | 0.20 / 0.61 | 0.13 / 0.48 | 0.20 / 0.61 | 2.42 / 7.09 | **0.13 / 0.50** | 0.17 / 0.63 | 2.08 / 8.31 |
| within, gap 10 | 0.25 / 0.79 | 0.14 / 0.53 | = terra | 2.59 / 6.80 | **0.16 / 0.57** | 0.18 / 0.67 | 2.17 / 10.2 |
| within, gap 20 | 0.34 / 1.11 | 0.17 / 0.62 | = terra | 2.74 / 7.20 | **0.19 / 0.66** | 0.19 / 0.71 | 1.68 / 8.39 |
| within, gap 40 | 0.44 / 1.76 | 0.23 / 0.82 | = terra | 3.05 / 6.87 | **0.22 / 0.78** | 0.24 / 0.87 | 1.18 / 5.40 |
| within, gap 80 | 0.51 / 2.48 | 0.29 / 1.06 | = terra | 3.10 / 6.76 | **0.24 / 0.90** | 0.29 / 1.10 | 0.61 / 2.98 |
| cross-line | 1.85 / 15.8 | 0.29 / 1.26 | = terra | 16.9 / 41.5 | **0.96 / 3.49** | 0.30 / 1.24 | 0.91 / 5.35 |
| crossing | 13.1 / 16.3 | 0.47 / 1.59 | = terra | 38.1 / 59.7 | **0.98 / 3.83** | 0.46 / 1.56 | 1.76 / 11.0 |

p99 within a line with rtk+ref: 1.2 (gap 1) → 4.5 px (gap 80). Cross-line and crossing p99 is about 30 px, from residual mismatches on wide baselines.
The error-based healthy definition gives the same picture to within ±0.1 px median (`noise_floor_by_group_source_pairs.csv`).

What drives it:
- **Interpolated poses are unusable for residuals** (terra_json at gap 1: 2.0 / 6.0 px; rtk_att0 equals terra_json at gap 1 because it uses the same interpolated attitude).
- **Attitude noise dominates any unrefined onboard prior** (rtk: 2.4–3.6 px median). Refinement removes it completely.
- **Terra reference poses are consistent within a line but not between lines.** Unrefined terra gives 13 px median on crossing pairs, and refinement brings it to 0.47 px.
- **After refinement, what's left is set by position error relative to the baseline.** That's why rtk+ref is as good as terra+ref within a line,
  where RTK's line bias cancels, but ~1 px worse across lines.

**By image position** (within a line, gaps 5–40; p50 / p90 / p99, rtk+ref): centre 0.13 / 0.43 / 1.45, mid 0.16 / 0.58 / 1.91, **edge 0.22 / 0.88 / 3.31** px
(`fig_noise_floor_vs_position.png`). Edges are about 2× worse: residual distortion, plus blur and vignetting in the corners. Thresholds should depend on image position.

**By flight segment** (healthy, p50 / p90, rtk+ref): cruise 0.15 / 0.58, diagonal 0.17 / 0.58, climb 0.12 / 0.42, turn 0.18 / 0.75 px.
Turns are only slightly worse once refined; unrefined terra in turns is 0.43 / 2.55.

**By class** (within a line, all gaps, p50 / p90, rtk+ref): airstrip 0.13 / 0.43, wild_field 0.12 / 0.41, dirt road 0.15 / 0.56, green_field 0.16 / 0.59,
background 0.17 / 0.66, paved road 0.18 / 0.76, container 0.20 / 0.78, car_park 0.22 / 0.77, umbrella 0.24 / 1.11, **roof 0.37 / 1.95** px.
Classes with vertical structure and edges (roofs, containers, umbrellas) are highest, which is consistent with occlusion and layering.

**Required pose quality: controlled sweep** (refined, 0.1° attitude; healthy p90 in px; `fig_sweep_position_noise.png`):

| σ position (m) | gap 5 | gap 10 | gap 20 | gap 40 | gap 80 | cross-line | crossing |
|---|---|---|---|---|---|---|---|
| 0.02 | 0.56 | 0.61 | 0.70 | 0.87 | 1.10 | 1.27 | 1.58 |
| 0.05 | 0.83 | 0.84 | 0.85 | 0.92 | 1.12 | 1.27 | 1.65 |
| 0.10 | 1.40 | 1.34 | 1.21 | 1.10 | 1.16 | 1.38 | 1.84 |
| 0.20 | 2.65 | 2.46 | 2.04 | 1.54 | 1.26 | 1.76 | 2.68 |
| 0.50 | 5.46 | 6.11 | 4.75 | 3.24 | 1.81 | 3.45 | 5.83 |
| 1.00 | 9.11 | 11.07 | 9.51 | 6.30 | 3.02 | 4.91 | 11.37 |

Longer baselines tolerate more position noise. This sweep uses white noise per frame. A common bias (as in real RTK within a line) cancels.

## 3. Unhealthy cells and separability

|across| p50 / p90 (px), rtk+ref:

| pairs | healthy (error-based) | **unhealthy (\|e\| ≥ 0.5 m)** | vegetation |
|---|---|---|---|
| within, gap 5 | 0.14 / 0.46 | 0.18 / 0.64 | 0.17 / 0.56 |
| within, gap 10 | 0.17 / 0.55 | 0.23 / 0.83 | 0.21 / 0.70 |
| within, gap 20 | 0.20 / 0.67 | 0.31 / 1.17 | 0.26 / 0.91 |
| within, gap 40 | 0.24 / 0.86 | 0.49 / 2.08 | 0.31 / 1.17 |
| within, gap 80 | 0.31 / 1.22 | 0.78 / 6.84 | 0.48 / 3.24 |
| cross-line | 1.06 / 3.99 | 0.85 / 2.42 | 1.18 / 6.17 |
| crossing | 1.03 / 5.03 | 0.65 / 2.65 | 0.55 / 2.48 |

Unhealthy cells grow with baseline faster than healthy ones, as expected for occlusion and layered structure.

Per-cell separability (per-cell median |across|, ≥ 3 matches; `separability.csv`, `fig_cell_score_cdf.png`):

| pairs | source | AUROC (unhealthy vs healthy) | detected at 5 % FPR |
|---|---|---|---|
| within, gap 1 | rtk+ref | 0.53 | 5 % |
| within, gap 10 | rtk+ref / terra+ref | 0.65 / 0.62 | 13 % / 11 % |
| within, gap 40 | rtk+ref / terra+ref | **0.75** / 0.69 | **30 %** / 22 % |
| within, gap 80 | rtk+ref / terra+ref | 0.71 / 0.66 | 29 % / 24 % (17 unhealthy cells only) |
| within, all gaps pooled | rtk+ref | 0.52 | 6 % |
| cross-line / crossing | rtk+ref | 0.41 / 0.33 | 1 % / 2 % |

Pooling all gaps destroys the signal. The short-gap matches outnumber the long-gap ones and carry little violation, so per-cell scores need to be computed per baseline.
On cross-line and crossing pairs, unhealthy cells score *lower* than healthy ones. Unhealthy cells there are mostly vegetation and edges that survive matching only where they look alike,
while healthy pavement suffers more from the RTK line bias.

**Vehicles** (within a line): median |across| 0.23–0.40 px, and only **2–3 % of vehicle matches exceed the healthy p99** (rtk+ref: 3 %).
→ Almost all labelled vehicles were parked. The few exceeding matches are consistent with moving cars, but without tracks there's no ground truth for motion.

**Along-line check (height):** triangulating with Terra poses and comparing with LiDAR on healthy ground gives a spread (|dz| p90) of 1.25 / 0.82 / 0.61 / 0.46 / 0.34 m at gaps 5 / 10 / 20 / 40 / 80.
That follows the predicted height-per-pixel scaling (1.03 → 0.065 m/px). Along-line displacement does encode height, as designed.
There's a **+0.26 m bias at the image centre rising to +0.4–0.67 m at the edges** with the calibration intrinsics (the JSON intrinsics give −2.1 m).
A small radial intrinsic error remains, which matters for height (Section 1 step 4) but not for the across-line noise floor.

## 4. Problems and limitations

1. **The unhealthy reference set doesn't match the across channel's purpose.** Error labels on this sequence are dominated by vegetation and edge
   surface-definition effects (Track A §6). There are no reflective/water cells and almost no confirmed moving objects, so the across channel's intended causes are barely present.
2. **Featureless areas** (airstrip interior, `fig_map_cell_across_rtk_ref.png` grey) produce *no matches*. That's a distinct signal (match density), not an across residual.
3. **Cross-line coverage is thin.** 41 of 107 cross-line pairs have < 40 matches, because opposite-heading views over forest match poorly.
4. **The noise-floor statistics include residual mismatches** in the tails (p99). The filtering is deliberately pose-independent, which costs some contamination.
5. **The RTK source has no lever arm** and a line-dependent bias (Track A Q5). The rtk attitude is a keyframe-slerp stand-in for the IMU until the rosbag is available.
6. **Single sequence, single day and lighting.** The thresholds above aren't validated elsewhere.

## 5. Recommended next steps

1. **Proceed to Section 1 step 2** (per-cell signals) with |across| as one signal among several, scored per baseline class
   (short gaps for motion, 40–80 frames for occlusion/layering) and normalised by image position. Add along-line height consistency, match density and texture for featureless and weak-geometry causes.
2. **Validate the moving/reflective causes on data that contains them:** e.g. an HKisland sequence (water) and a sequence with traffic (AMtown).
   Vehicle tracks or LiDAR-based motion labels would turn the vehicle flag into "moving vs parked".
3. **Self-calibrate the intrinsics** (focal + radial terms) on keyframe pairs before any height-error work (Q2). It should remove the +0.3–0.7 m radial height bias and halve the edge noise floor.
4. **Use RTK + rotation refinement as the onboard pose design.** Within a line it already reaches the reference floor. Cross-line use needs either RTK line-bias correction
   (e.g. a per-line offset from cross-line matches) or restricting the across channel to within-line pairs.
5. When the rosbag arrives: replace the slerp attitude with IMU attitude, and test a non-RTK GNSS against the sweep thresholds above.

## 6. Files

- **Tables:** `noise_floor_by_group_source_pairs.csv` (all groups × sources × pair groups), `noise_floor_by_image_position.csv`, `noise_floor_by_class.csv`,
  `noise_floor_by_segment.csv`, `separability.csv`, `vehicles.csv`, `along_height_check.csv`, `sweep_position_noise.csv`,
  `pairs.csv` (per pair: tags, match counts, refinement status and correction size per source), `match_info.csv`.
- **Figures:** `fig_noise_floor_vs_gap.png`, `fig_noise_floor_vs_position.png`, `fig_cell_score_cdf.png`, `fig_sweep_position_noise.png`, `fig_map_cell_across_rtk_ref.png`.
- **Code:** `aerodiag.poses.{terra,rtk,noisy,rosbag,refine}`, `aerodiag.flight.pairs`, `aerodiag.matching`, `aerodiag.geometry.epipolar`, `aerodiag.analysis.noise_floor`.
  18 unit tests pass.
