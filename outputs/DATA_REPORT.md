# DATA_REPORT — Phase 0 inspection, UAVScenes HKairport_GNSS03 (interval 1)

Generated 2026-11-08. All numbers below come from the scripts in `outputs/phase0/scripts/` (run with `./.venv`,
Python 3.12, numpy 2.5 / scipy 1.18 / OpenCV 5.0 / open3d 0.20). Intermediate tables and figures are in `outputs/phase0/`.
No data file was modified. `calibration_results.py` was parsed with `ast` (never imported).

---

## 0. Summary — what changes the plan

| # | Finding | Consequence |
|---|---|---|
| **F1** | **Only 913 of 3147 JSON poses are independent. 2234 are linear/slerp interpolations** between them. The reliable set is the **629 frames with `orig_idx % 5 == 0`** (orig_idx = 10 Hz counter including dropped frames). The other 284 non-interpolated frames are clustered in turns/line ends and are **not validated**. Interpolated poses are off by **5–20 px** (white, not drift) against image content. | Terra reference poses exist only every 0.5 s. 1-frame gaps (and 2–4) have **no reference pose**. The "terra" pose source and the RTK attitude (which is borrowed from Terra) must be restricted to keyframes, or B3 must change. **Decision needed (Q1).** |
| **F2** | **Images are NOT undistorted.** The JSON drops distortion (all zeros, f = 1478). With the calibration file's K + 5-coef distortion, the across-epipolar residual stays flat to the image edge. With the JSON model it grows 3× at r > 900 px and ~10× in the corners. | Use `HK_GNSS_calibration` K + dist for all projection/matching. Do not use JSON P3x3. |
| **F3** | Noise floor, first look: on cruise keyframe pairs (gap 5–20) with Terra poses and the calibrated camera model, the across-epipolar residual is **0.26–0.38 px median and 0.8–1.1 px p90** (all land cover, mostly trees). | Encouraging for the go/no-go gate, pending the proper B6 analysis. |
| **F4** | The Terra frame is the ENU frame declared in `terra_ply/metadata.xml` (origin lat 22.4150564, lon 114.0438757, h 179.664) **plus a ~1.8° tilt about the East axis** relative to RTK-ENU. LiDAR pavement planes are level within 0.4–0.8° in RTK-ENU but tilted 1.1–2.0° in Terra. | Confirms 2a: the Terra Z axis is not vertical. Never use z = const. |
| **F5** | After the similarity fit, the RTK–Terra disagreement is **per-flight-line**: 0.5 m (L2) to 2.4 m (L4). It's not removable by a similarity, time offset or lever arm. | "rtk" prior error is ~1–2.5 m and line-dependent. Which side is wrong is unresolved (Q6). |
| **F6** | `HKairport_GNSS03` maps to **`HK_GNSS_calibration`** in the file's own lookup table, **not** `HKairport_calibration`. | Use the lookup table, not the name-similar entry. |
| **F7** | LiDAR files are **XYZ only** (no intensity, no per-point time). | Deskewing is impossible (A1). Up to ~0.9 m along-track smear per scan. Mitigation in §2. |
| **F8** | **No water classes** (river/pool) appear anywhere in this sequence's labels. | The B2 "mostly water" fallback will never trigger here. "No-GT over water" is not testable on this sequence. |

---

## 1. Inventory (item 1)

| Path | Content | Count | Size |
|---|---|---|---|
| `calibration_results.py` | Python-syntax dicts (data) | 1 | 4.6 kB |
| `interval1_HKairport_GNSS03/interval1_CAM/` | `<sec>.<nsec>.jpg`, 2448×2048 RGB JPEG, ~0.9 MB | 3147 | ~2.8 GB |
| `interval1_HKairport_GNSS03/interval1_LIDAR/` | `image<ts>_lidar<ts>.txt` | 3147 | ~2.4 GB |
| `…/sampleinfos_interpolated.json` | list of 3147 dicts | 1 | 4.1 MB |
| `…/rtk_positions_raw.csv` | 1985 rows, 5 Hz | 1 | 260 kB |
| `…/posest_viz.png`, `…/rtk_positions_raw_viz.png` | dataset-provided quick-look plots (extra files, not in 2a) | 2 | — |
| `interval1_HKairport_GNSS03_Labels/interval1_LIDAR_label_id/` | same names as LiDAR, 1 int per line | 3147 | |
| `interval1_HKairport_GNSS03_Labels/interval1_LIDAR_label_color/` | same names, `R G B` per line | 3147 | 941 MB (both) |
| `HKairport_GNSS_Terra/` | `cloud_merged.ply`, `Mesh.ply`, `terra_ply/metadata.xml`, `terra_ply/Block/{Block.ply, Block.frag, 21× Block_i_j.jpg}` | — | 523 MB |

**No `cmap.py`.** I took the class names from the official UAVScenes `cmap.py` on GitHub
(`raw.githubusercontent.com/sijieaaa/UAVScenes/main/cmap.py`). All 13 observed ID→RGB pairs match it exactly.

**Timestamp matching:** I matched by nanosecond integer and found a perfect 1:1 between images, LiDAR, label-ID, label-colour and JSON entries
(3147 each). In every LiDAR filename the image and LiDAR timestamps are *identical*, so they were assigned, not independently stamped.
`SortedImageID` is contiguous 0..3146 and equals time order. **No image or LiDAR frame lacks a pose, and no pose lacks data.**

**Time span:** 2023-10-25 07:29:38.20 → 07:35:15.20 UTC (337.0 s). Δt = 100 ms ±0.2 ms, except **6 gaps present in all sources**
(the frames are simply absent): after SortedImageID 1017 (3.6 s), 1050 (3.1 s), 1363 (4.7 s), 1416 (1.6 s), 2232 (2.5 s), 2627 (7.5 s).
All gaps fall in turns.

**JSON:** keys as in 2a. A single P3x3 for all frames (fx = fy = 1478.055, cx = 1176.058, cy = 1047.518). K1..P2 all exactly 0. W×H 2448×2048.
T4x4 rotations are orthonormal to 1e-15. T4x4 is **camera→world** in the Terra frame (verified in §5), OpenCV camera axes.

**RTK CSV:** 1985 rows, Δt 200 ms, from 30.1 s before the first image to 29.7 s after the last. Easting/northing reproduce
lat/lon in **EPSG:32650 (UTM 50N)** to < 0.03 mm.

---

## 2. LiDAR files (item 2)

- **Format:** space-separated ASCII, 3 columns `x y z`, 6 decimals, metres. **No intensity, no per-point timestamp, no ring/tag, no label.**
- **Frame:** the Livox **sensor frame** (x = boresight, which points down on this mount). Ranges are 9.6–238.6 m. Median x is 33 m in the first frames (hover ~33 m AGL),
  ~80 m at cruise (≈ AGL), and 31 m at the end. Per-axis min/max: x 8.2…238.5, y −113…120, z −115…107.
- **Points/frame:** 19 322 – 33 750 (median 24 586). 80.95 M in total. No zero/NaN points.
- **Coverage:** the non-repetitive rosette covers the image centre densely but **does not reach the image corners**
  (see `phase0/overlay_panel.jpg`). Full ground coverage comes from overlapping frames.
- **Deskew (needed by A1):** impossible. There is no per-point time, and the Livox Avia scan order is non-repetitive, so time can't be inferred from the angle.
  Impact: a point measured δ ∈ [0, 100 ms] after the frame stamp is misplaced by v·δ, up to **0.9 m along-track** (mean ~0.45 m
  if the stamp marks the scan start; unknown). On flat ground this is an in-plane shift, so Z is almost unaffected. At height
  discontinuities (roof edges, tree crowns) it smears a ~0.9 m band. **Mitigation for A1:** estimate a single along-track
  time offset per scan by maximising agreement between overlapping scans, and/or treat edge cells with a slope-based
  uncertainty. Error estimate: ≤ 0.9 m · |∇z| vertical in a cell, ≈ 0 on flat ground.

## 3. Labels (item 3, item 10)

- Per-point, line-aligned with the LiDAR file: `label_id` (1 int) and `label_color` (3 ints, redundant with the ID). Point counts match in every frame.
- **No image masks in this download.** The upstream README mentions `interval1_CAM_label` (2D masks), which is not present here.
  If needed, masks can be produced by projecting labelled LiDAR (with the F2 camera model), but only within the LiDAR footprint.

| ID | name | LiDAR points | share | frames present |
|---:|---|---:|---:|---:|
| 13 | green_field | 38 723 872 | 47.84 % | 3147 |
| 0 | background | 19 687 596 | 24.32 % | 3077 |
| 10 | airstrip | 13 829 320 | 17.08 % | 2258 |
| 3 | paved_motor_road | 2 781 432 | 3.44 % | 1982 |
| 14 | wild_field | 2 512 271 | 3.10 % | 1526 |
| 1 | roof | 872 646 | 1.08 % | 1536 |
| 9 | container | 819 551 | 1.01 % | 1987 |
| 18 | car_park | 642 085 | 0.79 % | 1266 |
| 2 | dirt_motor_road | 639 913 | 0.79 % | 1648 |
| 11 | traffic_barrier | 312 285 | 0.39 % | 1665 |
| 20 | sedan *(dynamic)* | 92 393 | 0.11 % | 1552 |
| 16 | umbrella | 27 816 | 0.03 % | 1153 |
| 24 | truck *(dynamic)* | 9 152 | 0.01 % | 181 |

Shares are of all points summed over frames, so they are weighted by revisit and dominated by hover frames. A per-area share comes with the A2 grid.
Visually, `green_field` is mostly **tree canopy/forest** in the imagery, not lawn. The class name is misleading for "healthy ground" purposes.
`background` (24 %) is unlabelled mixed content. Healthy-ground candidates for B5(a): **airstrip, paved_motor_road, car_park, roof**
(+ maybe dirt_motor_road). Per-frame counts are in `phase0/lidar_label_counts_per_frame.csv`.

## 4. Calibration file (item 4)

Contents: six dicts (`AMtown_`, `AMvalley_`, `HKairport_`, `HKisland_`, `HK_GNSS_`, `Featureless_GNSS_calibration`), each with
`camera_intrinsic` (3×3, px), `camera_dist_coeffs` `[k1,k2,p1,p2,k3]` (OpenCV order, per the inline comment), `camera_ext_R` (3×3),
`camera_ext_t` (3, m). Plus `scenename_to_calibration`, mapping 23 scene names to these dicts. Header note: *"camera–LiDAR calibrations
from MARS-LVIG, not camera–3D map calibrations (those are in the .json)"*.

**Applies to HKairport_GNSS03:** `HK_GNSS_calibration` (via the lookup table):
```
K    = [[1444.43, 0, 1179.50], [0, 1444.34, 1044.90], [0, 0, 1]]
dist = [k1 -0.0560, k2 0.1180, p1 0.00122, p2 0.00064, k3 -0.0627]
R    = [[0.00363212,-0.999819,-0.0213618],[-0.0111679,0.0214512,-0.999591],[0.999879,0.00375613,-0.0111134]]
t    = [-0.0021928, 0.0470312, -0.0513126]  (m)
```
- **Direction: LiDAR→camera, `p_cam = R · p_lidar + t`, metres.** Under this reading 100 % of points land in front of the camera,
  at a median camera depth of 89.5 m (cruise). The inverse reading puts 54 % behind the camera. Overlays confirm it (§5).
- `R` is slightly non-orthonormal (det 0.99989, max |RRᵀ−I| 2.3e-4). It will be projected to SO(3) via SVD (< 0.01°).
- **vs JSON:** f differs by **+2.3 %** (1478.06 JSON vs 1444.4 calib), and the principal point by (−3.4, +2.6) px. JSON has zero distortion. See §6.
- The `HKairport_calibration` entry (f 1451, different R) is **not** the one mapped to this sequence. I did not use it.

## 5. Convention verification by projection (item 5)

- **Chain:** `X_world = T4x4[:3,:3] · (R·p_lidar + t) + T4x4[:3,3]`, with T4x4 = camera→world. Projection: OpenCV, F2 camera model.
- **Same frame** (tests extrinsic + intrinsics): LiDAR label boundaries follow image edges: airstrip edge line, airstrip/grass
  strip, concrete-pad outline (`phase0/overlay_panel.jpg`: climb 345, hover 755, cruise 1469, diagonal 2736; centre and corner crops).
- **Cross frame** (tests T4x4): LiDAR of frame i projected into image j tracks the scene (`phase0/overlays/lid1200_img1240_json.jpg`).
  Quantitatively (patch NCC search of image-i patches in image j at the projected location, keyframes): median offset 0.3 px
  across-epipolar. **No alternative convention was needed.**
- **F1 detail (pose interpolation).** The test exposed per-frame errors. Against a reference keyframe, other keyframes agree to
  0.0–0.1 px across-track and 1–3 px along-track. Interpolated frames are off by 5–20 px, uncorrelated frame to frame
  (`phase0/keyframe_test.csv`, `anchor_validation.csv`). Image-measured vs pose-predicted consecutive-frame shift: residual std
  6.4 px along / 2.5 px across (`flow_vs_pose.csv`). Structure (`p08`–`p11`, `phase0/anchors.csv`):
  - 2234 frames lie *exactly* (< 1 mm, < 0.001°) on linear-position / slerp-rotation interpolation between neighbouring anchors.
  - 629 anchors sit on `orig_idx % 5 == 0`, where `orig_idx = round((t − t₀)/100 ms)` counts dropped frames. This is the UAVScenes interval-5
    keyframe grid. One slot (SortedImageID 45) is itself interpolated.
  - 284 further anchors are clustered in turns and line ends (e.g. 991–1018, 1380–1438, 1810–1863, 2210–2314). In turns, even anchors showed
    ~10 px NCC offsets, but NCC is unreliable under the strong rotation there. **Status: unvalidated → I recommend not treating them as reference.**
  - Every frame is ≤ 0.5 s from an anchor.

## 6. Distortion verdict (item 6) — **images are distorted; use calib K + dist**

(a) The calibration file has non-zero distortion. The predicted displacement is 0 → −5.5 px at r = 800–1000 px → −7.6 px at the corners.
(b) Straight-line test (`p14`, 81 Hough segments ≥ 500 px from 290 frames): **inconclusive.** Only 12 segments were near the border, with a median
    bow of 0.34 px both raw and undistorted. The predicted bow is ~5 px only for ~2000 px lines along the top/bottom border; such lines are rare here.
(c) LiDAR-guided cross-frame residual vs radius (`p13`, 13 cruise keyframe pairs, gaps 5/10/20, ~80 k points with NCC > 0.85):

| radius (px) | across-epi median \|·\| JSON | calib | p90 JSON | calib |
|---|---|---|---|---|
| 0–300 | 0.25 | 0.24 | 0.84 | 0.82 |
| 300–600 | 0.34 | 0.27 | 0.98 | 0.85 |
| 600–900 | 0.38 | 0.32 | 1.08 | 0.91 |
| 900–1200 | **0.70** | **0.38** | 1.85 | 1.04 |
| >1200 | **3.18** | **1.10** | 4.92 | 1.98 |

Verdict: **not pre-undistorted.** The calib model is clearly better, and its residual is nearly flat with radius.
Caveat: under the calib model the along-epipolar residual has a bias that grows with radius (−0.4 px centre → −1.3 px at 900–1200 px).
This hints that the intrinsics aren't perfect, or that LiDAR scan skew is at work. Terra's own self-calibrated intrinsics (used for the T4x4) are not provided (see Q2).

Image quality notes: **visible motion blur in turns/diagonal frames and dark, vignetted corners** (`overlay_panel.jpg`).
Relevant to matching density near edges.

## 7. RTK analysis (item 7)

RTK was converted to local ENU about the Terra metadata origin, treating `alt` as WGS84 ellipsoidal height, then interpolated (linear, ns
timestamps) to image times. The fit uses 629 keyframes (`p15`, `p20`, `phase0/rtk_fit*.npz`, `rtk_residual_time.png`).

- **Unfitted:** Terra C − RTK-ENU median = (−0.27, +0.06, −0.24) m. So Terra's frame really is ~ENU at the declared origin, and
  `alt` is consistent with the height used for that origin.
- **Similarity RTK-ENU → Terra:** scale 0.99877, rotation 1.92°, of which **tilt 1.80°** (Terra Z vs ENU up, mostly about the East axis:
  ENU-up in Terra = (−0.0076, +0.0304, 0.9995)) and yaw −0.69°. Residual median 0.66 m, p95 2.46 m.
  The UTM fit gives the same tilt (1.80°) with scale 0.99807. UTM grid scale/convergence explain the difference.
  - The "all keyframes" fit is dominated by 164 hover keyframes. Excluding hover/climb/descent: scale 0.99866, tilt 1.73°, yaw −0.69°,
    residual median 1.31 m, p95 2.87 m. **I propose storing this flight-only fit** (`rtk_fit_flight.npz`) as the working RTK→Terra transform.
- **Residual per segment (median, m):** hover 0.27 · climb 0.36 · descent 0.46 · L0 0.54 · **L1 1.08 · L2 0.50 · L3 1.63 · L4 2.42 ·
  L5 (diagonal) 1.73** · turns 1.19. This is a line-dependent offset, so it is **non-rigid**: Terra block deformation, RTK error, or both.
- **Time offset:** best −150…−200 ms (median 0.64 vs 0.66 m at 0). Negligible improvement; not a timing problem.
- **Lever arm** (constant camera-frame offset, joint LS): (0.49, −0.04, 0.15) m, *worsens* the median (0.88 m). Not supported.
- **Altitude datum: undetermined.** The RTK `alt` is ~99.7 m at the first (hover) frame and 179.8 m at cruise. Terra's origin height (179.664) matches
  the cruise RTK altitude to ~0.1 m, so Terra and RTK use the same height reference. DJI RTK/Terra conventionally use WGS84
  ellipsoidal height, but nothing in the data proves it versus MSL (HK geoid undulation is a few metres). This only matters for
  absolute export. All work is in the Terra frame.

## 8. Frame-tilt check (item 8)

Robust plane fits to LiDAR points (cruise keyframes, F2 chain) in the Terra frame (`p16`):

| surface | extent | tilt vs Terra Z | tilt vs RTK-ENU up | resid std |
|---|---|---|---|---|
| airstrip (10), all | 228 × 114 m | 1.28° (toward +N) | **0.62°** | 0.24 m |
| airstrip, 40 m tiles (median of 8) | | 2.0° | **0.38°** | 0.09 m |
| paved_motor_road (3) | 227 × 183 m | 1.11° | 0.81° | 0.26 m |
| car_park (18) | 31 × 45 m | 0.74° | 1.26° | 0.23 m |

**Verdict:** the tilt is mainly in the **Terra frame**. Pavement is near-level in RTK-ENU (0.4–0.8°, plausible drainage
crossfall) but 1.1–2.0° tilted in Terra Z. Consistent with the 1.8° from §7. A "flat ground" prior must be a fitted plane/surface.

## 9. Flight segmentation (item 9)

Rules (`p17`, documented in code): smoothed Terra camera centres. Horizontal speed < 1.5 m/s → hover. |v_z| > 1 m/s → climb/descent.
Heading stable (< 8° over ±3 s) for ≥ 6 s at > 5 m/s → line, classed as parallel (within 15° of the dominant 6°/186° axis) or diagonal.
Everything else → turn. Result in `phase0/segments.csv`, plot in `phase0/segments.png`.

| segment | frames | notes |
|---|---|---|
| hover | 820 | 0–25 s at ~33 m AGL (Terra Z −47.7), and ~47–85 s at cruise height above the take-off point |
| climb | 165 | ~25–43 s, Terra Z −47.7 → 0 (RTK alt 133 → 180) |
| cruise_line | 1142 | **L0** (880–968, short, westbound, y≈114), **L1** (1094–1328, east, y≈201), **L2** (1453–1719, west, y≈124), **L3** (1887–2183, east, y≈61), **L4** (2339–2592, west, y≈−10). Line spacing ≈ 63–77 m, 9.0 m/s |
| diagonal | 143 | **L5** (2664–2806), heading 45°, crosses L2–L4 |
| turn | 698 | includes some straight line ends adjacent to timestamp gaps (rule artefact; harmless) |
| descent | 179 | ~320–337 s |

**Corrections to 2a:** the "climb during the first 60 s" is actually hover → climb → hover. There is a descent at the end. There are 4 long parallel lines
plus 1 short one. Camera tilt from nadir: median 6.2°, p95 16.7°, max 24.7°. Eastbound lines fly ~12° nose-down, westbound ~6°.
Terra camera Z varies by −3 … +5 m across lines at constant RTK altitude: the F4 tilt × northing.

## 10. Terra outputs (item 11)

- `cloud_merged.ply`: binary PLY, 15 143 794 points, xyz float32 + RGB. Extent x −445…195, y −132…315, z −176…+6 m (the low z are outliers/far terrain).
- `terra_ply/Block/Block.ply`: binary textured mesh, 1 268 985 vertices, 2 539 712 faces, per-face texcoords + `texnumber` → 21 textures
  `Block_i_j.jpg` (4096²). `Block.frag` is a Terra-internal binary (unused). `metadata.xml`: `SRS = ENU:22.415056378573908,114.04387569569714`,
  `SRSOrigin = 0,0,179.66438658301294`.
- `Mesh.ply`: ASCII CloudCompare re-export of the same mesh. Vertices identical to Block.ply (≤ 5e-7 m), with vertex colours.
- **Same frame as T4x4: confirmed.** LiDAR (cruise keyframes, F2 chain) → nearest cloud point: median 0.41 m, p90 0.88 m.
  **But there is a class-dependent vertical offset, LiDAR − Terra:** roof +0.03 m, green_field +0.11, background +0.11, paved road +0.31,
  **airstrip +0.60, wild_field +0.58** (IQR 0.3–0.5 m). Flat, low-texture surfaces sit highest above Terra. That is either a real Terra error
  (the thing we want to label) or a LiDAR range/extrinsic bias. A4/A5 will separate the global part. Flagged in Q4.

---

## 11. Blockers and open questions

- **Q1 (blocking for Track B design): keyframe-only reference poses.** Proposal:
  - The "terra" source exposes `is_reference` per frame. Only the 629 `orig%5==0` anchors count as reference. Interpolated frames are flagged and never used as truth.
  - B3 gaps become 5, 10, 20, 40, 80 frames between **keyframes** for terra/noisy_prior. Gap 1 (and 2–4) is evaluated only with
    priors that exist at every frame, i.e. RTK position + attitude refined by B2.
  - RTK source attitude: the spec says "Terra rotation + noise", but Terra rotation is interpolated on non-keyframes. Their 5–20 px errors equal ≈0.2–0.8° if attributed entirely to rotation; I haven't separated rotation from position error.
    Proposal: use slerp of keyframe rotations as the prior there. It's identical to the JSON, so I'd state its error explicitly and rely on
    B2 refinement. Alternative: restrict RTK experiments to keyframes until the rosbag IMU arrives.
- **Q2: camera model.** Use `HK_GNSS_calibration` K+dist (recommended). Optionally, in Track B, self-calibrate focal/distortion on
  keyframe pairs, since there's a residual radial along-line bias. OK to add as an optional step?
- **Q3:** the 284 non-grid anchors (turns): treat as non-reference (recommended) unless you know how UAVScenes produced them.
- **Q4:** the LiDAR-above-Terra offset of +0.6 m on airstrip vs +0.03 m on roofs. Is it acceptable to treat it as label signal after A4's
  global alignment? Or should I first check for a LiDAR range-scale or extrinsic-tilt bias (e.g. via roofs vs pavement at different
  off-nadir angles)? I recommend the check.
- **Q5:** the per-line RTK–Terra residual (0.5–2.4 m). I propose to arbitrate in Track A with LiDAR: check whether overlapping L3/L4 scans agree
  under Terra poses. If Terra is internally consistent, the RTK error is real and the "rtk" source should be modelled with line-dependent bias.
- **Q6:** no water in this sequence (F8). Keep the water fallback (configurable, inactive here). Should "background" (24 %) or forest
  be treated specially?
- **Q7:** I can't deskew (F7). Do you accept a per-scan along-track time-offset estimate plus documented ≤ 0.9 m·slope error?
- **Q8:** altitude datum is undetermined. It doesn't affect the work in the Terra frame. Is "assumed WGS84 ellipsoidal (DJI default), unverified" OK for export?
- **Environment:** `calibration_results.py` was initially missing from the working directory. It's present now and is what was parsed.
  Dependencies are installed in `./.venv` via uv (numpy, scipy, opencv-python-headless, open3d, pandas, pyarrow, matplotlib, pyproj,
  pyyaml, pytest, rosbags). GPU: RTX 5080 available, not needed so far.

## 12. Proposed package structure and plan (for approval before writing the package)

```
pyproject.toml                      # package "aerodiag", src layout, entry points below
configs/{data.yaml, track_a.yaml, track_b.yaml}
src/aerodiag/
  io/        timestamps.py (ns ints) · frames.py (FrameIndex: ts, paths, orig_idx, is_reference, segment)
             lidar.py · labels.py (cmap as data) · calib.py (ast parser, SO(3) projection, scene lookup)
             terra.py (PLY/mesh, metadata SRS) · rtk.py
  geometry/  camera.py (CameraModel: K+dist, project/undistort) · poses.py (Pose, SE3/Sim3, slerp)
             frames.py (RTK↔ENU↔Terra Sim3, UTM export) · epipolar.py (F/E from poses, Sampson,
             along/across decomposition) · plane.py
  poses/     base.py (PoseSource interface: pose(ts) → Pose + covariance/meta) · terra.py · rtk.py
             noisy.py (presets, seeded) · rosbag.py (placeholder, rosbags, ts matching) · refine.py (B2)
  flight/    segmentation.py · pairs.py (PairSelector: within/cross/crossing; tags gap, Δt, Δheading)
  grid/      ground_grid.py (GroundGrid: cell ids ↔ xy, configurable size) · rasterize.py
  labels/    lidar_map.py (A1) · cell_stats.py (A2) · terra_raster.py (A3) · align.py (A4) · table.py (A5/A7)
  matching/  features.py (SIFT/ORB via OpenCV, cached) · residuals.py (B4)
  analysis/  healthy.py (B5) · noise_floor.py (B6) · report.py
  cli/       phase0.py · track_a.py · track_b.py      (cached intermediates under outputs/cache/, keyed by config hash)
tests/       synthetic planar scene (across≈0, along→height), pose round-trips, calib direction,
             RTK→Terra Sim3, noise injection
```

Next steps after approval: (1) port Phase-0 findings into `io/`, `geometry/` and `flight/` with tests; (2) Track A: A1 with keyframe poses
only and dynamic removal, Q4 bias check, then A2–A7; (3) Track B in parallel: pose sources (with the Q1 decision), keyframe pairs,
B2 rotation refinement, B4–B7.
