# LABELS_REPORT — Track A: ground-truth error labels (HKairport_GNSS03)

Run: `aerodiag-track-a` (configs `configs/data.yaml`, `configs/track_a.yaml`). Every parameter is in `track_a.params.json`,
all numbers in `results.json`, intermediates in `outputs/cache/`. Working frame = Terra local frame, metres.
**Label definition:** `error = Terra_Z_aligned − LiDAR_Z` per 1 m cell. Positive means Terra is above LiDAR. LiDAR is the ground truth,
Terra is the reconstruction whose errors we predict.

## 0. Summary

- **94,581 cells have ground truth** (≥ 10 LiDAR points; median 64 points/cell). 41,324 cells are `no_gt`, all at the edge of the coverage.
  There's no water in this sequence, so no cells are `no_gt` because of water.
- **Hard surfaces are labelled tightly.** On airstrip, roads, car park and roofs the label |error| p90 is 0.25–0.36 m. On cells with reliable GT
  (`gt_good`) it's 0.20–0.31 m. Medians range from −0.02 (airstrip) to −0.23 m (car park).
- **Vegetation labels are dominated by GT definition.** Forest (`green_field`, 66 % of cells) and `background` have |error| p90 of 4–6 m.
  LiDAR returns penetrate the canopy, while photogrammetry reconstructs a smoothed top surface. A top-surface variant (`error_top`) only helps a little.
  **Vegetation labels should not be used as "healthy/unhealthy" evidence without the `gt_good` flag** (see §6).
- **The LiDAR GT is limited by Terra's own pose consistency between flight lines** (decimetres, up to ~0.6 m on oblique views).
  I measured this per cell (`gt_group_iqr`) and exposed it as a quality flag (`gt_good`), rather than hiding it.
- **Global (pre-alignment) offset:** Terra is **0.88 m below** LiDAR on stable cells. After robust ICP the residual on stable cells is 0.046 m median.
- **Q5 verdict:** Terra is internally consistent to roughly 0.2–1 m between lines, while RTK disagrees with Terra by line-dependent amounts of 0.5–2.4 m.
  **The RTK error is real** and should be modelled as line-dependent bias in Track B.

## 1. Decisions applied (from the Phase 0 approval)

| Q | Applied as |
|---|---|
| Q1 | The map uses only Terra-registered keyframes (`is_reference`: orig_idx % 5 == 0 anchors). |
| Q2 | Calibration-file camera model; self-calibration is reserved for Track B. |
| Q3 | Off-grid anchors aren't reference. Turn frames are additionally excluded from the GT map (see §3). |
| Q4 | LiDAR-bias check done (§5). The published calibration is kept; a refinement was tested and rejected. |
| Q5 | Line-to-line consistency measured with LiDAR (§5). |
| Q6 | `background` and `green_field` are reported separately everywhere. |
| Q7 | No per-point time: a global time offset was searched and found not identifiable, so δ = 0. The error budget is below. |
| Q8 | `rtk_to_working_sim3.json` labels the altitude "assumed WGS84 ellipsoidal, unverified". |

## 2. A1 — LiDAR map

- **Frames:** 426 reference keyframes (cruise 229, hover at altitude 114, diagonal 29, descent 28, climb 26), 10.9 M points.
  - Excluded: **turn keyframes (139)** and **low-altitude frames (64, median range < 45 m; the 33 m-AGL hover)**. The reason was found by
    `diag/group_bias.py` with them still included: on flat paved cells the low hover was a tight −0.40 m outlier (MAD 0.02 m) and
    produced a visible 15 m circle at the take-off point, and two turn runs were −0.94 and −1.02 m (MAD 0.65–0.75 m).
  - Cost: GT cells dropped from 132.6k to 94.6k, mostly forest at the outer edges.
- **Dynamic objects:** 17,452 sedan/truck points removed before accumulation.
- **Deskew (Q7):** there are no per-point timestamps, so a global offset δ was searched (X += v·δ), comparing east- and westbound lines.
  - With all classes the search was flat around 0 (median |Δz| 1.72–1.76 m for δ ∈ [−50, +50] ms, minimum near +20 ms).
  - With man-made classes only there were just ~150 cells, and the result was non-monotonic (`time_offset_search_manmade.csv`).
  - **δ is not identifiable from this data → δ = 0.**
  - Error budget: within-scan timing is uniform over 100 ms, so a point is misplaced along-track by up to ±0.45 m (std ≈ 0.26 m at 9 m/s) around the stamp position.
    If the stamp marks the scan start rather than the middle, there's an extra constant 0.45 m along-track shift. The vertical effect is about that shift × local slope:
    ≈ 0 on flat ground, and up to decimetres to metres at roof and canopy edges.
- **Sanity stacks** (`sanity_*.ply`, coloured by class), centred on the cruise stretch with the most paved surface:

| stack | frames | between-frame spread on flat 1 m cells (median / p90) | thickness, 0.5 m flat cells (median / p90) |
|---|---|---|---|
| 50 consecutive frames, JSON poses (incl. interpolated) | 1499–1548 | 0.033 / 0.057 m | 0.062 / 0.133 m |
| 39 consecutive keyframes (whole line L2) | 1454–1644 | 0.029 / 0.054 m | 0.050 / 0.080 m |

  **Verdict: clean, no doubling or smearing on ground surfaces within a line.** Interpolated poses don't visibly hurt stacking on flat
  ground, because their errors are mainly horizontal. Against the Terra mesh, before alignment, LiDAR sits **+0.75 to +1.04 m above it on every class**
  in this stretch (airstrip +0.94, roads +0.98, roof +0.91; IQR 0.10–0.16 m on hard surfaces). That's a uniform offset, removed by A4.
  `sanity_profile.png` shows a north–south transect: road and airstrip are parallel to the mesh, and the forest canopy LiDAR rises above the smoothed mesh.

## 3. A2 / A3 — grid and rasters

- **Grid:** origin (−303, −77), 1 m cells, 385 × 353 (`grid.json`, `aerodiag.grid.GroundGrid`).
- **LiDAR Z per cell is balanced:** the median over *viewing groups* (each flight line, and each contiguous hover/climb/descent run) of that group's cell median.
  A group needs at least 5 points to count. This stops one long hover from dominating a cell. The pooled median is kept as `lidar_z_pooled`.
  Also stored per cell: p10/p90, robust spread (1.4826·MAD), point count, frame and line counts, mean off-nadir angle, dominant class and its fraction.
- **GT uncertainty:** `gt_group_iqr` / `gt_group_range` are the IQR and range of the per-group medians.
  **`gt_good`** = at least 2 groups and IQR ≤ 0.20 m (16,811 cells).
- **Terra:** `terra_z` is the median Z of `cloud_merged.ply` points per cell. `terra_mesh_z` is a top-surface ray cast of the mesh at the cell centre (kept as a variant).
  89 GT cells have no Terra points.

## 4. A4 / A5 — alignment and labels

- **Stable cells** (≥ 50 points, spread ≤ 0.10 m, slope ≤ 0.15, group range ≤ 0.15 m, not dynamic): 2,466 cells.
- **Global offset before alignment** (median Terra − LiDAR on stable cells): **−0.884 m**. By class: airstrip −0.98, roads −1.00, container −0.95,
  background −0.83, green_field −0.74 m. Stored in `results.json → align.global_offset_pre_alignment`.
- **Robust point-to-plane ICP** (Tukey k = 0.15; correspondence 1.0 → 0.5 → 0.25 m; 0.25 m voxels), Terra → LiDAR on stable cells:
  translation (−0.124, −0.118, **+0.700**) m, rotation 0.140° (rx 0.112°, ry −0.082°, rz 0.017°). Final fitness 0.73, RMSE 0.14 m.
  The transform is in `terra_to_lidar_T.txt`. After alignment, the stable-cell residual is median −0.001 m and |median| 0.046 m.
  The 0.11° tilt is consistent with the small tilt difference expected between photogrammetric Z and the LiDAR map.
- **Label table** `labels.parquet`, one row per cell with LiDAR or Terra data. Columns:
  `cell_id, x, y, error, error_top, error_mesh, lidar_z, lidar_z_top, lidar_z_pooled, lidar_z_p10, lidar_z_p90, terra_z, terra_mesh_z,
  terra_z_raw, lidar_spread, gt_group_range, gt_group_iqr, n_groups, lidar_slope, point_count, n_frames, n_lines, mean_off_nadir_deg,
  no_gt, gt_good, stable, dominant_class, dominant_frac, terra_count, dominant_class_name`.

### Statistics — `error` (median label), all GT cells

Overall (94,492 cells): median +0.16, |e| p50 0.44, p90 4.26, p99 11.1 m. 64 % of cells have |e| > 0.25 m and 33 % have |e| > 1 m. Vegetation drives both.

| class | cells | median | MAD | \|e\| p50 | \|e\| p90 | \|e\| p99 | \|e\|>0.25 | \|e\|>1 |
|---|---|---|---|---|---|---|---|---|
| green_field (forest) | 62811 | 0.30 | 0.82 | 0.59 | 4.21 | 10.61 | 71% | 36% |
| background | 19114 | 0.12 | 0.81 | 0.57 | 6.29 | 13.12 | 65% | 41% |
| airstrip | 5023 | -0.02 | 0.17 | 0.12 | 0.36 | 0.59 | 24% | 0% |
| paved_motor_road | 3096 | -0.07 | 0.19 | 0.13 | 0.35 | 3.00 | 24% | 3% |
| wild_field | 1662 | -0.24 | 0.05 | 0.24 | 0.31 | 0.41 | 39% | 0% |
| container | 1345 | -0.02 | 0.24 | 0.17 | 1.09 | 7.39 | 37% | 12% |
| dirt_motor_road | 724 | -0.02 | 0.11 | 0.08 | 0.55 | 7.66 | 17% | 9% |
| car_park | 402 | -0.23 | 0.06 | 0.23 | 0.31 | 0.42 | 40% | 0% |
| roof | 227 | -0.12 | 0.07 | 0.12 | 0.25 | 2.14 | 10% | 5% |
| traffic_barrier | 52 | 0.09 | 0.18 | 0.13 | 0.35 | 0.53 | 27% | 0% |
| umbrella | 36 | -0.02 | 0.33 | 0.22 | 0.74 | 0.93 | 44% | 0% |

### Same, restricted to `gt_good` cells (16,811)

| class | cells | median | MAD | \|e\| p50 | \|e\| p90 | \|e\| p99 | \|e\|>0.25 | \|e\|>1 |
|---|---|---|---|---|---|---|---|---|
| green_field (forest) | 7158 | 0.06 | 0.23 | 0.17 | 1.47 | 7.84 | 39% | 13% |
| background | 3835 | -0.05 | 0.20 | 0.15 | 0.70 | 8.51 | 27% | 8% |
| airstrip | 1907 | -0.09 | 0.16 | 0.11 | 0.25 | 0.31 | 10% | 0% |
| paved_motor_road | 1218 | -0.12 | 0.16 | 0.15 | 0.29 | 0.36 | 18% | 0% |
| wild_field | 1106 | -0.23 | 0.04 | 0.23 | 0.29 | 0.42 | 31% | 0% |
| container | 772 | -0.05 | 0.18 | 0.13 | 0.45 | 1.88 | 23% | 4% |
| car_park | 344 | -0.23 | 0.06 | 0.23 | 0.31 | 0.38 | 42% | 0% |
| dirt_motor_road | 244 | 0.00 | 0.06 | 0.04 | 0.12 | 0.38 | 3% | 0% |
| roof | 187 | -0.12 | 0.07 | 0.12 | 0.20 | 1.15 | 5% | 2% |

**Top-surface variant `error_top`** (Terra − LiDAR p90): forest median −0.44 m with |e| p90 3.75 m, versus 4.21 m for the median label.
On hard surfaces it introduces a −0.16 to −0.43 m bias, which is expected because the p90 of a flat surface sits above it.
→ **Use `error` for hard surfaces. For vegetation neither label is a clean "reconstruction error"** (see §6).

**Error vs geometry** (`error_vs_slope.csv`, `error_vs_off_nadir.csv`): the median label rises with LiDAR slope, from −0.03 m at slope ≤ 0.1
to +0.95 m at slope > 1.2. That reflects canopy and edges plus within-scan smear. Off-nadir angle shows no monotonic trend.

**Small class-dependent offsets remain after alignment** on hard surfaces: car park and wild field −0.23 m, roof −0.12 m, airstrip −0.02 m.
They could be real Terra errors or line-dependent GT bias (§5). Both are within the GT uncertainty measured for those areas.

## 5. Consistency checks (Q4, Q5)

**Q4: is the LiDAR biased?** Per-viewing-group offset from the consensus on flat paved cells (`diag/group_bias.csv`):

| group | segment | cells | Δz median | MAD | median range | median off-nadir |
|---|---|---|---|---|---|---|
| L0 | line W | 2111 | −0.05 | 0.06 | 83 m | 17° |
| L1 | line E (views airstrip obliquely, looking S) | 805 | +0.12 | 0.14 | 91 m | 30° |
| L2 | line W (over airstrip) | 4171 | 0.00 | 0.03 | 83 m | 14° |
| L3 | line E (obliquely, looking N) | 2010 | −0.30 | 0.10 | 98 m | 33° |
| L4 | line W | 61 | −0.64 | 0.06 | 84 m | 14° |
| L5 | diagonal | 1432 | −0.03 | 0.07 | 83 m | 14° |
| climb / hover at altitude (4 runs) | | ~2000 each | 0.00 … +0.05 | 0.03–0.15 | 64–82 m | 11–14° |
| descent | | 2046 | +0.24 | 0.20 | 64 m | 15° |
| *(excluded)* low hover, 33 m AGL | | ~120 | −0.40 | 0.02 | 33 m | 5° |
| *(excluded)* 2 turn runs | | 457 / 86 | −0.94 / −1.02 | 0.65 / 0.75 | 91–107 m | 25–38° |

- **Near-nadir groups agree to ≤ 5 cm.** That rules out a meaningful LiDAR range-scale error and a pure extrinsic translation error.
- The opposite-sign offsets of L1 and L3 (both eastbound, looking south vs north) look like a roll-type angular error of ~4 mrad.
  A global LiDAR–camera rotation and range-scale refinement was tested (`diag/extrinsic_refine*.py`):
  - With the objective over all cells, cross-group disagreement only went from 0.616 to 0.579 m.
  - With the objective on flat paved cells, L3 was fixed, but it needed −9 mrad plus an implausible 0.8 % range scale, and L1's spread got worse (MAD 0.13 → 0.36 m).
  - **Conclusion: this is not a global calibration error. It's line-dependent pose error in the Terra keyframes. The published calibration is kept.**
  - The two automatic regressions in `results.json` (q4a, q4b) are ill-conditioned once low-altitude frames are excluded. They are superseded by this table.

**Q5: who is wrong, RTK or Terra?** Point-to-plane ICP between LiDAR from each pair of overlapping lines (`q5_line_consistency.csv`, fitness ≥ 0.7):
horizontal offsets 0.2–1.0 m, vertical 0.05–0.63 m, rotations 0.04–0.64°. The RTK − Terra camera offsets per line, from the flight-only Sim3, are
L0 (+0.53, +0.03), L1 (−0.71, +0.75), L2 (−0.11, +0.16), L3 (+1.63, −0.03), L4 (−2.24, −0.83), L5 (+1.56, +0.85) m (E, N).
They imply line-to-line differences up to 3.9 m (L3 vs L4), while LiDAR placed with Terra poses differs by only 0.19 / 0.73 m there.
**Verdict: Terra is consistent to ≲ 1 m between lines (decimetres on most pairs). The RTK solution carries a larger, line-dependent error of
0.5–2.4 m.** It shouldn't be treated as a better position than Terra. For Track B the "rtk" prior should be modelled as metre-level, line-dependent.
The Sim3 is stored in `outputs/rtk_to_working_sim3.json`: scale 0.99866, rotation 1.87°, tilt 1.73°, yaw −0.69°.

## 6. A6 — limitations

a. **LiDAR placed with Terra poses shares Terra's pose errors.** Labels capture matching-driven errors well (local surface errors) but underestimate
   pose-driven ones. Here this is measurable: line-dependent decimetre offsets show up as GT disagreement between viewing groups, and are exposed per cell
   in `gt_group_iqr` / `gt_good`. The non-rigid west–east ramp in `heat_error.png` (≈ −0.3 to +0.3 m across the open area) is part pose-driven
   Terra deformation and part GT inconsistency. The two can't be separated without an independent LiDAR trajectory (rosbag IMU/LiDAR odometry).
b. **"Z" is the working-frame axis,** tilted ~1.8° from vertical (Phase 0 F4). Local errors are unaffected by a rigid tilt.
   The ICP removed the residual 0.11° tilt between the two maps.
c. **Vegetation:** single-return LiDAR penetrates the canopy, so the per-cell median is not the surface photogrammetry is trying to reconstruct.
   Labels in forest mix "reconstruction error" with "surface-definition difference" (≫ 1 m).
d. **No deskew** (§2): up to ±0.45 m along-track smear per point. It mainly affects slopes and edges (label vs slope trend).
e. **Coverage:** turns and the low hover are excluded, so the outer ring of the scene has no GT. There's no water, so water behaviour can't be tested on this sequence.
f. **Class labels are not error labels.** `dominant_class` is the majority LiDAR label in the cell. Edge cells mix classes (`dominant_frac`).

## 7. Figures (`outputs/track_a/`)

`heat_lidar_z.png`, `heat_terra_z.png`, `heat_error.png` (label), `heat_error_top.png`, `heat_point_count.png`, `heat_no_gt.png`,
`heat_class.png`, `heat_gt_uncertainty.png`, `heat_gt_good.png`, `heat_stable.png`, `error_hist_by_class.png`, `sanity_profile.png`,
and the sanity point clouds `sanity_*.ply`.

## 8. Problems, open questions, next steps

1. **What should count as error-based "healthy/unhealthy" in Track B (B5b)?** Proposal:
   - healthy = `gt_good` & |error| ≤ 0.15 m & high texture
   - unhealthy = `gt_good` & |error| ≥ 0.5 m
   - vegetation excluded by default (limitation c), reported separately.
   On this sequence, hard surfaces have very few clearly unhealthy `gt_good` cells (|e| > 0.5 m is ≈ 0 % on airstrip, roads and roofs). Most "unhealthy" cells will be
   vegetation, container and edge cells. That will limit B6's unhealthy-side statistics.
2. **GT strength:** the label noise floor on hard surfaces is ~0.1–0.3 m. That's dominated by Terra line-to-line pose inconsistency, not LiDAR noise (3–5 cm).
   Labels can't resolve Terra errors smaller than that.
3. **The `rtk` pose source should carry a line-dependent metre-level bias** (Q5). This also affects what "RTK-grade" means in the noisy presets.
4. **Next:** Track B, as approved. Pose sources, epipolar decomposition and noise injection are already implemented and unit-tested
   (`aerodiag.poses`, `aerodiag.geometry.epipolar`). Track B starts with the class-based healthy definition (airstrip, roads, car park, roof).
