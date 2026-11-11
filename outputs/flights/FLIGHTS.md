# All flights

## Labels (Terra <-> this flight's LiDAR)

| flight | frames | gt_cells | stable_cells | terra_coverage | icp_fitness_25cm | icp_rmse_m | gt_good_cells | terra_abs_err_p50_m | frac_err_gt_0_5m |
|---|---|---|---|---|---|---|---|---|---|
| HKairport_GNSS03 | 3147 | 94581 | 2466 | 0.999 | 0.774 | 0.134 | 16811 | 0.164 | 0.140 |
| HKairport_GNSS02 | 3733 | 107109 | 2238 | 0.997 | 0.780 | 0.133 | 20143 | 0.153 | 0.153 |
| HKisland_GNSS03 | 2669 | 52179 | 88 | 0.998 | 0.178 | 0.157 | 12941 | 0.977 | 0.782 |
| HKisland_GNSS02 | 3796 | 60432 | 191 | 0.998 | 0.205 | 0.161 | 18150 | 2.798 | 0.933 |
| AMtown03 | 5599 | 331223 | 8029 | 1.000 | 0.737 | 0.134 | 39744 | 0.102 | 0.173 |

## Detection (spatial CV, boosted trees)

| flight | failure_auroc | failure_ci | failure_ap | failure_base | class_only_auroc | layered_auroc | vehicle_auroc | vehicle_ap | water_auroc | water_ap |
|---|---|---|---|---|---|---|---|---|---|---|
| HKairport_GNSS03 | 0.940 | [0.91, 0.96] | 0.763 | 0.219 | 0.663 | 0.977 | 0.938 | 0.302 | – | – |
| HKairport_GNSS02 | 0.923 | [0.90, 0.95] | 0.733 | 0.237 | 0.698 | 0.936 | 0.807 | 0.188 | – | – |
| HKisland_GNSS03 | – | – | – | – | – | – | – | – | 0.956 | 0.690 |
| HKisland_GNSS02 | – | – | – | – | – | – | – | – | 0.960 | 0.754 |
| AMtown03 | 0.918 | [0.90, 0.93] | 0.791 | 0.221 | 0.464 | 0.948 | 0.893 | 0.041 | – | – |

## How much / calibration / noise floor

| flight | mag_coverage | mag_bound_m | mag_certified | mag_cov_fail | failure_ece_cal | across_p90_within:20 | across_p90_within:40 | across_p90_cross_line |
|---|---|---|---|---|---|---|---|---|
| HKairport_GNSS03 | 0.899 | 0.314 | 0.700 | 0.652 | 0.012 | 0.741 | 0.974 | 2.956 |
| HKairport_GNSS02 | 0.895 | 0.259 | 0.620 | 0.650 | 0.009 | 0.883 | 1.004 | 3.990 |
| HKisland_GNSS03 | – | – | – | – | – | – | – | – |
| HKisland_GNSS02 | – | – | – | – | – | 1.013 | 1.057 | – |
| AMtown03 | 0.900 | 0.389 | 0.548 | 0.649 | 0.004 | 1.149 | 2.591 | – |

## Timing (fresh run)

| flight | frames | sweep_pairs | sweep_rows_M | t_track_a_s | t_track_b_s | t_step2_s | t_step3_s | onboard_min | t_total_min |
|---|---|---|---|---|---|---|---|---|---|
| HKairport_GNSS03 | 3147 | 781 | 14.024 | 37.209 | 158 | 147 | 52.115 | 5.946 | 6.609 |
| HKairport_GNSS02 | 3733 | 1304 | 24.301 | 62.182 | 246 | 218 | 56.122 | 8.671 | 9.750 |
| HKisland_GNSS03 | 2669 | 334 | 4.062 | 17.191 | 111 | 47.974 | 29.001 | 3.129 | 3.461 |
| HKisland_GNSS02 | 3796 | 731 | 9.183 | 35.918 | 173 | 89.495 | 31.611 | 4.907 | 5.542 |
| AMtown03 | 5599 | 1644 | 32.096 | 90.884 | 133 | 56.886 | 114 | 5.062 | 6.643 |
