"""aerodiag: pre-reconstruction failure diagnosis for aerial mapping.

Global conventions (apply everywhere unless a docstring says otherwise)
-----------------------------------------------------------------------
* Working frame ("world", W): the DJI Terra local frame of the T4x4 poses. Metres. Close to ENU about
  (lat 22.4150564, lon 114.0438757, h 179.664) but its Z axis is tilted ~1.8 deg from local vertical
  (Phase 0, DATA_REPORT F4). Never assume z = const is level.
* Camera frame (C): OpenCV, x right, y down, z forward (optical axis). Pixels: (u, v), origin at the top-left
  pixel centre.
* LiDAR frame (L): Livox Avia sensor frame, x = boresight. Calibration: p_C = R_CL p_L + t_CL (metres).
* Poses: T_wc (4x4) maps camera-frame points into the world frame: p_W = R_wc p_C + t_wc; t_wc = camera centre.
* Timestamps: int64 nanoseconds since the Unix epoch, parsed from strings (never via float64).
"""
__version__ = "0.1.0"
