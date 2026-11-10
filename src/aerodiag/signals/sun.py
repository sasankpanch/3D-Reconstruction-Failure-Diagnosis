"""Solar position (NOAA / Meeus simplified algorithm, ~0.01 deg). Inputs: int64 ns UTC timestamps, geodetic lat/lon.
Outputs: azimuth (deg, clockwise from north), elevation (deg, geometric, no refraction) and ENU unit vectors toward the sun."""
from __future__ import annotations
import numpy as np

def solar_position(t_ns, lat_deg, lon_deg):
    t_ns = np.atleast_1d(np.asarray(t_ns, dtype=np.int64))
    jd = (t_ns // 1_000_000_000).astype(np.float64) / 86400.0 + (t_ns % 1_000_000_000) * 1e-9 / 86400.0 + 2440587.5
    T = (jd - 2451545.0) / 36525.0
    L0 = np.mod(280.46646 + T * (36000.76983 + T * 0.0003032), 360)
    M = 357.52911 + T * (35999.05029 - 0.0001537 * T); Mr = np.radians(M)
    e = 0.016708634 - T * (0.000042037 + 0.0000001267 * T)
    C = np.sin(Mr) * (1.914602 - T * (0.004817 + 0.000014 * T)) + np.sin(2 * Mr) * (0.019993 - 0.000101 * T) + np.sin(3 * Mr) * 0.000289
    true_long = L0 + C
    omega = 125.04 - 1934.136 * T
    lam = true_long - 0.00569 - 0.00478 * np.sin(np.radians(omega))
    eps0 = 23 + (26 + (21.448 - T * (46.815 + T * (0.00059 - T * 0.001813))) / 60) / 60
    eps = eps0 + 0.00256 * np.cos(np.radians(omega))
    decl = np.arcsin(np.sin(np.radians(eps)) * np.sin(np.radians(lam)))
    y = np.tan(np.radians(eps) / 2) ** 2; L0r = np.radians(L0)
    eqt = 4 * np.degrees(y * np.sin(2 * L0r) - 2 * e * np.sin(Mr) + 4 * e * y * np.sin(Mr) * np.cos(2 * L0r)
                         - 0.5 * y * y * np.sin(4 * L0r) - 1.25 * e * e * np.sin(2 * Mr))           # minutes
    minutes_utc = np.mod(jd + 0.5, 1.0) * 1440.0
    tst = np.mod(minutes_utc + eqt + 4 * lon_deg, 1440.0)
    ha = np.radians(tst / 4 - 180); lat = np.radians(lat_deg)
    cosz = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha); zen = np.arccos(np.clip(cosz, -1, 1))
    az = np.degrees(np.arctan2(np.sin(ha), np.cos(ha) * np.sin(lat) - np.tan(decl) * np.cos(lat))) + 180
    return np.mod(az, 360), 90 - np.degrees(zen)

def sun_vector_enu(t_ns, lat_deg, lon_deg):
    az, el = solar_position(t_ns, lat_deg, lon_deg); a, e = np.radians(az), np.radians(el)
    return np.stack([np.cos(e) * np.sin(a), np.cos(e) * np.cos(a), np.sin(e)], -1)
