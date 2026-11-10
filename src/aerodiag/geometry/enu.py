"""Geodetic <-> local ENU <-> UTM helpers (pyproj). Heights: RTK 'alt' treated as WGS84 ellipsoidal height
(assumption, Q8: unverified; it only matters for absolute export)."""
from __future__ import annotations
import numpy as np
import pyproj

_ECEF = pyproj.Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
_ECEF_INV = pyproj.Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)

def _R_ecef_to_enu(lat0, lon0):
    la, lo = np.radians(lat0), np.radians(lon0)
    return np.array([[-np.sin(lo), np.cos(lo), 0],
                     [-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)],
                     [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]])

def geodetic_to_enu(lat, lon, h, lat0, lon0, h0) -> np.ndarray:
    X = np.array(_ECEF.transform(np.asarray(lon), np.asarray(lat), np.asarray(h))).T
    X0 = np.array(_ECEF.transform(lon0, lat0, h0))
    return (X - X0) @ _R_ecef_to_enu(lat0, lon0).T

def enu_to_geodetic(enu, lat0, lon0, h0):
    X0 = np.array(_ECEF.transform(lon0, lat0, h0))
    X = np.asarray(enu) @ _R_ecef_to_enu(lat0, lon0) + X0
    lon, lat, h = _ECEF_INV.transform(X[:, 0], X[:, 1], X[:, 2]); return np.c_[lat, lon, h]

def geodetic_to_utm(lat, lon, epsg: int = 32650):
    tr = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    e, n = tr.transform(np.asarray(lon), np.asarray(lat)); return np.c_[e, n]
