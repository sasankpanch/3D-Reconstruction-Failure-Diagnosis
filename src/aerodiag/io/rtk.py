"""RTK CSV (UAVScenes rtk_positions_raw.csv): scenename, headerstamp (epoch string -> ns), lat, lon, alt,
easting, northing (UTM 50N, EPSG:32650). 5 Hz. 'alt' datum: see configs (Q8)."""
from __future__ import annotations
import numpy as np, pandas as pd
from .timestamps import parse_ns_array
from ..geometry.enu import geodetic_to_enu

def load_rtk(path, origin=None) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"headerstamp": str})
    df["t_ns"] = parse_ns_array(df.headerstamp)
    df = df.sort_values("t_ns").reset_index(drop=True)
    if origin is not None:
        enu = geodetic_to_enu(df.lat.values, df.lon.values, df.alt.values, *origin)
        df["e"], df["n"], df["u"] = enu.T
    return df

def interpolate_rtk(df: pd.DataFrame, t_ns: np.ndarray, cols=("e", "n", "u")) -> np.ndarray:
    """Linear interpolation (int64 ns arithmetic relative to the first RTK sample). Outside the RTK span -> NaN."""
    t0 = df.t_ns.values[0]; tt = (df.t_ns.values - t0).astype(np.float64); q = (np.asarray(t_ns) - t0).astype(np.float64)
    out = np.stack([np.interp(q, tt, df[c].values) for c in cols], 1)
    out[(q < tt[0]) | (q > tt[-1])] = np.nan
    return out
