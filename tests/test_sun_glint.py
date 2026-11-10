import numpy as np
from aerodiag.signals.sun import solar_position, sun_vector_enu
from aerodiag.signals.glint import glint_risk

def _ns(y, mo, d, h, mi, s):
    import datetime as dt
    return int(dt.datetime(y, mo, d, h, mi, s, tzinfo=dt.timezone.utc).timestamp()) * 1_000_000_000

def test_solar_position_reference():
    # Reda & Andreas (2004) SPA example: 2003-10-17 12:30:30 local (UTC-7), Golden CO -> zenith 50.11162, azimuth 194.34024
    az, el = solar_position(_ns(2003, 10, 17, 19, 30, 30), 39.742476, -105.1786)
    assert abs(az[0] - 194.34024) < 0.15 and abs((90 - el[0]) - 50.11162) < 0.15

def test_sun_vector_and_glint_geometry():
    s = sun_vector_enu(_ns(2023, 10, 25, 7, 30, 0), 22.416, 114.044)[0]
    assert abs(np.linalg.norm(s) - 1) < 1e-12 and s[2] > 0 and s[0] < 0       # afternoon in HK: sun up, toward the west
    n = np.array([0, 0, 1.0]); mirror_cam = np.array([-s[0], -s[1], s[2]])     # camera at the mirror direction of the sun
    r, th = glint_risk(n, s, mirror_cam); assert th < 1e-6 and abs(r - 1) < 1e-9
    r2, th2 = glint_risk(n, s, np.array([0, 0, 1.0])); assert th2 > 10 and r2 < 0.2
