import numpy as np, pandas as pd, pytest
from pathlib import Path
from aerodiag.io.timestamps import parse_ns, ns_to_str
from aerodiag.io.calib import parse_calibration_file, load_scene_calibration
from aerodiag.io.rtk import interpolate_rtk
from aerodiag.geometry.enu import geodetic_to_enu, enu_to_geodetic
from aerodiag.geometry.se3 import Sim3
from aerodiag.poses.rtk import fit_rtk_to_working
from aerodiag.poses.noisy import inject_noise, NoiseSpec, NoisyPriorSource, PRESETS
from aerodiag.poses.terra import TerraPoseSource
from aerodiag.flight.anchors import detect_anchors
from aerodiag.grid.ground_grid import GroundGrid
from aerodiag.grid.rasterize import grouped_quantiles, grouped_mode, grouped_mad
from scipy.spatial.transform import Rotation as Rot
from conftest import nadir_pose

CALIB = Path(__file__).resolve().parents[1] / "calibration_results.py"

def test_timestamps_exact():
    s = "1698218978.199993134"; t = parse_ns(s)
    assert t == 1698218978199993134 and ns_to_str(t) == s and parse_ns(s + ".jpg") == t
    assert parse_ns("1698218978.2") == 1698218978200000000
    assert float(s) * 1e9 != t          # documents why float64 is not used

@pytest.mark.skipif(not CALIB.exists(), reason="calibration file not present")
def test_calibration_direction_and_entry():
    entries, lookup = parse_calibration_file(CALIB)
    assert lookup["HKairport_GNSS03"] == "HK_GNSS_calibration"
    c = load_scene_calibration(CALIB, "HKairport_GNSS03")
    assert c.entry_name == "HK_GNSS_calibration" and abs(np.linalg.det(c.R_CL) - 1) < 1e-12
    # LiDAR->camera: LiDAR boresight (+x_L, 80 m) must land in front of the camera on the optical axis
    pc = c.lidar_to_camera(np.array([[80.0, 0, 0]]))
    assert pc[0, 2] > 79 and np.hypot(*pc[0, :2]) < 2.5
    assert np.allclose(c.camera_to_lidar(pc), [[80, 0, 0]])

def test_enu_roundtrip():
    lat0, lon0, h0 = 22.415056378573908, 114.04387569569714, 179.66438658301294
    enu = geodetic_to_enu(np.array([22.416, 22.414]), np.array([114.043, 114.045]), np.array([180.0, 100.0]), lat0, lon0, h0)
    back = enu_to_geodetic(enu, lat0, lon0, h0)
    assert np.allclose(back, [[22.416, 114.043, 180.0], [22.414, 114.045, 100.0]], atol=1e-8)

def test_rtk_to_working_transform():
    """RTK->working Sim3 fit recovers a known transform from interpolated 5 Hz positions."""
    t_rtk = np.arange(0, 60_000_000_000, 200_000_000, dtype=np.int64) + 1_698_218_948_000_000_000
    tt = (t_rtk - t_rtk[0]) * 1e-9
    rtk = pd.DataFrame(dict(t_ns=t_rtk, e=9 * tt - 250, n=60 * np.sin(tt / 9), u=0.02 * tt))
    S = Sim3(0.9987, Rot.from_euler("xyz", [1.7, 0.4, -0.7], degrees=True).as_matrix(), np.array([-1.9, -1.07, 4.24]))
    t_img = t_rtk[0] + np.arange(5_000_000_000, 55_000_000_000, 100_000_000, dtype=np.int64) + 123
    C = S.apply(interpolate_rtk(rtk, t_img))
    sim, res = fit_rtk_to_working(rtk, t_img, C, np.ones(len(t_img), bool))
    assert abs(sim.s - S.s) < 1e-9 and np.allclose(sim.R, S.R, atol=1e-9) and res.max() < 1e-6
    assert np.isnan(interpolate_rtk(rtk, np.array([t_rtk[0] - 1]))).all()

def _traj(n=50):
    t = np.arange(n, dtype=np.int64) * 100_000_000
    return t, np.stack([nadir_pose([9 * i * 0.1, 0, 80], 0) for i in range(n)])

def test_noise_injection():
    t, T = _traj(2000)
    Tn, info = inject_noise(T, NoiseSpec(), seed=1); assert np.allclose(Tn, T)
    spec = NoiseSpec(pos_h_sigma=0.5, pos_v_sigma=1.0, att_sigma_deg=0.3)
    Tn, info = inject_noise(T, spec, seed=1); Tn2, _ = inject_noise(T, spec, seed=1)
    assert np.array_equal(Tn, Tn2)                                          # deterministic
    d = Tn[:, :3, 3] - T[:, :3, 3]
    assert abs(d[:, 0].std() - 0.5) < 0.05 and abs(d[:, 2].std() - 1.0) < 0.1
    ang = np.degrees(Rot.from_matrix(np.einsum("nji,njk->nik", T[:, :3, :3], Tn[:, :3, :3])).magnitude())
    assert abs(np.sqrt(np.mean(ang ** 2)) / np.sqrt(3) - 0.3) < 0.03       # per-axis sigma
    assert np.allclose(np.linalg.det(Tn[:, :3, :3]), 1)
    src = NoisyPriorSource(TerraPoseSource(t, T, np.ones(len(t), bool)), "gnss", seed=4)
    ps = src.get(t); assert ps.meta["spec"] == PRESETS["gnss"].__dict__ and ps.meta["seed"] == 4

def test_anchor_detection():
    t, T = _traj(41)
    keys = np.arange(0, 41, 5)
    rng = np.random.default_rng(0)
    T[keys, :3, 3] += rng.normal(0, 0.3, (len(keys), 3))       # independent keyframes
    from aerodiag.geometry.se3 import interpolate_poses
    T = interpolate_poses(t, t[keys], T[keys])
    a = detect_anchors(t, T); assert np.array_equal(np.where(a)[0], keys)

def test_grid_stats():
    g = GroundGrid.from_bounds(0, 0, 10, 5, 1.0); assert (g.nx, g.ny) == (10, 5)
    rng = np.random.default_rng(0); xy = rng.uniform(0, 10, (5000, 2)) * [1, 0.5]; z = rng.normal(0, 1, 5000)
    ids = g.cell_id(xy); cnt, q = grouped_quantiles(ids, z, g.n, (0.5, 0.9))
    k = ids[0]; assert np.isclose(q[0, k], np.median(z[ids == k])) and np.isclose(q[1, k], np.quantile(z[ids == k], 0.9))
    assert cnt.sum() == 5000 and np.isclose(grouped_mad(ids, z, g.n, q[0])[k], 1.4826 * np.median(np.abs(z[ids == k] - np.median(z[ids == k]))))
    lab = (xy[:, 0] > 5).astype(np.uint8); mode, frac = grouped_mode(ids, lab, g.n, 2)
    assert mode[g.cell_id(np.array([[7.5, 2.5]]))[0]] == 1
    assert np.allclose(g.centers()[g.cell_id(np.array([[3.2, 4.7]]))[0]], [3.5, 4.5])
