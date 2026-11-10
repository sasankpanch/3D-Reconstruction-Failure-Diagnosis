import numpy as np, cv2, pytest
from aerodiag.geometry.camera import CameraModel
from aerodiag.signals.sweep import sweep_pair
from conftest import nadir_pose

CFG = dict(patch_half_m=0.6, patch_samples=9, n_hyp_half=40, step_px=0.5, range_m=8.0, min_ref_std=1.0, minima_delta=0.1,
           minima_min_sep=2, periodic_cv=0.25, across_range_px=6, sigma_match_px=0.5, below_k_sigma=3.0, below_tol_m=0.5, glint_sigma_deg=5.0)
CAM = CameraModel(np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]]), np.array([-0.05, 0.1, 0, 0, -0.05]), 640, 480)

def texture(kind, seed=0):
    rng = np.random.default_rng(seed); res = 0.02; n = 6000           # 120 m x 120 m, 2 cm texels
    if kind == "random":
        t = cv2.GaussianBlur(rng.uniform(0, 255, (n, n)).astype(np.float32), (0, 0), 2.0)
        t = (t - t.mean()) / t.std() * 40 + 128
    else:                                                             # stripes along y (period 0.5 m), vary along x
        x = (np.arange(n) * res)[None, :].repeat(n, 0); t = (128 + 60 * np.sin(2 * np.pi * x / 0.5)).astype(np.float32)
    return t.astype(np.float32), res, -60.0

def render(T, zplane, tex, shift=(0.0, 0.0)):
    t, res, o = tex; v, u = np.mgrid[0:CAM.height, 0:CAM.width]; uv = np.c_[u.ravel(), v.ravel()].astype(float)
    b = np.c_[CAM.pixel_to_normalized(uv), np.ones(len(uv))] @ T[:3, :3].T; lam = (zplane - T[2, 3]) / b[:, 2]
    X = T[:3, 3] + lam[:, None] * b
    mx = ((X[:, 0] - shift[0] - o) / res).reshape(CAM.height, CAM.width).astype(np.float32)
    my = ((X[:, 1] - shift[1] - o) / res).reshape(CAM.height, CAM.width).astype(np.float32)
    return cv2.remap(t, mx, my, cv2.INTER_LINEAR)

def run(zplane, z0, kind="random", shift=(0.0, 0.0)):
    T1 = nadir_pose([-6, 0, 40]); T2 = nadir_pose([6, 0, 40]); tex = texture(kind)
    i1 = render(T1, zplane, tex); i2 = render(T2, zplane, tex, shift)
    g = np.stack(np.meshgrid(np.linspace(-8, 8, 9), np.linspace(-6, 6, 7)), -1).reshape(-1, 2)
    sun = np.array([[0.3, -0.4, 0.866]] * 2); nrm = np.tile([0, 0, 1.0], (len(g), 1))
    return sweep_pair(i1, i2, CAM, T1[:3, :3], T1[:3, 3], T2[:3, :3], T2[:3, 3], g, np.full(len(g), z0), nrm, sun, CFG)

def test_height_recovered_and_across_zero():
    o = run(zplane=0.0, z0=1.5)
    assert np.nanmedian(np.abs(o["z_best"] - 0.0)) < 0.05 and np.nanmedian(np.abs(o["across_px"])) < 0.15
    assert np.nanmedian(o["cost_min"]) < 0.1 and np.nanmax(o["periodic"]) == 0

def test_across_shift_detected():
    o = run(zplane=0.0, z0=0.0, shift=(0.0, 0.1))         # texture moved 0.1 m north in image 2 (baseline is east-west)
    expect = 0.1 * 600 / 40                                 # 1.5 px across the (horizontal) epipolar lines
    assert abs(np.nanmedian(np.abs(o["across_px"])) - expect) < 0.3

def test_periodic_texture_flagged():
    o = run(zplane=0.0, z0=0.0, kind="stripes")
    assert np.nanmean(o["periodic"]) > 0.5 and np.nanmedian(o["n_minima"]) >= 3

def test_below_terrain_flag():
    o = run(zplane=-3.0, z0=0.0)
    assert np.nanmean(o["below_terrain"]) > 0.9 and abs(np.nanmedian(o["z_best"]) + 3.0) < 0.1
