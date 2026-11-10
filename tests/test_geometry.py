import numpy as np
from scipy.spatial.transform import Rotation as Rot
from aerodiag.geometry.se3 import invert_T, make_T, world_to_camera, camera_to_world, interpolate_poses, umeyama, Sim3
from aerodiag.geometry.epipolar import relative_pose, decompose, triangulate_world, sampson, essential
from conftest import nadir_pose

def test_pose_roundtrip():
    rng = np.random.default_rng(0)
    T = make_T(Rot.random(5, random_state=1).as_matrix(), rng.normal(0, 100, (5, 3)))
    assert np.allclose(np.einsum("nij,njk->nik", T, invert_T(T)), np.eye(4), atol=1e-12)
    X = rng.normal(0, 50, (100, 3))
    for k in range(5):
        assert np.allclose(camera_to_world(T[k], world_to_camera(T[k], X)), X, atol=1e-9)

def test_camera_distortion_roundtrip(cam):
    uv = np.stack(np.meshgrid(np.linspace(0, 2447, 25), np.linspace(0, 2047, 21)), -1).reshape(-1, 2)
    back = cam.normalized_to_pixel(cam.pixel_to_normalized(uv))
    assert np.abs(back - uv).max() < 1e-4

def test_projection_convention(cam):
    T = nadir_pose([0, 0, 80])
    uv, ok = cam.project(world_to_camera(T, np.array([[0.0, 0, 0]])))
    assert ok[0] and np.allclose(uv[0], cam.K[:2, 2], atol=1e-9)           # nadir point -> principal point
    uv, _ = cam.project(world_to_camera(T, np.array([[10.0, 0, 0], [0, 10.0, 0]])))
    assert uv[0, 0] > cam.K[0, 2] and uv[1, 1] < cam.K[1, 2]                # East -> right, North -> up

def _scene(n=400, seed=0, terrain=lambda x, y: 0 * x):
    rng = np.random.default_rng(seed); xy = rng.uniform(-40, 40, (n, 2))
    return np.c_[xy, terrain(xy[:, 0], xy[:, 1])]

def _matches(cam, T1, T2, X):
    uv1, ok1 = cam.project(world_to_camera(T1, X)); uv2, ok2 = cam.project(world_to_camera(T2, X)); m = ok1 & ok2
    return cam.pixel_to_normalized(uv1[m]), cam.pixel_to_normalized(uv2[m]), X[m]

def test_static_plane_across_zero_and_height_recovered(cam):
    """Synthetic static planar scene, known poses: across ~ 0, along-line displacement recovers height."""
    X = _scene(terrain=lambda x, y: 0.0 * x)
    for gap_m, yaw2, tilt2 in [(0.9, 0, (0, 0)), (9.0, 0, (2, -3)), (36.0, 1.5, (5, 1)), (60.0, 90, (0, 4))]:
        T1 = nadir_pose([-gap_m / 2, 0, 80], 0, (1, 2)); T2 = nadir_pose([gap_m / 2, 3, 81], yaw2, tilt2)
        x1, x2, Xm = _matches(cam, T1, T2, X); R21, t21 = relative_pose(T1, T2)
        d = decompose(x1, x2, R21, t21, cam.f)
        assert np.abs(d["across_px"]).max() < 1e-6
        Xw = triangulate_world(x1, d["depth1"], T1)
        assert np.abs(Xw[:, 2] - Xm[:, 2]).max() < 1e-5
        assert np.all(d["along_px"] > 0)       # finite depth -> positive along (toward finite depths)

def test_along_encodes_height_and_motion_gives_across(cam):
    X = _scene(terrain=lambda x, y: 5 * np.sin(x / 7) + 0.1 * y)
    T1 = nadir_pose([-10, 0, 80]); T2 = nadir_pose([10, 0, 80]); R21, t21 = relative_pose(T1, T2)
    x1, x2, Xm = _matches(cam, T1, T2, X); d = decompose(x1, x2, R21, t21, cam.f)
    assert np.abs(triangulate_world(x1, d["depth1"], T1)[:, 2] - Xm[:, 2]).max() < 1e-5
    # higher terrain -> larger along displacement (closer to camera)
    assert np.corrcoef(Xm[:, 2], d["along_px"])[0, 1] > 0.99
    # an object moving perpendicular to the baseline between exposures -> across residual
    Xmov = Xm + np.array([0, 1.0, 0])             # 1 m north motion (baseline is east-west)
    uv2, _ = cam.project(world_to_camera(T2, Xmov)); d2 = decompose(x1, cam.pixel_to_normalized(uv2), R21, t21, cam.f)
    assert np.median(np.abs(d2["across_px"])) > 10    # ~1 m / 80 m * 1444 px = 18 px

def test_sampson_equals_across_for_small_rotation(cam):
    X = _scene(); T1 = nadir_pose([0, 0, 80]); T2 = nadir_pose([20, 0, 80]); R21, t21 = relative_pose(T1, T2)
    x1, x2, _ = _matches(cam, T1, T2, X); x2n = x2 + np.array([0, 2e-4])
    d = decompose(x1, x2n, R21, t21, cam.f)
    # Sampson spreads the error over both images -> ~ across / sqrt(2) when both line normals are similar
    assert np.allclose(np.abs(d["sampson_px"]) * np.sqrt(2), np.abs(d["across_px"]), rtol=0.05)

def test_interpolate_poses_endpoints():
    T = np.stack([nadir_pose([0, 0, 80], 0), nadir_pose([10, 0, 80], 20)]); t = np.array([0, 1_000_000_000], np.int64)
    Ti = interpolate_poses(np.array([0, 500_000_000, 1_000_000_000], np.int64), t, T)
    assert np.allclose(Ti[0], T[0]) and np.allclose(Ti[2], T[1]) and np.allclose(Ti[1, :3, 3], [5, 0, 80])

def test_umeyama_recovers_sim3():
    rng = np.random.default_rng(3); X = rng.normal(0, 100, (200, 3))
    S = Sim3(0.998, Rot.from_euler("xyz", [1.7, 0.2, -0.7], degrees=True).as_matrix(), np.array([-1.9, -1.1, 4.2]))
    est = umeyama(X, S.apply(X)); assert abs(est.s - S.s) < 1e-9 and np.allclose(est.R, S.R) and np.allclose(est.t, S.t)
    assert np.allclose(est.inverse().apply(S.apply(X)), X)
    assert abs(est.describe()["tilt_deg"] - np.degrees(np.arccos(S.R[2, 2]))) < 1e-9
