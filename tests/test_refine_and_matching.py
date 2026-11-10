import numpy as np
from scipy.spatial.transform import Rotation as Rot
from aerodiag.geometry.se3 import world_to_camera
from aerodiag.geometry.epipolar import relative_pose, decompose
from aerodiag.poses.refine import refine_rotation, refine_cv, apply_correction
from aerodiag.matching.match import local_consistency
from conftest import nadir_pose

def _pair(cam, seed=0, n=800, noise_px=0.3):
    rng = np.random.default_rng(seed); X = np.c_[rng.uniform(-40, 40, (n, 2)), rng.normal(0, 0.3, n)]
    T1 = nadir_pose([-4, 0, 80]); T2 = nadir_pose([4, 0, 80])
    u1, _ = cam.project(world_to_camera(T1, X)); u2, _ = cam.project(world_to_camera(T2, X))
    return T1, T2, cam.pixel_to_normalized(u1), cam.pixel_to_normalized(u2 + rng.normal(0, noise_px, u2.shape)), rng

def test_rotation_refinement_recovers_perturbation(cam):
    T1, T2, x1, x2, rng = _pair(cam)
    T2n = T2.copy(); T2n[:3, :3] = T2[:3, :3] @ Rot.from_euler("xyz", [0.1, -0.15, 0.2], degrees=True).as_matrix()
    R, t = relative_pose(T1, T2n); before = np.median(np.abs(decompose(x1, x2, R, t, cam.f)["across_px"]))
    w, info = refine_rotation(x1, x2, R, t, cam.f, rng=rng); R2, t2 = apply_correction(w, R, t)
    after = np.median(np.abs(decompose(x1, x2, R2, t2, cam.f)["across_px"]))
    assert before > 1.5 and after < 0.3
    assert abs(info["w_deg"] - 0.269) < 0.03

def test_refinement_on_exact_poses_is_noop(cam):
    T1, T2, x1, x2, rng = _pair(cam)
    R, t = relative_pose(T1, T2); fold, ws, _ = refine_cv(x1, x2, R, t, cam.f, folds=2, rng=rng)
    assert all(np.degrees(np.linalg.norm(w)) < 0.01 for w in ws) and set(np.unique(fold)) == {0, 1}

def test_local_consistency_keeps_coherent_motion_rejects_outliers():
    rng = np.random.default_rng(0); u1 = rng.uniform(0, 2000, (500, 2)); u2 = u1 + [50, 0]
    u2[:20] = rng.uniform(0, 2000, (20, 2))                       # gross mismatches
    ok = local_consistency(u1, u2)
    assert ok[:20].sum() <= 1 and ok[20:].mean() > 0.9
