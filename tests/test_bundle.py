import numpy as np
from scipy.spatial.transform import Rotation as Rot
from aerodiag.geometry.camera import CameraModel
from aerodiag.geometry.se3 import world_to_camera
from aerodiag.calib.bundle import BundleProblem
from conftest import nadir_pose

def test_bundle_recovers_focal_and_distortion_with_altitude_spread():
    rng = np.random.default_rng(0)
    K = np.array([[1460.0, 0, 1179.5], [0, 1460.0, 1044.9], [0, 0, 1]]); dist_true = np.array([-0.05, 0.11, 0.00122, 0.00064, -0.06])
    cam_true = CameraModel(K, dist_true, 2448, 2048)
    K0 = K.copy(); K0[0, 0] = K0[1, 1] = 1444.0; cam0 = CameraModel(K0, np.array([-0.056, 0.118, 0.00122, 0.00064, -0.0627]), 2448, 2048)
    X = np.c_[rng.uniform(-60, 60, (3000, 2)), rng.normal(0, 2.0, 3000)]
    poses = [nadir_pose([0, 0, h], 10, tuple(rng.normal(0, 1, 2))) for h in np.linspace(35, 80, 10)]      # climb
    poses += [nadir_pose([x, 5, 80], 10, tuple(rng.normal(0, 1, 2))) for x in np.linspace(-30, 30, 8)]     # line
    seg = np.r_[np.zeros(10, int), np.ones(8, int)]; lever = np.array([0.05, -0.1, 0.15])
    of, op, uv = [], [], []
    for f, T in enumerate(poses):
        u, ok = cam_true.project(world_to_camera(T, X)); ok &= (u[:, 0] > 5) & (u[:, 0] < 2443) & (u[:, 1] > 5) & (u[:, 1] < 2043)
        of.append(np.full(ok.sum(), f)); op.append(np.where(ok)[0]); uv.append(u[ok] + rng.normal(0, 0.3, (ok.sum(), 2)))
    of, op, uv = np.concatenate(of), np.concatenate(op), np.concatenate(uv)
    keep = np.bincount(op, minlength=len(X)) >= 3; m = keep[op]; remap = np.cumsum(keep) - 1
    Ts = np.stack(poses); a = Ts[:, :3, 3] + np.einsum("fij,j->fi", Ts[:, :3, :3], lever)          # antenna = centre + R l
    a[seg == 1] += [0.8, -0.5, 0.3]                                                                  # segment bias
    R0 = np.einsum("fij,fjk->fik", Ts[:, :3, :3], Rot.from_rotvec(rng.normal(0, np.radians(0.5), (len(Ts), 3))).as_matrix())
    bp = BundleProblem(a, R0, seg, of[m], remap[op[m]], uv[m], cam0)
    bp.solve(rounds=15)
    assert abs(bp.intr["f"] - 1460.0) < 1.5
    # distortion: compare displacement fields rather than coefficients
    xn = np.stack(np.meshgrid(np.linspace(-0.8, 0.8, 9), np.linspace(-0.7, 0.7, 9)), -1).reshape(-1, 2)
    assert np.abs(bp.camera().normalized_to_pixel(xn) - cam_true.normalized_to_pixel(xn)).max() < 1.0
    assert np.allclose(bp.o[1], [-0.8, 0.5, -0.3], atol=0.1) and np.allclose(bp.l, lever, atol=0.1)
