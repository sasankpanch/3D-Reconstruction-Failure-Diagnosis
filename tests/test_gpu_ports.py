"""Each CUDA port must reproduce its CPU reference (skipped without a CUDA device)."""
import numpy as np, pandas as pd, pytest
torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
from aerodiag import backend

@pytest.fixture(autouse=True)
def _reset():
    yield; backend.set_backend("auto")

def test_matching_identical():
    from aerodiag.matching import match as M
    r = np.random.default_rng(0); d1 = r.normal(size=(1500, 128)).astype(np.float32); d2 = d1 + r.normal(0, 0.3, d1.shape).astype(np.float32)
    d2[::3] = r.normal(size=d2[::3].shape); uv1 = r.uniform(0, 2000, (1500, 2)); uv2 = uv1 + 30 + r.normal(0, 1, uv1.shape); uv2[::7] += 200
    backend.set_backend("cpu"); a = M.match_descriptors(d1, d2); la = M.local_consistency(uv1, uv2)
    backend.set_backend("cuda"); b = M.match_descriptors(d1, d2); lb = M.local_consistency(uv1, uv2)
    assert set(map(tuple, a)) == set(map(tuple, b)) and np.array_equal(la, lb)

def test_accumulate_identical():
    from aerodiag.diagnosis.accumulate import accumulate_np
    from aerodiag.diagnosis.accumulate_cuda import accumulate as acc
    r = np.random.default_rng(1); n = 60000
    S = pd.DataFrame(dict(cell_id=r.integers(0, 3000, n), pair_type=r.choice(["within_line", "cross_line"], n), gap_frames=r.choice([1, 5, 10, 40], n)))
    for c in ["across_px", "cost_min", "sharpness", "flatness", "gain", "contrast_ratio", "dz_sigma", "glint_risk", "baseline_m",
              "heading_diff_deg", "ref_std"]:
        S[c] = r.normal(size=n).astype(np.float32)
    for c in ["periodic", "below_terrain", "at_boundary"]:
        S[c] = (r.uniform(size=n) < 0.2).astype(np.float32)
    S["n_minima"] = r.integers(1, 4, n).astype(np.float32)
    S.loc[r.uniform(size=n) < 0.1, "cost_min"] = np.nan; S.loc[r.uniform(size=n) < 0.01, "cost_min"] = np.inf; S.loc[r.uniform(size=n) < 0.05, "across_px"] = np.nan
    A = accumulate_np(S); B = acc(S)[A.columns]
    assert A.index.equals(B.index)
    for c in A.columns:
        a, b = A[c].astype(float).values, B[c].astype(float).values
        assert np.array_equal(np.isnan(a), np.isnan(b)), c
        m = np.isfinite(a) & np.isfinite(b); assert np.allclose(a[m], b[m], atol=1e-5), c

def test_motion_fit_and_bins_identical():
    from aerodiag.diagnosis import motion_reflect as mr, gbdt
    r = np.random.default_rng(2); n = 50000; cell = r.integers(0, 2000, n); X = np.c_[np.ones(n), r.uniform(0, 40, n), r.uniform(0, 30, n)]; y = r.normal(size=n)
    backend.set_backend("cpu"); a = mr._cell_fit(cell, X, y, 2000); F = r.normal(size=(5000, 7)); F[::5, 2] = np.nan; e1 = gbdt.make_bins(F)
    backend.set_backend("cuda"); b = mr._cell_fit(cell, X, y, 2000); e2 = gbdt.make_bins(F)
    assert np.allclose(a[0], b[0], equal_nan=True, atol=1e-10) and np.allclose(a[1], b[1], equal_nan=True, atol=1e-8)
    assert all(np.array_equal(u, v) for u, v in zip(e1, e2))

def test_batched_rotation_refinement_matches_scipy():
    from aerodiag.poses.refine import refine_rotation
    from aerodiag.poses.refine_cuda import refine_rotation_batch
    from scipy.spatial.transform import Rotation as Rot
    r = np.random.default_rng(3); items = []
    for p in range(6):
        X = np.c_[r.uniform(-40, 40, 800), r.uniform(-40, 40, 800), r.uniform(70, 90, 800)]                # camera-1 frame
        R = Rot.from_rotvec(r.normal(0, 0.003, 3)).as_matrix(); t = np.array([-8.0, 0.5, 0.1])
        x1 = X[:, :2] / X[:, 2:]; X2 = X @ R.T + t; x2 = X2[:, :2] / X2[:, 2:] + r.normal(0, 3e-4, (800, 2))
        Rp = Rot.from_rotvec(r.normal(0, 0.002, 3)).as_matrix() @ R                                          # perturbed prior
        items.append((x1, x2, Rp, t))
    cpu = [refine_rotation(*it, 1450.0, rng=np.random.default_rng(p)) for p, it in enumerate(items)]
    gpu = refine_rotation_batch(items, 1450.0, rngs=[np.random.default_rng(p) for p in range(6)])
    assert max(np.degrees(np.linalg.norm(a[0] - b[0])) for a, b in zip(cpu, gpu)) < 1e-5

def test_curve_shape_cuda_matches_numpy():
    from aerodiag.signals.sweep import curve_shape
    from aerodiag.signals.sweep_gpu import _curve_shape_t
    r = np.random.default_rng(4); n, K = 3000, 97; x = np.arange(K)
    cost = 0.5 + 0.3 * np.sin(x[None, :] / r.uniform(1.5, 6, (n, 1)) + r.uniform(0, 6, (n, 1))) + r.normal(0, 0.02, (n, K))
    cost[r.uniform(size=(n, K)) < 0.03] = np.nan; cost[:20] = np.nan; step = r.uniform(0.5, 2, n)
    a = curve_shape(cost, step, 0.1, 2, 0.25)
    b = {k: v.cpu().numpy() for k, v in _curve_shape_t(torch.as_tensor(cost, device="cuda"), torch.as_tensor(step, device="cuda"), 0.1, 2, 0.25).items()}
    for k in ["k_best", "n_minima", "periodic", "at_boundary"]:
        assert np.array_equal(a[k], b[k]), k
    for k in ["k_off", "cost_min", "sharpness", "flatness", "period_px"]:
        assert np.allclose(a[k], b[k], equal_nan=True, atol=1e-9), k

def test_rot90_keypoint_mapping_exact():
    from aerodiag.matching.features import rot90_points
    H, W = 5, 7; img = np.arange(H * W).reshape(H, W)
    for k in range(4):
        r = np.rot90(img, k)
        for x, y in [(2, 1), (6, 4), (0, 0), (3, 2)]:
            yy, xx = np.argwhere(r == img[y, x])[0]
            assert np.allclose(rot90_points(np.array([[xx, yy]], float), k, W, H)[0], [x, y])
