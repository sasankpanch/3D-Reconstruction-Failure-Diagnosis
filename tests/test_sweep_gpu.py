"""The CUDA sweep backend must reproduce the numpy backend (skipped without a CUDA device)."""
import numpy as np, pytest
torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
from aerodiag.signals.sweep import sweep_pair
from aerodiag.signals.sweep_gpu import GpuSweep
from test_sweep import CAM, CFG, nadir_pose, texture, render

@pytest.mark.parametrize("kind,shift,zplane", [("random", (0.0, 0.0), 0.0), ("random", (0.0, 0.1), 0.0), ("stripes", (0.0, 0.0), 0.0), ("random", (0.0, 0.0), -3.0)])
def test_gpu_matches_cpu(kind, shift, zplane):
    T1 = nadir_pose([-6, 0, 40]); T2 = nadir_pose([6, 0, 40]); tex = texture(kind)
    i1 = render(T1, zplane, tex); i2 = render(T2, zplane, tex, shift)
    g = np.stack(np.meshgrid(np.linspace(-8, 8, 9), np.linspace(-6, 6, 7)), -1).reshape(-1, 2)
    sun = np.array([[0.3, -0.4, 0.866]] * 2); nrm = np.tile([0, 0, 1.0], (len(g), 1)); z0 = np.full(len(g), 1.0)
    a = sweep_pair(i1, i2, CAM, T1[:3, :3], T1[:3, 3], T2[:3, :3], T2[:3, 3], g, z0, nrm, sun, CFG)
    b = sweep_pair(i1, i2, CAM, T1[:3, :3], T1[:3, 3], T2[:3, :3], T2[:3, 3], g, z0, nrm, sun, CFG, gpu=GpuSweep(CAM, i1, i2))
    assert np.array_equal(np.isfinite(a["cost_min"]), np.isfinite(b["cost_min"]))
    for k, tol in [("z_best", 0.02), ("across_px", 0.05), ("cost_min", 0.01), ("gain", 0.01)]:
        m = np.isfinite(a[k]) & np.isfinite(b[k]); assert m.mean() > 0.8 and np.median(np.abs(a[k][m] - b[k][m])) < tol, k
    assert np.nanmean(a["periodic"]) == pytest.approx(np.nanmean(b["periodic"]), abs=0.1)
