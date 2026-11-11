"""Tests for the step-3 additions: boosted trees, context features, conformal magnitude bounds, Platt calibration,
line-offset alignment, and spatial CV with a pluggable model."""
import numpy as np, pandas as pd
from aerodiag.diagnosis.gbdt import fit_gbdt, predict_gbdt
from aerodiag.diagnosis.context import add_context
from aerodiag.diagnosis.magnitude import magnitude_cv
from aerodiag.diagnosis.classifier import auroc, spatial_cv, blocks_of
from aerodiag.cli.calibrate import fit_platt, apply_platt, crossfit, ece
from aerodiag.calib.line_offsets import align
from aerodiag.grid.ground_grid import GroundGrid

def _data(n=4000, seed=0):
    rng = np.random.default_rng(seed); X = rng.normal(size=(n, 5))
    logit = 2 * X[:, 0] - 1.5 * (X[:, 1] > 0.5) + X[:, 2] * X[:, 3]           # non-linear signal
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(float); return X, y

def test_gbdt_learns_nonlinear_signal_and_handles_nan():
    X, y = _data(); X[::7, 1] = np.nan
    m = fit_gbdt(X[:3000], y[:3000], rounds=80); p = predict_gbdt(m, X[3000:])
    assert p.min() >= 0 and p.max() <= 1 and auroc(y[3000:], p) > 0.8

def test_gbdt_regression_mode():
    rng = np.random.default_rng(1); X = rng.normal(size=(3000, 3)); t = np.sin(X[:, 0]) * 2 + (X[:, 1] > 0)
    m = fit_gbdt(X[:2000], t[:2000], loss="l2", rounds=100); r = predict_gbdt(m, X[2000:])
    assert np.corrcoef(r, t[2000:])[0, 1] > 0.9

def test_context_features_local_mean_and_deviation():
    g = GroundGrid(x0=0.0, y0=0.0, cell=1.0, nx=10, ny=10)
    F = pd.DataFrame({"a": np.arange(100, dtype=float)}, index=np.arange(100)); F.loc[55, "a"] = np.nan
    C = add_context(F, g, ["a"], (3,))
    # interior cell 33 (ix=3, iy=3): 3x3 mean of a linear ramp = its own value
    assert np.isclose(C.loc[33, "a_ctx3"], 33.0) and np.isclose(C.loc[33, "a_dev3"], 0.0)
    assert np.isfinite(C.loc[55, "a_ctx3"]) and np.isnan(C.loc[55, "a_dev3"])        # NaN-aware mean, NaN deviation

def test_conformal_upper_bound_reaches_target_coverage():
    rng = np.random.default_rng(2); n = 6000; X = rng.normal(size=(n, 3)); e = np.exp(0.8 * X[:, 0] + rng.normal(0, 0.5, n)) * 0.2
    xy = rng.uniform(0, 500, (n, 2)); _, _, U, met = magnitude_cv(X, e, blocks_of(xy, 25.0), alpha=0.9, gbdt_kw=dict(rounds=60))
    assert abs(met["mondrian"]["coverage"] - 0.9) < 0.03 and met["spearman"] > 0.5

def test_platt_is_monotone_and_calibrates():
    rng = np.random.default_rng(3); n = 20000; z = rng.normal(size=n); y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.5 * z - 2)))).astype(float)
    s = 1 / (1 + np.exp(-3 * z))                                          # over-confident, wrongly centred scores
    ab = fit_platt(s, y); p = apply_platt(ab, s)
    assert ab[0] > 0 and np.all(np.diff(p[np.argsort(s)]) >= -1e-12)     # rank preserved
    assert ece(y, p)[0] < 0.01 < ece(y, s)[0]

def test_platt_rare_class_and_saturated_scores_converge():
    rng = np.random.default_rng(4); n = 50000; y = (rng.uniform(size=n) < 0.002).astype(float)
    s = np.where(y == 1, rng.uniform(0.5, 1, n), rng.uniform(0, 1e-9, n)); s[:100] = 1 - 1e-14   # separation + saturation
    a, b = fit_platt(s, y); assert np.isfinite(a) and np.isfinite(b) and a >= 0 and abs(a) < 1e3
    p = crossfit(s, y, np.arange(n) // 500); assert np.isfinite(p).mean() > 0.9

def test_line_offset_alignment_recovers_shift():
    rng = np.random.default_rng(5); xs, ys = np.meshgrid(np.arange(0, 80), np.arange(0, 80)); X = xs.ravel() + 0.5; Y = ys.ravel() + 0.5
    def z(x, y): return 3 * np.sin(x / 6) + 2 * np.cos(y / 5)             # terrain with relief
    a = pd.Series(z(X, Y), index=pd.MultiIndex.from_arrays([xs.ravel(), ys.ravel()]))
    d_true = np.array([1.2, -0.6, 0.4])                                    # b = a shifted: z_b(x) = z_a(x + d) - dz
    b = pd.Series(z(X + d_true[0], Y + d_true[1]) - d_true[2] + rng.normal(0, 0.02, len(X)), index=a.index)
    r = align(a, b); assert np.allclose(r["d"], d_true, atol=0.1) and r["cost"] < r["cost_at_zero"]

def test_spatial_cv_accepts_custom_model():
    X, y = _data(3000); xy = np.random.default_rng(6).uniform(0, 400, (3000, 2)); F = pd.DataFrame(index=np.arange(3000))
    oof, met = spatial_cv(F, X, pd.DataFrame({"h": y}), xy, 25.0, 5, 0, fit=lambda X_, y_: fit_gbdt(X_, y_, rounds=40), pred=predict_gbdt)
    assert oof.h.notna().mean() > 0.95 and met["h"]["auroc"] > 0.75
