"""CUDA step-3 models / bootstrap must reproduce the numpy reference; AP must be tie-aware."""
import numpy as np, pytest
from aerodiag.diagnosis.classifier import average_precision, auroc, fit_logreg_np, predict_np
from aerodiag.diagnosis.classifier import block_bootstrap as bb_dispatch
from aerodiag.diagnosis.gbdt import fit_gbdt_np, predict_gbdt_np
from aerodiag import backend

def test_ap_tie_aware_hand_computed():
    # thresholds s>=1: tp=1/2 (P=.5, R=.5); s>=0: tp=2/4 (P=.5, R=1) -> AP = .5*.5 + .5*.5 = .5, for any tie order
    assert average_precision(np.array([1, 0, 1, 0]), np.array([1.0, 1, 0, 0])) == pytest.approx(0.5)
    assert average_precision(np.array([0, 1, 0, 1]), np.array([1.0, 1, 0, 0])) == pytest.approx(0.5)
    y = np.array([1, 0, 1, 1, 0]); s = np.array([0.9, 0.8, 0.7, 0.3, 0.1])     # distinct: mean precision at positives
    assert average_precision(y, s) == pytest.approx((1 / 1 + 2 / 3 + 3 / 4) / 3)

torch = pytest.importorskip("torch")
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")

def _data(n=5000, seed=0):
    r = np.random.default_rng(seed); X = r.normal(size=(n, 6)); X[::9, 2] = np.nan
    z = 1.5 * X[:, 0] - (np.nan_to_num(X[:, 2]) > 0.3) + X[:, 1] * X[:, 3]; y = (r.uniform(size=n) < 1 / (1 + np.exp(-z))).astype(float)
    return X, y

@cuda
def test_logreg_cuda_matches_numpy():
    from aerodiag.diagnosis import models_cuda as mc
    X, y = _data(); X = np.nan_to_num(X)
    a = predict_np(fit_logreg_np(X, y, 10.0), X); b = mc.predict_logreg(mc.fit_logreg(X, y, 10.0), X)
    assert np.max(np.abs(a - b)) < 1e-6

@cuda
@pytest.mark.parametrize("loss", ["logistic", "l2"])
def test_gbdt_cuda_matches_numpy_and_models_interchange(loss):
    from aerodiag.diagnosis import models_cuda as mc
    X, y = _data(); t = y if loss == "logistic" else X[:, 0] * 2 + y
    kw = dict(rounds=40, loss=loss); mn = fit_gbdt_np(np.nan_to_num(X, nan=np.nan), t, **kw); mg = mc.fit_gbdt(X, t, **kw)
    a, b = predict_gbdt_np(mn, X), mc.predict_gbdt(mg, X)
    assert np.max(np.abs(a - b)) < 1e-6 and np.max(np.abs(predict_gbdt_np(mg, X) - b)) < 1e-9

@cuda
def test_bootstrap_cuda_matches_numpy():
    r = np.random.default_rng(1); n = 4000; y = (r.uniform(size=n) < 0.2).astype(float); blocks = r.integers(0, 120, n)
    for s in (r.normal(size=n) + y, np.round(r.normal(size=n) + y, 0)):         # continuous and heavily tied
        for fn in (auroc, average_precision):
            backend.set_backend("cpu"); a = bb_dispatch(y, s, blocks, fn)
            backend.set_backend("cuda"); b = bb_dispatch(y, s, blocks, fn)
            assert np.allclose(a, b, atol=1e-9), (fn.__name__, a, b)
    backend.set_backend("auto")
