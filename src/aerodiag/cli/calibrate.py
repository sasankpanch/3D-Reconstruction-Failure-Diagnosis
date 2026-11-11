"""Probability calibration of the step-3 heads. aerodiag-calibrate [--data configs/data.yaml]

The step-3 classifier is class-balanced, so its out-of-fold (OOF) outputs rank cells but are not probabilities.
We map them to probabilities with Platt scaling (logistic fit on logit(score)), cross-fitted over 25 m spatial blocks:
the mapping applied to a block is fitted on OOF scores of the other folds only, so no cell calibrates itself.
Reported per head: Brier score and expected calibration error (ECE, 10 equal-count bins) before/after, and a
reliability diagram. Calibration is monotone, so AUROC is unchanged.
Output: outputs/<scene>/step3/calibration.json, reliability_<head>.png, calibrated_probabilities.parquet"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from ..config import load_config
from ..grid.ground_grid import GroundGrid
from ..diagnosis.classifier import blocks_of, auroc

def logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12); return np.log(p / (1 - p))     # loose clip: avoids ties among confident scores

def fit_platt(s, y, iters=100, ridge=1e-3):
    """Unweighted logistic regression of y on logit(s): p = sigmoid(a * logit(s) + b), damped Newton (step halving until
    the penalised log-loss decreases; small ridge keeps it finite under separation). The slope is kept >= 0 so the
    mapping is monotone and never reverses the ranking."""
    z = logit(s); X = np.c_[z, np.ones_like(z)]; w = np.array([1.0, 0.0]); n = len(y)
    def loss(w):
        u = np.clip(X @ w, -30, 30); return float(np.mean(np.logaddexp(0, u) - y * u) + ridge * w[0] ** 2 / 2)
    L = loss(w)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30))); g = X.T @ (p - y) / n + np.r_[ridge * w[0], 0]
        H = (X * (p * (1 - p))[:, None]).T @ X / n + np.diag([ridge, 1e-9]); step = np.linalg.solve(H, g); t = 1.0
        while t > 1e-6 and loss(w - t * step) > L:
            t /= 2
        w_new = w - t * step; L_new = loss(w_new)
        if L - L_new < 1e-12:
            break
        w, L = w_new, L_new
    return max(float(w[0]), 0.0), float(w[1])

def apply_platt(ab, s):
    a, b = ab; return 1 / (1 + np.exp(-np.clip(a * logit(s) + b, -30, 30)))

def ece(y, p, bins=10):
    o = np.argsort(p); e = 0.0; rows = []
    for idx in np.array_split(o, bins):
        if len(idx):
            rows.append((float(p[idx].mean()), float(y[idx].mean()), len(idx))); e += len(idx) * abs(p[idx].mean() - y[idx].mean())
    return float(e / len(y)), rows

def crossfit(s, y, blocks, folds=5, seed=1):
    ub = np.unique(blocks); rng = np.random.default_rng(seed); f = dict(zip(ub, rng.integers(0, folds, len(ub))))
    fold = np.array([f[b] for b in blocks]); out = np.full(len(s), np.nan)
    for k in range(folds):
        tr, te = fold != k, fold == k
        if te.any() and (y[tr] == 1).sum() >= 5:
            out[te] = apply_platt(fit_platt(s[tr], y[tr]), s[te])
    return out

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="configs/data.yaml"); ap.add_argument("--cfg", default="configs/step3.yaml")
    ap.add_argument("--model", default="gbdt", choices=["logreg", "gbdt"]); a = ap.parse_args(argv)
    pre, tag = ("oofgb_", "_gbdt") if a.model == "gbdt" else ("oof_", "")
    d, cfg = load_config(a.data), load_config(a.cfg); o = Path(d["root"]) / d["paths"]["outputs"] / "step3"
    D = pd.read_parquet(o / "cell_diagnosis.parquet"); L = pd.read_parquet(o / "proxy_labels.parquet").reindex(D.index)
    g = GroundGrid.from_json((o.parent / "track_a" / "grid.json").read_text()); blocks = blocks_of(g.centers()[D.index.values], cfg["cv"]["block_m"])
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from ..analysis.plots import CAT, INK2, _style
    res = {}; cal = pd.DataFrame(index=D.index)
    for h in L.columns:
        s = D[f"{pre}{h}"].values; y = L[h].values; m = np.isfinite(s) & np.isfinite(y)
        pc = crossfit(s[m], y[m], blocks[m], cfg["cv"]["folds"]); ok = np.isfinite(pc)
        yy, s0, s1 = y[m][ok], s[m][ok], pc[ok]
        if len(yy) < 50:
            print(h, "skipped: too few calibrated cells"); continue
        e0, r0 = ece(yy, s0); e1, r1 = ece(yy, s1)
        res[h] = dict(n=int(ok.sum()), base_rate=float(yy.mean()), brier_raw=float(np.mean((s0 - yy) ** 2)), brier_cal=float(np.mean((s1 - yy) ** 2)),
                      brier_base_rate_only=float(np.mean((yy.mean() - yy) ** 2)), ece_raw=e0, ece_cal=e1,
                      auroc_raw=auroc(yy, s0), auroc_cal=auroc(yy, s1), reliability_cal=r1)
        col = np.full(len(D), np.nan); col[np.where(m)[0][ok]] = s1; cal[f"pcal_{h}"] = col
        fig, ax = plt.subplots(figsize=(4.2, 4.2)); _style(ax); ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--")
        for rows, c, lab in [(r0, CAT[1], f"raw (ECE {e0:.3f})"), (r1, CAT[0], f"calibrated (ECE {e1:.3f})")]:
            ax.plot([r[0] for r in rows], [r[1] for r in rows], "-o", color=c, lw=2, ms=6, label=lab)
        ax.set_xlabel("predicted probability"); ax.set_ylabel("observed frequency"); ax.set_title(f"{d['scene']}: {h} ({a.model})", fontsize=10)
        ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(o / f"reliability_{h}{tag}.png", dpi=150); plt.close(fig)
        print(h, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[h].items() if k != "reliability_cal"})
    cal.to_parquet(o / f"calibrated_probabilities{tag}.parquet"); (o / f"calibration{tag}.json").write_text(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
