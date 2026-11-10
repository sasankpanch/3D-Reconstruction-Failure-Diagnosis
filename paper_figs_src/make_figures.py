"""Paper figures (no pipeline changes). Fig 1: epipolar along/across concept. Fig 2: LiDAR-measured |height error|
vs out-of-fold predicted failure probability (each cell predicted by a model trained without its 25 m block)."""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, Rectangle
from aerodiag.config import load_config
from aerodiag.grid.ground_grid import GroundGrid
from aerodiag.diagnosis.classifier import prepare, fit_logreg, predict

INK, INK2, BLUE, RED, GREY = "#0b0b0b", "#52514e", "#2a78d6", "#e34948", "#9a9893"
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2})

# ---------------- Figure 1: concept
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.35), gridspec_kw=dict(width_ratios=[0.75, 1.4]))
for ax, title, (W, H) in zip(axes, ["Image 1", "Image 2"], [(10, 7), (14, 8)]):
    ax.add_patch(Rectangle((0, 0), W, H, fill=False, ec=INK2, lw=1)); ax.set_xlim(-0.3, W + 0.3); ax.set_ylim(-0.3, H + 0.9)
    ax.set_aspect("equal"); ax.axis("off"); ax.text(0.1, H + 0.15, title, color=INK, fontsize=9, weight="bold", va="bottom")
a = axes[0]
a.add_patch(Rectangle((4.3, 2.9), 1.4, 1.4, fc="#cde2fb", ec=BLUE, lw=1.2)); a.plot(5, 3.6, "o", color=BLUE, ms=4)
a.text(5, 2.5, "ground patch", ha="center", va="top", color=INK2, fontsize=7.5)
b = axes[1]
def on(t): return np.array([t, 1.6 + 0.242 * (t - 0.8)])
L0, L1 = on(0.8), on(13.2); b.plot([L0[0], L1[0]], [L0[1], L1[1]], "--", color=INK2, lw=1)
b.text(1.0, 1.25, "epipolar line", ha="left", va="top", color=INK2, fontsize=7.5)
p0, p1 = on(4.0), on(8.5)
n = np.array([-0.235, 0.972]); pv = p1 + 2.2 * n
b.add_patch(FancyArrowPatch(p0 + [0.3, 0.07], p1 - [0.3, 0.07], arrowstyle="-|>", mutation_scale=8, color=BLUE, lw=1.4))
b.add_patch(FancyArrowPatch(p1 + 0.3 * n, pv - 0.3 * n, arrowstyle="-|>", mutation_scale=8, color=RED, lw=1.4))
b.plot(*p0, "o", mfc="white", mec=INK2, ms=6, mew=1.2); b.plot(*p1, "o", color=BLUE, ms=6); b.plot(*pv, "o", color=RED, ms=6)
b.text(p0[0], p0[1] + 0.55, "predicted\n(terrain prior)", ha="center", va="bottom", color=INK2, fontsize=7)
mid = (p0 + p1) / 2; b.text(mid[0] + 0.3, mid[1] - 0.45, "along = height", ha="center", va="top", color=BLUE, fontsize=7.5, rotation=13.6)
b.text(p1[0] + 0.4, p1[1] - 0.4, "static match", ha="left", va="top", color=BLUE, fontsize=7.5)
b.text(p1[0] + 0.25, p1[1] + 1.55, "across = violation", ha="left", va="bottom", color=RED, fontsize=7.5)
b.text(pv[0] - 0.45, pv[1], "moving / reflective match", ha="right", va="center", color=RED, fontsize=7.5)
fig.tight_layout(w_pad=0.5); fig.savefig("figs/fig1_concept.pdf"); fig.savefig("figs/fig1_concept.png", dpi=200); plt.close(fig)

# ---------------- Figure 2: measured error vs out-of-fold predicted failure probability
dcfg = load_config("configs/data.yaml"); c3 = load_config("configs/step3.yaml")
grid = GroundGrid.from_json(open("outputs/track_a/grid.json").read())
F = pd.read_parquet("outputs/step3/cell_features.parquet"); lab = pd.read_parquet("outputs/step3/proxy_labels.parquet")
L = pd.read_parquet("outputs/track_a/labels.parquet").set_index("cell_id")
cols = [c for c in F.columns if c not in ("n_obs_fit", "pool_radius")]
xy = grid.centers()[F.index.values]; bm = c3["cv"]["block_m"]
blk = np.floor(xy[:, 0] / bm).astype(int) * 100003 + np.floor(xy[:, 1] / bm).astype(int)
ub = np.unique(blk); rng = np.random.default_rng(c3["seed"]); fo = dict(zip(ub, rng.integers(0, c3["cv"]["folds"], len(ub)))); fold = np.array([fo[b] for b in blk])
y = lab.failure.values; m = np.isfinite(y); p = np.full(len(F), np.nan)
for k in range(c3["cv"]["folds"]):
    tr = m & (fold != k); te = fold == k
    Xtr, cc, st = prepare(F[tr], cols); Xte, _, _ = prepare(F[te], cc, st)
    p[te] = predict(fit_logreg(Xtr, y[tr], l2=c3["cv"]["l2"]), Xte)
from aerodiag.diagnosis.classifier import auroc
print("AUROC of these out-of-fold maps on labelled cells:", round(auroc(y[m], p[m]), 3))
err = np.full(grid.n, np.nan); good = (~L.no_gt) & L.error.notna(); err[L.index[good]] = np.abs(L.error[good].values)
prob = np.full(grid.n, np.nan); prob[F.index.values] = p
both = np.isfinite(err) & np.isfinite(prob); err[~both] = np.nan; prob[~both] = np.nan
iy, ix = np.divmod(np.where(both)[0], grid.nx); y0, y1, x0, x1 = iy.min(), iy.max() + 1, ix.min(), ix.max() + 1
ext = [grid.x0 + x0, grid.x0 + x1, grid.y0 + y0, grid.y0 + y1]
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.75), sharey=True)
for ax, v, ttl, lo, hi, cl in [(axes[0], err, "Measured: |Terra $-$ LiDAR| height error", 0, 1.5, "m"),
                               (axes[1], prob, "Predicted: failure probability (out-of-fold)", 0, 1, "probability")]:
    cm = SEQ.copy(); cm.set_bad("#e6e5e1")
    im = ax.imshow(np.ma.masked_invalid(grid.to_image(v)[y0:y1, x0:x1]), origin="lower", extent=ext, cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
    ax.set_title(ttl, loc="left", fontsize=8.5, color=INK); ax.set_aspect("equal"); ax.tick_params(labelsize=7, colors=INK2)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, extend="max" if cl == "m" else "neither"); cb.set_label(cl, fontsize=7, color=INK2); cb.ax.tick_params(labelsize=6.5); cb.outline.set_visible(False)
    ax.set_xlabel("x (m)", fontsize=7.5, color=INK2)
axes[0].set_ylabel("y (m)", fontsize=7.5, color=INK2)
fig.tight_layout(w_pad=0.8); fig.savefig("figs/fig2_error_vs_prediction.pdf"); fig.savefig("figs/fig2_error_vs_prediction.png", dpi=200); plt.close(fig)
