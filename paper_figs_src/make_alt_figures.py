"""Alternative paper figures. A: AUROC of model vs baselines (spatial-CV, 95% block-bootstrap CI).
B: real image crop with Track-B matches coloured by across-epipolar residual (rtk+ref poses)."""
import json, numpy as np, pandas as pd, matplotlib, cv2
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
INK, INK2, BLUE, GREY, ORANGE = "#0b0b0b", "#52514e", "#2a78d6", "#9a9893", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2})
R = json.load(open("outputs/step3/results.json"))
cv = R["cv_metrics"]["failure"]; st = R["failure_by_stratum"]; bl = R["baselines"]["failure"]; cls = R["failure_class_only_baseline"]
rows = [("Full model (ours)", cv, BLUE), ("  within vegetation", st["vegetation (green_field)"], BLUE), ("  outside vegetation", st["non-vegetation"], BLUE),
        ("Photometric residual only", bl["photometric (long 1-NCC p50)"], GREY), ("Across-line residual only", bl["across (long |across| p50)"], GREY),
        ("Below-terrain flag only", bl["below-terrain fraction"], GREY), ("Land-cover class only", cls, ORANGE), ("Low texture only", bl["low_texture (textureless_frac)"], GREY)]
fig, ax = plt.subplots(figsize=(3.3, 2.2))
for k, (name, m, c) in enumerate(rows):
    y = len(rows) - 1 - k; lo, hi = m["auroc_ci95"]
    ax.plot([lo, hi], [y, y], color=c, lw=2, solid_capstyle="round"); ax.plot(m["auroc"], y, "o", color=c, ms=5)
    ax.text(hi + 0.012, y, f"{m['auroc']:.2f}", va="center", fontsize=7, color=INK2)
ax.axvline(0.5, color=INK2, lw=0.8, ls=":"); ax.text(0.505, -0.9, "chance", fontsize=6.5, color=INK2)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=7)
ax.set_xlim(0.45, 1.0); ax.set_ylim(-1.2, len(rows) - 0.4); ax.set_xlabel("AUROC, height error $\\geq$ 0.5 m (95% CI)", fontsize=7, color=INK2)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.tick_params(labelsize=7, colors=INK2); fig.tight_layout(); fig.savefig("figs/figA_baselines.pdf"); fig.savefig("figs/figA_baselines.png", dpi=220); plt.close(fig)

# ---- B: real crop
from aerodiag.config import load_config
from aerodiag.io.dataset import Dataset
ds = Dataset(load_config("configs/data.yaml"))
Rs = pd.read_parquet("outputs/track_b/residuals.parquet", columns=["pair_id", "u2", "v2", "cell_id", "across_rtk+ref"])
P = pd.read_csv("outputs/track_b/pairs.csv", index_col=0); L = pd.read_parquet("outputs/track_a/labels.parquet").set_index("cell_id")
Rs = Rs.join(P[["pair_type", "gap_frames", "j"]], on="pair_id"); Rs = Rs[(Rs.pair_type == "within_line") & (Rs.gap_frames == 40)].dropna(subset=["across_rtk+ref"])
Rs["cls"] = L.dominant_class.reindex(Rs.cell_id.where(Rs.cell_id >= 0)).values
score = Rs.groupby("pair_id").apply(lambda d: d.cls.isin([9, 1]).sum() + 0.2 * d.cls.isin([10, 3, 18]).sum(), include_groups=False)
pid = int(score.idxmax()); d = Rs[Rs.pair_id == pid]; j = int(d.j.iat[0])
img = cv2.cvtColor(ds.load_image(j), cv2.COLOR_BGR2RGB)
c = d[d.cls.isin([9, 1])]; cx, cy = np.median(c.u2), np.median(c.v2); hw, hh = 520, 360
x0, y0 = int(np.clip(cx - hw, 0, img.shape[1] - 2 * hw)), int(np.clip(cy - hh, 0, img.shape[0] - 2 * hh))
crop = img[y0:y0 + 2 * hh, x0:x0 + 2 * hw]; m = (d.u2 > x0) & (d.u2 < x0 + 2 * hw) & (d.v2 > y0) & (d.v2 < y0 + 2 * hh); e = d[m]
cm = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
fig, ax = plt.subplots(figsize=(3.3, 2.45)); ax.imshow(crop, alpha=0.75)
a = np.abs(e["across_rtk+ref"].values); o = np.argsort(a)
sc = ax.scatter(e.u2.values[o] - x0, e.v2.values[o] - y0, c=np.clip(a[o], 0, 3), cmap=cm, vmin=0, vmax=3, s=7, edgecolors="none")
ax.set_xticks([]); ax.set_yticks([])
cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02); cb.set_label("|across-line residual| (px)", fontsize=7, color=INK2); cb.ax.tick_params(labelsize=6.5); cb.outline.set_visible(False)
fig.tight_layout(); fig.savefig("figs/figB_real_across.pdf", dpi=220); fig.savefig("figs/figB_real_across.png", dpi=220); plt.close(fig)
print("pair", pid, "image", j, "matches in crop", len(e), "median |across| roofs/containers", round(float(np.median(np.abs(e[e.cls.isin([9,1])]['across_rtk+ref']))), 2),
      "paved", round(float(np.median(np.abs(e[e.cls.isin([10,3,18])]['across_rtk+ref']))), 2) if e.cls.isin([10,3,18]).any() else None)
