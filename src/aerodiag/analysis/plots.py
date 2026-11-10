"""Plot helpers (matplotlib, static PNG). Colour roles follow the project data-viz palette:
sequential = one blue hue light->dark; diverging = blue <-> neutral grey <-> red; categorical slots in fixed
order; semantic classes use the dataset's own class colours with a legend."""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm, TwoSlopeNorm

SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div_blue_red", ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f4b0a6", "#e34948", "#8f1d1d"])
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NAN_COLOR = "#d9d8d4"
INK, INK2 = "#0b0b0b", "#52514e"

def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8); ax.title.set_color(INK)

def heatmap(grid, values, path, title, cbar_label, cmap=SEQ, vmin=None, vmax=None, center=None, figsize=(11, 7.5)):
    img = np.ma.masked_invalid(grid.to_image(np.asarray(values, float)))
    cm = cmap.copy(); cm.set_bad(NAN_COLOR)
    fig, ax = plt.subplots(figsize=figsize)
    ext = [grid.x0, grid.x0 + grid.nx * grid.cell, grid.y0, grid.y0 + grid.ny * grid.cell]
    norm = TwoSlopeNorm(vcenter=center, vmin=vmin, vmax=vmax) if center is not None else None
    im = ax.imshow(img, origin="lower", extent=ext, cmap=cm, vmin=None if norm else vmin, vmax=None if norm else vmax,
                   norm=norm, interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, shrink=0.8, extend="both" if (vmin is not None or vmax is not None) else "neither")
    cb.set_label(cbar_label, color=INK2); cb.outline.set_visible(False)
    ax.set_aspect("equal"); ax.set_xlabel("working-frame x (m, ~East)"); ax.set_ylabel("working-frame y (m, ~North)")
    ax.set_title(title, loc="left", fontsize=11); _style(ax)
    ax.text(0.0, -0.09, "grey = no data", transform=ax.transAxes, fontsize=8, color=INK2)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)

def class_map(grid, cls, path, title, cmap_dict, names, figsize=(11, 7.5)):
    import matplotlib.patches as mpatches
    present = [c for c in sorted(set(np.unique(cls)) - {255})]
    lut = np.zeros(256, int) - 1
    for k, c in enumerate(present):
        lut[c] = k
    idx = np.where(cls == 255, np.nan, lut[cls]).astype(float)
    colors = [np.array(cmap_dict.get(c, (128, 128, 128))) / 255 for c in present]
    cm = ListedColormap(colors); cm.set_bad(NAN_COLOR)
    fig, ax = plt.subplots(figsize=figsize)
    ext = [grid.x0, grid.x0 + grid.nx * grid.cell, grid.y0, grid.y0 + grid.ny * grid.cell]
    ax.imshow(np.ma.masked_invalid(grid.to_image(idx)), origin="lower", extent=ext, cmap=cm,
              norm=BoundaryNorm(np.arange(len(present) + 1) - 0.5, len(present)), interpolation="nearest")
    ax.legend(handles=[mpatches.Patch(color=colors[k], label=f"{present[k]} {names(present[k])}") for k in range(len(present))],
              loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8)
    ax.set_aspect("equal"); ax.set_xlabel("working-frame x (m, ~East)"); ax.set_ylabel("working-frame y (m, ~North)")
    ax.set_title(title, loc="left", fontsize=11); _style(ax); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)

def mask_map(grid, mask, path, title, true_label, false_label, figsize=(11, 7.5)):
    import matplotlib.patches as mpatches
    cm = ListedColormap(["#cde2fb", "#184f95"])
    fig, ax = plt.subplots(figsize=figsize)
    ext = [grid.x0, grid.x0 + grid.nx * grid.cell, grid.y0, grid.y0 + grid.ny * grid.cell]
    ax.imshow(grid.to_image(mask.astype(float)), origin="lower", extent=ext, cmap=cm, vmin=0, vmax=1, interpolation="nearest")
    ax.legend(handles=[mpatches.Patch(color="#184f95", label=true_label), mpatches.Patch(color="#cde2fb", label=false_label)],
              loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9)
    ax.set_aspect("equal"); ax.set_xlabel("working-frame x (m, ~East)"); ax.set_ylabel("working-frame y (m, ~North)")
    ax.set_title(title, loc="left", fontsize=11); _style(ax); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
