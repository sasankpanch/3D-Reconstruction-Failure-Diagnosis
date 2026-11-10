"""Vectorised per-cell statistics (no Python loops over cells or points)."""
from __future__ import annotations
import numpy as np

def _grouped_sorted(ids: np.ndarray, z: np.ndarray, n_cells: int):
    m = ids >= 0; ids = ids[m]; z = z[m]
    order = np.lexsort((z, ids)); ids = ids[order]; z = z[order]
    counts = np.bincount(ids, minlength=n_cells); starts = np.r_[0, np.cumsum(counts)[:-1]]
    return z, counts, starts

def grouped_quantiles(ids, z, n_cells, qs=(0.5,)) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell quantiles (linear interpolation) of z. Returns (counts, (len(qs), n_cells) with NaN for empty)."""
    zs, counts, starts = _grouped_sorted(ids, z, n_cells)
    out = np.full((len(qs), n_cells), np.nan); has = counts > 0
    for k, q in enumerate(qs):
        pos = starts[has] + q * (counts[has] - 1); lo = np.floor(pos).astype(np.int64); hi = np.minimum(lo + 1, starts[has] + counts[has] - 1)
        w = pos - lo; out[k, has] = zs[lo] * (1 - w) + zs[hi] * w
    return counts, out

def grouped_mad(ids, z, n_cells, med: np.ndarray) -> np.ndarray:
    """Per-cell robust spread: 1.4826 * median(|z - median_cell|)."""
    m = ids >= 0; dev = np.abs(z[m] - med[ids[m]])
    _, q = grouped_quantiles(ids[m], dev, n_cells, (0.5,)); return 1.4826 * q[0]

def grouped_mode(ids, labels, n_cells, n_classes=256):
    """Per-cell dominant label and its fraction (255 / NaN for empty)."""
    m = ids >= 0
    h = np.bincount(ids[m].astype(np.int64) * n_classes + labels[m].astype(np.int64), minlength=n_cells * n_classes).reshape(n_cells, n_classes)
    tot = h.sum(1); mode = h.argmax(1).astype(np.uint8); mode[tot == 0] = 255
    frac = np.where(tot > 0, h.max(1) / np.maximum(tot, 1), np.nan)
    return mode, frac

def grouped_nunique(ids, values, n_cells) -> np.ndarray:
    m = ids >= 0; pairs = np.unique(np.c_[ids[m], values[m]], axis=0)
    return np.bincount(pairs[:, 0], minlength=n_cells)
