"""Coarse terrain (a DSM: canopy and roofs included) from SPARSE triangulated image points (bundle output).
Onboard-only: no LiDAR, no Terra. Median height per coarse cell -> hole filling (nearest) -> light smoothing ->
bilinear sampling at fine cells. Normals from the gradient of the coarse surface."""
from __future__ import annotations
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates
from ..grid.ground_grid import GroundGrid
from ..grid.rasterize import grouped_quantiles

class CoarseTerrain:
    def __init__(self, P: np.ndarray, coarse: float = 5.0, min_pts: int = 3, smooth_cells: float = 0.7, pad: float = 20.0):
        lo = P[:, :2].min(0); hi = P[:, :2].max(0)
        self.g = GroundGrid.from_bounds(lo[0], lo[1], hi[0], hi[1], coarse, pad)
        cnt, q = grouped_quantiles(self.g.cell_id(P[:, :2]), P[:, 2], self.g.n, (0.5,))
        z = q[0]; z[cnt < min_pts] = np.nan; img = self.g.to_image(z); self.support = self.g.to_image(cnt)
        miss = ~np.isfinite(img)
        if miss.any():
            _, (iy, ix) = distance_transform_edt(miss, return_indices=True); img = img[iy, ix]
        self.dist_to_data_m = distance_transform_edt(miss) * coarse
        self.z = gaussian_filter(img, smooth_cells) if smooth_cells > 0 else img
        gy, gx = np.gradient(self.z, coarse); self.gx, self.gy = gx, gy

    def _rc(self, xy):
        return [(xy[:, 1] - self.g.y0) / self.g.cell - 0.5, (xy[:, 0] - self.g.x0) / self.g.cell - 0.5]

    def height(self, xy):
        return map_coordinates(self.z, self._rc(xy), order=1, mode="nearest")

    def normal(self, xy):
        gx = map_coordinates(self.gx, self._rc(xy), order=1, mode="nearest"); gy = map_coordinates(self.gy, self._rc(xy), order=1, mode="nearest")
        n = np.stack([-gx, -gy, np.ones_like(gx)], -1); return n / np.linalg.norm(n, axis=-1, keepdims=True)

    def data_distance(self, xy):
        return map_coordinates(self.dist_to_data_m, self._rc(xy), order=0, mode="nearest")
