"""A3: rasterise the Terra REFERENCE reconstruction onto the ground grid.
z_terra: median Z of cloud_merged points per cell. z_mesh: top-surface Z of the Terra mesh at the cell centre
(vertical ray cast from above). Both in the working frame."""
from __future__ import annotations
import numpy as np
from ..grid.ground_grid import GroundGrid
from ..grid.rasterize import grouped_quantiles

def rasterize_points(P: np.ndarray, grid: GroundGrid, min_pts=1):
    ids = grid.cell_id(P[:, :2]); cnt, q = grouped_quantiles(ids, P[:, 2], grid.n, (0.5,))
    z = q[0]; z[cnt < min_pts] = np.nan; return z, cnt

def mesh_top_z(mesh, xy: np.ndarray, z_start: float = 200.0) -> np.ndarray:
    import open3d as o3d
    scene = o3d.t.geometry.RaycastingScene(); scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    rays = np.c_[xy, np.full(len(xy), z_start), np.zeros((len(xy), 2)), -np.ones(len(xy))].astype(np.float32)
    t = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    return np.where(np.isfinite(t), z_start - t, np.nan)

def mesh_dz_points(mesh, X: np.ndarray, up: float = 60.0) -> np.ndarray:
    """Signed vertical distance point - mesh (first mesh hit below point + up)."""
    import open3d as o3d
    scene = o3d.t.geometry.RaycastingScene(); scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    rays = np.c_[X[:, :2], X[:, 2] + up, np.zeros((len(X), 2)), -np.ones(len(X))].astype(np.float32)
    t = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    return np.where(np.isfinite(t), X[:, 2] - (X[:, 2] + up - t), np.nan)
