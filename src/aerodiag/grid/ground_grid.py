"""Ground grid in the working frame (x, y in metres; cell (ix, iy); cell_id = iy * nx + ix).
'Z' everywhere means the working-frame z axis (tilted ~1.8 deg from vertical; local errors are unaffected)."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import json
import numpy as np

@dataclass(frozen=True)
class GroundGrid:
    x0: float
    y0: float
    cell: float
    nx: int
    ny: int

    @staticmethod
    def from_bounds(xmin, ymin, xmax, ymax, cell=1.0, pad=0.0) -> "GroundGrid":
        x0 = np.floor((xmin - pad) / cell) * cell; y0 = np.floor((ymin - pad) / cell) * cell
        return GroundGrid(float(x0), float(y0), float(cell), int(np.ceil((xmax + pad - x0) / cell)),
                          int(np.ceil((ymax + pad - y0) / cell)))

    @property
    def n(self) -> int:
        return self.nx * self.ny

    def ixy(self, xy: np.ndarray):
        ix = np.floor((xy[:, 0] - self.x0) / self.cell).astype(np.int64)
        iy = np.floor((xy[:, 1] - self.y0) / self.cell).astype(np.int64)
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        return ix, iy, ok

    def cell_id(self, xy: np.ndarray) -> np.ndarray:
        """-1 outside the grid."""
        ix, iy, ok = self.ixy(xy); return np.where(ok, iy * self.nx + ix, -1)

    def centers(self) -> np.ndarray:
        iy, ix = np.divmod(np.arange(self.n), self.nx)
        return np.c_[self.x0 + (ix + 0.5) * self.cell, self.y0 + (iy + 0.5) * self.cell]

    def to_image(self, values: np.ndarray) -> np.ndarray:
        """(n,) -> (ny, nx) raster, row 0 = smallest y (use origin='lower' when plotting)."""
        return np.asarray(values).reshape(self.ny, self.nx)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(s: str) -> "GroundGrid":
        return GroundGrid(**json.loads(s))
