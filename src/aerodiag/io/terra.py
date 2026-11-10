"""DJI Terra outputs (REFERENCE reconstruction, never ground truth). All in the working frame (Phase 0 §10).
cloud_merged.ply: points xyz float32 + RGB.  terra_ply/Block/Block.ply: textured mesh (same vertices as Mesh.ply).
metadata.xml: SRS 'ENU:lat,lon', SRSOrigin '0,0,h'."""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np

def read_metadata(terra_dir) -> tuple[float, float, float]:
    txt = (Path(terra_dir) / "terra_ply" / "metadata.xml").read_text()
    lat, lon = map(float, re.search(r"<SRS>ENU:([^,]+),([^<]+)</SRS>", txt).groups())
    h = float(re.search(r"<SRSOrigin>[^,]+,[^,]+,([^<]+)</SRSOrigin>", txt).group(1))
    return lat, lon, h

def load_cloud(terra_dir) -> np.ndarray:
    import open3d as o3d
    pc = o3d.io.read_point_cloud(str(Path(terra_dir) / "cloud_merged.ply"))
    return np.asarray(pc.points, dtype=np.float64)

def load_mesh(terra_dir):
    import open3d as o3d
    block = Path(terra_dir) / "terra_ply" / "Block" / "Block.ply"           # single-block export (HK sequences)
    return o3d.io.read_triangle_mesh(str(block if block.exists() else Path(terra_dir) / "Mesh.ply"))  # multi-block (AMtown): merged mesh
