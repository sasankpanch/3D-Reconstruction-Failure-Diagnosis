"""Camera-LiDAR calibration parsed from calibration_results.py AS DATA (ast), never imported.

Per scene entry: camera_intrinsic K (3x3, px), camera_dist_coeffs [k1,k2,p1,p2,k3] (OpenCV order),
camera_ext_R / camera_ext_t: LiDAR -> camera, p_C = R_CL p_L + t_CL (metres)   [verified Phase 0 §4/§5].
Scene -> entry via the file's own `scenename_to_calibration` table (HKairport_GNSS03 -> HK_GNSS_calibration).
"""
from __future__ import annotations
import ast
from dataclasses import dataclass, field
import numpy as np

def parse_calibration_file(path) -> tuple[dict, dict]:
    tree = ast.parse(open(path).read())
    entries, lookup = {}, {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)):
            continue
        name = node.targets[0].id
        vals = node.value.values
        if all(isinstance(v, ast.Name) for v in vals):
            lookup = {ast.literal_eval(k): v.id for k, v in zip(node.value.keys, vals)}
        else:
            entries[name] = ast.literal_eval(node.value)
    return entries, lookup

def project_to_so3(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt

@dataclass
class CameraLidarCalib:
    entry_name: str
    K: np.ndarray          # 3x3
    dist: np.ndarray       # (5,) k1 k2 p1 p2 k3
    R_CL: np.ndarray       # 3x3, orthonormalised
    t_CL: np.ndarray       # (3,) metres
    R_raw: np.ndarray = field(repr=False, default=None)

    @property
    def T_CL(self) -> np.ndarray:
        T = np.eye(4); T[:3, :3] = self.R_CL; T[:3, 3] = self.t_CL; return T

    def lidar_to_camera(self, p_L: np.ndarray) -> np.ndarray:
        return p_L @ self.R_CL.T + self.t_CL

    def camera_to_lidar(self, p_C: np.ndarray) -> np.ndarray:
        return (p_C - self.t_CL) @ self.R_CL

def load_scene_calibration(path, scene: str) -> CameraLidarCalib:
    entries, lookup = parse_calibration_file(path)
    if scene not in lookup:
        raise KeyError(f"scene {scene!r} not in scenename_to_calibration ({sorted(lookup)})")
    name = lookup[scene]; e = entries[name]
    R = np.array(e["camera_ext_R"], float).reshape(3, 3)
    return CameraLidarCalib(name, np.array(e["camera_intrinsic"], float).reshape(3, 3),
                            np.array(e["camera_dist_coeffs"], float), project_to_so3(R),
                            np.array(e["camera_ext_t"], float), R)
