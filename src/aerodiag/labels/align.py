"""A4: align the Terra reconstruction to LiDAR with robust point-to-plane ICP, fitted only on STABLE cells.
Returns the rigid 4x4 transform T (applied to Terra points: X' = R X + t)."""
from __future__ import annotations
import numpy as np

def robust_icp(src: np.ndarray, dst: np.ndarray, max_corr=(1.0, 0.5, 0.25), tukey_k=0.15, voxel=0.25, init=None):
    import open3d as o3d
    reg = o3d.pipelines.registration
    ps = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src)).voxel_down_sample(voxel)
    pd_ = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(dst)).voxel_down_sample(voxel)
    pd_.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
    T = np.eye(4) if init is None else init; log = []
    for d in max_corr:
        r = reg.registration_icp(ps, pd_, d, T, reg.TransformationEstimationPointToPlane(reg.TukeyLoss(k=tukey_k)),
                                 reg.ICPConvergenceCriteria(max_iteration=80, relative_rmse=1e-7))
        T = r.transformation; log.append(dict(max_corr=d, fitness=r.fitness, rmse=r.inlier_rmse))
    return np.asarray(T), log

def describe_rigid(T: np.ndarray) -> dict:
    from scipy.spatial.transform import Rotation as Rot
    e = Rot.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)
    return dict(tx=T[0, 3], ty=T[1, 3], tz=T[2, 3], rx_deg=e[0], ry_deg=e[1], rz_deg=e[2],
                rot_deg=float(np.degrees(Rot.from_matrix(T[:3, :3]).magnitude())))
