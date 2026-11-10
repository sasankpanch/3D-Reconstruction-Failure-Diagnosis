import numpy as np, pytest
from scipy.spatial.transform import Rotation as Rot
from aerodiag.geometry.se3 import make_T
from aerodiag.geometry.camera import CameraModel

@pytest.fixture
def cam():
    K = np.array([[1444.43, 0, 1179.5], [0, 1444.34, 1044.9], [0, 0, 1]])
    return CameraModel(K, np.array([-0.056, 0.118, 0.00122, 0.00064, -0.0627]), 2448, 2048)

def nadir_pose(c, yaw_deg=0.0, tilt=(0.0, 0.0)):
    """Camera looking down (+z_cam = -Z_world), x_cam = East rotated by yaw, plus small tilts (deg)."""
    R0 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)   # cam x=E, y=-N (down in image = south), z=-Up
    R = Rot.from_euler("z", yaw_deg, degrees=True).as_matrix() @ R0 @ Rot.from_euler("xy", tilt, degrees=True).as_matrix()
    return make_T(R, np.asarray(c, float))
