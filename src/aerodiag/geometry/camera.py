"""Pinhole + Brown-Conrady (OpenCV 5-coef: k1,k2,p1,p2,k3) camera model, vectorised.

Pixel convention: OpenCV. Normalised coordinates x_n = (X/Z, Y/Z) in the camera frame (undistorted).
Phase 0 (F2): images are NOT undistorted; use the calibration-file K + dist ('calibration' model)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class CameraModel:
    K: np.ndarray
    dist: np.ndarray          # (5,) k1 k2 p1 p2 k3 ; zeros = pinhole
    width: int
    height: int

    @property
    def f(self) -> float:
        """Mean focal length (px), used to convert normalised residuals into pixels."""
        return float(0.5 * (self.K[0, 0] + self.K[1, 1]))

    def distort_normalized(self, xn: np.ndarray) -> np.ndarray:
        k1, k2, p1, p2, k3 = self.dist
        x, y = xn[:, 0], xn[:, 1]; r2 = x * x + y * y
        rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        return np.stack([xd, yd], 1)

    def normalized_to_pixel(self, xn: np.ndarray) -> np.ndarray:
        xd = self.distort_normalized(xn) if np.any(self.dist) else xn
        return np.stack([self.K[0, 0] * xd[:, 0] + self.K[0, 1] * xd[:, 1] + self.K[0, 2],
                         self.K[1, 1] * xd[:, 1] + self.K[1, 2]], 1)

    def pixel_to_normalized(self, uv: np.ndarray, iters: int = 30) -> np.ndarray:
        """Undistort pixels -> normalised coordinates (fixed-point iteration; < 1e-4 px error in-frame)."""
        uv = np.asarray(uv, float)
        yd = (uv[:, 1] - self.K[1, 2]) / self.K[1, 1]
        xd = (uv[:, 0] - self.K[0, 2] - self.K[0, 1] * yd) / self.K[0, 0]
        xdist = np.stack([xd, yd], 1)
        if not np.any(self.dist):
            return xdist
        k1, k2, p1, p2, k3 = self.dist; x = xdist.copy()
        for _ in range(iters):
            r2 = (x ** 2).sum(1); rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
            dx = 2 * p1 * x[:, 0] * x[:, 1] + p2 * (r2 + 2 * x[:, 0] ** 2)
            dy = p1 * (r2 + 2 * x[:, 1] ** 2) + 2 * p2 * x[:, 0] * x[:, 1]
            x = np.stack([(xdist[:, 0] - dx) / rad, (xdist[:, 1] - dy) / rad], 1)
        return x

    def project(self, X_c: np.ndarray, min_depth: float = 1e-6):
        """Camera-frame points (N,3) -> pixels (N,2), mask (in front of camera AND inside the image)."""
        z = X_c[:, 2]; front = z > min_depth
        xn = X_c[:, :2] / np.where(front, z, 1.0)[:, None]
        uv = self.normalized_to_pixel(xn)
        inside = front & (uv[:, 0] >= -0.5) & (uv[:, 0] < self.width - 0.5) & (uv[:, 1] >= -0.5) & (uv[:, 1] < self.height - 0.5)
        return uv, inside

    def bearing(self, uv: np.ndarray) -> np.ndarray:
        xn = self.pixel_to_normalized(uv); return np.c_[xn, np.ones(len(xn))]

    def radius(self, uv: np.ndarray) -> np.ndarray:
        """Distance (px) from the principal point."""
        return np.hypot(uv[:, 0] - self.K[0, 2], uv[:, 1] - self.K[1, 2])
