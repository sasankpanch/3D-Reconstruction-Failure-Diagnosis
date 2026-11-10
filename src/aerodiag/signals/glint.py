"""Glint (specular reflection) risk from sun position, viewing direction and surface normal.
theta = angle between the surface normal n and the half-vector h = unit(s + v) (s: toward the sun, v: toward the
camera). A mirror reflects the sun into the camera when theta = 0. risk = exp(-theta^2 / (2 sigma^2)).
The surface normal comes from the coarse terrain (onboard estimate), so tilted reflective surfaces are approximated."""
from __future__ import annotations
import numpy as np

def specular_angle_deg(normal, to_sun, to_cam):
    h = to_sun + to_cam; h /= np.linalg.norm(h, axis=-1, keepdims=True)
    n = normal / np.linalg.norm(normal, axis=-1, keepdims=True)
    return np.degrees(np.arccos(np.clip((n * h).sum(-1), -1, 1)))

def glint_risk(normal, to_sun, to_cam, sigma_deg=5.0):
    th = specular_angle_deg(normal, to_sun, to_cam); return np.exp(-0.5 * (th / sigma_deg) ** 2), th
