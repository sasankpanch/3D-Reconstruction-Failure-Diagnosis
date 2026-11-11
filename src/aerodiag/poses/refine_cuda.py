"""Batched rotation-only refinement on the GPU: all pairs solved at once.

Same problem as poses.refine.refine_rotation: minimise sum soft_l1(Sampson residual (px)) over a rotation correction
w (rotation vector, R21' = Exp(-w) R21, t21' = Exp(-w) t21, hence E(w) = Exp(-w) E0), matches subsampled to max_fit
with the same rng draw. Solver: Levenberg-Marquardt with IRLS weights of the soft_l1 loss (scipy's robust TRF
reaches the same minimum; differences are at the 1e-6 rad level, checked in tests), analytic-in-w finite-difference
Jacobian (3 columns), float64."""
from __future__ import annotations
import numpy as np
import torch

DEV = torch.device("cuda")

def _rodrigues(w):
    th = torch.linalg.norm(w, dim=-1, keepdim=True).clamp(min=1e-15); k = w / th; K = torch.zeros(w.shape[:-1] + (3, 3), dtype=w.dtype, device=w.device)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0], K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -k[..., 2], k[..., 1], k[..., 2], -k[..., 0], -k[..., 1], k[..., 0]
    I = torch.eye(3, dtype=w.dtype, device=w.device).expand_as(K); s = torch.sin(th)[..., None]; c = torch.cos(th)[..., None]
    return I + s * K + (1 - c) * (K @ K)

def _sampson(E, a, b):
    """E (P,3,3); a, b (P,M,3) homogeneous normalised points -> (P,M) signed Sampson distance."""
    Ex1 = a @ E.transpose(1, 2); Etx2 = b @ E; num = (b * Ex1).sum(-1)
    den = Ex1[..., 0] ** 2 + Ex1[..., 1] ** 2 + Etx2[..., 0] ** 2 + Etx2[..., 1] ** 2
    return num / torch.sqrt(den + 1e-300)

def refine_rotation_batch(items, f, f_scale_px=1.0, max_fit=1500, iters=50, rngs=None):
    """items: list of (x1, x2, R21, t21) (normalised coords). Returns list of (w, info) like refine_rotation."""
    P = len(items); sel = []
    for p, (x1, x2, R, t) in enumerate(items):
        if rngs is not None and rngs[p] is not None and len(x1) > max_fit:
            s = rngs[p].choice(len(x1), max_fit, replace=False); sel.append((x1[s], x2[s]))
        else:
            sel.append((x1, x2))
    M = max(len(a) for a, _ in sel); A = np.zeros((P, M, 3)); B = np.zeros((P, M, 3)); mask = np.zeros((P, M), bool)
    for p, (a, b) in enumerate(sel):
        A[p, :len(a), :2] = a; A[p, :len(a), 2] = 1; B[p, :len(b), :2] = b; B[p, :len(b), 2] = 1; mask[p, :len(a)] = True
    E0 = np.stack([_skew(t) @ R for _, _, R, t in items])
    a = torch.as_tensor(A, device=DEV); b = torch.as_tensor(B, device=DEV); m = torch.as_tensor(mask, device=DEV).double(); E0 = torch.as_tensor(E0, device=DEV)
    s2 = f_scale_px ** 2
    def resid(w):                                                        # (P,M) residual in px (0 on padding)
        return _sampson(_rodrigues(-w) @ E0, a, b) * f * m
    def cost(r):
        return (2 * (torch.sqrt(1 + r ** 2 / s2) - 1) * s2).sum(1) * 0.5  # scipy: 0.5 * sum rho(f^2)
    w = torch.zeros(P, 3, dtype=torch.float64, device=DEV); r = resid(w); c0 = cost(r); r0 = r.clone(); C = c0.clone()
    lam = torch.full((P,), 1e-3, dtype=torch.float64, device=DEV); h = 1e-7
    for _ in range(iters):
        J = torch.stack([(resid(w + h * torch.eye(3, dtype=torch.float64, device=DEV)[k]) - r) / h for k in range(3)], -1)   # (P,M,3)
        wt = 1 / torch.sqrt(1 + r ** 2 / s2) * m                         # soft_l1 IRLS weight rho'(z)
        JtW = J.transpose(1, 2) * wt[:, None, :]; H = JtW @ J; g = (JtW @ r[..., None])[..., 0]
        Hd = H + lam[:, None, None] * torch.diag_embed(torch.diagonal(H, dim1=1, dim2=2).clamp(min=1e-12))
        step = -torch.linalg.solve(Hd, g[..., None])[..., 0]; wn = w + step; rn = resid(wn); Cn = cost(rn)
        better = Cn < C
        w = torch.where(better[:, None], wn, w); r = torch.where(better[:, None], rn, r); C = torch.where(better, Cn, C)
        lam = torch.where(better, lam / 3, lam * 4)
        if bool((step.abs().max(1).values < 1e-10).all()):
            break
    W = w.cpu().numpy(); R0 = r0.abs().cpu().numpy(); R1 = r.abs().cpu().numpy(); out = []
    for p in range(P):
        n = int(mask[p].sum()); out.append((W[p], dict(cost0=float(np.median(R0[p, :n])), cost1=float(np.median(R1[p, :n])), nfev=-1,
                                                       w_deg=float(np.degrees(np.linalg.norm(W[p]))), n_fit=n)))
    return out

def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

def pixel_to_normalized(cam, uv, iters=30):
    """GPU undistortion, same fixed-point iteration as CameraModel.pixel_to_normalized (float64)."""
    uv = torch.as_tensor(np.asarray(uv, np.float64), device=DEV); K = cam.K; k1, k2, p1, p2, k3 = [float(v) for v in cam.dist]
    yd = (uv[:, 1] - K[1, 2]) / K[1, 1]; xd = (uv[:, 0] - K[0, 2] - K[0, 1] * yd) / K[0, 0]; x, y = xd.clone(), yd.clone()
    if any(cam.dist):
        for _ in range(iters):
            r2 = x * x + y * y; rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
            dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x); dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
            x, y = (xd - dx) / rad, (yd - dy) / rad
    return torch.stack([x, y], 1).cpu().numpy()
