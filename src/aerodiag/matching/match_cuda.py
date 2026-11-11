"""CUDA versions of the pose-independent matching filters (exact ports of matching.match):
brute-force L2 2-NN with Lowe ratio and mutual check, and the local displacement-consistency filter (k-NN in image 1,
median and MAD over neighbours with the same interpolated median as numpy)."""
from __future__ import annotations
import numpy as np
import torch

DEV = torch.device("cuda")

def match_descriptors(d1, d2, ratio=0.8, mutual=True, block=4096):
    if len(d1) < 2 or len(d2) < 2:
        return np.zeros((0, 2), int)
    a = torch.as_tensor(np.asarray(d1, np.float32), device=DEV); b = torch.as_tensor(np.asarray(d2, np.float32), device=DEV)
    nn1, dist1, dist2 = [], [], []
    for s in range(0, len(a), block):                                   # 2-NN of each d1 row in d2 (L2 distance)
        D = torch.cdist(a[s:s + block], b); v, k = torch.topk(D, 2, dim=1, largest=False)
        nn1.append(k[:, 0]); dist1.append(v[:, 0]); dist2.append(v[:, 1])
    nn1, dist1, dist2 = torch.cat(nn1), torch.cat(dist1), torch.cat(dist2)
    keep = dist1 < ratio * dist2; q = torch.nonzero(keep).squeeze(1); t = nn1[keep]
    if mutual and len(q):
        back = []
        for s in range(0, len(b), block):                               # 1-NN of each d2 row in d1
            back.append(torch.cdist(b[s:s + block], a).argmin(1))
        back = torch.cat(back); m = back[t] == q; q, t = q[m], t[m]
    return torch.stack([q, t], 1).cpu().numpy().astype(int)

def local_consistency(uv1, uv2, k=12, tol_px=12.0, tol_mad=4.0):
    if len(uv1) <= k:
        return np.ones(len(uv1), bool)
    p1 = torch.as_tensor(np.asarray(uv1, np.float64), device=DEV); d = torch.as_tensor(np.asarray(uv2, np.float64), device=DEV) - p1
    nn = torch.topk(torch.cdist(p1, p1), k + 1, dim=1, largest=False).indices[:, 1:]   # self excluded (distance 0 first)
    dn = d[nn]                                                           # (n, k, 2)
    med = torch.quantile(dn, 0.5, dim=1)                                 # interpolated median == np.median
    mad = 1.4826 * torch.quantile((dn - med[:, None, :]).abs(), 0.5, dim=1)
    tol = torch.clamp(tol_mad * torch.linalg.norm(mad, dim=1), min=tol_px)
    return (torch.linalg.norm(d - med, dim=1) < tol).cpu().numpy()
