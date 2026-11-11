"""CUDA backend for the plane sweep (PyTorch). Same arithmetic as the numpy path in sweep.py; only the heavy parts
run on the GPU: projecting every (cell, hypothesis, sample) point into image 2, bilinear sampling, and the NCC cost.
Geometry is done in float64 relative to camera 2 and cast to float32 only for sampling; bilinear sampling is exact
(grid_sample, align_corners=True), whereas cv2.remap quantises sub-pixel positions to 1/32 px, so tiny differences
from the CPU path are expected (checked by tests/test_sweep_gpu.py)."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as Fnn

class GpuSweep:
    def __init__(self, cam, img1, img2, device="cuda"):
        self.dev = torch.device(device); self.cam = cam
        self.K = torch.tensor(cam.K, dtype=torch.float64, device=self.dev); self.dist = [float(v) for v in cam.dist]
        self.W, self.H = cam.width, cam.height
        self.img = [torch.as_tensor(np.ascontiguousarray(im, np.float32), device=self.dev)[None, None] for im in (img1, img2)]

    def _px(self, Xc):
        """camera-frame points (..., 3) float64 tensor -> distorted pixels (..., 2); NaN behind the camera."""
        z = Xc[..., 2]; zz = torch.where(z > 1e-6, z, torch.full_like(z, float("nan")))
        x = Xc[..., 0] / zz; y = Xc[..., 1] / zz; k1, k2, p1, p2, k3 = self.dist; r2 = x * x + y * y
        rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x); yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        K = self.K; return torch.stack([K[0, 0] * xd + K[0, 1] * yd + K[0, 2], K[1, 1] * yd + K[1, 2]], -1)

    def _sample_t(self, which, uv):
        """bilinear sample (exact) at pixel positions uv (..., 2) tensor; NaN outside the image or for NaN uv."""
        shp = uv.shape[:-1]; u = uv[..., 0].reshape(-1); v = uv[..., 1].reshape(-1)
        bad = ~torch.isfinite(u) | ~torch.isfinite(v) | (u < 0) | (v < 0) | (u > self.W - 1) | (v > self.H - 1)
        g = torch.stack([2 * u / (self.W - 1) - 1, 2 * v / (self.H - 1) - 1], -1).to(torch.float32)
        g = torch.nan_to_num(g, nan=-2.0)[None, None]
        s = Fnn.grid_sample(self.img[which], g, mode="bilinear", padding_mode="zeros", align_corners=True)[0, 0, 0]
        s = torch.where(bad, torch.full_like(s, float("nan")), s); return s.reshape(shp)

    def sample(self, which, uv):
        """numpy in / numpy out (float32)."""
        return self._sample_t(which, torch.as_tensor(np.asarray(uv, np.float64), device=self.dev)).cpu().numpy()

    def hypothesis_cost(self, D, Z, c1, R2, c2, ref):
        """D (N,S,3) ray directions from camera 1, Z (N,K) hypothesis heights, ref (N,S) reference samples.
        Returns cost = 1 - NCC (N,K) with NaN where any sample is invalid, plus the image-2 pixel positions of every
        hypothesis kept on the GPU for gathering the best one."""
        t = lambda a: torch.as_tensor(np.asarray(a, np.float64), device=self.dev)
        D, Z, c1t, R2t, c2t = t(D), t(Z), t(c1), t(R2), t(c2)
        lam = (Z[:, :, None] - c1t[2]) / D[:, None, :, 2]                                   # (N,K,S)
        Xc = ((c1t - c2t) + lam[..., None] * D[:, None]) @ R2t                               # camera-2 frame (N,K,S,3)
        uv2 = self._px(Xc); tgt = self._sample_t(1, uv2)                                    # (N,K,S)
        r = torch.as_tensor(np.asarray(ref, np.float32), device=self.dev)[:, None, :]
        valid = torch.isfinite(tgt).all(-1) & torch.isfinite(r).all(-1)
        tg = torch.nan_to_num(tgt); rr = torch.nan_to_num(r)
        am = rr - rr.mean(-1, keepdim=True); bm = tg - tg.mean(-1, keepdim=True)
        ncc = (am * bm).sum(-1) / torch.clamp(torch.sqrt((am * am).sum(-1) * (bm * bm).sum(-1)), min=1e-6)
        cost = torch.where(valid, 1 - ncc, torch.full_like(ncc, float("nan")))
        self._uv2, self._tgt = uv2, tgt
        return cost.cpu().numpy().astype(np.float64)

    def pixel_to_normalized(self, uv, iters=30):
        """undistort pixels (n,2) -> normalised coordinates, same fixed-point iteration as CameraModel (float64)."""
        uv = torch.as_tensor(np.asarray(uv, np.float64), device=self.dev); K = self.K; k1, k2, p1, p2, k3 = self.dist
        yd = (uv[:, 1] - K[1, 2]) / K[1, 1]; xd = (uv[:, 0] - K[0, 2] - K[0, 1] * yd) / K[0, 0]; x, y = xd.clone(), yd.clone()
        if any(self.dist):
            for _ in range(iters):
                r2 = x * x + y * y; rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
                dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x); dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
                x, y = (xd - dx) / rad, (yd - dy) / rad
        return torch.stack([x, y], 1).cpu().numpy()

    def ncc_ref(self, ref, uv):
        """NCC between ref (N,S) and image-2 samples at uv (N,A,S,2) -> (N,A); NaN where any sample is invalid
        (matches sweep._ncc whenever all samples are finite)."""
        t = self._sample_t(1, torch.as_tensor(np.asarray(uv, np.float64), device=self.dev))
        r = torch.as_tensor(np.asarray(ref, np.float32), device=self.dev)[:, None, :]
        valid = torch.isfinite(t).all(-1) & torch.isfinite(r).all(-1); tg = torch.nan_to_num(t); rr = torch.nan_to_num(r)
        am = rr - rr.mean(-1, keepdim=True); bm = tg - tg.mean(-1, keepdim=True)
        n = (am * bm).sum(-1) / torch.clamp(torch.sqrt((am * am).sum(-1) * (bm * bm).sum(-1)), min=1e-6)
        return torch.where(valid, n, torch.full_like(n, float("nan"))).cpu().numpy().astype(np.float64)

    def gather_best(self, kb):
        """image-2 pixels (N,S,2) and samples (N,S) at the best hypothesis index kb (N,)."""
        k = torch.as_tensor(kb, device=self.dev, dtype=torch.long); i = torch.arange(len(k), device=self.dev)
        return self._uv2[i, k].cpu().numpy(), self._tgt[i, k].cpu().numpy().astype(np.float64)
