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
    def __init__(self, cam, img1, img2, device="cuda", precision="float32"):
        """precision: geometry of the hypothesis projections. float32 (default; consumer and Jetson GPUs run float64
        ~32-64x slower) works on coordinates relative to camera 1 / 2, so metre-scale offsets keep ~1e-6 m precision
        (~1e-4 px); float64 is the bit-for-bit reference of the numpy path."""
        self.dev = torch.device(device); self.cam = cam; self.gdt = torch.float32 if precision == "float32" else torch.float64
        self.K = torch.tensor(cam.K, dtype=torch.float64, device=self.dev); self.dist = [float(v) for v in cam.dist]
        self.W, self.H = cam.width, cam.height
        self.img = [torch.as_tensor(np.ascontiguousarray(im, np.float32), device=self.dev)[None, None] for im in (img1, img2)]

    def _px(self, Xc):
        """camera-frame points (..., 3) float64 tensor -> distorted pixels (..., 2); NaN behind the camera."""
        z = Xc[..., 2]; zz = torch.where(z > 1e-6, z, torch.full_like(z, float("nan")))
        x = Xc[..., 0] / zz; y = Xc[..., 1] / zz; k1, k2, p1, p2, k3 = self.dist; r2 = x * x + y * y
        rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x); yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        K = self.K.to(Xc.dtype); return torch.stack([K[0, 0] * xd + K[0, 1] * yd + K[0, 2], K[1, 1] * yd + K[1, 2]], -1)

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
        t = lambda a: torch.as_tensor(np.asarray(a, np.float64), device=self.dev).to(self.gdt)
        # relative quantities formed in float64 on the host, then cast: heights above camera 1, camera-1 -> camera-2 offset
        D, Zr, base, R2t = t(D), t(np.asarray(Z, np.float64) - float(c1[2])), t(np.asarray(c1, np.float64) - np.asarray(c2, np.float64)), t(R2)
        lam = Zr[:, :, None] / D[:, None, :, 2]                                               # (N,K,S)
        Xc = (base + lam[..., None] * D[:, None]) @ R2t                                       # camera-2 frame (N,K,S,3)
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

# ---------------------------------------------------------------- fully-GPU sweep (every step of sweep.sweep_pair)
def _nanmedian_rows(c):
    """numpy.nanmedian along dim 1 (mean of the two middle values; +-inf kept; NaN rows -> NaN)."""
    n = (~torch.isnan(c)).sum(1); s, _ = torch.sort(torch.where(torch.isnan(c), torch.full_like(c, float("inf")), c), 1)
    # NaN were mapped to +inf and sort last; genuine +inf values are among the first n entries and stay correct
    lo = ((n - 1).clamp(min=0) // 2); hi = (n // 2).clamp(max=c.shape[1] - 1); i = torch.arange(len(c), device=c.device)
    a, b = s[i, lo], s[i, hi]; m = torch.where(a == b, a, (a + b) / 2)
    return torch.where(n > 0, m, torch.full_like(m, float("nan")))

def _nanstat(x):
    """row-wise nanmean, nanstd (ddof 0) of (N,S)."""
    ok = ~torch.isnan(x); n = ok.sum(1).double(); xs = torch.where(ok, x, torch.zeros_like(x))
    mu = xs.sum(1) / n; var = (torch.where(ok, (x - mu[:, None]) ** 2, torch.zeros_like(x))).sum(1) / n
    return mu, torch.sqrt(var)

def _ncc_nan(ref, b):
    """sweep._ncc with NaN skipped exactly as numpy: ref (N,S) finite rows, b (N,A,S) -> (N,A)."""
    am = ref - torch.nanmean(ref, -1, keepdim=True); okb = ~torch.isnan(b)
    nb = okb.sum(-1, keepdim=True); bmean = torch.where(okb, b, torch.zeros_like(b)).sum(-1, keepdim=True) / nb
    bm = torch.where(okb, b - bmean, torch.zeros_like(b)); amb = am[:, None, :]
    num = torch.where(okb, amb * bm, torch.zeros_like(bm)).sum(-1)
    den = torch.sqrt(torch.nansum(am * am, -1)[:, None] * (bm * bm).sum(-1))
    return num / torch.clamp(torch.nan_to_num(den, nan=0.0), min=1e-6)

def _parabola_t(c, k):
    n = c.shape[1]; i = torch.arange(len(k), device=c.device); km = (k - 1).clamp(0, n - 1); kp = (k + 1).clamp(0, n - 1)
    a, b, cc = c[i, km], c[i, k], c[i, kp]; den = a - 2 * b + cc
    off = torch.where((k > 0) & (k < n - 1) & (den.abs() > 1e-9), 0.5 * (a - cc) / den, torch.zeros_like(den))
    return off.clamp(-0.5, 0.5), den

def _curve_shape_t(cost, step_px, minima_delta, min_sep, periodic_cv):
    """sweep.curve_shape on the GPU (same minima merging, periodicity and parabola rules)."""
    n, K = cost.shape; fin = torch.isfinite(cost); cfill = torch.where(fin, cost, torch.full_like(cost, float("inf")))
    k = torch.argmin(cfill, 1); i = torch.arange(n, device=cost.device); cmin = cfill[i, k]
    off, den = _parabola_t(torch.where(fin, cost, torch.full_like(cost, 2.0)), k)
    sharp = den / torch.clamp(step_px, min=1e-6) ** 2; flat = _nanmedian_rows(cost) - cmin
    loc = torch.zeros_like(cfill, dtype=torch.bool)
    loc[:, 1:-1] = (cfill[:, 1:-1] <= cfill[:, :-2]) & (cfill[:, 1:-1] <= cfill[:, 2:])
    loc[:, 0] = cfill[:, 0] <= cfill[:, 1]; loc[:, -1] = cfill[:, -1] <= cfill[:, -2]
    near = loc & (cfill <= cmin[:, None] + minima_delta)
    col = torch.arange(K, device=cost.device)[None, :].expand(n, K); NEG = -10 ** 9
    def prev_excl(mask):                                                  # column of the previous True in the row (exclusive)
        idx = torch.where(mask, col, torch.full_like(col, NEG)); cm = torch.cummax(idx, 1).values
        return torch.cat([torch.full((n, 1), NEG, device=cost.device, dtype=cm.dtype), cm[:, :-1]], 1)
    pn = prev_excl(near); new = near & ((pn == NEG) | (col - pn >= min_sep))  # merge minima closer than min_sep (to previous raw minimum)
    nm = new.sum(1); pk = prev_excl(new); spm = new & (pk != NEG); sp = torch.where(spm, (col - pk).double(), torch.zeros_like(cost))
    cnt = spm.sum(1).double(); mu = sp.sum(1) / cnt.clamp(min=1); sd = torch.sqrt(torch.clamp((torch.where(spm, (sp - mu[:, None]) ** 2, torch.zeros_like(sp))).sum(1) / cnt.clamp(min=1), min=0))
    per = (nm >= 3) & (sd / torch.clamp(mu, min=1e-9) < periodic_cv); period = torch.where(per, mu * step_px, torch.full_like(mu, float("nan")))
    return dict(k_best=k, k_off=off, cost_min=cmin, sharpness=sharp, flatness=flat, n_minima=nm, periodic=per, period_px=period,
                at_boundary=(k == 0) | (k == K - 1))

def sweep_pair_cuda(g: "GpuSweep", R1, c1, R2, c2, xy, z0, normals, sun, cfg):
    """All of sweep.sweep_pair on the GPU. Geometry relative to camera 1 in g.gdt; per-cell results float64."""
    dev, f64, gd = g.dev, torch.float64, g.gdt; T = lambda a, dt=f64: torch.as_tensor(np.asarray(a, np.float64), device=dev).to(dt)
    h, n_s, K2 = cfg["patch_half_m"], cfg["patch_samples"], cfg["n_hyp_half"]
    o = torch.linspace(-h, h, n_s, dtype=f64, device=dev); dy, dx = torch.meshgrid(o, o, indexing="ij"); dx, dy = dx.reshape(-1), dy.reshape(-1)
    N, S = len(xy), len(dx); ctr = S // 2; c1d = np.asarray(c1, np.float64)
    xyr = T(np.asarray(xy, np.float64) - c1d[:2]); z0t = T(z0); z0r = z0t - c1d[2]; R1t, R2t = T(R1), T(R2); b21 = T(c1d - np.asarray(c2, np.float64))
    P0 = torch.stack([xyr[:, :1] + dx, xyr[:, 1:2] + dy, z0r[:, None].expand(N, S)], -1)                 # relative to camera 1
    uv1 = g._px(P0 @ R1t); ref = g._sample_t(0, uv1).double()
    mu_r, sd_r = _nanstat(ref); ok = torch.isfinite(ref).all(1) & (sd_r > cfg["min_ref_std"])
    bn = torch.as_tensor(g.pixel_to_normalized(torch.nan_to_num(uv1).reshape(-1, 2).cpu().numpy()), device=dev)
    D = (torch.cat([bn, torch.ones(len(bn), 1, dtype=f64, device=dev)], 1) @ R1t.T).reshape(N, S, 3)
    def centre_cam2(zr):                                                   # centre-ray point at relative height zr, in camera 2
        lam = zr / D[:, ctr, 2]; return (b21 + lam[:, None] * D[:, ctr]) @ R2t
    p0 = g._px(centre_cam2(z0r)); p1 = g._px(centre_cam2(z0r + 1.0)); ppm = torch.linalg.norm(p1 - p0, dim=-1)
    dz = torch.maximum(cfg["step_px"] / torch.clamp(ppm, min=1e-6), torch.full_like(ppm, cfg["range_m"] / K2))
    ks = torch.arange(-K2, K2 + 1, device=dev, dtype=f64); Zr = z0r[:, None] + dz[:, None] * ks           # relative heights (N,K)
    # hypotheses: projection, sampling, NCC (geometry in g.gdt)
    Dg = D.to(gd); lam = Zr.to(gd)[:, :, None] / Dg[:, None, :, 2]; Xc = (b21.to(gd) + lam[..., None] * Dg[:, None]) @ R2t.to(gd)
    uv2 = g._px(Xc); tgt = g._sample_t(1, uv2); r = ref.float()[:, None, :]
    valid = torch.isfinite(tgt).all(-1) & torch.isfinite(r).all(-1); tg = torch.nan_to_num(tgt); rr = torch.nan_to_num(r)
    am = rr - rr.mean(-1, keepdim=True); bm = tg - tg.mean(-1, keepdim=True)
    ncc = (am * bm).sum(-1) / torch.clamp(torch.sqrt((am * am).sum(-1) * (bm * bm).sum(-1)), min=1e-6)
    cost = torch.where(valid, 1 - ncc, torch.full_like(ncc, float("nan"))).double()
    step_px = dz * ppm; cs = _curve_shape_t(cost, step_px, cfg["minima_delta"], cfg["minima_min_sep"], cfg["periodic_cv"])
    kb = cs["k_best"]; i = torch.arange(N, device=dev); z_best_r = z0r + dz * (ks[kb] + cs["k_off"])
    uvb = uv2[i, kb].double(); tb = tgt[i, kb].double()
    mu_b, sd_b = _nanstat(tb); gain = torch.log(mu_b / torch.clamp(mu_r, min=1e-3)); contrast = sd_b / torch.clamp(sd_r, min=1e-3)
    # across-epipolar residual at the best hypothesis
    lamp = (z_best_r[:, None] + 0.5) / D[..., 2]; uvp = g._px((b21 + lamp[..., None] * D) @ R2t)
    e = (uvp - uvb)[:, ctr]; e = e / torch.clamp(torch.linalg.norm(e, dim=1, keepdim=True), min=1e-9); nrm = torch.stack([-e[:, 1], e[:, 0]], 1)
    A = cfg["across_range_px"]; sh = torch.arange(-A, A + 1, device=dev, dtype=f64)
    uva = uvb[:, None] + sh[None, :, None, None] * nrm[:, None, None, :]
    na = _ncc_nan(ref, g._sample_t(1, uva).double())
    ka = torch.argmax(torch.where(torch.isfinite(na), na, torch.full_like(na, -2.0)), 1); offa, _ = _parabola_t(-torch.nan_to_num(na, nan=-2.0), ka)
    across = sh[ka] + offa; across = torch.where((ka == 0) | (ka == len(sh) - 1), torch.full_like(across, float("nan")), across)
    across_gain = na[i, ka] - na[i, len(sh) // 2]
    sigma_z = cfg["sigma_match_px"] / torch.clamp(ppm, min=1e-6); dzz = z_best_r - z0r
    below = z_best_r < z0r - torch.clamp(cfg["below_k_sigma"] * sigma_z, min=cfg["below_tol_m"])
    # glint (both views)
    nt = T(normals); nt = nt / torch.linalg.norm(nt, dim=1, keepdim=True); Xcell = torch.cat([xyr, z0r[:, None]], 1)
    def glint(sv, vcam):
        v = vcam / torch.linalg.norm(vcam, dim=1, keepdim=True); hv = T(sv)[None, :] + v; hv = hv / torch.linalg.norm(hv, dim=1, keepdim=True)
        th = torch.rad2deg(torch.arccos(torch.clamp((nt * hv).sum(1), -1, 1))); return torch.exp(-0.5 * (th / cfg["glint_sigma_deg"]) ** 2), th
    g1, th1 = glint(sun[0], -Xcell); g2, th2 = glint(sun[1], -b21[None, :] - Xcell)
    Kc = g.K; r1 = torch.hypot(uv1[:, ctr, 0].double() - Kc[0, 2], uv1[:, ctr, 1].double() - Kc[1, 2]); r2 = torch.hypot(uvb[:, ctr, 0] - Kc[0, 2], uvb[:, ctr, 1] - Kc[1, 2])
    z_best = z_best_r + c1d[2]
    out = dict(across_px=across, across_ncc_gain=across_gain, cost_min=cs["cost_min"], sharpness=cs["sharpness"], flatness=cs["flatness"],
               n_minima=cs["n_minima"].double(), periodic=cs["periodic"].double(), period_px=cs["period_px"], at_boundary=cs["at_boundary"].double(),
               z_best=z_best, dz=dzz, sigma_z=sigma_z, dz_sigma=dzz / torch.clamp(sigma_z, min=1e-3), below_terrain=below.double(),
               photo_residual=cs["cost_min"], gain=gain, contrast_ratio=contrast, px_per_m=ppm, step_px=step_px, ref_std=sd_r,
               glint_risk=torch.maximum(g1, g2), specular_deg=torch.minimum(th1, th2), r1=r1, r2=r2)
    stacked = torch.stack([torch.where(ok, v.double(), torch.full_like(v.double(), float("nan"))) for v in out.values()]).float().cpu().numpy()
    return {k_: stacked[n_] for n_, k_ in enumerate(out)}
