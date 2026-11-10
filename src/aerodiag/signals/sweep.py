"""Per-cell plane sweep (1D stereo search along the epipolar line) and the step-2 signals.

For each cell visible in both images of a pair:
  * reference patch: a horizontal ground patch (n x n samples, half-size h metres) at the coarse terrain height z0,
    projected into image 1 and sampled there (fixed);
  * hypotheses: horizontal planes z_k = z0 + (k - K/2) * dz, where dz is chosen so that image 2 moves ~step_px per step
    (capped to +-range_m); each image-1 sample ray is intersected with plane z_k and projected into image 2
    -> the search runs exactly along the epipolar line;
  * cost c_k = 1 - NCC(reference, image-2 samples).
Signals (all per cell x pair):
  across_px          residual across the epipolar line: at the best hypothesis, image-2 samples are shifted
                     perpendicular to the local epipolar direction (+-across_range px), sub-pixel NCC peak.
  curve shape        cost_min, z_best (sub-step parabola), sharpness (curvature at the minimum, per px^2),
                     flatness (median cost - min cost), n_minima (local minima within minima_delta of the minimum,
                     >= 2 steps apart), periodic (>= 3 minima with near-constant spacing), period_px, at_boundary.
  photometric        photo_residual = cost_min; gain = log(mean2/mean1); contrast_ratio = std2/std1 at the best match.
  height plausibility dz = z_best - z0; sigma_z = sigma_match_px / (px per metre); dz_sigma = dz / sigma_z;
                     below_terrain = z_best < z0 - max(k_sigma * sigma_z, below_tol_m).
  glint risk          max over the two views of exp(-theta^2 / 2 sigma^2), theta = angle(normal, half-vector(sun, view)).
Pixels are raw (distorted) image pixels; geometry uses the calibrated camera model."""
from __future__ import annotations
import numpy as np, cv2
from ..geometry.camera import CameraModel
from .glint import glint_risk

def _sample(img, uv, width=16384):
    """Bilinear sampling of a float32 image at arbitrary-shaped pixel arrays (NaN outside). cv2.remap needs maps with
    both dimensions < 32767, so samples are laid out in rows of `width`."""
    shp = uv.shape[:-1]; u = uv[..., 0].astype(np.float32).ravel(); v = uv[..., 1].astype(np.float32).ravel(); n = len(u)
    rows = max(1, -(-n // width)); pad = rows * width - n
    u = np.r_[u, np.full(pad, -1e4, np.float32)].reshape(rows, width); v = np.r_[v, np.full(pad, -1e4, np.float32)].reshape(rows, width)
    u = np.nan_to_num(u, nan=-1e4); v = np.nan_to_num(v, nan=-1e4)
    m = cv2.remap(img, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    return m.ravel()[:n].reshape(shp)

def _project(cam: CameraModel, R, c, X):
    Xc = (X - c) @ R; z = Xc[..., 2]; zz = np.where(z > 1e-6, z, np.nan)
    xn = np.stack([Xc[..., 0] / zz, Xc[..., 1] / zz], -1); sh = xn.shape
    return cam.normalized_to_pixel(xn.reshape(-1, 2)).reshape(sh)

def _ncc(a, b, axis=-1):
    am = a - np.nanmean(a, axis, keepdims=True); bm = b - np.nanmean(b, axis, keepdims=True)
    num = np.nansum(am * bm, axis); den = np.sqrt(np.nansum(am * am, axis) * np.nansum(bm * bm, axis))
    return num / np.maximum(den, 1e-6)

def _parabola(c, k):
    """sub-step offset of the extremum of samples c[k-1], c[k], c[k+1] (arrays along axis 1)."""
    n = c.shape[1]; i = np.arange(len(k)); km = np.clip(k - 1, 0, n - 1); kp = np.clip(k + 1, 0, n - 1)
    a, b, cc = c[i, km], c[i, k], c[i, kp]; den = a - 2 * b + cc
    off = np.where((k > 0) & (k < n - 1) & (np.abs(den) > 1e-9), 0.5 * (a - cc) / den, 0.0)
    return np.clip(off, -0.5, 0.5), den

def curve_shape(cost, step_px, minima_delta=0.1, min_sep=2, periodic_cv=0.25):
    """Vectorised cost-curve descriptors. cost: (n, K)."""
    n, K = cost.shape; cfill = np.where(np.isfinite(cost), cost, np.inf); k = np.argmin(cfill, 1); i = np.arange(n)
    cmin = cfill[i, k]; off, den = _parabola(np.where(np.isfinite(cost), cost, 2.0), k)
    sharp = den / np.maximum(step_px, 1e-6) ** 2
    flat = np.nanmedian(cost, 1) - cmin
    loc = np.zeros_like(cfill, bool)
    loc[:, 1:-1] = (cfill[:, 1:-1] <= cfill[:, :-2]) & (cfill[:, 1:-1] <= cfill[:, 2:])
    loc[:, 0] = cfill[:, 0] <= cfill[:, 1]; loc[:, -1] = cfill[:, -1] <= cfill[:, -2]
    near = loc & (cfill <= cmin[:, None] + minima_delta)
    # merge minima closer than min_sep steps (keep first of each cluster)
    nm = np.zeros(n, int); per = np.zeros(n, bool); period = np.full(n, np.nan)
    rows, cols = np.nonzero(near)
    if len(rows):
        order = np.lexsort((cols, rows)); rows, cols = rows[order], cols[order]
        new = np.r_[True, (rows[1:] != rows[:-1]) | (cols[1:] - cols[:-1] >= min_sep)]
        rows, cols = rows[new], cols[new]
        nm = np.bincount(rows, minlength=n)
        st = np.r_[0, np.flatnonzero(np.diff(rows)) + 1]
        for s0, e0 in zip(st, np.r_[st[1:], len(rows)]):
            if e0 - s0 >= 3:
                sp = np.diff(cols[s0:e0]); cv = sp.std() / max(sp.mean(), 1e-9)
                if cv < periodic_cv:
                    per[rows[s0]] = True; period[rows[s0]] = sp.mean() * step_px[rows[s0]]
    return dict(k_best=k, k_off=off, cost_min=cmin, sharpness=sharp, flatness=flat, n_minima=nm, periodic=per,
                period_px=period, at_boundary=(k == 0) | (k == K - 1))

def sweep_pair(img1, img2, cam, R1, c1, R2, c2, xy, z0, normals, sun, cfg):
    """All signals for cells (xy, z0) for one pair. Returns dict of per-cell arrays (NaN where invalid)."""
    h, n_s, K2 = cfg["patch_half_m"], cfg["patch_samples"], cfg["n_hyp_half"]
    o = np.linspace(-h, h, n_s); dx, dy = [a.ravel() for a in np.meshgrid(o, o)]
    N = len(xy); S = len(dx)
    P0 = np.stack([xy[:, 0:1] + dx, xy[:, 1:2] + dy, np.repeat(z0[:, None], S, 1)], -1)         # (N,S,3)
    uv1 = _project(cam, R1, c1, P0); ref = _sample(img1, uv1)
    ok = np.isfinite(ref).all(1) & (np.nanstd(ref, 1) > cfg["min_ref_std"])
    # rays from camera 1 through the reference samples
    d = (uv1.reshape(-1, 2)); bn = cam.pixel_to_normalized(np.nan_to_num(d)); D = np.c_[bn, np.ones(len(bn))] @ R1.T
    D = D.reshape(N, S, 3)
    # pixels per metre in image 2 along the centre ray (local epipolar scale)
    ctr = S // 2
    def centre_at(z):                                                                            # centre-ray point at height z (N,)
        lam = (z - c1[2]) / D[:, ctr, 2]; return c1 + lam[:, None] * D[:, ctr]
    p0 = _project(cam, R2, c2, centre_at(z0)); p1 = _project(cam, R2, c2, centre_at(z0 + 1.0))
    ppm = np.linalg.norm(p1 - p0, axis=-1)                                                       # px per metre of height
    dz = np.maximum(cfg["step_px"] / np.maximum(ppm, 1e-6), cfg["range_m"] / K2)
    ks = np.arange(-K2, K2 + 1); Z = z0[:, None] + dz[:, None] * ks                             # (N,K)
    lam = (Z[:, :, None] - c1[2]) / D[:, None, :, 2]; X = c1 + lam[..., None] * D[:, None]       # (N,K,S,3)
    uv2 = _project(cam, R2, c2, X); tgt = _sample(img2, uv2)
    ncc = _ncc(ref[:, None, :], tgt); cost = 1 - ncc
    cost[~np.isfinite(tgt).all(-1)] = np.nan
    step_px = dz * ppm
    cs = curve_shape(cost, step_px, cfg["minima_delta"], cfg["minima_min_sep"], cfg["periodic_cv"])
    kb = cs["k_best"]; i = np.arange(N)
    z_best = z0 + dz * (ks[kb] + cs["k_off"])
    # photometric at the best hypothesis
    tb = tgt[i, kb]; gain = np.log(np.nanmean(tb, 1) / np.maximum(np.nanmean(ref, 1), 1e-3))
    contrast = np.nanstd(tb, 1) / np.maximum(np.nanstd(ref, 1), 1e-3)
    # across-epipolar residual at the best hypothesis
    Xb = X[i, kb]; uvb = uv2[i, kb]                                                              # (N,S,2)
    lamp = (z_best[:, None] + 0.5 - c1[2]) / D[..., 2]; uvp = _project(cam, R2, c2, c1 + lamp[..., None] * D)
    e = (uvp - uvb)[:, ctr]; e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9); nrm = np.stack([-e[:, 1], e[:, 0]], 1)
    sh = np.arange(-cfg["across_range_px"], cfg["across_range_px"] + 1)
    uva = uvb[:, None] + sh[None, :, None, None] * nrm[:, None, None, :]                       # (N,A,S,2)
    ta = _sample(img2, uva); na = _ncc(ref[:, None, :], ta)
    ka = np.argmax(np.where(np.isfinite(na), na, -2), 1); offa, _ = _parabola(-np.nan_to_num(na, nan=-2), ka)
    across = sh[ka] + offa; across[(ka == 0) | (ka == len(sh) - 1)] = np.nan                      # peak outside search range
    across_gain = na[i, ka] - na[i, len(sh) // 2]
    # height plausibility
    sigma_z = cfg["sigma_match_px"] / np.maximum(ppm, 1e-6); dzz = z_best - z0
    below = z_best < z0 - np.maximum(cfg["below_k_sigma"] * sigma_z, cfg["below_tol_m"])
    # glint (both views)
    Xc = np.c_[xy, z0]; v1 = c1 - Xc; v2 = c2 - Xc; v1 /= np.linalg.norm(v1, axis=1, keepdims=True); v2 /= np.linalg.norm(v2, axis=1, keepdims=True)
    g1, th1 = glint_risk(normals, sun[0], v1, cfg["glint_sigma_deg"]); g2, th2 = glint_risk(normals, sun[1], v2, cfg["glint_sigma_deg"])
    out = dict(across_px=across, across_ncc_gain=across_gain, cost_min=cs["cost_min"], sharpness=cs["sharpness"], flatness=cs["flatness"],
               n_minima=cs["n_minima"].astype(float), periodic=cs["periodic"].astype(float), period_px=cs["period_px"],
               at_boundary=cs["at_boundary"].astype(float), z_best=z_best, dz=dzz, sigma_z=sigma_z, dz_sigma=dzz / np.maximum(sigma_z, 1e-3),
               below_terrain=below.astype(float), photo_residual=cs["cost_min"], gain=gain, contrast_ratio=contrast,
               px_per_m=ppm, step_px=step_px, ref_std=np.nanstd(ref, 1), glint_risk=np.maximum(g1, g2), specular_deg=np.minimum(th1, th2),
               r1=np.hypot(uv1[:, ctr, 0] - cam.K[0, 2], uv1[:, ctr, 1] - cam.K[1, 2]), r2=np.hypot(uvb[:, ctr, 0] - cam.K[0, 2], uvb[:, ctr, 1] - cam.K[1, 2]))
    for k_ in out:
        out[k_] = np.where(ok, out[k_], np.nan).astype(np.float32)
    return out
