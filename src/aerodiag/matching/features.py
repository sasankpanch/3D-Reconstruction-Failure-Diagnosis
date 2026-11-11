"""RootSIFT features with on-disk cache (outputs/cache/features_<type><n>/<frame name>.npz).
Extractor (data config `features_type`): 'sift' = OpenCV SIFT on the CPU (reference); 'sift_cuda' = Kornia SIFT on the
GPU (same RootSIFT descriptor and keypoint convention, but its own detector, so keypoints differ from OpenCV's).
Keypoints in DISTORTED pixel coordinates (raw image); undistortion happens in geometry code.
texture = grey-level std in a square window around each keypoint (patch texture)."""
from __future__ import annotations
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd, cv2

def _extract(args):
    img_path, out_path, max_features, win = args
    if Path(out_path).exists():
        return out_path
    g = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create(nfeatures=max_features)
    kp, des = sift.detectAndCompute(g, None)
    uv = np.array([k.pt for k in kp], np.float32).reshape(-1, 2)
    des = des.astype(np.float32) if des is not None else np.zeros((0, 128), np.float32)
    des /= (des.sum(1, keepdims=True) + 1e-7); des = np.sqrt(des)          # RootSIFT
    gf = g.astype(np.float32); m = cv2.boxFilter(gf, -1, (win, win)); m2 = cv2.boxFilter(gf * gf, -1, (win, win))
    sd = np.sqrt(np.maximum(m2 - m * m, 0))
    ij = np.clip(np.round(uv).astype(int), 0, [g.shape[1] - 1, g.shape[0] - 1])
    tex = sd[ij[:, 1], ij[:, 0]].astype(np.float32)
    tmp = str(out_path) + ".tmp.npz"; np.savez(tmp, uv=uv, des=des.astype(np.float16), tex=tex); Path(tmp).replace(out_path)
    return out_path

def _texture(g, uv, win):
    gf = g.astype(np.float32); m = cv2.boxFilter(gf, -1, (win, win)); m2 = cv2.boxFilter(gf * gf, -1, (win, win))
    sd = np.sqrt(np.maximum(m2 - m * m, 0)); ij = np.clip(np.round(uv).astype(int), 0, [g.shape[1] - 1, g.shape[0] - 1])
    return sd[ij[:, 1], ij[:, 0]].astype(np.float32)

def rot90_points(uv, k, W, H):
    """Map points from an image rotated by np.rot90(img, k) back to the original image (W x H) pixel coordinates.
    np.rot90(img, 1): rotated[r, c] = img[c, W - 1 - r]  ->  x = W - 1 - y', y = x' (applied k times)."""
    uv = np.asarray(uv, np.float64).copy(); widths = [W if i % 2 == 0 else H for i in range(5)]
    for i in range(k % 4, 0, -1):                 # undo quarter turn i: the image before it had width widths[i - 1]
        x, y = uv[:, 0].copy(), uv[:, 1].copy(); uv[:, 0] = widths[i - 1] - 1 - y; uv[:, 1] = x
    return uv

def heading_quarter_turns(ds, sign=1):
    """Onboard heading normalisation: quarter turns k per frame so that every image is rotated to a common flight
    direction (within +-45 deg). Heading = RTK track direction (velocity over +-1.5 s); hovers keep the last heading."""
    from ..calib.onboard import rtk_enu
    enu = rtk_enu(ds); t = ds.frames.t_ns.values / 1e9; n = len(t); hd = np.full(n, np.nan)
    lo = np.searchsorted(t, t - 1.5); hi = np.clip(np.searchsorted(t, t + 1.5, "right") - 1, 0, n - 1)
    d = enu[hi, :2] - enu[lo, :2]; sp = np.linalg.norm(d, axis=1) / np.maximum(t[hi] - t[lo], 1e-3)
    ok = sp > 2.0; hd[ok] = np.degrees(np.arctan2(d[ok, 0], d[ok, 1]))                  # degrees from north, clockwise
    hd = pd.Series(hd).ffill().bfill().fillna(0.0).values
    return (sign * np.round(hd / 90.0)).astype(int) % 4

def _extract_cuda(items, max_features, win, turns=None):
    """Kornia SIFT on the GPU; JPEG decoding and the texture map overlap on CPU threads. turns: optional quarter
    turns per item (heading normalisation); keypoints are mapped back to original pixel coordinates exactly."""
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from kornia.feature import SIFTFeature, get_laf_center
    sf = SIFTFeature(num_features=max_features, upright=False, rootsift=True, device=torch.device("cuda"))
    load = lambda a: (a, cv2.imread(str(a[0]), cv2.IMREAD_GRAYSCALE))
    with ThreadPoolExecutor(4) as ex, torch.inference_mode():
        for n, ((img_path, out_path), g) in enumerate(ex.map(load, items)):
            k = 0 if turns is None else int(turns[n]); gr = np.ascontiguousarray(np.rot90(g, k)) if k else g
            lafs, resp, des = sf(torch.as_tensor(gr, device="cuda").float()[None, None] / 255.0)
            uv = get_laf_center(lafs)[0].cpu().numpy(); d = des[0].cpu().numpy().astype(np.float32)
            uv = (rot90_points(uv, k, g.shape[1], g.shape[0]) if k else uv).astype(np.float32)
            tmp = str(out_path) + ".tmp.npz"; np.savez(tmp, uv=uv, des=d.astype(np.float16), tex=_texture(g, uv, win)); Path(tmp).replace(out_path)

class FeatureStore:
    def __init__(self, ds, cache_dir, max_features=8000, win=21, workers=12, kind=None):
        self.kind = kind or ds.cfg.get("features_type", "sift"); assert self.kind in ("sift", "sift_cuda")
        self.rotate = ds.cfg.get("features_rotate", 0)                  # 0 = off, +1 / -1 = heading normalisation sign
        name = f"features_sift{max_features}" if self.kind == "sift" else f"features_siftcuda{max_features}" + (f"_rot{self.rotate:+d}" if self.rotate else "")
        self.ds = ds; self.dir = Path(cache_dir) / name
        self.dir.mkdir(parents=True, exist_ok=True); self.max_features, self.win, self.workers = max_features, win, workers

    def path(self, i: int) -> Path:
        return self.dir / f"{self.ds.frames.name.iat[i]}.npz"

    def ensure(self, frames):
        todo = [i for i in sorted(set(int(f) for f in frames)) if not self.path(i).exists()]
        if todo and self.kind == "sift_cuda":
            turns = heading_quarter_turns(self.ds, self.rotate)[todo] if self.rotate else None
            _extract_cuda([(self.ds.p["cam_dir"] / f"{self.ds.frames.name.iat[i]}.jpg", self.path(i)) for i in todo], self.max_features, self.win, turns)
        elif todo:
            args = [(self.ds.p["cam_dir"] / f"{self.ds.frames.name.iat[i]}.jpg", self.path(i), self.max_features, self.win) for i in todo]
            with Pool(self.workers) as p:
                p.map(_extract, args, chunksize=2)
        return len(todo)

    def load(self, i: int) -> dict:
        with np.load(self.path(i)) as z:
            return dict(uv=z["uv"], des=z["des"].astype(np.float32), tex=z["tex"])
