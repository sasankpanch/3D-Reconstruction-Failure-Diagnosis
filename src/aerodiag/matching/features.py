"""RootSIFT features with on-disk cache (outputs/cache/features/<frame name>.npz).
Keypoints in DISTORTED pixel coordinates (raw image); undistortion happens in geometry code.
texture = grey-level std in a square window around each keypoint (patch texture)."""
from __future__ import annotations
from multiprocessing import Pool
from pathlib import Path
import numpy as np, cv2

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

class FeatureStore:
    def __init__(self, ds, cache_dir, max_features=8000, win=21, workers=12):
        self.ds = ds; self.dir = Path(cache_dir) / f"features_sift{max_features}"; self.dir.mkdir(parents=True, exist_ok=True)
        self.max_features, self.win, self.workers = max_features, win, workers

    def path(self, i: int) -> Path:
        return self.dir / f"{self.ds.frames.name.iat[i]}.npz"

    def ensure(self, frames):
        todo = [i for i in sorted(set(int(f) for f in frames)) if not self.path(i).exists()]
        if todo:
            args = [(self.ds.p["cam_dir"] / f"{self.ds.frames.name.iat[i]}.jpg", self.path(i), self.max_features, self.win) for i in todo]
            with Pool(self.workers) as p:
                p.map(_extract, args, chunksize=2)
        return len(todo)

    def load(self, i: int) -> dict:
        with np.load(self.path(i)) as z:
            return dict(uv=z["uv"], des=z["des"].astype(np.float32), tex=z["tex"])
