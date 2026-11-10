"""Tiny on-disk cache for intermediates: outputs/cache/<name>_<hash>.npz. Re-runs skip completed work."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .config import config_hash

class Cache:
    def __init__(self, root: str | Path):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
    def path(self, name: str, key) -> Path:
        return self.root / f"{name}_{config_hash(key)}.npz"
    def get(self, name: str, key):
        p = self.path(name, key)
        if p.exists():
            with np.load(p, allow_pickle=False) as z:
                return {k: z[k] for k in z.files}
        return None
    def put(self, name: str, key, arrays: dict) -> Path:
        p = self.path(name, key); tmp = p.with_suffix(".tmp.npz")
        np.savez(tmp, **arrays); tmp.replace(p); return p
    def get_or_compute(self, name: str, key, fn):
        r = self.get(name, key)
        if r is None:
            r = fn(); self.put(name, key, r)
        return r
