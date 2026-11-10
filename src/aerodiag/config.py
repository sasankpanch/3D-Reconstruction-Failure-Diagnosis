"""YAML config loading, hashing (for cache keys) and parameter logging."""
from __future__ import annotations
import copy, hashlib, json, os, platform, sys, time
from pathlib import Path
import yaml

def load_config(*paths: str | Path, overrides: dict | None = None) -> dict:
    """Load and deep-merge YAML files in order; `overrides` (dotted keys allowed) applied last."""
    cfg: dict = {}
    for p in paths:
        with open(p) as fh:
            _merge(cfg, yaml.safe_load(fh) or {})
    for k, v in (overrides or {}).items():
        d = cfg
        *head, last = k.split(".")
        for h in head:
            d = d.setdefault(h, {})
        d[last] = v
    return cfg

def _merge(a: dict, b: dict) -> dict:
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _merge(a[k], v)
        else:
            a[k] = copy.deepcopy(v)
    return a

def config_hash(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]

def write_params(out_dir: str | Path, name: str, params: dict) -> Path:
    """Log every parameter next to an output (JSON), with environment info."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rec = dict(params=params, created=time.strftime("%Y-%m-%dT%H:%M:%S"), python=sys.version.split()[0],
               platform=platform.platform(), argv=sys.argv)
    try:
        import numpy, scipy, cv2, open3d
        rec["versions"] = dict(numpy=numpy.__version__, scipy=scipy.__version__, opencv=cv2.__version__, open3d=open3d.__version__)
    except Exception:  # pragma: no cover
        pass
    p = out_dir / f"{name}.params.json"
    p.write_text(json.dumps(rec, indent=2, default=str))
    return p
