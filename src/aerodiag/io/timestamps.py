"""Timestamp parsing. Epoch strings like '1698218978.199993134' -> int64 ns. float64 cannot represent epoch
times at ns resolution (spacing ~238 ns at 1.7e9 s), so strings are split, never converted through float."""
from __future__ import annotations
import numpy as np

def parse_ns(s: str) -> int:
    s = s.strip()
    if s.endswith(".jpg") or s.endswith(".txt"):
        s = s[:-4]
    if "." in s:
        sec, frac = s.split(".", 1)
    else:
        sec, frac = s, "0"
    if not (sec.lstrip("-").isdigit() and frac.isdigit()):
        raise ValueError(f"not an epoch timestamp: {s!r}")
    return int(sec) * 1_000_000_000 + int(frac.ljust(9, "0")[:9])

def parse_ns_array(strings) -> np.ndarray:
    return np.array([parse_ns(s) for s in strings], dtype=np.int64)

def ns_to_str(t: int) -> str:
    t = int(t); return f"{t // 1_000_000_000}.{t % 1_000_000_000:09d}"

def rel_seconds(t_ns: np.ndarray, t0: int) -> np.ndarray:
    """Seconds relative to t0, computed in int64 first (exact) then converted."""
    return (np.asarray(t_ns, dtype=np.int64) - np.int64(t0)).astype(np.float64) * 1e-9
