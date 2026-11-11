"""Compute backend selection. 'auto' (default) = CUDA through PyTorch when a GPU is available, else numpy on the CPU.
The numpy code is the reference implementation; every CUDA path computes the same quantities (float64 where results
must match) and is checked against it in tests/. Override with AERODIAG_BACKEND=cpu|cuda|auto or set_backend()."""
from __future__ import annotations
import os

_STATE = {"mode": os.environ.get("AERODIAG_BACKEND", "auto"), "resolved": None, "pid": None}

def set_backend(mode: str):
    assert mode in ("auto", "cpu", "cuda"); _STATE.update(mode=mode, resolved=None)

def use_cuda() -> bool:
    """Resolved once per process. Initialises CUDA in this process when it returns True. A process forked after CUDA
    was initialised cannot use it, so forked workers fall back to the CPU reference (parallel CPU stages stay safe)."""
    if _STATE["resolved"] is not None and _STATE["pid"] != os.getpid():
        return False
    if _STATE["resolved"] is None:
        ok = False
        if _STATE["mode"] != "cpu":
            try:
                import torch
                ok = torch.cuda.is_available()
            except Exception:
                ok = False
            if _STATE["mode"] == "cuda" and not ok:
                raise RuntimeError("AERODIAG backend 'cuda' requested but no CUDA device / PyTorch available")
        _STATE["resolved"] = ok; _STATE["pid"] = os.getpid()
    return _STATE["resolved"]

def device():
    import torch
    return torch.device("cuda")
