"""CUDA port of diagnosis.accumulate (same outputs, same column names). Grouped statistics on the GPU in float64:
counts, means, maxima and linearly interpolated quantiles that skip NaN, exactly as pandas groupby median/quantile."""
from __future__ import annotations
import numpy as np, pandas as pd
import torch
from .accumulate import baseline_class, ABS, PER_CLASS

DEV = torch.device("cuda")

class Groups:
    """Rows -> dense group index for the cells present (sorted cell ids, like pandas groupby)."""
    def __init__(self, cell_ids: np.ndarray):
        self.cells, inv = np.unique(cell_ids, return_inverse=True); self.G = len(self.cells)
        self.g = torch.as_tensor(inv, device=DEV)

    def count(self, mask=None):
        g = self.g if mask is None else self.g[mask]; return torch.bincount(g, minlength=self.G).double()

    def sum(self, v, mask=None):
        v = v.double(); g = self.g
        if mask is not None:
            g, v = g[mask], v[mask]
        return torch.zeros(self.G, dtype=torch.float64, device=DEV).index_add_(0, g, v)

    def mean(self, v, mask=None):
        """NaN-skipping mean (pandas mean)."""
        ok = ~torch.isnan(v) if mask is None else (mask & ~torch.isnan(v)); n = self.count(ok)
        return torch.where(n > 0, self.sum(torch.where(ok, v, torch.zeros_like(v)), ok) / n.clamp(min=1), torch.full_like(n, float("nan")))

    def max(self, v, mask=None):
        ok = ~torch.isnan(v) if mask is None else (mask & ~torch.isnan(v))
        out = torch.full((self.G,), -float("inf"), dtype=torch.float64, device=DEV).scatter_reduce_(0, self.g[ok], v[ok].double(), "amax")
        return torch.where(self.count(ok) > 0, out, torch.full_like(out, float("nan")))

    def quantiles(self, v, qs, mask=None):
        """NaN-skipping linear-interpolated quantiles per group -> (G, len(qs)); NaN for empty groups."""
        ok = ~torch.isnan(v) if mask is None else (mask & ~torch.isnan(v)); g = self.g[ok]; x = v[ok].double()
        o = torch.argsort(x, stable=True); g, x = g[o], x[o]; o = torch.argsort(g, stable=True); g, x = g[o], x[o]   # by (g, x)
        n = torch.bincount(g, minlength=self.G); start = torch.cumsum(n, 0) - n; out = []
        for q in qs:
            pos = q * (n - 1).clamp(min=0).double(); lo = torch.floor(pos).long(); hi = torch.clamp(lo + 1, max=(n - 1).clamp(min=0))
            frac = pos - lo.double(); il = (start + lo).clamp(max=max(len(x) - 1, 0)); ih = (start + hi).clamp(max=max(len(x) - 1, 0))
            if len(x):
                xa, xb = x[il], x[ih]
                if q == 0.5:                                              # pandas median(): mean of the two middle values
                    val = torch.where(frac == 0, xa, (xa + xb) / 2)
                else:                                                     # pandas quantile(): a + (b - a) * frac
                    val = torch.where(frac == 0, xa, xa + (xb - xa) * frac)
            else:
                val = torch.zeros(self.G, dtype=torch.float64, device=DEV)
            out.append(torch.where(n > 0, val, torch.full_like(val, float("nan"))))
        return torch.stack(out, 1)

def accumulate(S: pd.DataFrame, max_rows=12_000_000) -> pd.DataFrame:
    """Large flights are processed in chunks of whole cells (cells are independent, so results are identical)."""
    if len(S) <= max_rows:
        return _accumulate(S)
    need = ["cell_id", "pair_type", "gap_frames"] + sorted(set(PER_CLASS) | {"cost_min", "gain", "glint_risk", "baseline_m", "heading_diff_deg",
                                                                          "periodic", "n_minima", "below_terrain", "at_boundary", "ref_std"})
    cid = S.cell_id.values; order = np.argsort(cid, kind="stable"); sc = cid[order]
    bounds = np.searchsorted(sc, np.unique(sc)[::max(1, int(np.ceil(len(np.unique(sc)) * max_rows / len(S))))])
    bounds = np.r_[bounds, len(sc)]; out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b > a:
            out.append(_accumulate(S.iloc[order[a:b]][need]))
    F = pd.concat(out).sort_index(); F.index.name = "cell_id"
    order = list(out[0].columns[: out[0].columns.get_loc("glint_gain_excess") + 1])
    for b in ["short", "mid", "long", "cross"]:
        order += [f"{b}_n"] + [f"{b}_{k}_p{q}" for k in PER_CLASS for q in (50, 90)]
    return F[[c for c in order if c in F.columns]]

def _accumulate(S: pd.DataFrame) -> pd.DataFrame:
    t = lambda c: torch.as_tensor(np.ascontiguousarray(S[c].values, dtype=np.float32), device=DEV).double()   # float64 on the GPU
    G = Groups(S.cell_id.values); idx = pd.Index(G.cells, name="cell_id")
    bcls = baseline_class(S.pair_type, S.gap_frames); cols = {k: (t(k).abs() if k in ABS else t(k)) for k in set(PER_CLASS) | {"cost_min", "gain"}}
    for k in ("glint_risk", "baseline_m", "heading_diff_deg", "periodic", "n_minima", "below_terrain", "at_boundary", "ref_std"):
        cols[k] = t(k)
    valid = ~torch.isnan(cols["cost_min"]); n_all = G.count(); nv = G.count(valid)
    np_ = lambda x: x.cpu().numpy()
    base = pd.DataFrame(index=idx)
    base["n_pairs"] = np_(n_all).astype(np.int64)
    base["textureless_frac"] = np_((n_all - nv) / n_all)
    base["glint_max"] = np_(G.max(cols["glint_risk"])); base["max_baseline_m"] = np_(G.max(cols["baseline_m"])); base["max_heading_diff"] = np_(G.max(cols["heading_diff_deg"]))
    base["n_valid"] = np_(nv)
    for name, v in [("periodic_frac", cols["periodic"]), ("below_frac", cols["below_terrain"]), ("boundary_frac", cols["at_boundary"])]:
        base[name] = np_(G.mean(v, valid))
    base["multi_min_frac"] = np_(G.mean((cols["n_minima"] >= 2).double(), valid))          # NaN counts as False (as in pandas apply)
    base["ref_std_med"] = np_(G.quantiles(cols["ref_std"], (0.5,), valid)[:, 0])
    gr = cols["glint_risk"]; gain = cols["gain"]
    hi = G.quantiles(gain, (0.5,), valid & (gr >= 0.3))[:, 0]; lo = G.quantiles(gain, (0.5,), valid & (gr < 0.05))[:, 0]
    base["glint_frac"] = np_(G.mean((gr >= 0.3).double(), valid))
    base["glint_gain_excess"] = np_(hi - lo)
    base.loc[base.n_valid == 0, ["periodic_frac", "multi_min_frac", "below_frac", "boundary_frac", "ref_std_med", "glint_frac"]] = np.nan
    out = [base]
    for b in ["short", "mid", "long", "cross"]:
        mb = valid & torch.as_tensor(bcls == b, device=DEV); nb = G.count(mb)
        f = pd.DataFrame(index=idx); f[f"{b}_n"] = np_(nb)
        for k in PER_CLASS:
            q = G.quantiles(cols[k], (0.5, 0.9), mb); f[f"{b}_{k}_p50"] = np_(q[:, 0]); f[f"{b}_{k}_p90"] = np_(q[:, 1])
        f = f[f[f"{b}_n"] > 0]; f[f"{b}_n"] = f[f"{b}_n"].astype(np.int64); out.append(f)
    F = pd.concat(out, axis=1); F["n_valid"] = F.n_valid.fillna(0)
    return F
