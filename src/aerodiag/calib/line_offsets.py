"""Per-flight-line RTK offset correction from image tie geometry (onboard only: images, timestamps, RTK).

RTK is precise within a flight line but carries a line-dependent bias (decimetres to metres). Within a line the bias is
common to all frames, so within-line two-view triangulation is self-consistent but shifted by that line's bias.
  1. per line: rotation-refined within-line pairs (gap >= min_gap) -> triangulated points -> 1 m height raster
     (median per cell);
  2. per overlapping line pair (a, b): 3D shift d_ab = o_b - o_a that best aligns b's raster onto a's
     (horizontal grid search, vertical = robust median, cost = median |dz| over common cells); horizontal shifts are
     only trusted where the cost has a clear minimum (relief present), else only the vertical component is used;
  3. weighted least squares for one offset per line (gauge: weighted mean offset = 0, so the RTK->working similarity
     is unchanged), then frames outside lines get the offset interpolated in time between line mid-times.
Validation (onboard): across-epipolar residual of cross-line pairs before/after (rotation-refined), which must drop,
and within-line pairs, which must stay unchanged. Output: <out>/line_offsets/offsets.json, frame_offsets.npy, report.
Usage: python -m aerodiag.calib.line_offsets configs/data.yaml"""
from __future__ import annotations
import sys, json, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial.transform import Rotation as Rot
from ..config import load_config
from ..io.dataset import Dataset
from ..geometry.epipolar import relative_pose, decompose, triangulate_world
from ..poses.refine import refine_rotation

_G = {}

def _pair(pid):
    T, P, R, cam = _G["T"], _G["P"], _G["R"], _G["cam"]; off = _G.get("off")
    d = R.get(pid)
    if d is None or len(d) < 40:
        return pid, None
    i, j = int(P.i[pid]), int(P.j[pid]); Ti, Tj = T[i].copy(), T[j].copy()
    if off is not None:
        Ti[:3, 3] += off[i]; Tj[:3, 3] += off[j]
    x1 = cam.pixel_to_normalized(d[:, :2]); x2 = cam.pixel_to_normalized(d[:, 2:])
    Rr, t = relative_pose(Ti, Tj); w, _ = refine_rotation(x1, x2, Rr, t, cam.f, rng=np.random.default_rng(pid))
    Tj[:3, :3] = Tj[:3, :3] @ Rot.from_rotvec(w).as_matrix(); Rr, t = relative_pose(Ti, Tj); dd = decompose(x1, x2, Rr, t, cam.f)
    out = dict(across=np.abs(dd["across_px"]))
    if _G.get("triangulate"):
        ok = (dd["depth1"] > 10) & (dd["depth1"] < 300) & (np.abs(dd["across_px"]) < 2.0)
        out["pts"] = triangulate_world(x1[ok], dd["depth1"][ok], Ti)
    return pid, out

def raster(pts, cell=1.0):
    k = np.floor(pts[:, :2] / cell).astype(np.int64); df = pd.DataFrame(dict(x=k[:, 0], y=k[:, 1], z=pts[:, 2]))
    g = df.groupby(["x", "y"]).z; r = g.median()[g.size() >= 2]; return r

def _dense(r):
    x = np.array([k[0] for k in r.index]); y = np.array([k[1] for k in r.index]); x0, y0 = x.min(), y.min()
    A = np.full((y.max() - y0 + 1, x.max() - x0 + 1), np.nan); A[y - y0, x - x0] = r.values; return A, x0, y0

def _sample(A, x0, y0, X, Y):
    """Bilinear sample of raster A (cell centres at integer+0.5) at metric X, Y; NaN if any neighbour is missing."""
    u = X - x0 - 0.5; v = Y - y0 - 0.5; iu = np.floor(u).astype(int); iv = np.floor(v).astype(int); fu = u - iu; fv = v - iv
    ok = (iu >= 0) & (iv >= 0) & (iu < A.shape[1] - 1) & (iv < A.shape[0] - 1); out = np.full(len(X), np.nan)
    iu, iv, fu, fv = iu[ok], iv[ok], fu[ok], fv[ok]
    out[ok] = (A[iv, iu] * (1 - fu) * (1 - fv) + A[iv, iu + 1] * fu * (1 - fv) + A[iv + 1, iu] * (1 - fu) * fv + A[iv + 1, iu + 1] * fu * fv)
    return out

def align(ra, rb, search=3.0):
    """Shift d = (dx, dy, dz) moving b onto a: minimise median |z_a(x + dxy) - z_b(x) - dz| (coarse 0.25 m grid, then
    0.05 m refinement). Returns d, cost, cost at zero shift, median cost over the search window (relief), n."""
    A, x0, y0 = _dense(ra); bx = np.array([k[0] for k in rb.index]) + 0.5; by = np.array([k[1] for k in rb.index]) + 0.5; bz = rb.values
    def cost(dx, dy):
        za = _sample(A, x0, y0, bx + dx, by + dy); m = np.isfinite(za)
        if m.sum() < 200:
            return np.inf, np.nan, 0
        dz = np.median(za[m] - bz[m]); return float(np.median(np.abs(za[m] - bz[m] - dz))), float(dz), int(m.sum())
    grid = [(dx, dy) for dx in np.arange(-search, search + 1e-9, 0.25) for dy in np.arange(-search, search + 1e-9, 0.25)]
    C = {g: cost(*g) for g in grid}; fin = [v[0] for v in C.values() if np.isfinite(v[0])]
    if not fin:
        return None
    b = min(C, key=lambda g: C[g][0])
    for _ in range(2):                                                       # local refinement
        loc = [(b[0] + dx, b[1] + dy) for dx in np.arange(-0.2, 0.21, 0.05) for dy in np.arange(-0.2, 0.21, 0.05)]
        for g in loc:
            C[g] = cost(*g)
        b = min(C, key=lambda g: C[g][0])
    c, dz, n = C[b]; c0, dz0, _ = C.get((0.0, 0.0), cost(0.0, 0.0))
    return dict(d=np.array([b[0], b[1], dz]), cost=c, cost_at_zero=c0, cost_median=float(np.median(fin)), n=n, dz_at_zero=dz0)

def main(argv=None):
    cfgp = (argv or sys.argv[1:])[0]; dcfg = load_config(cfgp); ds = Dataset(dcfg); out = Path(ds.out) / "line_offsets"; out.mkdir(exist_ok=True, parents=True)
    from ..cli.track_b import TrackB
    tb = TrackB.__new__(TrackB); tb.ds = ds; tb.cfg = load_config(Path(dcfg["root"]) / "configs/track_b.yaml"); tb.dcfg = dcfg; tb.results = {}
    T = tb.build_sources()["rtk"].T_wc; f = ds.frames; line = f.line_id.values
    P = pd.read_csv(Path(ds.out) / "track_b/pairs_selected.csv", index_col=0)
    Rm = pd.read_parquet(Path(ds.out) / "track_b/residuals.parquet", columns=["pair_id", "u1", "v1", "u2", "v2"])
    R = {int(p): d[["u1", "v1", "u2", "v2"]].values.astype(float) for p, d in Rm.groupby("pair_id")}
    cam = ds.camera; t0 = time.time(); res = {}
    wl = P[(P.pair_type == "within_line") & (P.gap_frames >= 20) & (line[P.i.values] >= 0) & (line[P.i.values] == line[P.j.values])]
    _G.update(T=T, P=P, R=R, cam=cam, triangulate=True, off=None)
    with Pool(12) as pool:
        tri = dict(pool.map(_pair, wl.index.tolist(), chunksize=4))
    lines = sorted(set(line[line >= 0])); pts = {L: [] for L in lines}
    for pid, o in tri.items():
        if o is not None:
            pts[line[int(P.i[pid])]].append(o["pts"])
    ras = {L: raster(np.concatenate(v)) for L, v in pts.items() if v}
    print(time.strftime("%H:%M:%S"), "per-line rasters:", {L: len(r) for L, r in ras.items()}, f"{time.time() - t0:.0f}s", flush=True)
    edges = []
    for a in ras:
        for b in ras:
            if a < b:
                r = align(ras[a], ras[b])
                if r is None:
                    continue
                relief = r["cost_median"] / max(r["cost"], 1e-3)               # how pronounced the horizontal minimum is
                r.update(a=int(a), b=int(b), relief_ratio=float(relief), horizontal_trusted=bool(relief >= 1.5)); edges.append(r)
                print(f"  lines {a}-{b}: n={r['n']} shift(b->a) {np.round(r['d'], 2)} cost {r['cost']:.3f} (at 0: {r['cost_at_zero']:.3f}) relief {relief:.2f}", flush=True)
    # least squares: o_b - o_a = -d_ab   (d aligns b onto a, i.e. b's points must move by d -> b's bias is -d relative to a)
    idx = {L: k for k, L in enumerate(ras)}; n = len(idx); O = np.zeros((n, 3))
    for c in range(3):
        rows, rhs, wts = [], [], []
        for e in edges:
            if c < 2 and not e["horizontal_trusted"]:
                continue
            row = np.zeros(n); row[idx[e["b"]]] = 1; row[idx[e["a"]]] = -1; rows.append(row); rhs.append(-e["d"][c]); wts.append(np.sqrt(e["n"]) / max(e["cost"], 0.05))
        if not rows:
            continue
        Aw = np.array(rows) * np.array(wts)[:, None]; bw = np.array(rhs) * np.array(wts)
        Aw = np.vstack([Aw, np.ones(n) * 10 * max(wts)]); bw = np.r_[bw, 0]               # gauge: mean offset 0
        O[:, c] = np.linalg.lstsq(Aw, bw, rcond=None)[0]
    # note: d_ab compensates b relative to a, so the correction to ADD to line L's camera centres is -o_L ... we define
    # O as each line's bias; the correction is -O.
    linked = {e["a"] for e in edges} | {e["b"] for e in edges}
    corr = {int(L): (-O[idx[L]]).tolist() for L in ras if L in linked}          # lines without overlap: interpolated below
    print("per-line correction (m, added to camera centres):", {L: np.round(v, 2).tolist() for L, v in corr.items()}, flush=True)
    # frame offsets. The horizontal per-line corrections are explained by a camera-RTK time offset: a frame's position
    # error is v * dt (v = RTK velocity), which flips sign with flight direction. Fit dt (one scalar) to the per-line
    # horizontal corrections and apply v * dt to EVERY frame (turns and diagonals included, no interpolation).
    # Each line additionally keeps its residual horizontal bias. Vertical: per-line correction (RTK height bias), interpolated in time between line mid-times elsewhere.
    t = f.t_ns.values.astype(float); C = T[:, :3, 3]; V = np.gradient(C, t / 1e9, axis=0)
    Ls = sorted(corr); vL = np.array([V[line == L].mean(0) for L in Ls]); cL = np.array([corr[L] for L in Ls])
    A = np.r_[vL[:, 0], vL[:, 1]]; bb = np.r_[cL[:, 0] - cL[:, 0].mean(), cL[:, 1] - cL[:, 1].mean()]
    A = A - np.r_[np.full(len(Ls), vL[:, 0].mean()), np.full(len(Ls), vL[:, 1].mean())]
    dt = float(A @ bb / (A @ A)) if len(Ls) >= 2 else 0.0
    rms0 = float(np.sqrt(np.mean(bb ** 2))); rms1 = float(np.sqrt(np.mean((bb - dt * A) ** 2)))
    print(f"camera-RTK time offset dt = {dt:+.3f} s; horizontal per-line correction rms {rms0:.2f} m -> {rms1:.2f} m after v*dt", flush=True)
    off_time = V * dt; off_time[:, 2] = 0.0
    off = off_time.copy()
    # residual per-line horizontal bias (what v*dt does not explain), applied to that line's frames only (no
    # interpolation into turns / diagonals, which is what broke the purely per-line version on AMtown)
    for L in corr:
        m = line == L; off[m, :2] += np.array(corr[L][:2]) - V[m, :2].mean(0) * dt
    mids = sorted((np.median(t[line == L]), L) for L in corr); mt = np.array([m for m, _ in mids]); mz = np.array([corr[L][2] for _, L in mids])
    off[:, 2] = np.interp(t, mt, mz)
    for L in corr:
        off[line == L, 2] = corr[L][2]
    off_time[:, 2] = off[:, 2]
    variants = {"time_offset": off_time, "time_offset+line_residual": off}
    res_dt = dict(time_offset_s=dt, horizontal_rms_before_m=rms0, horizontal_rms_after_m=rms1)
    # model selection on ONBOARD validation only: across residual of cross-line + crossing pairs; select on even pair
    # ids, report on odd pair ids (no reuse of the selection set for the reported number)
    def across_of(pids, o):
        _G.update(triangulate=False, off=o)
        with Pool(12) as pool:
            rr = pool.map(_pair, pids, chunksize=4)
        a = [r[1]["across"] for r in rr if r[1] is not None]
        return np.concatenate(a) if a else np.array([np.nan])
    xp = P[P.pair_type.isin(["cross_line", "crossing"])].index.values; sel_ids, rep_ids = xp[xp % 2 == 0].tolist(), xp[xp % 2 == 1].tolist()
    score = {k: float(np.nanpercentile(across_of(sel_ids, o), 90)) for k, o in variants.items()}
    choice = min(score, key=score.get); off = variants[choice]
    print("model selection (cross pairs, even ids), across p90:", {k: round(v, 2) for k, v in score.items()}, "->", choice, flush=True)
    np.save(out / "frame_offsets.npy", off)
    val = dict(selection_p90=score, choice=choice)
    for name, sel in [("cross_line", P.pair_type == "cross_line"), ("crossing", P.pair_type == "crossing"),
                      ("within_line", (P.pair_type == "within_line") & P.gap_frames.isin([5, 20, 40]))]:
        ids = P[sel].index.values; pids = (ids[ids % 2 == 1] if name != "within_line" else ids).tolist()[:200]   # held-out half
        for tag, o in [("before", None), ("after", off)]:
            a = across_of(pids, o); val[f"{name}_{tag}"] = dict(p50=float(np.nanmedian(a)), p90=float(np.nanpercentile(a, 90)), n=int(len(a)))
        print(f"  {name} (held-out): across p50/p90 before {val[name + '_before']['p50']:.2f}/{val[name + '_before']['p90']:.2f}"
              f"  after {val[name + '_after']['p50']:.2f}/{val[name + '_after']['p90']:.2f} px", flush=True)
    res = dict(edges=[{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in e.items()} for e in edges], correction_m=corr, time_offset=res_dt, validation=val, method=__doc__)
    (out / "offsets.json").write_text(json.dumps(res, indent=2, default=float))

if __name__ == "__main__":
    main()
