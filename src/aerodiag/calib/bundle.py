"""Bundle adjustment with RTK-anchored camera centres (onboard data only: image matches + GNSS positions).

Model (frame f, track p):
    R_f = R0_f Exp(w_f)                         camera->world rotation (R0 = prior / initial guess only)
    c_f = a_f + o_seg(f) - R_f l                camera centre: antenna position a (RTK, ENU) + per-segment offset o
                                                (absorbs segment-wise RTK bias; one segment fixed = gauge) - lever arm l
    x   = distort(f_px, k1, k2, k3, p1, p2, cx, cy; R_f^T (X_p - c_f))
Unknowns: intrinsics subset (default f, k1, k2, k3), lever arm l (camera frame), segment offsets, per-frame rotation
corrections w, points X. Points are initialised by least-squares ray intersection, then everything is solved jointly by robust (Huber)
least squares with an analytic sparse Jacobian. Outlier observations rejected between rounds.
Metric scale and absolute orientation come from the RTK positions; focal length is separable from flying height only
when the frames span different altitudes (climb/descent)."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation as Rot
from ..geometry.camera import CameraModel

INTR = ["f", "k1", "k2", "k3"]

def _cam(base: CameraModel, intr: dict) -> CameraModel:
    K = base.K.copy(); d = base.dist.copy()
    if "f" in intr:
        K[0, 0] = K[1, 1] = intr["f"]
    for k, idx in (("k1", 0), ("k2", 1), ("k3", 4)):
        if k in intr:
            d[idx] = intr[k]
    return CameraModel(K, d, base.width, base.height)

@dataclass
class BundleProblem:
    a: np.ndarray                 # (F,3) antenna positions, ENU metres
    R0: np.ndarray                # (F,3,3) initial camera->world rotations
    seg: np.ndarray               # (F,) segment index
    obs_f: np.ndarray             # (M,) frame index (0..F-1)
    obs_p: np.ndarray             # (M,) track index
    obs_uv: np.ndarray            # (M,2) observed (distorted) pixels
    cam0: CameraModel
    estimate: tuple = ("f", "k1", "k2", "k3")
    estimate_lever: bool = True
    estimate_offsets: bool = True
    fixed_segment: int = 0
    lever0: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # state
    def __post_init__(self):
        F = len(self.a); self.S = int(self.seg.max()) + 1; self.N = int(self.obs_p.max()) + 1
        self.w = np.zeros((F, 3)); self.o = np.zeros((self.S, 3)); self.l = np.asarray(self.lever0, float).copy()
        self.intr = {"f": self.cam0.K[0, 0], "k1": self.cam0.dist[0], "k2": self.cam0.dist[1], "k3": self.cam0.dist[4]}
        self.X = np.zeros((self.N, 3)); self.valid = np.ones(len(self.obs_f), bool)

    # ---------------------------------------------------------------- geometry
    def poses(self, w=None, o=None, l=None):
        w = self.w if w is None else w; o = self.o if o is None else o; l = self.l if l is None else l
        R = np.einsum("fij,fjk->fik", self.R0, Rot.from_rotvec(w).as_matrix())
        c = self.a + o[self.seg] - np.einsum("fij,j->fi", R, l)
        return R, c

    def camera(self, intr=None):
        return _cam(self.cam0, self.intr if intr is None else intr)

    def project(self, R, c, cam, X, of, op):
        Xc = np.einsum("mji,mj->mi", R[of], X[op] - c[of])
        uv = cam.normalized_to_pixel(Xc[:, :2] / Xc[:, 2:3]); return uv, Xc[:, 2]

    def residuals_all(self):
        R, c = self.poses(); uv, z = self.project(R, c, self.camera(), self.X, self.obs_f, self.obs_p)
        r = uv - self.obs_uv; r[z <= 0] = 1e3; return r

    # ---------------------------------------------------------------- (a) points
    def triangulate(self):
        R, c = self.poses(); cam = self.camera(); m = self.valid
        xn = cam.pixel_to_normalized(self.obs_uv[m]); d = np.einsum("mij,mj->mi", R[self.obs_f[m]], np.c_[xn, np.ones(len(xn))])
        d /= np.linalg.norm(d, axis=1, keepdims=True); P = np.eye(3)[None] - d[:, :, None] * d[:, None, :]
        cc = c[self.obs_f[m]]; A = np.zeros((self.N, 3, 3)); b = np.zeros((self.N, 3))
        np.add.at(A, self.obs_p[m], P); np.add.at(b, self.obs_p[m], np.einsum("mij,mj->mi", P, cc))
        cnt = np.bincount(self.obs_p[m], minlength=self.N); ok = cnt >= 2
        A[~ok] = np.eye(3); self.X = np.linalg.solve(A, b[..., None])[..., 0]; self.point_ok = ok

    # ---------------------------------------------------------------- (b) joint solve, analytic sparse Jacobian
    def _layout(self):
        F = len(self.a); segs = [s for s in range(self.S) if s != self.fixed_segment] if self.estimate_offsets else []
        n_i = len(self.estimate); n_l = 3 if self.estimate_lever else 0
        return F, segs, n_i, n_l

    def _pack(self):
        F, segs, n_i, n_l = self._layout()
        g = [np.array([self.intr[nm] for nm in self.estimate])] + ([self.l] if n_l else []) + [self.o[s] for s in segs]
        return np.concatenate(g + [self.w.ravel(), self.X.ravel()])

    def _unpack(self, x):
        F, segs, n_i, n_l = self._layout(); k = 0
        intr = dict(self.intr); intr.update({n: x[k + i] for i, n in enumerate(self.estimate)}); k += n_i
        l = x[k:k + 3] if n_l else self.l; k += n_l
        o = self.o.copy()
        for s in segs:
            o[s] = x[k:k + 3]; k += 3
        w = x[k:k + 3 * F].reshape(F, 3); k += 3 * F
        return intr, l, o, w, x[k:].reshape(-1, 3)

    def _proj_and_jac(self, x, of, op, want_jac):
        F, segs, n_i, n_l = self._layout(); intr, l, o, w, X = self._unpack(x); cam = _cam(self.cam0, intr)
        R = np.einsum("fij,fjk->fik", self.R0, Rot.from_rotvec(w).as_matrix())
        Rt = np.swapaxes(R, 1, 2)[of]; q = np.einsum("mij,mj->mi", Rt, X[op] - self.a[of] - o[self.seg[of]])
        Xc = q + l; z = Xc[:, 2]; xn = Xc[:, 0] / z; yn = Xc[:, 1] / z
        k1, k2, p1, p2, k3 = cam.dist; fpx = cam.K[0, 0]
        r2 = xn * xn + yn * yn; rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        xd = xn * rad + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn); yd = yn * rad + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
        uv = np.c_[fpx * xd + cam.K[0, 2], fpx * yd + cam.K[1, 2]]
        if not want_jac:
            return uv, z, None
        M = len(of); g = 2 * k1 + 4 * k2 * r2 + 6 * k3 * r2 ** 2
        Dxx = rad + g * xn * xn + 2 * p1 * yn + 6 * p2 * xn; Dxy = g * xn * yn + 2 * p1 * xn + 2 * p2 * yn
        Dyx = g * xn * yn + 2 * p1 * xn + 2 * p2 * yn; Dyy = rad + g * yn * yn + 6 * p1 * yn + 2 * p2 * xn
        # d(u,v)/d(Xc): (M,2,3)
        dn = np.zeros((M, 2, 3)); dn[:, 0, 0] = 1 / z; dn[:, 0, 2] = -xn / z; dn[:, 1, 1] = 1 / z; dn[:, 1, 2] = -yn / z
        Dd = np.stack([np.stack([Dxx, Dxy], -1), np.stack([Dyx, Dyy], -1)], 1)
        JXc = fpx * np.einsum("mab,mbc->mac", Dd, dn)
        blocks, cols = [], []
        # intrinsics
        dint = {"f": np.stack([xd, yd], 1), "k1": fpx * np.stack([xn * r2, yn * r2], 1), "k2": fpx * np.stack([xn * r2 ** 2, yn * r2 ** 2], 1),
                "k3": fpx * np.stack([xn * r2 ** 3, yn * r2 ** 3], 1)}
        k = 0
        for nm in self.estimate:
            blocks.append(dint[nm][:, :, None]); cols.append(np.full((M, 1), k)); k += 1
        if n_l:
            blocks.append(JXc); cols.append(np.tile(np.arange(k, k + 3), (M, 1))); k += 3
        segpos = {s: k + 3 * j for j, s in enumerate(segs)}; k += 3 * len(segs)
        sg = self.seg[of]; has = np.array([s in segpos for s in sg])
        if has.any():
            Jo = -np.einsum("mac,mcd->mad", JXc, Rt)
            blocks.append(np.where(has[:, None, None], Jo, 0.0)); base = np.array([segpos.get(s, 0) for s in sg])
            cols.append(base[:, None] + np.arange(3))
        # rotation: R <- R Exp(d) gives q <- Exp(-d) q ~ q + [q]x d  (right perturbation, small-angle Jacobian)
        qx = np.zeros((M, 3, 3)); qx[:, 0, 1] = -q[:, 2]; qx[:, 0, 2] = q[:, 1]; qx[:, 1, 0] = q[:, 2]; qx[:, 1, 2] = -q[:, 0]; qx[:, 2, 0] = -q[:, 1]; qx[:, 2, 1] = q[:, 0]
        blocks.append(np.einsum("mac,mcd->mad", JXc, qx)); cols.append(k + 3 * of[:, None] + np.arange(3)); k += 3 * F
        blocks.append(np.einsum("mac,mcd->mad", JXc, Rt)); cols.append(k + 3 * op[:, None] + np.arange(3))
        return uv, z, (blocks, cols)

    def _obs_blocks(self, x, of, op):
        """Per-observation residual-model pieces: uv (M,2), z, Jg (M,2,G) with column ids cg (M,G) into the global
        block (padded with the trash column n_g), Jp (M,2,3)."""
        F, segs, n_i, n_l = self._layout(); n_g = n_i + n_l + 3 * len(segs) + 3 * F
        uv, z, (B, C) = self._proj_and_jac(x, of, op, True)
        Jp, Cp = B[-1], C[-1]; Bg, Cg = B[:-1], C[:-1]
        Jg = np.concatenate(Bg, axis=2); cg = np.concatenate(Cg, axis=1)
        # (observations of the fixed segment carry an all-zero offset block; it contributes nothing)
        return uv, z, Jg, cg, Jp, n_g

    def optimize_lm(self, iters=30, huber=1.0, lam0=1e-3, log=None):
        m = self.valid & self.point_ok[self.obs_p]; of, op, uvo = self.obs_f[m], self.obs_p[m], self.obs_uv[m]
        x = self._pack(); N = self.N; n_glob = len(x) - 3 * N; lam = lam0
        def cost_of(xx):
            uv, z, _ = self._proj_and_jac(xx, of, op, False); r = uv - uvo; e = np.linalg.norm(r, axis=1); e[z <= 0] = 1e3
            return np.sum(np.where(e <= huber, 0.5 * e ** 2, huber * (e - 0.5 * huber)))
        cost = cost_of(x)
        for it in range(iters):
            uv, z, Jg, cg, Jp, n_g = self._obs_blocks(x, of, op); r = uv - uvo
            e = np.linalg.norm(r, axis=1); w = np.where(e <= huber, 1.0, huber / np.maximum(e, 1e-12)); w[z <= 0] = 0
            G = Jg.shape[2]; M = len(of)
            Wg = Jg * w[:, None, None]; Wp = Jp * w[:, None, None]
            Hgg_o = np.einsum("mai,maj->mij", Wg, Jg); bg_o = np.einsum("mai,ma->mi", Wg, r)
            Hpp_o = np.einsum("mai,maj->mij", Wp, Jp); bp_o = np.einsum("mai,ma->mi", Wp, r)
            E_o = np.einsum("mai,maj->mij", Wg, Jp)                                   # (M,G,3)
            ng1 = n_g + 1
            Hgg = np.bincount((cg[:, :, None] * ng1 + cg[:, None, :]).ravel(), Hgg_o.ravel(), ng1 * ng1).reshape(ng1, ng1)[:n_g, :n_g]
            bg = np.bincount(cg.ravel(), bg_o.ravel(), ng1)[:n_g]
            Hpp = np.zeros((N, 3, 3)); np.add.at(Hpp, op, Hpp_o); bp = np.zeros((N, 3)); np.add.at(bp, op, bp_o)
            while True:
                Hd = Hgg + lam * np.diag(np.diag(Hgg) + 1e-9)
                Hpd = Hpp + lam * (np.einsum("nii->ni", Hpp)[:, :, None] * np.eye(3)) + 1e-9 * np.eye(3)
                Hpi = np.linalg.inv(Hpd)
                # Schur complement: S = Hgg - sum_p F_p Hpp^-1 F_p^T, with F_p = sum_{i in p} scatter(E_i)
                order = np.argsort(op, kind="stable"); ops = op[order]; starts = np.r_[0, np.flatnonzero(np.diff(ops)) + 1]
                lens = np.diff(np.r_[starts, len(ops)]); pos = np.arange(len(ops)) - np.repeat(starts, lens)
                Sacc = np.zeros(ng1 * ng1); rhs = bg.copy()
                EHi = np.einsum("mgk,mkl->mgl", E_o, Hpi[op])                         # E_i Hpp^-1
                rhs -= np.bincount(cg.ravel(), np.einsum("mgk,mk->mg", EHi, bp[op]).ravel(), ng1)[:n_g]
                for k in range(int(lens.max())):
                    sel = (pos + k) < np.repeat(lens, lens); ia = order[sel]; ib = order[np.flatnonzero(sel) + k]
                    blk = np.einsum("mgk,mhk->mgh", EHi[ia], E_o[ib])
                    idx = (cg[ia][:, :, None] * ng1 + cg[ib][:, None, :]).ravel()
                    Sacc += np.bincount(idx, blk.ravel(), ng1 * ng1)
                    if k > 0:
                        idx2 = (cg[ib][:, :, None] * ng1 + cg[ia][:, None, :]).ravel()
                        Sacc += np.bincount(idx2, np.swapaxes(blk, 1, 2).ravel(), ng1 * ng1)
                S = Hd - Sacc.reshape(ng1, ng1)[:n_g, :n_g]
                try:
                    dg = -np.linalg.solve(S, rhs)
                except np.linalg.LinAlgError:
                    lam *= 10; continue
                dgp = np.r_[dg, 0.0]
                Edg = np.einsum("mgk,mg->mk", E_o, dgp[cg]); t = np.zeros((N, 3)); np.add.at(t, op, Edg)
                dp = -np.einsum("nij,nj->ni", Hpi, bp + t)
                xn = x.copy(); xn[:n_glob] += dg; xn[n_glob:] += dp.ravel()
                cn = cost_of(xn)
                if cn < cost:
                    x, cost, lam = xn, cn, max(lam / 3, 1e-9); break
                lam *= 5
                if lam > 1e8:
                    break
            if log:
                log(dict(it=it, cost=cost, lam=lam))
            if lam > 1e8 or (it > 0 and abs(prev - cost) / max(cost, 1e-12) < 1e-7):
                break
            prev = cost
        self.intr, self.l, self.o, self.w, self.X = self._unpack(x)
        return cost

    def reject(self, k=4.0, floor_px=2.0):
        e = np.linalg.norm(self.residuals_all(), axis=1); v = e[self.valid]
        thr = max(floor_px, k * 1.4826 * np.median(np.abs(v - np.median(v))) + np.median(v))
        self.valid = e < thr; return thr

    def solve(self, rounds=12, tol=1e-6, log=None):
        hist = []
        self.triangulate(); self.reject(k=6, floor_px=8)
        for it in range(rounds):
            if it == 0 or not hasattr(self, "point_ok"):
                self.triangulate()
            self.optimize_lm(); thr = self.reject()
            self.point_ok = self.point_ok & (np.bincount(self.obs_p[self.valid], minlength=self.N) >= 2)
            e = np.linalg.norm(self.residuals_all(), axis=1)[self.valid]
            h = dict(round=it, rms_px=float(np.sqrt(np.mean(e ** 2))), median_px=float(np.median(e)), n_valid=int(self.valid.sum()),
                     reject_thr_px=float(thr), **{k: float(v) for k, v in self.intr.items()}, lever=self.l.round(4).tolist())
            hist.append(h)
            if log:
                log(h)
            if it > 2 and abs(hist[-1]["f"] - hist[-2]["f"]) / hist[-1]["f"] < tol and abs(hist[-1]["rms_px"] - hist[-2]["rms_px"]) < 1e-3:
                break
        return hist
