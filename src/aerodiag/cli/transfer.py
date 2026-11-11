"""Cross-scene transfer: train each head on one scene, apply unchanged to another (no retraining, no recalibration).

aerodiag-transfer --scenes configs/data.yaml configs/data_town.yaml [configs/data_island.yaml] [--heads failure layered vehicle]

Features: step-3 per-cell features (onboard signals only). Two normalisations are reported:
  train     standardisation statistics from the TRAINING scene applied to the test scene;
  own_rank  each scene's features replaced by their within-scene percentile rank (unsupervised: uses only that scene's
            own onboard features, never its labels), which removes scene-specific scale (altitude, texture, RTK quality).
Labels on the test scene are used only for scoring. Metrics: AUROC / AP with 95% block-bootstrap CIs (25 m blocks
of the test scene). Within-scene spatial-CV numbers are reported alongside for reference.
Output: outputs/transfer/results.json, outputs/transfer/TRANSFER.md"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd
from ..config import load_config, write_params
from ..grid.ground_grid import GroundGrid
from ..diagnosis.classifier import prepare, fit_logreg, predict, full_metrics, blocks_of
from ..diagnosis.gbdt import fit_gbdt, predict_gbdt

def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)

def load_scene(cfg_path):
    d = load_config(cfg_path); o = Path(d["root"]) / d["paths"]["outputs"]
    F = pd.read_parquet(o / "step3" / "cell_features.parquet"); L = pd.read_parquet(o / "step3" / "proxy_labels.parquet")
    g = GroundGrid.from_json((o / "track_a" / "grid.json").read_text())
    r = json.loads((o / "step3" / "results.json").read_text())
    return dict(name=d["scene"], F=F, L=L, xy=g.centers()[F.index.values], within=r["cv_metrics"], within_gbdt=r.get("cv_metrics_gbdt", {}))

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenes", nargs="+", default=["configs/data.yaml", "configs/data_town.yaml"])
    ap.add_argument("--heads", nargs="+", default=["failure", "layered", "vehicle"])
    ap.add_argument("--norms", nargs="+", default=["train", "own_rank"])
    ap.add_argument("--models", nargs="+", default=["logreg", "gbdt"])
    ap.add_argument("--cfg", default="configs/step3.yaml"); ap.add_argument("--out", default="outputs/transfer")
    a = ap.parse_args(argv); cfg = load_config(a.cfg); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    S = [load_scene(p) for p in a.scenes]; res = {}; rows = []
    common = sorted(set.intersection(*[set(c for c in s["F"].columns if c not in ("n_obs_fit", "pool_radius")) for s in S]))
    rank = lambda F: F[common].astype(float).replace([np.inf, -np.inf], np.nan).rank(pct=True)
    for model_name in a.models:
     for norm in a.norms:
      for tr in S:
        for te in S:
            if tr is te:
                continue
            Ftr, Fte = (rank(tr["F"]), rank(te["F"])) if norm == "own_rank" else (tr["F"], te["F"])
            for h in a.heads:
                if h not in tr["L"] or h not in te["L"]:
                    continue
                ytr = tr["L"][h].values; m = np.isfinite(ytr)
                if (ytr[m] == 1).sum() < 30:
                    continue
                Xtr, cols, stats = prepare(Ftr, common)                        # columns decided on TRAIN only
                Xte, _, _ = prepare(Fte, cols, stats)
                if model_name == "gbdt":                                       # trees: raw (or ranked) values, no scaling
                    raw = lambda F_: F_[cols].astype(float).replace([np.inf, -np.inf], np.nan).values
                    s = predict_gbdt(fit_gbdt(raw(Ftr)[m], ytr[m], **cfg.get("gbdt", {})), raw(Fte))
                else:
                    s = predict(fit_logreg(Xtr[m], ytr[m], l2=cfg["cv"]["l2"]), Xte)
                yte = te["L"][h].values
                met = full_metrics(yte, s, blocks_of(te["xy"], cfg["cv"]["block_m"]), cfg["seed"])
                wi = (te["within_gbdt"] if model_name == "gbdt" else te["within"]).get(h, {})
                key = f"{model_name}:{norm}:{tr['name']} -> {te['name']}:{h}"; res[key] = dict(transfer=met, within_scene_cv_on_test=wi)
                rows.append(dict(model=model_name, norm=norm, train=tr["name"], test=te["name"], head=h, n=met["n"], pos=met["positives"], base=met["base_rate"],
                                 auroc=met["auroc"], auroc_ci95=met["auroc_ci95"], ap=met["ap"], within_auroc=wi.get("auroc"), within_ap=wi.get("ap")))
                log(key, "AUROC %.3f [%.3f, %.3f]  AP %.3f (base %.3f) | within-scene AUROC %s" % (
                    met["auroc"], *met["auroc_ci95"], met["ap"], met["base_rate"], None if wi.get("auroc") is None else round(wi["auroc"], 3)))
    T = pd.DataFrame(rows)
    (out / "results.json").write_text(json.dumps(res, indent=2, default=float))
    f = lambda v: f"{v:.3f}" if isinstance(v, float) else (f"[{v[0]:.2f}, {v[1]:.2f}]" if isinstance(v, (list, tuple)) else str(v))
    md = ["# Cross-scene transfer (train on one scene, test on another; no retraining)", "", f"Common features: {len(common)}", "",
          "| model | norm | train | test | head | n | positives | base rate | AUROC | 95% CI | AP | within-scene AUROC | within-scene AP |", "|" + "---|" * 13]
    md += ["| " + " | ".join(f(r[k]) for k in ["model", "norm", "train", "test", "head", "n", "pos", "base", "auroc", "auroc_ci95", "ap", "within_auroc", "within_ap"]) + " |" for r in rows]
    (out / "TRANSFER.md").write_text("\n".join(md)); write_params(out, "transfer", dict(scenes=a.scenes, heads=a.heads, norms=a.norms, models=a.models, step3=cfg))
    print("\n".join(md))

if __name__ == "__main__":
    main()
