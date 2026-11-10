import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *
R=pd.read_csv(f"{OUT}/rtk_resid.csv"); S=pd.read_csv(f"{OUT}/segments.csv"); A=pd.read_csv(f"{OUT}/anchors.csv")
d=R.join(S[["segment","line_id","t_s"]]).join(A[["is_anchor","orig_mod5"]]); d=d[d.is_anchor&d.orig_mod5.eq(0)]
d["grp"]=np.where(d.line_id>=0,d.segment+"_L"+d.line_id.astype(str),d.segment)
print(d.groupby("grp").resid_enu.describe(percentiles=[.5,.95])[["count","50%","95%","max"]].round(2))
fig,ax=plt.subplots(figsize=(12,4)); 
for g,s in d.groupby("segment"): ax.scatter(s.t_s,s.resid_enu,s=5,label=g)
ax.set_xlabel("time since first image (s)"); ax.set_ylabel("|Terra C - sim(RTK)| (m)"); ax.set_title("RTK->Terra similarity residual, keyframes"); ax.legend(); plt.tight_layout(); plt.savefig(f"{OUT}/rtk_residual_time.png",dpi=100)
