import numpy as np, pandas as pd
from scipy.spatial.transform import Rotation as Rot, Slerp
from common import *
F=load_frames(); T=F["T"]; C=T[:,:3,3]; t=F["t"].astype(np.int64); n=len(t); tt=(t-t[0])/1e9
orig=np.round((t-t[0])/1e8).astype(int); print("orig idx max",orig.max(),"residual ms max",np.abs((t-t[0])/1e6-orig*100).max())
best=None
for ph in range(5):
    K=np.where(orig%5==ph)[0]
    Ci=np.stack([np.interp(tt,tt[K],C[K,k]) for k in range(3)],1)
    Ri=Slerp(tt[K],Rot.from_matrix(T[K,:3,:3]))(np.clip(tt,tt[K][0],tt[K][-1]))
    dp=np.linalg.norm(C-Ci,axis=1); dr=np.degrees((Ri.inv()*Rot.from_matrix(T[:,:3,:3])).magnitude())
    nk=np.setdiff1d(np.arange(n),K); nk=nk[(nk>K[0])&(nk<K[-1])]
    bad=nk[(dp[nk]>1e-3)|(dr[nk]>1e-3)]
    print(f"phase {ph}: keyframes {len(K)}, non-key {len(nk)}, deviating non-key {len(bad)}, max dp {dp[nk].max():.2e} max dr {dr[nk].max():.2e}", bad[:20])
    if best is None or len(bad)<best[1]: best=(ph,len(bad),K)
ph,_,K=best; kf=np.zeros(n,bool); kf[K]=True
# frames before first / after last keyframe (extrapolated?)
print("first/last keyframe",K[0],K[-1],"n frames",n)
pd.DataFrame(dict(i=np.arange(n),t_ns=t,orig_idx=orig,is_keyframe=kf)).to_csv(f"{OUT}/keyframes.csv",index=False)
print("keyframes per segment of SortedImageID: total",kf.sum())
