import numpy as np, pandas as pd
from scipy.spatial.transform import Rotation as Rot, Slerp
from scipy.interpolate import CubicSpline
from common import *
F=load_frames(); T=F["T"]; C=T[:,:3,3]; t=F["t"].astype(np.int64); n=len(t); tt=(t-t[0])/1e9
K=np.arange(0,n,5)
for name in ["linear","cubic"]:
    if name=="linear": Ci=np.stack([np.interp(tt,tt[K],C[K,k]) for k in range(3)],1)
    else: Ci=CubicSpline(tt[K],C[K])(tt)
    Ri=Slerp(tt[K],Rot.from_matrix(T[K,:3,:3]))(np.clip(tt,tt[K][0],tt[K][-1]))
    dp=np.linalg.norm(C-Ci,axis=1); dr=np.degrees((Ri.inv()*Rot.from_matrix(T[:,:3,:3])).magnitude())
    df=pd.DataFrame(dict(i=np.arange(n),mod=np.arange(n)%5,dp=dp,dr=dr))
    print(name); print(df[df["mod"]!=0].groupby("mod")[["dp","dr"]].quantile([.5,.9,.99]).unstack().round(4))
    bad=df[(df["mod"]!=0)&((df.dp>0.01)|(df.dr>0.01))]; print("non-key frames deviating:",len(bad),"ranges:")
    if len(bad):
        b=bad.i.to_numpy(); br=np.split(b,np.where(np.diff(b)>5)[0]+1); print([(x[0],x[-1]) for x in br])
    df.to_csv(f"{OUT}/interp_model_{name}.csv",index=False)
