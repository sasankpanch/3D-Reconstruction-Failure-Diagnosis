"""Detect Terra keyframes vs interpolated poses: interpolated frames lie on the (linear/slerp) interpolation of
their neighbours => near-zero time-normalised 2nd difference of position and rotation."""
import numpy as np, pandas as pd
from scipy.spatial.transform import Rotation as Rot, Slerp
from common import *
F=load_frames(); T=F["T"]; C=T[:,:3,3]; t=F["t"].astype(np.int64); n=len(t)
# test: is frame i the linear interpolation (in time) of i-1 and i+1 ?
a=np.full(n,np.nan); b=np.full(n,np.nan)
for i in range(1,n-1):
    w=(t[i]-t[i-1])/(t[i+1]-t[i-1])
    a[i]=np.linalg.norm(C[i]-((1-w)*C[i-1]+w*C[i+1]))
    R=Rot.from_matrix(T[[i-1,i+1],:3,:3]); s=Slerp([0,1],R)(w)
    b[i]=np.degrees((s.inv()*Rot.from_matrix(T[i,:3,:3])).magnitude())
df=pd.DataFrame(dict(i=np.arange(n),mod5=np.arange(n)%5,pos_dev=a,rot_dev=b,t_ns=t))
print(df.groupby("mod5")[["pos_dev","rot_dev"]].describe(percentiles=[.5,.9]).T.round(4))
# kink frames = keyframes candidates
kf=(df.pos_dev>1e-3)|(df.rot_dev>1e-3)
print("kink frames",kf.sum(),"mod5 of kinks",np.bincount(df.i[kf]%5,minlength=5))
ki=df.i[kf].to_numpy(); print("diff of kink indices",np.unique(np.diff(ki),return_counts=True))
# where do non-5 spacings occur
d=np.diff(ki); bad=np.where(d!=5)[0]; print("non-5 spacing at",[(ki[j],ki[j+1]) for j in bad][:40])
df["kink"]=kf; df.to_csv(f"{OUT}/keyframe_detection.csv",index=False)
