"""Anchor (independently registered) frames: greedy removal of frames that equal linear/slerp interpolation of
their current neighbouring anchors (tol 1 mm, 1e-3 deg)."""
import numpy as np, pandas as pd
from scipy.spatial.transform import Rotation as Rot
from common import *
F=load_frames(); T=F["T"]; C=T[:,:3,3]; t=F["t"].astype(np.int64); n=len(t); R=Rot.from_matrix(T[:,:3,:3])
orig=np.round((t-t[0])/1e8).astype(int)
def dev(a,i,b):
    w=(t[i]-t[a])/(t[b]-t[a]); dp=np.linalg.norm(C[i]-((1-w)*C[a]+w*C[b]))
    rel=R[a].inv()*R[b]; Ri=R[a]*Rot.from_rotvec(w*rel.as_rotvec()); return dp,np.degrees((Ri.inv()*R[i]).magnitude())
anch=list(range(n)); changed=True
while changed:
    changed=False; keep=[anch[0]]
    k=1
    while k<len(anch)-1:
        a,i,b=keep[-1],anch[k],anch[k+1]; dp,dr=dev(a,i,b)
        if dp<1e-3 and dr<1e-3: changed=True
        else: keep.append(i)
        k+=1
    keep.append(anch[-1]); anch=keep
A=np.array(anch); print("anchors",len(A),"of",n); is_a=np.zeros(n,bool); is_a[A]=True
print("anchors with orig%5==0:",(orig[A]%5==0).sum(),"; anchors not %5:",(orig[A]%5!=0).sum())
odd=A[orig[A]%5!=0]; br=np.split(odd,np.where(np.diff(odd)>6)[0]+1); print("non-%5 anchor clusters (SortedImageID ranges, count):",[(int(x[0]),int(x[-1]),len(x)) for x in br])
k5=np.where(orig%5==0)[0]; miss=k5[~is_a[k5]]; print("orig%5==0 frames that are NOT anchors (interpolated keyframe slots):",len(miss), miss[:50])
# anchor spacing in time
dt=np.diff(t[A])/1e9; print("anchor spacing s: ",np.unique(np.round(dt,1),return_counts=True))
# max distance of any frame to nearest anchor in time
import bisect
dist=np.array([np.min(np.abs(t[A]-ti)) for ti in t])/1e9; print("time to nearest anchor: max",dist.max(),"hist",np.unique(np.round(dist,1),return_counts=True))
pd.DataFrame(dict(i=np.arange(n),t_ns=t,orig_idx=orig,is_anchor=is_a,orig_mod5=orig%5,dt_to_anchor_s=dist)).to_csv(f"{OUT}/anchors.csv",index=False)
