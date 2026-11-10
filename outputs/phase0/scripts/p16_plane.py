"""Item 8: plane fit to LiDAR airstrip (10) / paved road (3) / car park (18) points in the Terra frame
(keyframes only, extrinsic LiDAR->cam then T4x4). Report tilt of plane normal from Terra +Z and from RTK-ENU up."""
import numpy as np, pandas as pd
from common import *
F=load_frames(); key,Kc,dist,Re,te=get_hk_calib(); U,_,Vt=np.linalg.svd(Re); Re=U@Vt
A=pd.read_csv(f"{OUT}/anchors.csv"); kf=np.where(A.is_anchor&A.orig_mod5.eq(0))[0]; kf=kf[(kf>900)&(kf<2950)]
fit=np.load(f"{OUT}/rtk_fit.npz"); R_enu=fit["R_enu"]   # maps ENU -> Terra (rotation part)
pts={10:[],3:[],18:[]}
for i in kf[::2]:
    x,lab=load_lidar(F["names"][i]); T=F["T"][i]; Xw=(x@Re.T+te)@T[:3,:3].T+T[:3,3]
    for c in pts: pts[c].append(Xw[lab==c][::4])
def plane(P):
    for _ in range(3):
        c=P.mean(0); _,_,vt=np.linalg.svd(P-c,full_matrices=False); n=vt[2]*np.sign(vt[2][2]); r=(P-c)@n; s=1.4826*np.median(np.abs(r)); P=P[np.abs(r)<3*s+1e-3]
    return n,c,len(P),np.std((P-c)@n)
up_enu_in_terra=R_enu@np.array([0,0,1.])
print("RTK-ENU up axis expressed in Terra frame:",up_enu_in_terra.round(4),"angle to Terra +Z",np.degrees(np.arccos(up_enu_in_terra[2])).round(3))
for c,L in pts.items():
    P=np.vstack(L); n,cen,m,sd=plane(P)
    print(f"class {c} {CMAP[c]}: {m} pts, extent E {np.ptp(P[:,0]):.0f} m N {np.ptp(P[:,1]):.0f} m; normal {n.round(4)}; tilt vs Terra Z {np.degrees(np.arccos(n[2])):.3f} deg (dir az {np.degrees(np.arctan2(n[1],n[0])):.0f}); tilt vs ENU up {np.degrees(np.arccos(n@up_enu_in_terra)):.3f} deg; plane resid std {sd:.2f} m")
# local planes: airstrip in tiles of 40 m
P=np.vstack(pts[10]); tiles=np.floor(P[:,:2]/40).astype(int); out=[]
for k in np.unique(tiles,axis=0):
    m=(tiles==k).all(1)
    if m.sum()>3000: n,_,_,sd=plane(P[m]); out.append((np.degrees(np.arccos(n[2])),np.degrees(np.arccos(n@up_enu_in_terra)),sd))
o=np.array(out); print("airstrip 40m tiles:",len(o),"median tilt vs TerraZ",np.median(o[:,0]).round(2),"vs ENU-up",np.median(o[:,1]).round(2),"median resid std",np.median(o[:,2]).round(3))
np.save(f"{OUT}/airstrip_pts_sample.npy",P[::10])
