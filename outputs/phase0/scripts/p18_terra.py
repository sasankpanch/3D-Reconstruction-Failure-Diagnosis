"""Item 11: Terra outputs - extents and frame check vs LiDAR (keyframes) placed with T4x4 + extrinsic."""
import numpy as np, pandas as pd, open3d as o3d
from scipy.spatial import cKDTree
from common import *
TD=f"{ROOT}/HKairport_GNSS_Terra"
pc=o3d.io.read_point_cloud(f"{TD}/cloud_merged.ply"); P=np.asarray(pc.points); print("cloud_merged",P.shape,"min",P.min(0).round(1),"max",P.max(0).round(1))
m=o3d.io.read_triangle_mesh(f"{TD}/terra_ply/Block/Block.ply"); V=np.asarray(m.vertices); print("Block.ply verts",V.shape,"tris",np.asarray(m.triangles).shape,"min",V.min(0).round(1),"max",V.max(0).round(1),"has uv",m.has_triangle_uvs())
m2=o3d.io.read_triangle_mesh(f"{TD}/Mesh.ply"); V2=np.asarray(m2.vertices); print("Mesh.ply verts",V2.shape,"tris",np.asarray(m2.triangles).shape,"identical vertices to Block.ply:",np.allclose(V,V2,atol=1e-3) if V.shape==V2.shape else "shape differs","max abs diff",np.abs(V-V2).max() if V.shape==V2.shape else None)
F=load_frames(); key,Kc,dist,Re,te=get_hk_calib(); U,_,Vt=np.linalg.svd(Re); Re=U@Vt
A=pd.read_csv(f"{OUT}/anchors.csv"); S=pd.read_csv(f"{OUT}/segments.csv")
kf=np.where(A.is_anchor&A.orig_mod5.eq(0)&S.segment.eq("cruise_line"))[0]
tree=cKDTree(P); rows=[]
for i in kf[::6]:
    x,lab=load_lidar(F["names"][i]); keep=(lab!=20)&(lab!=24); x=x[keep][::5]; lab=lab[keep][::5]; T=F["T"][i]
    Xw=(x@Re.T+te)@T[:3,:3].T+T[:3,3]; d,j=tree.query(Xw,distance_upper_bound=3.0)
    ok=np.isfinite(d); dz=np.full(len(d),np.nan); dz[ok]=Xw[ok,2]-P[j[ok],2]
    for k in np.where(ok)[0][::3]: rows.append((i,lab[k],d[k],dz[k]))
df=pd.DataFrame(rows,columns="i lab d dz".split())
print("LiDAR(keyframes, cruise) -> Terra cloud NN dist: median",df.d.median().round(3),"p90",df.d.quantile(.9).round(3),"| dz (LiDAR-Terra) median",df.dz.median().round(3))
print(df.groupby("lab").agg(n=("d","size"),d_med=("d","median"),dz_med=("dz","median"),dz_iqr=("dz",lambda s:s.quantile(.75)-s.quantile(.25))).rename(index=CMAP).round(3))
# dz vs position: tilt between LiDAR-in-Terra-frame and Terra cloud?
Xs=[]
