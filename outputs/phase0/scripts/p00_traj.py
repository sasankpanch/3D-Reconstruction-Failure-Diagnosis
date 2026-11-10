import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *
F=load_frames(); T=F["T"]; C=T[:,:3,3]; t=(F["t"]-F["t"][0])/1e9
R=T[:,:3,:3]; print("det R range",np.linalg.det(R).min(),np.linalg.det(R).max(),"orthonorm err",np.abs(np.einsum('nij,nik->njk',R,R)-np.eye(3)).max())
z=R[:,:,2]; print("cam z-axis median",np.median(z,0)); tilt=np.degrees(np.arccos(-z[:,2])); print("tilt from -Z median/p95/max",np.median(tilt),np.percentile(tilt,95),tilt.max())
v=np.gradient(C,t,axis=0); sp=np.linalg.norm(v[:,:2],axis=1); hd=np.degrees(np.arctan2(v[:,1],v[:,0]))
for i in range(0,len(t),100): print(i,f"t={t[i]:6.1f} C={C[i].round(1)} speed={sp[i]:.1f} head={hd[i]:6.1f} vz={v[i,2]:.2f} tilt={tilt[i]:.1f}")
fig,ax=plt.subplots(1,2,figsize=(14,6)); s=ax[0].scatter(C[:,0],C[:,1],c=np.arange(len(C)),s=2); plt.colorbar(s,ax=ax[0]); ax[0].set_aspect("equal")
for i in range(0,len(C),250): ax[0].annotate(str(i),C[i,:2],fontsize=7)
ax[1].plot(t,C[:,2]); ax[1].set_xlabel("t s"); ax[1].set_ylabel("Terra Z"); plt.savefig(f"{OUT}/traj_quick.png",dpi=90)
