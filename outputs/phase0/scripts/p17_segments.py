"""Item 9: flight segmentation from Terra camera centres (smoothed over keyframes). Rules (documented):
 speed<1.5 m/s -> hover; |vz|>1 m/s -> climb/descent; else heading-stable (|dheading| over +-3 s < 8 deg) runs
 lasting >= 6 s -> line (parallel if heading within 15 deg of the dominant E/W axis, diagonal otherwise), rest -> turn."""
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from common import *
F=load_frames(); C=F["T"][:,:3,3]; t=(F["t"]-F["t"][0])/1e9; n=len(t)
Cs=uniform_filter1d(C,11,axis=0,mode="nearest"); v=np.gradient(Cs,t,axis=0); v[np.diff(t,prepend=-1)>1]=np.nan
v=pd.DataFrame(v).ffill().bfill().to_numpy(); sp=np.linalg.norm(v[:,:2],axis=1); hd=np.degrees(np.arctan2(v[:,1],v[:,0]))
seg=np.array(["turn"]*n,dtype=object)
hv=sp<1.5; seg[hv]="hover"; seg[np.abs(v[:,2])>1.0]=np.where(v[np.abs(v[:,2])>1.0,2]>0,"climb","descent")
# heading stability
uh=np.unwrap(np.radians(hd)); stab=np.zeros(n,bool)
for i in range(n):
    m=np.abs(t-t[i])<=3; stab[i]=np.degrees(np.ptp(uh[m]))<8
cand=(seg=="turn")&stab&(sp>5)
# runs
lid=np.full(n,-1); runs=[]; i=0
while i<n:
    if cand[i]:
        j=i
        while j+1<n and cand[j+1] and t[j+1]-t[j]<1.0: j+=1
        if t[j]-t[i]>=6: runs.append((i,j))
        i=j+1
    else: i+=1
dom=np.median(np.mod(hd[cand],180)); print("dominant line axis heading (mod 180)",dom.round(1))
info=[]
for k,(a,b) in enumerate(runs):
    h=np.degrees(np.angle(np.mean(np.exp(1j*np.radians(hd[a:b+1])))))
    dlt=np.abs((np.mod(h-dom+90,180))-90)
    kind="cruise_line" if dlt<15 else "diagonal"
    seg[a:b+1]=kind; lid[a:b+1]=k
    info.append((k,kind,a,b,round(t[b]-t[a],1),round(h,1),round(np.median(sp[a:b+1]),2),np.round(np.median(C[a:b+1],0),1)))
for r in info: print(r)
df=pd.DataFrame(dict(i=np.arange(n),t_s=t,segment=seg,line_id=lid,speed=sp,heading=hd,vz=v[:,2]))
df.to_csv(f"{OUT}/segments.csv",index=False); print(df.segment.value_counts())
cols={"hover":"grey","climb":"tab:orange","descent":"tab:brown","turn":"tab:red","cruise_line":"tab:blue","diagonal":"tab:green"}
fig,ax=plt.subplots(1,2,figsize=(15,6.5))
for s,c in cols.items():
    m=seg==s; ax[0].scatter(C[m,0],C[m,1],s=3,c=c,label=f"{s} ({m.sum()})"); ax[1].scatter(t[m],C[m,2],s=3,c=c)
for k,kind,a,b,*_ in info: ax[0].annotate(f"L{k}",C[(a+b)//2,:2],fontsize=10,weight="bold")
ax[0].set_aspect("equal"); ax[0].legend(markerscale=4,fontsize=8); ax[0].set_xlabel("Terra X (m, ~East)"); ax[0].set_ylabel("Terra Y (m, ~North)"); ax[0].set_title("Camera path (Terra frame) by segment")
ax[1].set_xlabel("time since first image (s)"); ax[1].set_ylabel("Terra Z (m)"); ax[1].set_title("Height over time")
plt.tight_layout(); plt.savefig(f"{OUT}/segments.png",dpi=110)
