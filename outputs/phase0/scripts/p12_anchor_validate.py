import numpy as np, cv2, pandas as pd
from p03_projection import *
A=pd.read_csv(f"{OUT}/anchors.csv"); isA=A.is_anchor.to_numpy()
rng=np.random.default_rng(2); out=[]
refs=[1200,1400,1820,2280,2450,2850]
for r in refs:
    if not isA[r]: r=int(np.where(isA)[0][np.argmin(np.abs(np.where(isA)[0]-r))])
    x,lab=load_lidar(F["names"][r]); m=(lab!=20)&(lab!=24); x=x[m]; x=x[rng.choice(len(x),2000,replace=False)]
    gr=cv2.cvtColor(load_img(F["names"][r]),cv2.COLOR_BGR2GRAY); Xw=world(r,x); uvr=proj(lid2cam(x),"json")
    for i in range(r-8,r+9):
        if i==r or F["t"][max(i,r)]-F["t"][min(i,r)]>1.5e9: continue
        gi=cv2.cvtColor(load_img(F["names"][i]),cv2.COLOR_BGR2GRAY); uvi=proj(to_cam(i,Xw),"json")
        o=ncc_offsets(gr,gi,uvr,uvi,half=15,search=30); g=o[np.isfinite(o[:,0])&(o[:,2]>0.85)]
        if len(g)<50: continue
        out.append((r,i,bool(isA[i]),len(g),np.median(g[:,0]),np.median(g[:,1])))
d=pd.DataFrame(out,columns="ref i anchor n du dv".split()); d["e"]=np.hypot(d.du,d.dv)
print(d.round(2).to_string()); print(d.groupby(["ref","anchor"]).e.median().round(2)); print(d.groupby("anchor")[["e"]].describe().round(2))
d.to_csv(f"{OUT}/anchor_validation.csv",index=False)
