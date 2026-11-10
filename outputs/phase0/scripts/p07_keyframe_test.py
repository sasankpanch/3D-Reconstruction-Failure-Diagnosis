"""Per-frame pose consistency relative to a fixed reference frame r: project LiDAR of r into image i,
NCC-search image-r patches in image i. offset(i) ~ pose_error(i) - pose_error(r) in px (on ground)."""
import numpy as np, cv2, pandas as pd, sys
from p03_projection import *
rng=np.random.default_rng(1); out=[]
for r in [1200,1203,2450]:
    x,lab=load_lidar(F["names"][r]); m=(lab!=20)&(lab!=24); x=x[m]; x=x[rng.choice(len(x),2500,replace=False)]
    gr=cv2.cvtColor(load_img(F["names"][r]),cv2.COLOR_BGR2GRAY); Xw=world(r,x); uvr=proj(lid2cam(x),"json")
    for i in range(r-12,r+13):
        gi=cv2.cvtColor(load_img(F["names"][i]),cv2.COLOR_BGR2GRAY); uvi=proj(to_cam(i,Xw),"json")
        o=ncc_offsets(gr,gi,uvr,uvi,half=15,search=30); g=o[np.isfinite(o[:,0])&(o[:,2]>0.85)]
        out.append((r,i,i%5,len(g),np.median(g[:,0]),np.median(g[:,1])))
        print(r,i,i%5,len(g),round(np.median(g[:,0]),2),round(np.median(g[:,1]),2),flush=True)
pd.DataFrame(out,columns="r i mod5 n du dv".split()).to_csv(f"{OUT}/keyframe_test.csv",index=False)
