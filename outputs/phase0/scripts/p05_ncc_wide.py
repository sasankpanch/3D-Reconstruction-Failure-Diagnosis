import numpy as np, cv2, pandas as pd
from p03_projection import *
rng=np.random.default_rng(0)
for i,j in [(1200,1201),(1200,1202),(1200,1205),(1200,1210),(1550,1551),(1550,1560),(2450,2451),(1250,1251)]:
    x,lab=load_lidar(F["names"][i]); m=(lab!=20)&(lab!=24); x=x[m]; x=x[rng.choice(len(x),3000,replace=False)]
    gi=cv2.cvtColor(load_img(F["names"][i]),cv2.COLOR_BGR2GRAY); gj=cv2.cvtColor(load_img(F["names"][j]),cv2.COLOR_BGR2GRAY)
    Xw=world(i,x); uvi=proj(lid2cam(x),"json"); uvj=proj(to_cam(j,Xw),"json")
    o=ncc_offsets(gi,gj,uvi,uvj,half=15,search=30); g=o[np.isfinite(o[:,0])&(o[:,2]>0.9)]
    # also: pure image-to-image shift predicted by poses for a pixel at image center
    flow=np.median(uvj-uvi,0)
    print(f"{i}->{j} n={len(g)} du med {np.median(g[:,0]):6.2f} IQR {np.percentile(g[:,0],25):6.2f}..{np.percentile(g[:,0],75):6.2f}  dv med {np.median(g[:,1]):5.2f}  pose-predicted flow {flow.round(1)}  ratio {np.median(g[:,0])/flow[0] if flow[0] else 0:.3f}")
