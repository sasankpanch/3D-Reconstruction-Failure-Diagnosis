"""Consecutive-frame image shift (phase correlation, central crop, downsampled) vs pose-predicted shift of the
image centre's ground point (uses LiDAR median depth). Diagnoses pose/image timing jitter."""
import numpy as np, cv2, pandas as pd
from p03_projection import *
rows=[]; prev=None
idx=list(range(1100,1330))+list(range(2380,2480))
cache={}
def gray(i):
    g=cv2.cvtColor(load_img(F["names"][i]),cv2.COLOR_BGR2GRAY).astype(np.float32); return cv2.resize(g,None,fx=0.5,fy=0.5)
for i in idx:
    j=i+1
    if F["t"][j]-F["t"][i]>1.5e8: continue
    gi,gj=gray(i),gray(j); win=cv2.createHanningWindow(gi.shape[::-1],cv2.CV_32F)
    (dx,dy),resp=cv2.phaseCorrelate(gi,gj,win); dx*=2; dy*=2
    x,lab=load_lidar(F["names"][i]); pc=lid2cam(x); d=np.median(pc[:,2])
    # pose prediction for points across image (homography of ground at depth d): use LiDAR pts directly
    Xw=world(i,x); uvi=proj(pc,"json"); uvj=proj(to_cam(j,Xw),"json"); fl=np.median(uvj-uvi,0)
    rows.append((i,(F["t"][j]-F["t"][i])/1e6,dx,dy,resp,fl[0],fl[1],d))
df=pd.DataFrame(rows,columns="i dt_ms img_dx img_dy resp pose_dx pose_dy depth".split())
df["ex"]=df.img_dx-df.pose_dx; df["ey"]=df.img_dy-df.pose_dy
df.to_csv(f"{OUT}/flow_vs_pose.csv",index=False)
pd.set_option("display.width",200); print(df.iloc[::6].round(2).to_string())
print(df[["img_dx","pose_dx","ex","ey"]].describe().round(2))
print("corr(ex_t, ex_t+1)",np.corrcoef(df.ex[:-1],df.ex[1:])[0,1])
