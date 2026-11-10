"""Item 5 figure v2: rasterise same-frame LiDAR labels (5x5 splats, calib model), draw class boundaries (yellow)
and the airstrip/paved/carpark/roof region boundaries over the image. Crops: centre and two corners."""
import numpy as np, cv2, pandas as pd
from common import *
F=load_frames(); key,Kc,dist,Re,te=get_hk_calib(); U,_,Vt=np.linalg.svd(Re); Re=U@Vt
A=pd.read_csv(f"{OUT}/anchors.csv"); S=pd.read_csv(f"{OUT}/segments.csv"); kfm=A.is_anchor&A.orig_mod5.eq(0)
def pick(seg,q):
    c=np.where(kfm&S.segment.eq(seg))[0]; return int(c[int(q*(len(c)-1))])
frames=[("climb",pick("climb",.6)),("hover-alt",pick("hover",.7)),("cruise",pick("cruise_line",.3)),("diagonal",pick("diagonal",.5))]
rows=[]
for nm,i in frames:
    img=load_img(F["names"][i]); x,lab=load_lidar(F["names"][i]); pc=x@Re.T+te
    uv,_=cv2.projectPoints(pc.reshape(-1,1,3),np.zeros(3),np.zeros(3),Kc,dist); uv=np.round(uv[:,0]).astype(int)
    R=np.full((2048,2448),255,np.uint8); ok=(uv[:,0]>=0)&(uv[:,0]<2448)&(uv[:,1]>=0)&(uv[:,1]<2048)
    order=np.argsort(-pc[:,2]); 
    for k in order[ok[order]]: pass
    u,v,l=uv[ok,0],uv[ok,1],lab[ok]
    for dy in range(-3,4):
        for dx in range(-3,4):
            R[np.clip(v+dy,0,2047),np.clip(u+dx,0,2447)]=l
    R=cv2.medianBlur(R,7)
    e=(cv2.morphologyEx(R,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))>0)&(R!=255)&(cv2.erode(R,np.ones((3,3),np.uint8))!=255)
    o=img.copy(); o[e]=(0,255,255)
    full=cv2.resize(o,(612,512)); crops=[o[824:1224,1024:1424],o[0:400,0:400],o[1648:2048,2048:2448]]
    row=np.hstack([full]+[cv2.resize(c,(512,512),interpolation=cv2.INTER_NEAREST) for c in crops])
    cv2.putText(row,f"{nm} frame {i}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2); rows.append(row)
cv2.imwrite(f"{OUT}/overlay_panel.jpg",np.vstack(rows),[cv2.IMWRITE_JPEG_QUALITY,85]); print(frames)
