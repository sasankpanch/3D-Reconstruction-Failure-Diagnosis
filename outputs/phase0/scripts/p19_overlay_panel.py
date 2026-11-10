"""Item 5 figure: same-frame LiDAR overlay (label colours) with calib model (HK_GNSS K+dist), centre and corner crops,
for keyframes in cruise / turn / diagonal / climb. Plus cross-frame (gap 20 keyframes) overlay."""
import numpy as np, cv2, pandas as pd
from common import *
F=load_frames(); key,Kc,dist,Re,te=get_hk_calib(); U,_,Vt=np.linalg.svd(Re); Re=U@Vt
A=pd.read_csv(f"{OUT}/anchors.csv"); S=pd.read_csv(f"{OUT}/segments.csv"); kfm=A.is_anchor&A.orig_mod5.eq(0)
def pick(seg,q=0.5):
    c=np.where(kfm&S.segment.eq(seg))[0]; return int(c[int(q*(len(c)-1))])
frames=[("cruise",pick("cruise_line",.3)),("cruise2",pick("cruise_line",.8)),("turn",pick("turn",.5)),("diagonal",pick("diagonal",.5)),("climb",pick("climb",.6))]
tiles=[]
for nm,i in frames:
    img=load_img(F["names"][i]); x,lab=load_lidar(F["names"][i]); pc=x@Re.T+te
    uv,_=cv2.projectPoints(pc.reshape(-1,1,3),np.zeros(3),np.zeros(3),Kc,dist); uv=uv[:,0]
    o=img.copy()
    for (u,v),l in zip(uv.astype(int),lab):
        if 0<=u<2448 and 0<=v<2048: c=CRGB.get(int(l),(255,255,255)); o[v,u]=(c[2],c[1],c[0])
    o=cv2.dilate(o,np.ones((2,2),np.uint8)) if False else o
    mix=img.copy(); msk=np.any(o!=img,axis=2); mix[msk]=o[msk]
    full=cv2.resize(mix,(612,512)); cx=mix[824:1224,1024:1424]; cr=mix[1648:2048,2048:2448]; cl=mix[0:400,0:400]
    row=np.hstack([full,cv2.resize(cx,(512,512)),cv2.resize(cl,(512,512)),cv2.resize(cr,(512,512))])
    cv2.putText(row,f"{nm} frame {i}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2); tiles.append(row)
cv2.imwrite(f"{OUT}/overlay_panel.jpg",np.vstack(tiles),[cv2.IMWRITE_JPEG_QUALITY,85]); print([f for f in frames])
