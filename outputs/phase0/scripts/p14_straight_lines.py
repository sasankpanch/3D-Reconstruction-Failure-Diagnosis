"""Item 6(b): straightness of long straight features. Detect long segments (HoughLinesP on Canny), collect edge
pixels within 2 px of each, fit a straight line in raw pixels and in calib-undistorted pixels; compare RMS and
signed sagitta (bow) for segments near the image border vs centre."""
import numpy as np, cv2, pandas as pd
from common import *
F=load_frames(); key,Kc,dist,_,_=get_hk_calib()
rows=[]
for idx in range(900,2900,7):
    g=cv2.cvtColor(load_img(F["names"][idx]),cv2.COLOR_BGR2GRAY); e=cv2.Canny(cv2.GaussianBlur(g,(5,5),1.5),60,150)
    L=cv2.HoughLinesP(e,1,np.pi/720,150,minLineLength=500,maxLineGap=6)
    if L is None: continue
    ys,xs=np.nonzero(e); P=np.c_[xs,ys].astype(float)
    for x1,y1,x2,y2 in L.reshape(-1,4)[:15]:
        a=np.array([x1,y1],float); b=np.array([x2,y2],float); d=(b-a)/np.linalg.norm(b-a); nrm=np.array([-d[1],d[0]])
        s=(P-a)@d; off=(P-a)@nrm; m=(s>0)&(s<np.linalg.norm(b-a))&(np.abs(off)<2.5); Q=P[m]
        if len(Q)<500: continue
        mid=(a+b)/2; rmid=np.hypot(mid[0]-Kc[0,2],mid[1]-Kc[1,2])
        res={}
        for nm,pts in [("raw",Q),("undist",cv2.undistortPoints(Q.reshape(-1,1,2),Kc,dist,P=Kc)[:,0])]:
            c=pts.mean(0); u,sv,vt=np.linalg.svd(pts-c); dd=vt[0]; nn=vt[1]; t_=(pts-c)@dd; o=(pts-c)@nn
            # sagitta: fit quadratic o = k t^2 + ..., bow over the segment length
            co=np.polyfit(t_,o,2); Lh=(t_.max()-t_.min())/2; res[nm]=(np.sqrt(np.mean(o**2)),co[0]*Lh**2)
        rows.append((idx,rmid,np.linalg.norm(b-a),len(Q),*res["raw"],*res["undist"]))
d=pd.DataFrame(rows,columns="frame rmid len n rms_raw sag_raw rms_und sag_und".split()); d.to_csv(f"{OUT}/straight_lines.csv",index=False)
d["zone"]=pd.cut(d.rmid,[0,500,850,2000],labels=["centre","mid","border"])
print(len(d),"segments"); print(d.groupby("zone",observed=True)[["rms_raw","rms_und"]].median().round(3))
d["|sag_raw|"]=d.sag_raw.abs(); d["|sag_und|"]=d.sag_und.abs()
print(d.groupby("zone",observed=True)[["|sag_raw|","|sag_und|"]].agg(["median","count"]).round(2))
d["improve"]=d["|sag_raw|"]-d["|sag_und|"]
print("border segments: fraction where undistortion reduces |bow|:",(d[d.zone=="border"].improve>0).mean().round(2), " median reduction px:",d[d.zone=="border"].improve.median().round(2))
print("rms ratio und/raw border median:",(d[d.zone=="border"].rms_und/d[d.zone=="border"].rms_raw).median().round(3))
