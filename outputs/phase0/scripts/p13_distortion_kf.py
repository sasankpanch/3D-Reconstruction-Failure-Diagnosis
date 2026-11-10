"""Item 6(c): LiDAR-guided cross-frame NCC residual vs image radius, keyframes (orig%5==0) only, cruise.
Models: 'json' = JSON P3x3, zero distortion; 'calib' = HK_GNSS K + dist (applied to BOTH projections)."""
import numpy as np, cv2, pandas as pd
from p03_projection import *
A=pd.read_csv(f"{OUT}/anchors.csv"); kf=np.where(A.orig_mod5.eq(0)&A.is_anchor)[0]
def near(i): return int(kf[np.argmin(np.abs(kf-i))])
rng=np.random.default_rng(3); rows=[]
bases=[1150,1200,1250,1300,1500,1560,1620,1950,2050,2150,2400,2480,2560]
for b in bases:
    i=near(b)
    for g in [5,10,20]:
        j=near(i+g)
        if F["t"][j]-F["t"][i]>3e9 or j==i: continue
        x,lab=load_lidar(F["names"][i]); m=(lab!=20)&(lab!=24); x=x[m]; lab=lab[m]; s=rng.choice(len(x),min(5000,len(x)),replace=False); x=x[s]; lab=lab[s]
        gi=cv2.cvtColor(load_img(F["names"][i]),cv2.COLOR_BGR2GRAY); gj=cv2.cvtColor(load_img(F["names"][j]),cv2.COLOR_BGR2GRAY); Xw=world(i,x)
        for model in ["json","calib"]:
            uvi=proj(lid2cam(x),model); pcj=to_cam(j,Xw); uvj=proj(pcj,model)
            o=ncc_offsets(gi,gj,uvi,uvj,half=12,search=15)
            # epipolar direction in image j for across/along decomposition: direction of motion of pt with depth
            uvj2=proj(to_cam(j,Xw+ (Xw-F["T"][i,:3,3])*0.02),model); ed=uvj2-uvj; ed/=np.linalg.norm(ed,axis=1,keepdims=True)+1e-12
            for k in np.where(np.isfinite(o[:,0])&(o[:,2]>0.85))[0]:
                al=o[k,0]*ed[k,0]+o[k,1]*ed[k,1]; ac=-o[k,0]*ed[k,1]+o[k,1]*ed[k,0]
                rows.append((i,j,j-i,model,uvi[k,0],uvi[k,1],uvj[k,0],uvj[k,1],o[k,0],o[k,1],al,ac,o[k,2],lab[k]))
        print(i,j,flush=True)
d=pd.DataFrame(rows,columns="i j gap model ui vi uj vj du dv along across ncc lab".split())
d.to_parquet(f"{OUT}/distortion_ncc_kf.parquet")
d["rj"]=np.hypot(d.uj-1224,d.vj-1024); d["rb"]=pd.cut(d.rj,[0,300,600,900,1200,1700])
q=lambda s: pd.Series({"n":len(s),"med|across|":np.median(np.abs(s)),"p90|across|":np.percentile(np.abs(s),90),"med_across":np.median(s)})
print(d.groupby(["model","rb"],observed=True).across.apply(q).unstack().round(2))
print(d.groupby(["model","rb"],observed=True).along.apply(q).unstack().round(2))
print(d.groupby(["model","gap"],observed=True).across.apply(q).unstack().round(2))
