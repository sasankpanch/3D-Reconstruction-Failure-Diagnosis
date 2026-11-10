"""Item 5/6 quantitative: for LiDAR pts of frame i, project into images i and j under model M; NCC-search the
image-i patch in image j. Offset = disagreement between geometry and image content (px). Report vs radius."""
import numpy as np, cv2, pandas as pd
from p03_projection import *
rows=[]
pairs=[(1150,1160),(1200,1220),(1250,1290),(1550,1570),(1600,1640),(2000,2020),(2050,2090),(2450,2470),(2500,2540),(2750,2770),(1200,1201),(1200,1205)]
rng=np.random.default_rng(0)
for i,j in pairs:
    x,lab=load_lidar(F["names"][i]); keep=(lab!=20)&(lab!=24); x=x[keep]; lab=lab[keep]
    sel=rng.choice(len(x),min(6000,len(x)),replace=False); x=x[sel]; lab=lab[sel]
    gi=cv2.cvtColor(load_img(F["names"][i]),cv2.COLOR_BGR2GRAY); gj=cv2.cvtColor(load_img(F["names"][j]),cv2.COLOR_BGR2GRAY)
    Xw=world(i,x)
    for model in ["json","calib"]:
        uvi=proj(lid2cam(x),model); uvj=proj(to_cam(j,Xw),model)
        o=ncc_offsets(gi,gj,uvi,uvj,half=12,search=10)
        r=np.hypot(uvi[:,0]-1224,uvi[:,1]-1024)
        for k in range(len(o)):
            if np.isfinite(o[k,0]): rows.append((i,j,model,uvi[k,0],uvi[k,1],r[k],o[k,0],o[k,1],o[k,2],lab[k]))
    print(i,j,"done")
df=pd.DataFrame(rows,columns="i j model ui vi r du dv ncc lab".split()); df.to_parquet(f"{OUT}/ncc_offsets.parquet")
d=df[df.ncc>0.8].copy(); d["e"]=np.hypot(d.du,d.dv); d["rb"]=pd.cut(d.r,[0,300,600,900,1200,1600])
print(d.groupby(["model","rb"],observed=True).e.describe(percentiles=[.5,.9]).round(2))
print(d.groupby(["model","i","j"],observed=True).agg(n=("e","size"),med=("e","median"),du=("du","median"),dv=("dv","median")).round(2))
