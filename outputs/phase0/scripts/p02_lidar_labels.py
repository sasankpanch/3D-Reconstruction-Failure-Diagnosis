"""Phase 0 items 2,3,10: LiDAR txt columns/ranges and label stats over all frames."""
import os, numpy as np, pandas as pd
from multiprocessing import Pool
ROOT="/home/edgar/Desktop/CC"; D=f"{ROOT}/interval1_HKairport_GNSS03/interval1_LIDAR"; L=f"{ROOT}/interval1_HKairport_GNSS03_Labels"
files=sorted(os.listdir(D))
def one(f):
    with open(f"{D}/{f}") as fh: first=fh.readline()
    x=pd.read_csv(f"{D}/{f}",sep=r"\s+",header=None,engine="c").to_numpy()
    li=pd.read_csv(f"{L}/interval1_LIDAR_label_id/{f}",sep=r"\s+",header=None).to_numpy()
    lc=pd.read_csv(f"{L}/interval1_LIDAR_label_color/{f}",sep=r"\s+",header=None).to_numpy()
    assert len(x)==len(li)==len(lc), f
    ids,cnt=np.unique(li[:,0],return_counts=True)
    pairs=np.unique(np.c_[li[:,0],lc],axis=0)
    r=np.linalg.norm(x[:,:3],axis=1)
    return dict(f=f,n=len(x),ncol=x.shape[1],licol=li.shape[1],lccol=lc.shape[1],dec=len(first.split()[0].split(".")[-1]),
        mn=x.min(0),mx=x.max(0),med=np.median(x,0),rmin=r.min(),rmax=r.max(),nzero=int((r<0.1).sum()),
        ids=dict(zip(ids.tolist(),cnt.tolist())),pairs=pairs)
with Pool(20) as p: res=p.map(one,files,chunksize=20)
n=np.array([r["n"] for r in res]); print("points/frame min/med/max",n.min(),np.median(n),n.max(),"total",n.sum())
print("ncols lidar",{r["ncol"] for r in res},"labid cols",{r["licol"] for r in res},"labcol cols",{r["lccol"] for r in res},"decimals",{r["dec"] for r in res})
mn=np.min([r["mn"] for r in res],0); mx=np.max([r["mx"] for r in res],0); med=np.median([r["med"] for r in res],0)
print("xyz global min",mn,"max",mx,"median of medians",med); print("range min",min(r["rmin"] for r in res),"max",max(r["rmax"] for r in res),"zero pts",sum(r["nzero"] for r in res))
tot={}
for r in res:
    for k,v in r["ids"].items(): tot[k]=tot.get(k,0)+v
pairs=np.unique(np.vstack([r["pairs"] for r in res]),axis=0)
print("id->color pairs:"); 
S=sum(tot.values())
for k in sorted(tot): print(f"  id {k:3d} color {pairs[pairs[:,0]==k][:,1:].tolist()} count {tot[k]:10d} share {100*tot[k]/S:6.2f}% frames {sum(k in r['ids'] for r in res)}")
pd.DataFrame([dict(frame=i,file=r["f"],n=r["n"],**{f"id{k}":r["ids"].get(k,0) for k in sorted(tot)}) for i,r in enumerate(res)]).to_csv(f"{ROOT}/outputs/phase0/lidar_label_counts_per_frame.csv",index=False)
# per-frame median x (range along sensor x) over time
mx_=np.array([r["med"][0] for r in res]); print("median sensor-x per frame: first",mx_[:3],"idx600",mx_[600],"last",mx_[-3:])
