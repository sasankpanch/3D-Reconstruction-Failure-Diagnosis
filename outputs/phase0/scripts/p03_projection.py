"""Item 5/6: verify conventions by projection.
Same-frame overlay tests extrinsic + intrinsics (+distortion). Cross-frame overlay (LiDAR of frame i -> image j)
tests the T4x4 convention. Quantitative: local NCC search of image-i patch around each projected LiDAR pt in image j."""
import sys, numpy as np, cv2
from common import *
F=load_frames(); key,Kc,dist,Re,te=get_hk_calib(); Kj=F["K"]
print("calib entry used:",key); print("Kcalib\n",Kc,"\ndist",dist,"\nKjson\n",Kj)
print("R_ext det",np.linalg.det(Re),"orth err",np.abs(Re@Re.T-np.eye(3)).max())
def lid2cam(x): return x@Re.T+te                   # LiDAR->camera (hypothesis A)
def lid2cam_inv(x): return (x-te)@Re               # camera->LiDAR interpretation (hypothesis B): p_c = R^T (p_l - t)
x,lab=load_lidar(F["names"][1200])
for nm,f in [("A L->C",lid2cam),("B C->L",lid2cam_inv)]:
    pc=f(x); print(nm,"cam-frame median",np.median(pc,0).round(2),"frac z>0",(pc[:,2]>0).mean())
def proj(pc,model):
    if model=="json": uv=pc@Kj.T; return uv[:,:2]/uv[:,2:]
    uv,_=cv2.projectPoints(pc.reshape(-1,1,3),np.zeros(3),np.zeros(3),Kc,dist); return uv[:,0]
def world(i,x):  # cam->world hypothesis
    return lid2cam(x)@F["T"][i,:3,:3].T+F["T"][i,:3,3]
def to_cam(j,Xw, inv=False):
    T=F["T"][j]
    if inv: return Xw@T[:3,:3].T+T[:3,3]  # if T were world->cam
    return (Xw-T[:3,3])@T[:3,:3]
def overlay(img,uv,lab,path,step=1):
    o=img.copy(); h,w=o.shape[:2]; m=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
    for (u,v),l in zip(uv[m][::step].astype(int),lab[m][::step]):
        c=CRGB.get(int(l),(255,255,255)); cv2.circle(o,(u,v),3,(c[2],c[1],c[0]),-1)
    cv2.imwrite(path,cv2.resize(cv2.addWeighted(img,0.45,o,0.55,0),None,fx=0.5,fy=0.5))
ncc_rows=[]
def ncc_offsets(gi,gj,uvi,uvj,half=10,search=8):
    """For each point: patch in image i at uvi, search in image j around uvj; return best offset (du,dv) and NCC."""
    h,w=gi.shape; res=[]
    for (ui,vi),(uj,vj) in zip(uvi.astype(int),uvj.astype(int)):
        if not(half<=ui<w-half and half<=vi<h-half and half+search<=uj<w-half-search and half+search<=vj<h-half-search): res.append((np.nan,)*3); continue
        P=gi[vi-half:vi+half+1,ui-half:ui+half+1]; S=gj[vj-half-search:vj+half+search+1,uj-half-search:uj+half+search+1]
        if P.std()<4: res.append((np.nan,)*3); continue
        r=cv2.matchTemplate(S,P,cv2.TM_CCOEFF_NORMED); _,mv,_,ml=cv2.minMaxLoc(r)
        # subpixel parabola
        dx,dy=ml; sx=sy=0.
        if 0<dx<r.shape[1]-1: a,b,c=r[dy,dx-1],r[dy,dx],r[dy,dx+1]; sx=0.5*(a-c)/(a-2*b+c+1e-12)
        if 0<dy<r.shape[0]-1: a,b,c=r[dy-1,dx],r[dy,dx],r[dy+1,dx]; sy=0.5*(a-c)/(a-2*b+c+1e-12)
        res.append((dx+sx-search,dy+sy-search,mv))
    return np.array(res)
if __name__=="__main__":
    import os; os.makedirs(f"{OUT}/overlays",exist_ok=True)
    tests=[(1200,1200),(1200,1210),(1200,1240),(1600,1600),(1000,1000),(1400,1400),(2750,2750),(400,400),(2000,1300) ]
    for i,j in tests:
        x,lab=load_lidar(F["names"][i]); keep=lab!=20; keep&=lab!=24
        Xw=world(i,x)
        for model in ["json","calib"]:
            pc=to_cam(j,Xw); uv=proj(pc,model)
            overlay(load_img(F["names"][j]),uv[pc[:,2]>0],lab[pc[:,2]>0],f"{OUT}/overlays/lid{i}_img{j}_{model}.jpg")
        print("written",i,j)
