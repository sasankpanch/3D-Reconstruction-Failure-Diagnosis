"""Item 7: RTK -> Terra similarity. Two RTK representations: UTM50N (E,N,alt) as given; and local ENU computed
from lat/lon/alt about Terra metadata origin (lat0,lon0, h0=179.664). Interpolate RTK (5 Hz) to image times."""
import numpy as np, pandas as pd, pyproj, json
from scipy.spatial.transform import Rotation as Rot
from common import *
F=load_frames(); T=F["T"]; C=T[:,:3,3]; t=F["t"]; A=pd.read_csv(f"{OUT}/anchors.csv")
r=pd.read_csv(f"{D}/rtk_positions_raw.csv",dtype={"headerstamp":str}); r["t"]=r.headerstamp.map(ts_ns)
# UTM check
tr=pyproj.Transformer.from_crs("EPSG:4326","EPSG:32650",always_xy=True); e,nn=tr.transform(r.lon.values,r.lat.values)
print("UTM50N recompute max diff E,N (m):",np.abs(e-r.easting).max(),np.abs(nn-r.northing).max())
lat0,lon0,h0=22.415056378573908,114.04387569569714,179.66438658301294
ecef=pyproj.Transformer.from_crs("EPSG:4979","EPSG:4978",always_xy=True)
def enu(lat,lon,h):
    X=np.array(ecef.transform(lon,lat,h)).T; X0=np.array(ecef.transform(lon0,lat0,h0))
    la,lo=np.radians(lat0),np.radians(lon0)
    Rm=np.array([[-np.sin(lo),np.cos(lo),0],[-np.sin(la)*np.cos(lo),-np.sin(la)*np.sin(lo),np.cos(la)],[np.cos(la)*np.cos(lo),np.cos(la)*np.sin(lo),np.sin(la)]])
    return (X-X0)@Rm.T
r[["e_","n_","u_"]]=enu(r.lat.values,r.lon.values,r.alt.values)
def interp(cols,tq):
    tt=(r.t.values-r.t.values[0])/1e9; q=(tq-r.t.values[0])/1e9
    return np.stack([np.interp(q,tt,r[c].values) for c in cols],1)
def umeyama(X,Y,scale=True):
    mx,my=X.mean(0),Y.mean(0); Xc,Yc=X-mx,Y-my; U,S,Vt=np.linalg.svd(Yc.T@Xc/len(X)); D_=np.eye(3)
    if np.linalg.det(U@Vt)<0: D_[2,2]=-1
    R=U@D_@Vt; s=(S*np.diag(D_)).sum()/Xc.var(0).sum() if scale else 1.0; t_=my-s*R@mx; return s,R,t_
def describe(R):
    ang=np.degrees(Rot.from_matrix(R).magnitude()); z=R[:,2]; tilt=np.degrees(np.arccos(np.clip(z[2],-1,1)))
    yaw=np.degrees(np.arctan2(R[1,0],R[0,0])); return ang,tilt,yaw
key=A.is_anchor.values&(A.orig_mod5.values==0)
res={}
for name,cols in [("utm",["easting","northing","alt"]),("enu",["e_","n_","u_"])]:
    P=interp(cols,t)
    for dt_ms in [0]:
        s,R,tt_=umeyama(P[key],C[key]); Y=s*P@R.T+tt_; e_=np.linalg.norm(Y-C,axis=1)
        ang,tilt,yaw=describe(R)
        print(f"{name}: scale {s:.5f} rot {ang:.3f} deg  tilt(z-axis) {tilt:.3f} deg  yaw {yaw:.3f} deg  t {tt_.round(2)}  resid med {np.median(e_[key]):.3f} p95 {np.percentile(e_[key],95):.3f} max {e_[key].max():.2f}")
        res[name]=dict(scale=s,R=R.tolist(),t=tt_.tolist(),resid=e_)
    # raw comparison without fitting (for ENU: is Terra == ENU?)
    if name=="enu":
        dd=C-P; print("Terra C - ENU(RTK) unfitted: median",np.median(dd[key],0).round(2),"std",dd[key].std(0).round(2))
# time offset sweep (enu)
for dt_ms in range(-400,401,50):
    P=interp(["e_","n_","u_"],t+dt_ms*10**6); s,R,tt_=umeyama(P[key],C[key]); e_=np.linalg.norm(s*P[key]@R.T+tt_-C[key],axis=1)
    print(f"  time offset {dt_ms:+4d} ms: median {np.median(e_):.3f} p95 {np.percentile(e_,95):.3f}")
# lever arm: C = s R p + t + Rcam l  (l in camera frame) -> linear LS in (s R p + t) fixed, solve l, iterate
P=interp(["e_","n_","u_"],t); s,R,tt_=umeyama(P[key],C[key])
for it in range(5):
    Rc=T[key,:3,:3]; resid=C[key]-(s*P[key]@R.T+tt_); l=np.linalg.lstsq(Rc.reshape(-1,3),resid.reshape(-1),rcond=None)[0]
    Cc=C[key]-np.einsum('nij,j->ni',Rc,l); s,R,tt_=umeyama(P[key],Cc)
e_=np.linalg.norm(Cc-(s*P[key]@R.T+tt_),axis=1); print("lever arm (camera frame, m):",l.round(3),"resid med",np.median(e_).round(3),"p95",np.percentile(e_,95).round(3))
np.savez(f"{OUT}/rtk_fit.npz",**{k:np.array(v) for k,v in dict(s_utm=res["utm"]["scale"],R_utm=res["utm"]["R"],t_utm=res["utm"]["t"],s_enu=res["enu"]["scale"],R_enu=res["enu"]["R"],t_enu=res["enu"]["t"]).items()})
pd.DataFrame(dict(i=np.arange(len(t)),resid_utm=res["utm"]["resid"],resid_enu=res["enu"]["resid"])).to_csv(f"{OUT}/rtk_resid.csv",index=False)
print("alt during hover at start (frames 0-200) RTK alt",r.alt.values[:5].round(2),"; geoid N (EGM96/2008) at site can be checked; Terra origin h0",h0)
