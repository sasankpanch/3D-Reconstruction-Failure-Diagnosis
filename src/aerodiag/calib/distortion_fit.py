"""Onboard radial-distortion fit: images + RTK only (stand-in attitude), per-pair rotation-only refinement.
Calibration pairs: climb / descent / turn / diagonal segments; cruise lines held out for evaluation."""
import sys; sys.path.insert(0,'src'); import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot
from aerodiag.config import load_config; from aerodiag.io.dataset import Dataset
from aerodiag.geometry.camera import CameraModel; from aerodiag.geometry.epipolar import relative_pose, decompose
from aerodiag.poses.refine import refine_rotation
from aerodiag.cli.track_b import TrackB
cfgp=sys.argv[1]; dcfg=load_config(cfgp); ds=Dataset(dcfg); out=ds.out
tb=TrackB.__new__(TrackB); tb.ds=ds; tb.cfg=load_config('configs/track_b.yaml'); tb.dcfg=dcfg; tb.results={}
T=tb.build_sources()["rtk"].T_wc
P=pd.read_csv(out/'track_b/pairs_selected.csv',index_col=0); f=ds.frames
R=pd.read_parquet(out/'track_b/residuals.parquet',columns=['pair_id','u1','v1','u2','v2','tex'])
cal=P[P.pair_type.isin(['turn','climb','crossing'])|((P.pair_type=='within_line')&f.segment.values[P.i.values].__eq__('diagonal'))]
ev=P[(P.pair_type=='within_line')&(f.segment.values[P.i.values]=='cruise_line')&P.gap_frames.isin([5,10,20,40])]
rng=np.random.default_rng(0)
def grp(S,n): S=S.sample(min(n,len(S)),random_state=0); return [(int(P.i[p]),int(P.j[p]),d[['u1','v1']].values.astype(float),d[['u2','v2']].values.astype(float)) for p,d in R[R.pair_id.isin(S.index)].groupby('pair_id') if len(d)>=40]
G=grp(cal,200); E=grp(ev,200); print('cal pairs',len(G),'eval pairs',len(E), cal.pair_type.value_counts().to_dict())
K=ds.calib.K; p12=ds.calib.dist[2:4]
def across(dist,groups,refine=True):
    cam=CameraModel(K,np.r_[dist[0],dist[1],p12,dist[2] if len(dist)>2 else 0.0],2448,2048); a=[]
    for i,j,u1,u2 in groups:
        x1=cam.pixel_to_normalized(u1); x2=cam.pixel_to_normalized(u2); Rr,t=relative_pose(T[i],T[j])
        if refine:
            w,_=refine_rotation(x1,x2,Rr,t,cam.f,rng=np.random.default_rng(0)); Tj=T[j].copy(); Tj[:3,:3]=T[j][:3,:3]@Rot.from_rotvec(w).as_matrix(); Rr,t=relative_pose(T[i],Tj)
        a.append(np.abs(decompose(x1,x2,Rr,t,cam.f)['across_px']))
    return np.concatenate(a)
obj=lambda d: float(np.mean(np.minimum(across(d,G),5.0)**2))      # truncated (robust) mean square
for nm,d in [('file',ds.calib.dist[[0,1,4]]),('zero',np.zeros(3))]:
    a=across(d,G); e=across(d,E); print(nm,'cal p50/p90 %.2f/%.2f'%(np.median(a),np.percentile(a,90)),' EVAL(cruise) p50/p90 %.2f/%.2f'%(np.median(e),np.percentile(e,90)))
r=minimize(obj,np.zeros(3),method='Nelder-Mead',options=dict(xatol=1e-3,fatol=1e-4,maxiter=120))
a=across(r.x,G); e=across(r.x,E)
print('fit k1,k2,k3',r.x.round(4),'cal p50/p90 %.2f/%.2f'%(np.median(a),np.percentile(a,90)),' EVAL(cruise) p50/p90 %.2f/%.2f'%(np.median(e),np.percentile(e,90)))
import json; (out/'onboard_distortion.json').write_text(json.dumps(dict(k1=r.x[0],k2=r.x[1],k3=r.x[2],p1=p12[0],p2=p12[1],K='calibration file',method=__doc__,cal_pairs=len(G),eval_pairs=len(E),eval_p50=float(np.median(e)),eval_p90=float(np.percentile(e,90))),indent=2))
