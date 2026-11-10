"""Shared Phase-0 helpers (throwaway inspection code; the real package comes after approval).
Conventions: T4x4 (JSON) hypothesised camera->world (Terra local frame, metres); OpenCV camera (x right, y down, z fwd).
Extrinsic (calibration_results.py, HK_GNSS_calibration): p_cam = R @ p_lidar + t (metres) -- verified in p03.
Timestamps: int64 nanoseconds since Unix epoch."""
import ast, json, numpy as np, pandas as pd, os
ROOT="/home/edgar/Desktop/CC"; D=f"{ROOT}/interval1_HKairport_GNSS03"; L=f"{ROOT}/interval1_HKairport_GNSS03_Labels"; OUT=f"{ROOT}/outputs/phase0"
def ts_ns(s):
    a,b=s.split("."); return int(a)*10**9+int(b.ljust(9,"0")[:9])
def load_calib(path=f"{ROOT}/calibration_results.py"):
    tree=ast.parse(open(path).read()); out={}
    for node in tree.body:
        if isinstance(node,ast.Assign) and isinstance(node.value,ast.Dict):
            name=node.targets[0].id
            try: out[name]=ast.literal_eval(node.value)
            except ValueError:  # dict of Name references (scenename_to_calibration)
                out[name]={ast.literal_eval(k):v.id for k,v in zip(node.value.keys,node.value.values)}
    return out
def load_frames():
    js=json.load(open(f"{D}/sampleinfos_interpolated.json")); js=sorted(js,key=lambda e:e["SortedImageID"])
    T=np.array([e["T4x4"] for e in js]); t=np.array([ts_ns(e["OriginalImageName"][:-4]) for e in js],dtype=np.int64)
    names=[e["OriginalImageName"][:-4] for e in js]; K=np.array(js[0]["P3x3"])
    return dict(T=T,t=t,names=names,K=K)
def lidar_file(name): return f"image{name}_lidar{name}.txt"
def load_lidar(name):
    f=lidar_file(name)
    x=pd.read_csv(f"{D}/interval1_LIDAR/{f}",sep=r"\s+",header=None).to_numpy(float)
    lab=pd.read_csv(f"{L}/interval1_LIDAR_label_id/{f}",header=None).to_numpy()[:,0]
    return x,lab
def load_img(name):
    import cv2; return cv2.imread(f"{D}/interval1_CAM/{name}.jpg")
def get_hk_calib():
    c=load_calib(); key=c["scenename_to_calibration"]["HKairport_GNSS03"]; cc=c[key]
    return key, np.array(cc["camera_intrinsic"]).reshape(3,3), np.array(cc["camera_dist_coeffs"]), np.array(cc["camera_ext_R"]).reshape(3,3), np.array(cc["camera_ext_t"])
CMAP={0:'background',1:'roof',2:'dirt_motor_road',3:'paved_motor_road',4:'river',5:'pool',6:'bridge',9:'container',10:'airstrip',11:'traffic_barrier',13:'green_field',14:'wild_field',15:'solar_board',16:'umbrella',17:'transparent_roof',18:'car_park',19:'paved_walk',20:'sedan',24:'truck'}
CRGB={0:(0,0,0),1:(119,11,32),2:(180,165,180),3:(128,64,128),9:(250,170,30),10:(81,0,81),11:(102,102,156),13:(107,142,35),14:(210,180,140),16:(153,153,153),18:(250,170,160),20:(0,0,142),24:(0,0,70)}
