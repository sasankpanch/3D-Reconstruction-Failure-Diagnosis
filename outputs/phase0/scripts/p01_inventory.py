"""Phase 0 item 1-3: inventory + timestamp matching. Timestamps kept as int nanoseconds."""
import json, os, re, glob, sys
from collections import Counter
import numpy as np
import pandas as pd

ROOT = "/home/edgar/Desktop/CC"
OUT = f"{ROOT}/outputs/phase0"
D = f"{ROOT}/interval1_HKairport_GNSS03"
L = f"{ROOT}/interval1_HKairport_GNSS03_Labels"

def ts_ns(s):
    sec, frac = s.split(".")
    return int(sec) * 10**9 + int(frac.ljust(9, "0")[:9])

imgs = sorted(os.listdir(f"{D}/interval1_CAM"))
lid = sorted(os.listdir(f"{D}/interval1_LIDAR"))
lab_id = sorted(os.listdir(f"{L}/interval1_LIDAR_label_id"))
lab_col = sorted(os.listdir(f"{L}/interval1_LIDAR_label_color"))
print("counts img/lidar/labid/labcol", len(imgs), len(lid), len(lab_id), len(lab_col))
rx = re.compile(r"image(\d+\.\d+)_lidar(\d+\.\d+)\.txt$")
img_ts = {ts_ns(f[:-4]) for f in imgs}
lid_pairs = [rx.match(f).groups() for f in lid]
same = sum(a == b for a, b in lid_pairs)
print("lidar names where image ts == lidar ts:", same, "/", len(lid_pairs))
lid_img_ts = {ts_ns(a) for a, b in lid_pairs}
js = json.load(open(f"{D}/sampleinfos_interpolated.json"))
js_ts = {ts_ns(e["OriginalImageName"][:-4]) for e in js}
print("img-json", len(img_ts - js_ts), "json-img", len(js_ts - img_ts))
print("lidar-img", len(lid_img_ts - img_ts), "img-lidar", len(img_ts - lid_img_ts))
print("labid==lidar names", lab_id == lid, "labcol==lidar", lab_col == lid)
ids = [e["SortedImageID"] for e in js]
print("SortedImageID contiguous", ids == list(range(len(js))))
t = np.array(sorted(js_ts), dtype=np.int64)
print("json sorted by ts == SortedImageID order", [ts_ns(e["OriginalImageName"][:-4]) for e in sorted(js, key=lambda e: e["SortedImageID"])] == list(t))
dt = np.diff(t) / 1e9
print("span s", (t[-1] - t[0]) / 1e9, "dt median", np.median(dt), "min", dt.min(), "max", dt.max())
gaps = np.where(dt > 0.15)[0]
for g in gaps: print("  gap after idx", g, f"{dt[g]:.3f}s")
print("dt hist", np.histogram(dt[dt < 0.15], bins=[0, .08, .095, .105, .12, .15]))
# intrinsics constant?
P = np.array([e["P3x3"] for e in js]); print("P3x3 unique rows:", np.unique(P.reshape(len(P), -1), axis=0).shape[0])
print("dist nonzero:", sum(any(e[k] != 0 for k in ["K1","K2","K3","P1","P2"]) for e in js), "W,H set", {(e["Width"], e["Height"]) for e in js}, "scenes", {e["OriginalSceneName"] for e in js})
# image sizes / formats
from PIL import Image
for f in [imgs[0], imgs[len(imgs)//2]]:
    im = Image.open(f"{D}/interval1_CAM/{f}"); print(f, im.size, im.mode, im.format, os.path.getsize(f"{D}/interval1_CAM/{f}"))
# RTK
r = pd.read_csv(f"{D}/rtk_positions_raw.csv", dtype={"headerstamp": str})
r["t_ns"] = r.headerstamp.map(ts_ns)
print("rtk rows", len(r), "scenes", r.scenename.unique(), "dt median", np.median(np.diff(r.t_ns))/1e9, "max", np.diff(r.t_ns).max()/1e9)
print("rtk start-first img", (r.t_ns.iloc[0]-t[0])/1e9, "end-last", (r.t_ns.iloc[-1]-t[-1])/1e9)
print(r.describe().T)
print("utc first image", pd.to_datetime(t[0], unit="ns", utc=True), "last", pd.to_datetime(t[-1], unit="ns", utc=True))
pd.DataFrame({"t_ns": t}).to_csv(f"{OUT}/frame_timestamps_ns.csv", index=False)
