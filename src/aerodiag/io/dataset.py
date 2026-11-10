"""Dataset access: frame table (timestamps, names, reference flags, flight segments), poses, LiDAR, labels, images.

Frames are matched across sources by int64-ns timestamp (Phase 0: perfect 1:1 match)."""
from __future__ import annotations
import json, re
from functools import cached_property
from pathlib import Path
import numpy as np, pandas as pd
from .timestamps import parse_ns
from .calib import load_scene_calibration, CameraLidarCalib
from .terra import read_metadata
from ..geometry.camera import CameraModel
from ..flight.anchors import reference_mask
from ..flight.segmentation import segment_flight

class Dataset:
    def __init__(self, cfg: dict):
        self.cfg = cfg; self.root = Path(cfg["root"]); p = cfg["paths"]
        self.p = {k: self.root / v for k, v in p.items()}
        self.out = self.p["outputs"]

    # ---------------------------------------------------------------- frames / poses
    @cached_property
    def _json(self):
        js = json.load(open(self.p["sampleinfos"]))
        js = sorted(js, key=lambda e: e["SortedImageID"])
        assert [e["SortedImageID"] for e in js] == list(range(len(js)))
        return js

    @cached_property
    def T_wc_json(self) -> np.ndarray:
        """Terra T4x4 (camera->world) for every frame, INCLUDING interpolated ones. Use `frames.is_reference`."""
        return np.array([e["T4x4"] for e in self._json], dtype=np.float64)

    @cached_property
    def frames(self) -> pd.DataFrame:
        names = [e["OriginalImageName"][:-4] for e in self._json]
        t = np.array([parse_ns(s) for s in names], dtype=np.int64)
        assert np.all(np.diff(t) > 0)
        rc = self.cfg["reference_poses"]
        ref = reference_mask(t, self.T_wc_json, rc["keyframe_modulo"], rc["frame_period_ns"], rc["anchor_tol_m"],
                             rc["anchor_tol_deg"], rc["exclude_off_grid_anchors"])
        seg = segment_flight(t, self.T_wc_json[:, :3, 3], **self.cfg["segmentation"])
        df = pd.DataFrame(dict(name=names, t_ns=t, **ref)); df = pd.concat([df, seg], axis=1)
        df.index.name = "frame"
        return df

    # ---------------------------------------------------------------- calibration / camera
    @cached_property
    def calib(self) -> CameraLidarCalib:
        return load_scene_calibration(self.p["calibration"], self.cfg["scene"])

    @cached_property
    def camera(self) -> CameraModel:
        e = self._json[0]
        if self.cfg["camera"]["model"] == "json":
            return CameraModel(np.array(e["P3x3"]), np.zeros(5), e["Width"], e["Height"])
        dist = self.calib.dist
        if self.cfg["camera"].get("dist_override") is not None:      # onboard-estimated distortion (k1 k2 p1 p2 k3)
            dist = np.asarray(self.cfg["camera"]["dist_override"], float)
        return CameraModel(self.calib.K, dist, e["Width"], e["Height"])

    @cached_property
    def terra_origin(self):
        return read_metadata(self.p["terra_dir"])

    # ---------------------------------------------------------------- per-frame data
    @cached_property
    def _lidar_names(self) -> dict:
        """image name -> LiDAR file name. Files are 'image<t_img>_lidar<t_lidar>.txt'; t_lidar equals t_img on the
        HK sequences but differs by ~10-20 ms on AMtown, so the map is read from the directory listing."""
        m = {}
        for f in self.p["lidar_dir"].iterdir():
            g = re.match(r"image(.+?)_lidar(.+?)\.txt$", f.name)
            if g:
                m[g.group(1)] = f.name
        return m

    def lidar_name(self, i: int) -> str:
        return self._lidar_names[self.frames.name.iat[i]]

    def lidar_t_ns(self, i: int) -> int:
        return parse_ns(re.match(r"image.+?_lidar(.+?)\.txt$", self.lidar_name(i)).group(1))

    def load_lidar(self, i: int) -> np.ndarray:
        """(N,3) float64, LiDAR sensor frame, metres."""
        return pd.read_csv(self.p["lidar_dir"] / self.lidar_name(i), sep=r"\s+", header=None,
                           dtype=np.float64, engine="c").to_numpy()

    def load_labels(self, i: int) -> np.ndarray:
        return pd.read_csv(self.p["label_id_dir"] / self.lidar_name(i), header=None, dtype=np.int32,
                           engine="c").to_numpy()[:, 0].astype(np.uint8)

    def load_image(self, i: int, gray: bool = False) -> np.ndarray:
        import cv2
        img = cv2.imread(str(self.p["cam_dir"] / f"{self.frames.name.iat[i]}.jpg"),
                         cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(self.frames.name.iat[i])
        return img

    def lidar_world(self, i: int, T_wc: np.ndarray | None = None) -> np.ndarray:
        """LiDAR points of frame i in the working frame: p_W = T_wc (R_CL p_L + t_CL)."""
        T = self.T_wc_json[i] if T_wc is None else T_wc
        pc = self.calib.lidar_to_camera(self.load_lidar(i))
        return pc @ T[:3, :3].T + T[:3, 3]
