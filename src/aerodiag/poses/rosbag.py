"""'rosbag' source - PLACEHOLDER until the MARS-LVIG bag is available. Reads IMU attitude and a GNSS fix with
the pure-Python `rosbags` library (no ROS install) and matches to image timestamps by time (never by index).

NOT EXERCISED: no bag exists in the workspace; topic names / message types must be confirmed on the real bag.
The working-frame mapping (IMU->camera rotation, GNSS->working Sim3) must be estimated like the RTK fit."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .base import PoseSource, PoseSet

class RosbagPoseSource(PoseSource):
    name = "rosbag"

    def __init__(self, bag_path, imu_topic: str | None = None, gnss_topic: str | None = None, max_dt_ns: int = 20_000_000):
        self.bag_path = Path(bag_path); self.imu_topic = imu_topic; self.gnss_topic = gnss_topic; self.max_dt_ns = max_dt_ns

    def read_topic(self, topic: str):
        """Yield (t_ns header stamp, message) for one topic."""
        from rosbags.highlevel import AnyReader
        if not self.bag_path.exists():
            raise FileNotFoundError(f"rosbag not available: {self.bag_path}")
        with AnyReader([self.bag_path]) as reader:
            conns = [c for c in reader.connections if c.topic == topic]
            for conn, t_rec, raw in reader.messages(connections=conns):
                msg = reader.deserialize(raw, conn.msgtype)
                st = msg.header.stamp; yield int(st.sec) * 1_000_000_000 + int(st.nanosec), msg

    def get(self, t_ns: np.ndarray) -> PoseSet:
        raise NotImplementedError("rosbag pose source: bag not yet available (see module docstring)")
