"""Multi-view tracks from pairwise matches (union-find over (frame, keypoint) nodes).
Tracks containing two different keypoints of the same frame are dropped (inconsistent)."""
from __future__ import annotations
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

def build_tracks(pair_frames: np.ndarray, k1: np.ndarray, k2: np.ndarray, pair_of_match: np.ndarray, min_views: int = 3):
    """pair_frames: (P,2) frame indices per pair; k1/k2/pair_of_match: per match.
    Returns obs arrays (frame, kp, track) for consistent tracks with >= min_views."""
    fi = pair_frames[pair_of_match, 0].astype(np.int64); fj = pair_frames[pair_of_match, 1].astype(np.int64)
    n1 = fi * 100000 + k1; n2 = fj * 100000 + k2
    nodes, inv = np.unique(np.r_[n1, n2], return_inverse=True); m = len(n1)
    g = coo_matrix((np.ones(m), (inv[:m], inv[m:])), shape=(len(nodes), len(nodes)))
    _, lab = connected_components(g, directed=False)
    frame = nodes // 100000; kp = nodes % 100000
    # consistency: one keypoint per frame per track
    key = lab * 10000 + frame; _, cnt = np.unique(key, return_counts=True)
    bad_key = np.unique(key)[cnt > 1]; bad_tracks = np.unique(bad_key // 10000)
    views = np.bincount(lab)
    ok = (views[lab] >= min_views) & ~np.isin(lab, bad_tracks)
    tr, tinv = np.unique(lab[ok], return_inverse=True)
    return dict(frame=frame[ok], kp=kp[ok], track=tinv, n_tracks=len(tr), n_dropped_inconsistent=int(len(bad_tracks)))
