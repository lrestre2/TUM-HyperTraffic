"""Node features, BEV proximity edges, and hyperedge construction for one frame.

See claude.md for the node feature layout and the five hyperedge types.
This module builds the four single-frame hyperedge types: H1 (proximity),
H3 (converging, needs per-object velocity), H4 (VRU proximity), and H5
(V2X match, R4 only, needs per-object sensor provenance). H2 (tracklet)
needs multi-frame context and lives in `data.dataset.build_tracklet_pairs`
alongside the training pipeline instead.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import radius_neighbors_graph

from data.dataset import Frame, Kinematics

# 8-class one-hot order, matches configs/default.yaml `classes:`
CLASSES = ["CAR", "TRUCK", "VAN", "TRAILER", "BUS", "PEDESTRIAN", "BICYCLE", "MOTORCYCLE"]

# OpenLABEL occlusion_level -> 0=NOT, 1=PARTIAL, 2=FULL
OCCLUSION_MAP = {
    "NOT_OCCLUDED": 0,
    "PARTIALLY_OCCLUDED": 1,
    "MOSTLY_OCCLUDED": 2,
    "FULLY_OCCLUDED": 2,
}

FEATURE_NAMES = (
    ["x", "y", "z", "yaw", "l", "w", "h", "vx", "vy", "speed", "num_points", "occ_emb"]
    + [f"cls_{c}" for c in CLASSES]
)


def yaw_from_quaternion(qx: float, qy: float, qz: float, qw: float) -> float:
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))


def class_onehot(cls: str) -> np.ndarray:
    vec = np.zeros(len(CLASSES), dtype=np.float32)
    if cls in CLASSES:
        vec[CLASSES.index(cls)] = 1.0
    return vec


def bev_xy(frame: Frame) -> np.ndarray:
    """(N, 2) array of (x, y) BEV positions, in frame.objects order."""
    return np.array([[obj.val[0], obj.val[1]] for obj in frame.objects], dtype=np.float64)


def build_node_features(frame: Frame, kinematics: dict[str, Kinematics] | None = None) -> np.ndarray:
    """(N, F) node feature matrix h_i, in frame.objects order (see claude.md)."""
    kinematics = kinematics or {}
    rows = []
    for obj in frame.objects:
        x, y, z = obj.xyz
        qx, qy, qz, qw = obj.quat
        yaw = yaw_from_quaternion(qx, qy, qz, qw)
        l, w, h = obj.lwh
        kin = kinematics.get(obj.uuid)
        vx, vy, speed = (kin.vx, kin.vy, kin.speed) if kin else (0.0, 0.0, 0.0)
        occ_emb = OCCLUSION_MAP.get(obj.occlusion, 2)
        row = np.concatenate([
            [x, y, z, yaw, l, w, h, vx, vy, speed, float(obj.num_points), float(occ_emb)],
            class_onehot(obj.cls),
        ])
        rows.append(row)
    return np.stack(rows).astype(np.float32) if rows else np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)


def bev_edges(xy: np.ndarray, radius: float = 10.0) -> list[tuple[int, int]]:
    """Undirected edge list for the BEV proximity graph at distance <= radius."""
    if len(xy) == 0:
        return []
    graph = radius_neighbors_graph(xy, radius=radius, mode="connectivity")
    coo = graph.tocoo()
    return sorted({(min(i, j), max(i, j)) for i, j in zip(coo.row, coo.col) if i != j})


def build_h1_hyperedges(xy: np.ndarray, radius: float = 10.0) -> np.ndarray:
    """H1 membership: connected components of the BEV graph at d <= radius.

    Isolated nodes form their own singleton hyperedge (the doc's "k=1
    fallback") — that falls out for free, since an isolated node is its
    own connected component.

    Returns an (N,) array of hyperedge ids, one per node.
    """
    if len(xy) == 0:
        return np.zeros(0, dtype=np.int64)
    graph = radius_neighbors_graph(xy, radius=radius, mode="connectivity")
    _, ids = connected_components(graph, directed=False)
    return ids


def hyperedge_cluster_sizes(hyperedge_ids: np.ndarray) -> np.ndarray:
    """(N,) hyperedge ids -> (N,) size of each node's own cluster.

    E.g. `build_h1_hyperedges` ids of `[0, 0, 1, 2, 2, 2]` -> sizes
    `[2, 2, 1, 3, 3, 3]`: nodes 0/1 share a 2-member cluster, node 2 is
    isolated, nodes 3/4/5 share a 3-member cluster.
    """
    if len(hyperedge_ids) == 0:
        return np.zeros(0, dtype=np.int64)
    counts = np.bincount(hyperedge_ids)
    return counts[hyperedge_ids]


# Proxy-task labels for the H1 cluster-size classification task (see
# `eval/evaluate.py --task cluster_size`): predicting how many other objects
# share a node's own BEV-proximity cluster. Unlike speed-state (a bucketed
# function of the node's own vx/vy/speed feature), cluster size cannot be
# read off a single node's feature vector — it requires knowing every other
# object's position in the scene, so a linear probe can only succeed here if
# the encoder's message-passing actually aggregated neighbor information.
# Bucket edges were chosen from the empirical R2 cluster-size distribution
# (isolated=30.5%, pair=19.3%, small_group(3-4)=25.5%, crowd(5+)=24.7%).
CLUSTER_SIZE_NAMES = ["isolated", "pair", "small_group", "crowd"]
CLUSTER_SIZE_LABELS = [0, 1, 2, 3]


def cluster_size_label(size: int) -> int:
    """Bucket an H1 cluster size into the 4 `CLUSTER_SIZE_NAMES` classes."""
    if size <= 1:
        return 0
    if size == 2:
        return 1
    if size <= 4:
        return 2
    return 3


def build_incidence_matrix(hyperedge_ids: np.ndarray) -> np.ndarray:
    """Hard-membership incidence matrix H, shape (N_objects, N_hyperedges)."""
    n = len(hyperedge_ids)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)
    num_edges = int(hyperedge_ids.max()) + 1
    H = np.zeros((n, num_edges), dtype=np.float32)
    H[np.arange(n), hyperedge_ids] = 1.0
    return H


def build_adjacency_matrix(edges: list[tuple[int, int]], n: int) -> np.ndarray:
    """Dense symmetric (N, N) 0/1 adjacency matrix from an undirected edge list.

    Pairwise counterpart to `build_incidence_matrix`, used by the vanilla GCN
    baseline (ablation: pairwise BEV edges only, no hyperedges).
    """
    A = np.zeros((n, n), dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    return A


# ---------------------------------------------------------------------------
# H3 — Converging: connected components of *projected* (t + horizon) BEV positions
# ---------------------------------------------------------------------------

def project_xy(xy: np.ndarray, vx: np.ndarray, vy: np.ndarray, horizon: float = 1.5) -> np.ndarray:
    """BEV positions projected `horizon` seconds forward at constant velocity.

    Objects with no derived velocity (vx=vy=0 — stationary or unknown) simply
    project to their current position, which is correct: a stopped object can
    still anchor a converging cluster if others are projected toward it.
    """
    return xy + horizon * np.stack([vx, vy], axis=1)


def build_h3_hyperedges(xy: np.ndarray, vx: np.ndarray, vy: np.ndarray, horizon: float = 1.5, radius: float = 10.0) -> np.ndarray:
    """H3 membership: connected components of the BEV graph over projected positions.

    Same connected-components construction as H1 (`build_h1_hyperedges`),
    applied to positions projected `horizon` seconds ahead instead of current
    positions — objects whose projected cells overlap are "converging".
    Reuses the H1 `bev_threshold` as the cell radius; there is no separate
    cell-size hyperparameter in configs/default.yaml.
    """
    return build_h1_hyperedges(project_xy(xy, vx, vy, horizon), radius=radius)


# ---------------------------------------------------------------------------
# H4 — VRU proximity: vehicle-within-15m-of-VRU, ped-within-3m-of-ped
# ---------------------------------------------------------------------------

# "pedestrian/cyclist" per claude.md's H4 rule. Kept separate from eval.py's
# broader VRU_CLASSES (which adds MOTORCYCLE for stratified *reporting*) —
# that's a different purpose than this hyperedge-membership rule.
H4_VEHICLE_CLASSES = {"CAR", "TRUCK", "VAN", "TRAILER", "BUS", "MOTORCYCLE"}
H4_VRU_CLASSES = {"PEDESTRIAN", "BICYCLE"}


def build_h4_incidence(
    xy: np.ndarray,
    classes: list[str],
    vru_radius_veh: float = 15.0,
    vru_radius_ped: float = 3.0,
) -> np.ndarray:
    """H4 membership: one hyperedge per VRU (pedestrian/cyclist) node, containing
    itself plus any vehicle within `vru_radius_veh` and — pedestrians only —
    any other pedestrian within `vru_radius_ped`.

    Unlike H1/H3 (a partition via connected components), H4 hyperedges are
    node-centered and overlap by construction: a vehicle near two VRUs sits in
    two different H4 columns. VRUs with nothing nearby still get a singleton
    column (themselves only).
    """
    n = len(classes)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)
    vru_idx = [i for i in range(n) if classes[i] in H4_VRU_CLASSES]
    if not vru_idx:
        return np.zeros((n, 0), dtype=np.float32)

    dists = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    columns = []
    for i in vru_idx:
        members = {i}
        members.update(
            j for j in range(n)
            if classes[j] in H4_VEHICLE_CLASSES and dists[i, j] <= vru_radius_veh
        )
        if classes[i] == "PEDESTRIAN":
            members.update(
                j for j in range(n)
                if j != i and classes[j] == "PEDESTRIAN" and dists[i, j] <= vru_radius_ped
            )
        col = np.zeros(n, dtype=np.float32)
        col[list(members)] = 1.0
        columns.append(col)
    return np.stack(columns, axis=1)


# ---------------------------------------------------------------------------
# H5 — V2X match (R4 only): cross-provenance cooperative-perception grouping
# ---------------------------------------------------------------------------

# NOTE on this hyperedge's design: claude.md describes H5 as "roadside +
# vehicle obs of the SAME object". That assumes two separate per-sensor node
# entries to match, mirroring R2's north/south pairing. R4's actual labels
# (confirmed against both the mini dataset and the upstream dev-kit) are
# defined on the *registered* (fused) point cloud — one box per real object,
# not two. There is no second node to match against.
#
# This approximates the intended cooperative-perception structure using each
# box's `sensor_id` provenance attribute instead: objects tagged
# "s110_lidar_ouster_south" (roadside) are grouped with nearby objects that
# carry a different provenance tag (proxy for vehicle-observed). This is a
# judgment call, not a certainty from the schema — see conversation notes
# before relying on it for a paper claim.
H5_ROADSIDE_SENSOR = "s110_lidar_ouster_south"
H5_VEHICLE_WEIGHT = 0.6


def build_h5_incidence(
    xy: np.ndarray,
    sensor_ids: list[str],
    radius: float = 10.0,
    vehicle_weight: float = 0.6,
) -> np.ndarray:
    """H5 membership: one hyperedge per roadside-tagged object, containing itself
    (weight 1.0) plus any non-roadside-tagged object within `radius` (weight
    `vehicle_weight`, the soft V2X weight — initialised here, learned downstream
    same as claude.md's incidence-matrix design).
    """
    n = len(sensor_ids)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)
    roadside_idx = [i for i in range(n) if sensor_ids[i] == H5_ROADSIDE_SENSOR]
    if not roadside_idx:
        return np.zeros((n, 0), dtype=np.float32)

    dists = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    columns = []
    for i in roadside_idx:
        col = np.zeros(n, dtype=np.float32)
        col[i] = 1.0
        for j in range(n):
            if j != i and sensor_ids[j] != H5_ROADSIDE_SENSOR and dists[i, j] <= radius:
                col[j] = vehicle_weight
        columns.append(col)
    return np.stack(columns, axis=1)


def stack_incidence(matrices: list[np.ndarray], n: int) -> np.ndarray:
    """Concatenate several (n, E_k) incidence matrices column-wise into one (n, sum E_k) H.

    Empty (E_k=0) matrices are skipped; if every matrix is empty, returns an
    (n, 0) array rather than erroring.
    """
    cols = [m for m in matrices if m.shape[1] > 0]
    if not cols:
        return np.zeros((n, 0), dtype=np.float32)
    return np.concatenate(cols, axis=1)
