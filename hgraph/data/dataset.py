"""OpenLABEL frame parser and north/south (or roadside/vehicle) positive-pair loader.

Each TUMTraf label file is one ASAM OpenLABEL 1.0.0 JSON document holding a
single frame keyed by frame number, e.g. `openlabel.frames["250"]`. See
claude.md for the schema and hyperedge/feature design this module feeds.
"""

from __future__ import annotations

import bisect
import glob
import json
import os
import random
import re
from dataclasses import dataclass, field

FILENAME_TS_RE = re.compile(r"^(\d+)_(\d+)_")


@dataclass
class ObjectAnnotation:
    uuid: str
    cls: str
    val: list[float]  # [x, y, z, qx, qy, qz, qw, l, w, h]
    num_points: int
    occlusion: str    # raw OpenLABEL occlusion_level, e.g. "NOT_OCCLUDED"
    sensor_id: str = ""  # R4 provenance tag, e.g. "s110_lidar_ouster_south"; used by H5

    @property
    def xyz(self) -> tuple[float, float, float]:
        return self.val[0], self.val[1], self.val[2]

    @property
    def quat(self) -> tuple[float, float, float, float]:
        return self.val[3], self.val[4], self.val[5], self.val[6]

    @property
    def lwh(self) -> tuple[float, float, float]:
        return self.val[7], self.val[8], self.val[9]


@dataclass
class Frame:
    path: str
    frame_id: str
    timestamp: float  # seconds
    weather: str | None
    time_of_day: str | None
    objects: list[ObjectAnnotation] = field(default_factory=list)


def parse_filename_timestamp(path: str) -> float:
    """Recover the capture timestamp from a `<sec>_<nsec>_<sensor>.json` filename."""
    name = os.path.basename(path)
    m = FILENAME_TS_RE.match(name)
    if not m:
        raise ValueError(f"cannot parse timestamp from filename: {name}")
    sec, nsec = m.groups()
    return int(sec) + int(nsec) * 1e-9


def _get_attr(attrs: dict, group: str, name: str, default=None):
    for item in attrs.get(group, []):
        if item["name"] == name:
            return item["val"]
    return default


def load_frame(path: str) -> Frame:
    """Parse one OpenLABEL JSON label file into a `Frame`."""
    with open(path) as f:
        data = json.load(f)
    ol = data["openlabel"]
    (frame_id, frame_data), = ol["frames"].items()

    props = frame_data.get("frame_properties", {})
    objects = []
    for uuid, obj in frame_data.get("objects", {}).items():
        od = obj["object_data"]
        cuboid = od["cuboid"]
        attrs = cuboid.get("attributes", {})
        num_points = _get_attr(attrs, "num", "num_points", default=0)
        occlusion = _get_attr(attrs, "text", "occlusion_level", default="NOT_OCCLUDED")
        sensor_id = _get_attr(attrs, "text", "sensor_id", default="")
        objects.append(ObjectAnnotation(
            uuid=uuid,
            cls=od["type"],
            val=list(cuboid["val"]),
            num_points=int(num_points),
            occlusion=occlusion,
            sensor_id=sensor_id,
        ))

    return Frame(
        path=path,
        frame_id=frame_id,
        timestamp=float(props["timestamp"]),
        weather=props.get("weather_type"),
        time_of_day=props.get("time_of_day"),
        objects=objects,
    )


def list_frame_paths(sensor_dir: str) -> list[str]:
    """Label-file paths in a sensor's `labels_point_clouds/<sensor>/` dir, filename-sorted."""
    return sorted(glob.glob(os.path.join(sensor_dir, "*.json")))


def iter_frames(sensor_dir: str):
    for path in list_frame_paths(sensor_dir):
        yield load_frame(path)


def load_sequence(sensor_dir: str) -> list[Frame]:
    """All frames for one sensor, sorted by capture timestamp (for track building)."""
    frames = [load_frame(p) for p in list_frame_paths(sensor_dir)]
    frames.sort(key=lambda fr: fr.timestamp)
    return frames


@dataclass
class FramePair:
    a_path: str
    b_path: str
    dt: float  # |timestamp_a - timestamp_b|, seconds


def load_pairs(dir_a: str, dir_b: str, max_dt: float = 0.10) -> list[FramePair]:
    """Match two sensor streams into positive pairs by nearest timestamp.

    R2's north/south lidars (and R4's roadside/vehicle lidars) are not
    hardware-synced, so filenames almost never share an exact timestamp —
    a pair is the closest frame in `dir_b` within `max_dt` seconds of each
    frame in `dir_a`. Typical R2 north/south gaps are ~20 ms with a ~90 ms
    worst case, so the 0.10 s default keeps all matches while rejecting
    frames with no real counterpart.
    """
    paths_a = list_frame_paths(dir_a)
    paths_b = list_frame_paths(dir_b)
    if not paths_a or not paths_b:
        return []

    ts_b = [parse_filename_timestamp(p) for p in paths_b]
    order = sorted(range(len(paths_b)), key=lambda i: ts_b[i])
    paths_b = [paths_b[i] for i in order]
    ts_b = [ts_b[i] for i in order]

    pairs = []
    for path_a in paths_a:
        t_a = parse_filename_timestamp(path_a)
        idx = bisect.bisect_left(ts_b, t_a)
        best_j, best_dt = None, None
        for j in (idx - 1, idx):
            if 0 <= j < len(ts_b):
                dt = abs(ts_b[j] - t_a)
                if best_dt is None or dt < best_dt:
                    best_j, best_dt = j, dt
        if best_j is not None and best_dt <= max_dt:
            pairs.append(FramePair(a_path=path_a, b_path=paths_b[best_j], dt=best_dt))
    return pairs


# ---------------------------------------------------------------------------
# Velocity / acceleration derivation (per-UUID, across consecutive frames)
# ---------------------------------------------------------------------------

@dataclass
class Kinematics:
    vx: float
    vy: float
    speed: float       # vxy = |v|, m/s
    accel: float | None  # axy, m/s^2 — None if no valid previous speed
    gap_flag: int         # 1 if dt > gap_flag_threshold (missing frame in track)


def speed_label(vxy: float) -> int:
    """Pseudo-label bucket for speed-state classification (no annotation needed)."""
    if vxy < 0.5:
        return 0  # stopped
    elif vxy < 3.0:
        return 1  # slow
    elif vxy < 10.0:
        return 2  # moving
    else:
        return 3  # fast


def derive_kinematics(
    frames: list[Frame],
    dt_max: float = 0.30,
    vxy_max: float = 40.0,
    axy_max: float = 10.0,
    gap_flag_threshold: float = 0.15,
    full_occlusion_num_points: int = 2,
) -> list[dict[str, Kinematics]]:
    """Per-frame uuid -> Kinematics, derived from consecutive-frame displacement.

    `frames` must be one sensor's sequence, timestamp-sorted (see
    `load_sequence`). A track transition is only used when
    `0 < dt <= dt_max` and the resulting speed is `<= vxy_max`; objects
    with `num_points <= full_occlusion_num_points` (position inferred, not
    measured) never get a velocity. Missing UUIDs (no valid previous
    observation) are simply absent from that frame's dict.
    """
    results: list[dict[str, Kinematics]] = [dict() for _ in frames]
    last_seen: dict[str, tuple[float, float, float, float]] = {}  # uuid -> (t, x, y, prev_speed)

    for i, fr in enumerate(frames):
        out = results[i]
        seen_this_frame = set()
        for obj in fr.objects:
            seen_this_frame.add(obj.uuid)
            x, y, _ = obj.xyz

            if obj.num_points <= full_occlusion_num_points:
                # Position inferred, not measured — skip velocity derivation,
                # but still refresh last_seen so a later real observation
                # can compute a valid dt against a real timestamp.
                last_seen[obj.uuid] = (fr.timestamp, x, y, None)
                continue

            prev = last_seen.get(obj.uuid)
            last_seen[obj.uuid] = (fr.timestamp, x, y, None)  # placeholder, updated below

            if prev is not None:
                t_prev, x_prev, y_prev, speed_prev = prev
                dt = fr.timestamp - t_prev
                gap_flag = 1 if dt > gap_flag_threshold else 0
                if 0 < dt <= dt_max:
                    vx = (x - x_prev) / dt
                    vy = (y - y_prev) / dt
                    vxy = (vx ** 2 + vy ** 2) ** 0.5
                    if vxy <= vxy_max:
                        accel = None
                        if speed_prev is not None:
                            a = (vxy - speed_prev) / dt
                            if abs(a) <= axy_max:
                                accel = a
                        out[obj.uuid] = Kinematics(vx=vx, vy=vy, speed=vxy, accel=accel, gap_flag=gap_flag)
                        last_seen[obj.uuid] = (fr.timestamp, x, y, vxy)

    return results


# ---------------------------------------------------------------------------
# H2 — Tracklet: STRL pairs across the "last K frames of the same UUID"
# ---------------------------------------------------------------------------

@dataclass
class TrackletPair:
    frame_idx_a: int  # earlier frame
    frame_idx_b: int  # later frame (>= frame_idx_a)
    node_idx_a: int    # index into frames[frame_idx_a].objects
    node_idx_b: int    # index into frames[frame_idx_b].objects
    uuid: str


def build_tracklet_pairs(
    frames: list[Frame], k: int = 5, rng: random.Random | None = None
) -> list[TrackletPair]:
    """H2 tracklet pairs for the STRL temporal loss (training.loss.strl_mse).

    For each appearance of an object, pair it with one earlier appearance of
    the *same* UUID sampled from the last `k` frames (claude.md's "last K=5
    frames of same UUID" tracklet window). Sampling within the window, rather
    than always using the immediately preceding frame, follows STRL's
    temporally-distant-positive design (Huang et al., ICCV 2021) — it's a
    trajectory-history hyperedge, not just adjacent-frame smoothing.

    `frames` must be one sensor's sequence, timestamp-sorted (see
    `load_sequence`). This is deliberately separate from graph_builder.py's
    single-frame hyperedge constructors: H2 needs multi-frame context, so it
    produces (frame_idx, node_idx) pairs for the training loop to embed and
    compare, not an incidence-matrix column.
    """
    rng = rng or random.Random(0)
    history: dict[str, list[tuple[int, int]]] = {}
    pairs: list[TrackletPair] = []

    for t, frame in enumerate(frames):
        for node_idx, obj in enumerate(frame.objects):
            window = [(fi, ni) for fi, ni in history.get(obj.uuid, []) if t - fi <= k]
            if window:
                fi, ni = rng.choice(window)
                pairs.append(TrackletPair(frame_idx_a=fi, frame_idx_b=t, node_idx_a=ni, node_idx_b=node_idx, uuid=obj.uuid))
            history.setdefault(obj.uuid, []).append((t, node_idx))

    return pairs
