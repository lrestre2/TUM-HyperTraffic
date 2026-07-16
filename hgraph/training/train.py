"""Self-supervised training loop: encoder over north/south (R2) positive
pairs, NT-Xent contrastive loss on the pooled scene embedding. Supports
either the HGNN encoder (H1, or H1+H3+H4 with --hyperedges full) or the
vanilla GCN baseline (pairwise BEV edges), for the claude.md ablation.
Optionally adds the H2 tracklet STRL loss on top of either encoder (--strl).

See claude.md "Full pipeline" and "Second sanity check: NT-Xent loss
decreases over the first 10 batches." Run from the `hgraph/` directory:

    python -m training.train --model hgnn --hyperedges full --strl \\
        --scenarios r2_s01 r2_s02 r2_s03 --val-scenarios r2_s04
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import random
import time

import numpy as np
import torch
import yaml

from data.dataset import build_tracklet_pairs, derive_kinematics, load_frame, load_pairs, load_sequence
from data.graph_builder import (
    FEATURE_NAMES,
    bev_edges,
    bev_xy,
    build_adjacency_matrix,
    build_h1_hyperedges,
    build_h3_hyperedges,
    build_h4_incidence,
    build_h5_incidence,
    build_incidence_matrix,
    build_node_features,
    stack_incidence,
)
from models.gcn_baseline import GCNModel
from models.hgnn import HGNNModel
from training.loss import combined_loss, strl_mse

MODEL_CLASSES = {"hgnn": HGNNModel, "gcn": GCNModel}

# hgraph/training/train.py -> hgraph/ -> project root. `paths.*` in
# configs/default.yaml are written relative to the project root (see
# `output_dir: hgraph/outputs`), independent of the caller's cwd.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def resolve_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def r2_sensor_dirs(r2_root: str, scenarios: list[str]) -> list[tuple[str, str]]:
    """(north_dir, south_dir) pair of label dirs for each requested r2 scenario."""
    dirs = []
    for scenario in scenarios:
        matches = glob.glob(os.path.join(r2_root, scenario, "*", "labels_point_clouds"))
        if not matches:
            raise FileNotFoundError(f"no labels_point_clouds under {r2_root}/{scenario}")
        base = matches[0]
        dirs.append((
            os.path.join(base, "s110_lidar_ouster_north"),
            os.path.join(base, "s110_lidar_ouster_south"),
        ))
    return dirs


def r4_sensor_dir(r4_root: str, split: str) -> str:
    """R4's single registered (fused roadside+vehicle) label dir for one split."""
    matches = glob.glob(os.path.join(r4_root, split, "labels_point_clouds", "*registered*"))
    if not matches:
        raise FileNotFoundError(f"no registered labels_point_clouds under {r4_root}/{split}")
    return matches[0]


def build_pair_index(r2_root: str, scenarios: list[str], max_dt: float) -> list:
    pairs = []
    for north_dir, south_dir in r2_sensor_dirs(r2_root, scenarios):
        pairs.extend(load_pairs(north_dir, south_dir, max_dt=max_dt))
    return pairs


def build_kinematics_and_tracklets(
    r2_root: str, scenarios: list[str], kin_cfg: dict, tracklet_k: int, seed: int = 0
) -> tuple[dict, list[tuple[str, int, str, int]]]:
    """Per-sensor-sequence kinematics (for H3 velocity + node features) and H2
    tracklet pairs, across every sensor stream in `scenarios`.

    Returns (kin_index, tracklet_pairs): `kin_index` maps label-file path ->
    {uuid: Kinematics}; `tracklet_pairs` is a flat list of
    (path_a, node_idx_a, path_b, node_idx_b) ready for the STRL loss —
    resolved from frame indices to paths here so the training loop doesn't
    need to keep each sequence's frame list around.
    """
    rng = random.Random(seed)
    kin_index: dict = {}
    tracklet_pairs: list[tuple[str, int, str, int]] = []
    for north_dir, south_dir in r2_sensor_dirs(r2_root, scenarios):
        for sensor_dir in (north_dir, south_dir):
            frames = load_sequence(sensor_dir)
            kin_list = derive_kinematics(frames, **kin_cfg)
            for frame, kin in zip(frames, kin_list):
                kin_index[frame.path] = kin
            for tp in build_tracklet_pairs(frames, k=tracklet_k, rng=rng):
                tracklet_pairs.append(
                    (frames[tp.frame_idx_a].path, tp.node_idx_a, frames[tp.frame_idx_b].path, tp.node_idx_b)
                )
    return kin_index, tracklet_pairs


def build_r4_kinematics_and_tracklets(
    r4_root: str, split: str, kin_cfg: dict, tracklet_k: int, seed: int = 0
) -> tuple[dict, list[tuple[str, int, str, int]]]:
    """Same as `build_kinematics_and_tracklets`, for R4's single registered
    (fused roadside+vehicle) label sequence — no north/south pairing to do.
    """
    rng = random.Random(seed)
    frames = load_sequence(r4_sensor_dir(r4_root, split))
    kin_list = derive_kinematics(frames, **kin_cfg)
    kin_index = {frame.path: kin for frame, kin in zip(frames, kin_list)}
    tracklet_pairs = [
        (frames[tp.frame_idx_a].path, tp.node_idx_a, frames[tp.frame_idx_b].path, tp.node_idx_b)
        for tp in build_tracklet_pairs(frames, k=tracklet_k, rng=rng)
    ]
    return kin_index, tracklet_pairs


def run_epoch_r4(
    model: HGNNModel | GCNModel,
    tracklet_pairs: list[tuple[str, int, str, int]],
    cfg: dict,
    model_type: str,
    kin_index: dict,
    optimizer: torch.optim.Optimizer | None = None,
    device: str = "cpu",
    max_batches: int | None = None,
    batch_losses: list[float] | None = None,
) -> float:
    """R4 has one fused view per frame (see H5's design notes) — no second
    view to run NT-Xent across. Training here is H2's STRL temporal loss
    alone: pull embeddings of the same tracked object, at two points in its
    last `tracklet_k` frames, together. Structure (H1+H3+H4+H5) still comes
    from `--hyperedges full`; this only replaces the *loss*, not the encoder.
    """
    training = optimizer is not None
    model.train(training)
    bev_threshold = cfg["hyperedges"]["bev_threshold"]
    batch_size = cfg["train"]["batch_size"]
    h4_cfg = cfg["hyperedges"]

    rng = np.random.default_rng(0)
    order = rng.permutation(len(tracklet_pairs)) if training else np.arange(len(tracklet_pairs))

    total_loss, n_batches = 0.0, 0
    for start in range(0, len(order), batch_size):
        if max_batches is not None and n_batches >= max_batches:
            break
        batch_idx = order[start:start + batch_size]
        if len(batch_idx) < 1:
            continue

        with torch.set_grad_enabled(training):
            s1s, s2s = [], []
            for i in batch_idx:
                path_a, node_a, path_b, node_b = tracklet_pairs[i]
                xa, Ha = frame_to_graph(path_a, bev_threshold, model_type, hyperedges="full", kin_index=kin_index, h4_cfg=h4_cfg, include_h5=True)
                xb, Hb = frame_to_graph(path_b, bev_threshold, model_type, hyperedges="full", kin_index=kin_index, h4_cfg=h4_cfg, include_h5=True)
                if xa.shape[0] == 0 or xb.shape[0] == 0:
                    continue
                xa, Ha, xb, Hb = xa.to(device), Ha.to(device), xb.to(device), Hb.to(device)
                emb_a = model.embed_nodes(xa, Ha)
                emb_b = model.embed_nodes(xb, Hb)
                s1s.append(emb_a[node_a])
                s2s.append(emb_b[node_b])

            if not s1s:
                continue

            loss = strl_mse(torch.stack(s1s), torch.stack(s2s))

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        if batch_losses is not None:
            batch_losses.append(loss.item())

    return total_loss / max(n_batches, 1)


def frame_to_graph(
    path: str,
    bev_threshold: float,
    model_type: str,
    hyperedges: str = "h1",
    kin_index: dict | None = None,
    h4_cfg: dict | None = None,
    include_h5: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One label file -> (node_features, structure) tensors.

    `structure` is the incidence matrix for "hgnn" (H1 alone, or H1+H3+H4,
    +H5 when `include_h5` — R4 only, see `H5_ROADSIDE_SENSOR` — when
    `hyperedges="full"`), or the plain BEV adjacency matrix for "gcn"
    (ablation: same features, no hyperedges). H3 needs derived velocity, so
    `kin_index` (from `build_kinematics_and_tracklets`) also feeds real
    vx/vy/speed into the node features here — previously always zero.
    """
    frame = load_frame(path)
    kin = (kin_index or {}).get(path, {})
    x = build_node_features(frame, kin)
    xy = bev_xy(frame)
    if model_type == "hgnn":
        mats = [build_incidence_matrix(build_h1_hyperedges(xy, radius=bev_threshold))]
        if hyperedges == "full":
            vx = np.array([kin[o.uuid].vx if o.uuid in kin else 0.0 for o in frame.objects])
            vy = np.array([kin[o.uuid].vy if o.uuid in kin else 0.0 for o in frame.objects])
            h3_ids = build_h3_hyperedges(xy, vx, vy, horizon=h4_cfg["project_horizon"], radius=bev_threshold)
            mats.append(build_incidence_matrix(h3_ids))
            classes = [o.cls for o in frame.objects]
            mats.append(build_h4_incidence(xy, classes, h4_cfg["vru_radius_veh"], h4_cfg["vru_radius_ped"]))
            if include_h5:
                sensor_ids = [o.sensor_id for o in frame.objects]
                mats.append(build_h5_incidence(xy, sensor_ids, radius=bev_threshold, vehicle_weight=h4_cfg["v2x_init_weight"]))
        struct = stack_incidence(mats, len(frame.objects))
    else:
        edges = bev_edges(xy, radius=bev_threshold)
        struct = build_adjacency_matrix(edges, len(frame.objects))
    return torch.from_numpy(x), torch.from_numpy(struct)


def run_epoch(
    model: HGNNModel | GCNModel,
    pairs: list,
    cfg: dict,
    model_type: str,
    optimizer: torch.optim.Optimizer | None = None,
    device: str = "cpu",
    max_batches: int | None = None,
    batch_losses: list[float] | None = None,
    hyperedges: str = "h1",
    kin_index: dict | None = None,
    tracklet_pairs: list[tuple[str, int, str, int]] | None = None,
    lambda_strl: float = 0.5,
) -> float:
    """One pass over `pairs`; trains if `optimizer` is given, else evaluates under no_grad.

    `max_batches` caps the number of batches (for smoke tests); `batch_losses`,
    if given, is appended with the per-batch loss (for the NT-Xent sanity check).
    When `tracklet_pairs` is given (H2), each batch also samples up to
    `batch_size` tracklet pairs and adds the STRL temporal loss on their node
    embeddings (claude.md combined objective: L = L_nt_xent + lambda * L_strl).
    """
    training = optimizer is not None
    model.train(training)
    bev_threshold = cfg["hyperedges"]["bev_threshold"]
    batch_size = cfg["train"]["batch_size"]
    tau = cfg["loss"]["tau"]
    h4_cfg = cfg["hyperedges"]

    rng = np.random.default_rng(0)
    order = rng.permutation(len(pairs)) if training else np.arange(len(pairs))

    def to_graph(path: str) -> tuple[torch.Tensor, torch.Tensor]:
        return frame_to_graph(path, bev_threshold, model_type, hyperedges=hyperedges, kin_index=kin_index, h4_cfg=h4_cfg)

    total_loss, n_batches = 0.0, 0
    for start in range(0, len(order), batch_size):
        if max_batches is not None and n_batches >= max_batches:
            break
        batch_idx = order[start:start + batch_size]
        if len(batch_idx) < 2:
            continue  # NT-Xent needs >= 2 samples in the batch for negatives

        with torch.set_grad_enabled(training):
            z1s, z2s = [], []
            for i in batch_idx:
                pair = pairs[i]
                x1, H1 = to_graph(pair.a_path)
                x2, H2 = to_graph(pair.b_path)
                if x1.shape[0] == 0 or x2.shape[0] == 0:
                    continue
                x1, H1 = x1.to(device), H1.to(device)
                x2, H2 = x2.to(device), H2.to(device)
                _, p1 = model(x1, H1)
                _, p2 = model(x2, H2)
                z1s.append(p1)
                z2s.append(p2)

            if len(z1s) < 2:
                continue

            z1 = torch.stack(z1s)
            z2 = torch.stack(z2s)

            strl_pairs = None
            if tracklet_pairs:
                sample = rng.choice(len(tracklet_pairs), size=min(batch_size, len(tracklet_pairs)), replace=False)
                s1s, s2s = [], []
                for j in sample:
                    path_a, node_a, path_b, node_b = tracklet_pairs[j]
                    xa, Ha = to_graph(path_a)
                    xb, Hb = to_graph(path_b)
                    xa, Ha, xb, Hb = xa.to(device), Ha.to(device), xb.to(device), Hb.to(device)
                    emb_a = model.embed_nodes(xa, Ha)
                    emb_b = model.embed_nodes(xb, Hb)
                    s1s.append(emb_a[node_a])
                    s2s.append(emb_b[node_b])
                if s1s:
                    strl_pairs = (torch.stack(s1s), torch.stack(s2s))

            loss = combined_loss(z1, z2, tau=tau, lambda_strl=lambda_strl, strl_pairs=strl_pairs)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        if batch_losses is not None:
            batch_losses.append(loss.item())

    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"))
    parser.add_argument("--dataset", choices=["r2", "r4"], default="r2")
    parser.add_argument("--model", choices=["hgnn", "gcn"], default="hgnn")
    parser.add_argument(
        "--hyperedges", choices=["h1", "full"], default="h1",
        help="hgnn only: 'h1' (proximity, original ablation) or 'full' (H1+H3+H4). "
             "Ignored for --dataset r4, which always trains full H1+H3+H4+H5.",
    )
    parser.add_argument("--strl", action="store_true", help="r2 only: add the H2 tracklet STRL loss on top of NT-Xent")
    parser.add_argument("--scenarios", nargs="+", default=["r2_s01", "r2_s02", "r2_s03"])
    parser.add_argument("--val-scenarios", nargs="+", default=["r2_s04"])
    parser.add_argument("--r4-train-split", default="train")
    parser.add_argument("--r4-val-split", default="val")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = args.device
    epochs = args.epochs or cfg["train"]["epochs"]
    kin_cfg = {k: cfg["kinematics"][k] for k in
               ("dt_max", "vxy_max", "axy_max", "gap_flag_threshold", "full_occlusion_num_points")}
    tracklet_k = cfg["hyperedges"]["tracklet_k"]

    if args.dataset == "r4":
        # R4's labels are pre-fused (one box per real object, see H5's design
        # notes) — no second view for NT-Xent, so H1+H3+H4+H5 structure is
        # trained with the H2 STRL temporal loss alone.
        r4_root = resolve_path(cfg["paths"]["r4_root"])
        train_kin, train_tracklets = build_r4_kinematics_and_tracklets(r4_root, args.r4_train_split, kin_cfg, tracklet_k)
        val_kin, val_tracklets = build_r4_kinematics_and_tracklets(r4_root, args.r4_val_split, kin_cfg, tracklet_k, seed=1)
        print(f"model: {args.model}  dataset: r4 (full H1+H3+H4+H5, STRL-only)  "
              f"train tracklet pairs: {len(train_tracklets)}  val tracklet pairs: {len(val_tracklets)}  device: {device}")
    else:
        r2_root = resolve_path(cfg["paths"]["r2_root"])
        max_dt = cfg["pairing"]["max_dt"]
        train_pairs = build_pair_index(r2_root, args.scenarios, max_dt)
        val_pairs = build_pair_index(r2_root, args.val_scenarios, max_dt)
        print(f"model: {args.model}  hyperedges: {args.hyperedges}  strl: {args.strl}  "
              f"train pairs: {len(train_pairs)}  val pairs: {len(val_pairs)}  device: {device}")

        # H3 needs real velocity (not the zero default) and H2/STRL needs tracklet
        # pairs, so build both whenever either feature is requested.
        train_kin, train_tracklets = ({}, [])
        val_kin, val_tracklets = ({}, [])
        if args.hyperedges == "full" or args.strl:
            train_kin, train_tracklets = build_kinematics_and_tracklets(r2_root, args.scenarios, kin_cfg, tracklet_k)
            val_kin, val_tracklets = build_kinematics_and_tracklets(r2_root, args.val_scenarios, kin_cfg, tracklet_k, seed=1)
            print(f"H2 tracklet pairs: train {len(train_tracklets)}  val {len(val_tracklets)}")

    in_dim = len(FEATURE_NAMES)
    model_cls = MODEL_CLASSES[args.model]
    model = model_cls(
        in_dim=in_dim,
        encoder_dim=cfg["model"]["encoder_dim"],
        projector_dim=cfg["model"]["projector_dim"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    output_dir = resolve_path(cfg["paths"]["output_dir"])
    os.makedirs(output_dir, exist_ok=True)
    if args.dataset == "r4":
        tag = args.model + "_r4_v2x"
    else:
        tag = args.model + ("_full" if args.hyperedges == "full" else "") + ("_strl" if args.strl else "")
    log_path = os.path.join(output_dir, f"train_log_{tag}.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "seconds"])

    lambda_strl = cfg["loss"]["lambda_strl"] if args.strl else 0.0
    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        if args.dataset == "r4":
            train_loss = run_epoch_r4(model, train_tracklets, cfg, args.model, train_kin, optimizer=optimizer, device=device)
            val_loss = run_epoch_r4(model, val_tracklets, cfg, args.model, val_kin, optimizer=None, device=device)
        else:
            train_loss = run_epoch(
                model, train_pairs, cfg, args.model, optimizer=optimizer, device=device,
                hyperedges=args.hyperedges, kin_index=train_kin,
                tracklet_pairs=train_tracklets if args.strl else None, lambda_strl=lambda_strl,
            )
            val_loss = run_epoch(
                model, val_pairs, cfg, args.model, optimizer=None, device=device,
                hyperedges=args.hyperedges, kin_index=val_kin,
                tracklet_pairs=val_tracklets if args.strl else None, lambda_strl=lambda_strl,
            )
        elapsed = time.time() - t0

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, f"{elapsed:.1f}"])
        print(f"epoch {epoch:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  ({elapsed:.1f}s)")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, f"{tag}_best.pt"))

    torch.save(model.state_dict(), os.path.join(output_dir, f"{tag}_last.pt"))


if __name__ == "__main__":
    main()
