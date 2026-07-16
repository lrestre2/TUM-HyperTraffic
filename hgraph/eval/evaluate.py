"""Linear-probe evaluation: frozen encoder node embeddings -> speed-state classification.

See claude.md ablation protocol: three splits (random 80/20, scenario
holdout, day/night holdout) and stratified reports — VRU-only,
low-point-count (<=5 pts), occlusion-stratified — since overall macro-F1
alone is insufficient. Run from `hgraph/`:

    python -m eval.evaluate --checkpoint outputs/hgnn_best.pt --model hgnn
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score

from data.dataset import derive_kinematics, load_sequence, speed_label
from data.graph_builder import (
    FEATURE_NAMES,
    OCCLUSION_MAP,
    bev_edges,
    bev_xy,
    build_adjacency_matrix,
    build_h1_hyperedges,
    build_h3_hyperedges,
    build_h4_incidence,
    build_incidence_matrix,
    build_node_features,
    stack_incidence,
)
from models.gcn_baseline import GCNModel
from models.hgnn import HGNNModel

# hgraph/eval/evaluate.py -> hgraph/ -> project root. `paths.*` in
# configs/default.yaml are written relative to the project root (see
# `output_dir: hgraph/outputs`), independent of the caller's cwd.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def resolve_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


VRU_CLASSES = {"PEDESTRIAN", "BICYCLE", "MOTORCYCLE"}
SPEED_STATE_NAMES = ["stopped", "slow", "moving", "fast"]
SPEED_LABELS = [0, 1, 2, 3]
OCC_NAMES = {0: "NOT", 1: "PARTIAL", 2: "FULL"}


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def r2_scenario_dirs(r2_root: str, scenarios: list[str]) -> dict[str, dict[str, str]]:
    """scenario -> {'north': dir, 'south': dir}"""
    out = {}
    for scenario in scenarios:
        matches = glob.glob(os.path.join(r2_root, scenario, "*", "labels_point_clouds"))
        if not matches:
            raise FileNotFoundError(f"no labels_point_clouds under {r2_root}/{scenario}")
        base = matches[0]
        out[scenario] = {
            "north": os.path.join(base, "s110_lidar_ouster_north"),
            "south": os.path.join(base, "s110_lidar_ouster_south"),
        }
    return out


def extract_examples(
    sensor_dir: str, scenario: str, cfg: dict, model, device: str, hyperedges: str = "h1"
) -> list[dict]:
    """Every object with a derived speed label in one sensor sequence -> a labeled example."""
    bev_threshold = cfg["hyperedges"]["bev_threshold"]
    kin_cfg = cfg["kinematics"]
    is_hgnn = isinstance(model, HGNNModel)

    frames = load_sequence(sensor_dir)
    kinematics_per_frame = derive_kinematics(
        frames,
        dt_max=kin_cfg["dt_max"],
        vxy_max=kin_cfg["vxy_max"],
        axy_max=kin_cfg["axy_max"],
        gap_flag_threshold=kin_cfg["gap_flag_threshold"],
        full_occlusion_num_points=kin_cfg["full_occlusion_num_points"],
    )

    examples = []
    model.eval()
    with torch.no_grad():
        for frame, kin in zip(frames, kinematics_per_frame):
            if not kin or len(frame.objects) == 0:
                continue
            x = build_node_features(frame, kin)
            xy = bev_xy(frame)
            x_t = torch.from_numpy(x).to(device)

            if is_hgnn:
                mats = [build_incidence_matrix(build_h1_hyperedges(xy, radius=bev_threshold))]
                if hyperedges == "full":
                    vx = np.array([kin[o.uuid].vx if o.uuid in kin else 0.0 for o in frame.objects])
                    vy = np.array([kin[o.uuid].vy if o.uuid in kin else 0.0 for o in frame.objects])
                    h3_ids = build_h3_hyperedges(
                        xy, vx, vy, horizon=cfg["hyperedges"]["project_horizon"], radius=bev_threshold
                    )
                    mats.append(build_incidence_matrix(h3_ids))
                    classes = [o.cls for o in frame.objects]
                    mats.append(build_h4_incidence(
                        xy, classes, cfg["hyperedges"]["vru_radius_veh"], cfg["hyperedges"]["vru_radius_ped"]
                    ))
                struct = torch.from_numpy(stack_incidence(mats, len(frame.objects))).to(device)
            else:
                edges = bev_edges(xy, radius=bev_threshold)
                struct = torch.from_numpy(build_adjacency_matrix(edges, len(frame.objects))).to(device)

            node_embeds = model.embed_nodes(x_t, struct).cpu().numpy()

            for i, obj in enumerate(frame.objects):
                k = kin.get(obj.uuid)
                if k is None:
                    continue
                examples.append({
                    "embedding": node_embeds[i],
                    "label": speed_label(k.speed),
                    "scenario": scenario,
                    "time_of_day": frame.time_of_day,
                    "cls": obj.cls,
                    "num_points": obj.num_points,
                    "occ_emb": OCCLUSION_MAP.get(obj.occlusion, 2),
                })
    return examples


# ---------------------------------------------------------------------------
# Splits (claude.md ablation protocol)
# ---------------------------------------------------------------------------

def random_split(examples: list[dict], test_frac: float = 0.2, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(examples))
    n_test = int(len(examples) * test_frac)
    test_ids = set(idx[:n_test].tolist())
    train = [e for i, e in enumerate(examples) if i not in test_ids]
    test = [e for i, e in enumerate(examples) if i in test_ids]
    return train, test


def scenario_holdout_split(examples: list[dict], train_scenarios: set[str], test_scenarios: set[str]):
    train = [e for e in examples if e["scenario"] in train_scenarios]
    test = [e for e in examples if e["scenario"] in test_scenarios]
    return train, test


def day_night_split(examples: list[dict], night_values: tuple[str, ...] = ("NIGHT",)):
    """Train on non-night frames, test on night frames — driven by the OpenLABEL
    `time_of_day` attribute, not scenario name (R2: s01/s02=DUSK, s03=DAY, s04=NIGHT)."""
    train = [e for e in examples if e["time_of_day"] not in night_values]
    test = [e for e in examples if e["time_of_day"] in night_values]
    return train, test


# ---------------------------------------------------------------------------
# Linear probe + stratified reporting
# ---------------------------------------------------------------------------

def fit_linear_probe(train: list[dict], seed: int = 0) -> LogisticRegression:
    X = np.stack([e["embedding"] for e in train])
    y = np.array([e["label"] for e in train])
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(X, y)
    return clf


def _subset_report(clf: LogisticRegression, examples: list[dict], name: str) -> dict:
    if not examples:
        return {"name": name, "n": 0}
    X = np.stack([e["embedding"] for e in examples])
    y_true = np.array([e["label"] for e in examples])
    y_pred = clf.predict(X)
    return {
        "name": name,
        "n": len(examples),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=SPEED_LABELS, zero_division=0),
        "per_class_f1": f1_score(y_true, y_pred, average=None, labels=SPEED_LABELS, zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=SPEED_LABELS).tolist(),
    }


def evaluate_split(train: list[dict], test: list[dict], split_name: str) -> dict:
    clf = fit_linear_probe(train)

    reports = {"overall": _subset_report(clf, test, "overall")}
    reports["vru_only"] = _subset_report(clf, [e for e in test if e["cls"] in VRU_CLASSES], "vru_only")
    reports["low_point_count"] = _subset_report(
        clf, [e for e in test if e["num_points"] <= 5], "low_point_count(<=5pts)"
    )
    for occ_id, occ_name in OCC_NAMES.items():
        key = f"occ_{occ_name.lower()}"
        reports[key] = _subset_report(clf, [e for e in test if e["occ_emb"] == occ_id], f"occlusion={occ_name}")

    return {"split": split_name, "n_train": len(train), "n_test": len(test), "reports": reports}


def print_report(result: dict) -> None:
    print(f"\n=== {result['split']}  (train n={result['n_train']}, test n={result['n_test']}) ===")
    for rep in result["reports"].values():
        if rep["n"] == 0:
            print(f"  {rep['name']:<24s} n=0 (empty)")
            continue
        per_class = [round(v, 3) for v in rep["per_class_f1"]]
        print(
            f"  {rep['name']:<24s} n={rep['n']:6d}  macro_F1={rep['macro_f1']:.3f}  "
            f"per_class_F1({','.join(SPEED_STATE_NAMES)})={per_class}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", choices=["hgnn", "gcn"], default="hgnn")
    parser.add_argument(
        "--hyperedges", choices=["h1", "full"], default="h1",
        help="hgnn only: must match the --hyperedges the checkpoint was trained with",
    )
    parser.add_argument("--scenarios", nargs="+", default=["r2_s01", "r2_s02", "r2_s03", "r2_s04"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = args.device
    in_dim = len(FEATURE_NAMES)

    model_cls = HGNNModel if args.model == "hgnn" else GCNModel
    model = model_cls(
        in_dim=in_dim,
        encoder_dim=cfg["model"]["encoder_dim"],
        projector_dim=cfg["model"]["projector_dim"],
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    dirs = r2_scenario_dirs(resolve_path(cfg["paths"]["r2_root"]), args.scenarios)
    examples = []
    for scenario, sensors in dirs.items():
        for sensor_dir in sensors.values():
            examples.extend(extract_examples(sensor_dir, scenario, cfg, model, device, hyperedges=args.hyperedges))
    print(f"total labeled examples: {len(examples)}")

    results = []

    train, test = random_split(examples, test_frac=0.2)
    results.append(evaluate_split(train, test, "random_80_20"))

    train, test = scenario_holdout_split(examples, {"r2_s01", "r2_s02", "r2_s03"}, {"r2_s04"})
    results.append(evaluate_split(train, test, "scenario_holdout(s01-03->s04)"))

    train, test = day_night_split(examples)
    results.append(evaluate_split(train, test, "day_night_holdout(->NIGHT)"))

    for r in results:
        print_report(r)


if __name__ == "__main__":
    main()
