"""Predict stress-strain curves for every generated candidate.

The default prediction path intentionally uses the validated predictor bundled
under ``external_predictor``. The older in-tree GeoLoop-StressNet path is kept
only as an explicit diagnostic mode so manuscript screening does not silently
switch to an unvalidated surrogate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.dataset import load_config, read_curve
from src.models.stressnet import GeoLoopStressNet
from src.utils.io_utils import pore_records_to_array, read_jsonl
from src.utils.normalization import FeatureNormalizer


def resolve_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def normalize_predictnet_local(pores: np.ndarray, cube_size: float) -> np.ndarray:
    """Return the 9-channel local tensor expected by ``external_predictor/model.py``.

    The validated PredictNet was trained with 3 xyz channels plus 6 feature
    channels.  Its preprocessing script inserts and then removes an Euler column;
    generated candidates do not store that column, so this adapter directly
    constructs the post-removal channel order:
    ``x, y, z, area, volume, anisotropy, phi, theta, sphericity``.
    """

    pores = np.asarray(pores, dtype=np.float32)
    if pores.size == 0:
        return np.zeros((0, 9), dtype=np.float32)
    local = np.stack(
        [
            pores[:, 0],
            pores[:, 1],
            pores[:, 2],
            pores[:, 4],
            pores[:, 5],
            pores[:, 6],
            pores[:, 7],
            (pores[:, 8] + 3.6) % 3.6,
            pores[:, 9],
        ],
        axis=1,
    ).astype(np.float32)
    local[:, :3] /= float(cube_size)
    local[:, 3:5] = np.log1p(np.clip(local[:, 3:5], 0.0, None)) / np.log(200.0)
    local[:, 6] /= 0.9
    local[:, 7] /= 3.6
    return local


def sample_or_pad(local: np.ndarray, npoints: int, rng: np.random.Generator) -> np.ndarray:
    if local.shape[0] >= npoints:
        choice = rng.choice(local.shape[0], npoints, replace=False)
        return local[np.sort(choice)].astype(np.float32)
    padded = np.zeros((npoints, local.shape[1]), dtype=np.float32)
    padded[: local.shape[0]] = local
    return padded


def inverse_predictnet_response(
    normalized: np.ndarray,
    target_response: np.ndarray,
) -> np.ndarray:
    eps = 1e-8
    safe = np.clip(np.asarray(target_response, dtype=np.float32), eps, None)
    min_val = float(np.min(safe))
    max_val = float(np.max(safe))
    if max_val <= min_val:
        max_val = min_val + eps
    return np.exp(normalized * (np.log(max_val) - np.log(min_val)) + np.log(min_val))


def inverse_predictnet_response_with_scaler(
    normalized: np.ndarray,
    scalers: dict,
) -> np.ndarray:
    if scalers.get("label_mode") != "global_log1p_standard" or "scaler_label" not in scalers:
        raise RuntimeError("scalers.bin does not contain global_log1p_standard scaler_label")
    log_stress = scalers["scaler_label"].inverse_transform(np.asarray(normalized, dtype=np.float32).reshape(1, -1))
    return np.clip(np.expm1(log_stress[0]), 0.0, None).astype(np.float32)


def predict_with_external_predictnet(args: argparse.Namespace, cfg: dict, project_root: Path) -> None:
    if not args.target_curve:
        raise SystemExit(
            "--target-curve is required when --predictor-mode external_predictnet is used. "
            "The validated predictor uses the target curve min/max for label inversion."
        )
    rules = cfg["screening"]
    checkpoint_path = resolve_project_path(project_root, rules["porenet_checkpoint"])
    scaler_path = resolve_project_path(project_root, rules["porenet_scalers"])
    for path in (checkpoint_path, scaler_path):
        if not path.exists():
            raise FileNotFoundError(path)

    predictor_code_dir = checkpoint_path.parents[2]
    sys.path.insert(0, str(predictor_code_dir))
    try:
        from joblib import load
        from model import PredictNet
    finally:
        if sys.path[0] == str(predictor_code_dir):
            sys.path.pop(0)

    scalers = load(scaler_path)
    coarse_scaler = scalers["scaler_coarse"]
    label_mode = cfg.get("screening", {}).get(
        "porenet_label_mode",
        scalers.get("label_mode", "per_curve_log_minmax"),
    )
    device = torch.device(args.device)
    model = PredictNet(feature_channels=9, global_channels=3, num_classes=51).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=True)
    model.eval()

    _, target_response = read_curve(
        Path(args.target_curve),
        cfg["data"]["curve_columns"],
        51,
    )
    target_frame = pd.read_csv(args.target_curve)
    strain_column = cfg["data"]["curve_columns"][0]
    if strain_column in target_frame.columns:
        strain = np.linspace(
            float(target_frame[strain_column].min()),
            float(target_frame[strain_column].max()),
            51,
        )
    else:
        strain = np.linspace(0.0, 0.6, 51)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(args.input)
    npoints = int(args.predictnet_points)
    cube_size = float(cfg["data"]["cube_size_mm"])
    rng = np.random.default_rng(int(args.seed))
    for record in records:
        pores = pore_records_to_array(record["pores"])
        local = normalize_predictnet_local(pores, cube_size)
        local = sample_or_pad(local, npoints, rng)
        if not record.get("macro") or len(record["macro"]) < 5:
            raise ValueError(f"{record['candidate_id']} has no 5-value macro vector")
        macro = np.asarray(record["macro"], dtype=np.float32)
        coarse = np.asarray([[macro[0], macro[3], macro[4]]], dtype=np.float32)
        coarse = coarse_scaler.transform(coarse).astype(np.float32)
        with torch.no_grad():
            prediction = model(
                torch.from_numpy(coarse).to(device),
                torch.from_numpy(local).unsqueeze(0).to(device),
            )[0].cpu().numpy()
        if label_mode == "global_log1p_standard":
            stress = inverse_predictnet_response_with_scaler(prediction, scalers)
        else:
            stress = inverse_predictnet_response(prediction, target_response)
        prediction_path = output_dir / f"{record['candidate_id']}.csv"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"Strain": strain, "Stress": stress}).to_csv(
            prediction_path,
            index=False,
            encoding="utf-8-sig",
        )
    metadata = {
        "predictor_mode": "external_predictnet",
        "predictor_code_dir": str(predictor_code_dir.resolve()),
        "checkpoint": str(checkpoint_path),
        "scaler": str(scaler_path),
        "label_mode": label_mode,
        "target_curve": str(Path(args.target_curve).resolve()),
        "npoints": npoints,
        "seed": int(args.seed),
    }
    (output_dir / "prediction_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} validated PredictNet stress predictions to {output_dir}")


def predict_with_internal_stressnet(args: argparse.Namespace, cfg: dict) -> None:
    validator_cfg = cfg["response_validator"]
    checkpoint_path = Path(validator_cfg["checkpoint"])
    normalizer = FeatureNormalizer.load(validator_cfg["normalizer"])
    if normalizer.state.curve_kind != "stress_strain":
        raise RuntimeError("Stress validator normalizer has the wrong curve_kind")
    device = torch.device(args.device)
    model = GeoLoopStressNet(
        pore_feature_dim=cfg["model"]["pore_feature_dim"],
        macro_dim=cfg["model"]["macro_dim"], set_hidden=cfg["model"]["set_hidden"],
        hidden=validator_cfg["hidden"], curve_points=cfg["data"]["curve_points"],
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    max_pores = int(cfg["data"]["max_pores"])
    strain = np.linspace(0.0, normalizer.state.strain_max, cfg["data"]["curve_points"])
    records = read_jsonl(args.input)
    for record in records:
        pores = pore_records_to_array(record["pores"])
        if len(pores) > max_pores:
            pores = pores[np.argsort(pores[:, 5])[-max_pores:]]
        normalized = normalizer.transform_pores(pores)
        padded = np.zeros((max_pores, normalized.shape[1]), dtype=np.float32)
        mask = np.zeros(max_pores, dtype=bool)
        padded[:len(normalized)] = normalized
        mask[:len(normalized)] = True
        if not record.get("macro"):
            raise ValueError(f"{record['candidate_id']} has no macro vector")
        macro = normalizer.transform_macro(np.asarray(record["macro"], dtype=np.float32))
        with torch.no_grad():
            prediction = model(
                torch.from_numpy(padded).unsqueeze(0).to(device),
                torch.from_numpy(mask).unsqueeze(0).to(device),
                torch.from_numpy(macro).unsqueeze(0).to(device),
            )[0].cpu().numpy()
        stress = normalizer.inverse_response(prediction)
        prediction_path = output_dir / f"{record['candidate_id']}.csv"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"Strain": strain, "Stress": stress}).to_csv(
            prediction_path, index=False, encoding="utf-8-sig"
        )
    metadata = {
        "predictor_mode": "internal_stressnet_diagnostic",
        "checkpoint": str(checkpoint_path.resolve()),
        "normalizer": str(Path(validator_cfg["normalizer"]).resolve()),
    }
    (output_dir / "prediction_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} internal diagnostic stress predictions to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-curve", help="required for the validated external PredictNet path")
    parser.add_argument(
        "--predictor-mode",
        choices=["external_predictnet", "internal_stressnet"],
        default="external_predictnet",
    )
    parser.add_argument("--predictnet-points", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    cfg, project_root = load_config(args.config)
    if args.predictor_mode == "external_predictnet":
        predict_with_external_predictnet(args, cfg, project_root)
    else:
        predict_with_internal_stressnet(args, cfg)


if __name__ == "__main__":
    main()
