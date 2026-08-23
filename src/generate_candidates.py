"""Generate candidates from a target response curve or perform a real-data dry run."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataset import load_config, read_curve, read_pores
from src.utils.io_utils import pore_array_to_records, write_jsonl
from src.utils.metrics import geometry_metrics
from src.utils.normalization import FeatureNormalizer, MACRO_COLUMNS


def build_model(cfg: dict):
    from src.models.poregen_cvae import GeoLoopPoreGen
    from src.models.poregen_slot_cvae import GeoLoopPoreGenV2

    model_cfg = dict(cfg["model"])
    architecture = model_cfg.pop("architecture", "slot_linear_v1")
    if architecture == "slot_query_v2":
        return GeoLoopPoreGenV2(max_pores=cfg["data"]["max_pores"], **model_cfg)
    if architecture != "slot_linear_v1":
        raise ValueError(f"Unknown GeoLoop-PoreGen architecture: {architecture}")
    return GeoLoopPoreGen(max_pores=cfg["data"]["max_pores"], **model_cfg)


def make_record(candidate_id: str, pores: np.ndarray, macro: np.ndarray | None,
                source: dict, cube_size: float) -> dict:
    return {
        "schema_version": "geoloop-candidate-v1",
        "candidate_id": candidate_id,
        "units": {"coordinates": "mm", "area": "mm^2", "volume": "mm^3", "angles": "rad"},
        "pore_columns": ["x", "y", "z", "eq_diameter", "area", "volume",
                         "anisotropy", "phi", "theta", "sphericity"],
        "macro": None if macro is None else [float(x) for x in macro],
        "pores": pore_array_to_records(pores),
        "geometry_metrics": geometry_metrics(pores, cube_size),
        "source": source,
    }


def select_pore_indices(
    presence: np.ndarray,
    physical_pores: np.ndarray,
    macro: np.ndarray | None,
    cfg: dict,
    cube_size: float,
) -> np.ndarray:
    generation_cfg = cfg.get("generation", {})
    mode = generation_cfg.get("presence_selection", "threshold")
    threshold = float(generation_cfg.get("presence_threshold", 0.5))
    if mode == "threshold":
        return np.flatnonzero(presence >= threshold)

    order = np.argsort(-presence)
    min_pores = int(cfg.get("screening", {}).get("min_pores", 1))
    max_pores = min(int(cfg.get("screening", {}).get("max_pores", len(presence))), len(presence))
    count_scale = float(generation_cfg.get("presence_count_scale", 1.0))
    expected_count = int(round(float(presence.sum()) * count_scale))
    expected_count = int(np.clip(expected_count, min_pores, max_pores))
    if mode == "topk_expected_count":
        return np.sort(order[:expected_count])

    if mode == "topk_porosity":
        if macro is not None and len(macro) >= 5:
            target_porosity = float(macro[4])
            cumulative_volume = np.cumsum(physical_pores[order, 5])
            target_volume = max(target_porosity, 0.0) * float(cube_size) ** 3
            porosity_count = int(np.searchsorted(cumulative_volume, target_volume, side="left") + 1)
            count = max(expected_count, porosity_count)
        else:
            count = expected_count
        count = int(np.clip(count, min_pores, max_pores))
        return np.sort(order[:count])

    raise ValueError(f"Unknown generation.presence_selection: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--target-curve")
    parser.add_argument("--samples", type=int)
    parser.add_argument("--temperature", type=float,
                        help="latent sampling temperature; lower favors quality, higher diversity")
    parser.add_argument("--output", default=None)
    parser.add_argument("--dry-run-real-sample", metavar="NAME",
                        help="round-trip one real pore cloud; does not claim model generation")
    args = parser.parse_args()
    cfg, root = load_config(args.config)
    output = Path(args.output or cfg["generation"]["output_dir"]) / "candidates.jsonl"
    cube_size = float(cfg["data"]["cube_size_mm"])

    if args.dry_run_real_sample:
        name = args.dry_run_real_sample
        pores = read_pores(root / cfg["data"]["pore_dir"] / f"{name}.csv")
        global_df = pd.read_csv(root / cfg["data"]["global_csv"]).set_index("name")
        macro = global_df.loc[name, MACRO_COLUMNS].to_numpy(float)
        record = make_record(
            f"roundtrip_{name}", pores, macro,
            {"type": "real_data_roundtrip", "sample": name,
             "note": "Interface test only; not a generated scientific result."}, cube_size,
        )
        write_jsonl([record], output)
        print(f"Wrote real-data round-trip candidate: {output}")
        return

    checkpoint_path = Path(cfg["generation"]["checkpoint"])
    normalizer_path = Path(cfg["generation"]["normalizer"])
    if not checkpoint_path.is_absolute():
        checkpoint_path = (root / checkpoint_path).resolve()
    if not normalizer_path.is_absolute():
        normalizer_path = (root / normalizer_path).resolve()
    if not args.target_curve:
        raise SystemExit("--target-curve is required unless --dry-run-real-sample is used")
    if not checkpoint_path.exists() or not normalizer_path.exists():
        raise FileNotFoundError(
            "GeoLoop-PoreGen checkpoint/normalizer missing. Run `python -m src.train` first. "
            "The legacy checkpoint is intentionally not loaded with strict=False."
        )
    normalizer = FeatureNormalizer.load(normalizer_path)
    import torch
    strain, response = read_curve(Path(args.target_curve), cfg["data"]["curve_columns"],
                                  cfg["data"]["curve_points"])
    curve = torch.from_numpy(normalizer.transform_curve(strain, response)).unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    checkpoint_kind = ckpt.get("config", {}).get("data", {}).get("curve_kind")
    if checkpoint_kind and checkpoint_kind != cfg["data"]["curve_kind"]:
        raise RuntimeError(
            f"Checkpoint task is {checkpoint_kind!r}, but config requests "
            f"{cfg['data']['curve_kind']!r}. Retrain instead of mixing tasks."
        )
    if normalizer.state.curve_kind != cfg["data"]["curve_kind"]:
        raise RuntimeError(
            "Normalizer curve_kind does not match config. Use the normalizer saved "
            "with this stress-strain checkpoint."
        )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    n = args.samples or cfg["generation"]["candidates_per_target"]
    temperature = args.temperature if args.temperature is not None else cfg["generation"].get("temperature", 1.0)
    with torch.no_grad():
        out = model.generate(curve.to(device), n, temperature=temperature)
    pores_norm = out["pores"].cpu().numpy()
    presence = torch.sigmoid(out["presence_logits"]).cpu().numpy()
    macro = normalizer.inverse_macro(out["macro"].cpu().numpy())
    records = []
    for i in range(n):
        all_physical = normalizer.inverse_pores(pores_norm[i])
        selected = select_pore_indices(presence[i], all_physical, macro[i], cfg, cube_size)
        physical = all_physical[selected]
        records.append(make_record(
            f"candidate_{i:05d}", physical, macro[i],
            {"type": "geoloop_poregen", "model_name": cfg["model_name"],
             "release_version": cfg["release_version"],
             "temperature": float(temperature),
             "presence_selection": cfg.get("generation", {}).get("presence_selection", "threshold"),
             "target_curve": str(Path(args.target_curve).resolve()),
             "checkpoint": str(checkpoint_path.resolve())}, cube_size,
        ))
    write_jsonl(records, output)
    print(f"Wrote {len(records)} candidates: {output}")


if __name__ == "__main__":
    main()
