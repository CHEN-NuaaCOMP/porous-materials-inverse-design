"""Response-guided latent candidate generation using the frozen validated PredictNet."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from src.data.dataset import load_config, predictnet_target_label, read_curve
from src.generate_candidates import build_model, make_record, select_pore_indices
from src.models.predictnet_regularizer import FrozenPredictNetRegularizer
from src.utils.io_utils import write_jsonl
from src.utils.normalization import FeatureNormalizer


CODE_ROOT = Path(__file__).resolve().parents[1]


def resolve_code_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else CODE_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--target-curve", required=True)
    parser.add_argument("--target-porosity", type=float)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=16, help="number of optimized latent restarts")
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--response-weight", type=float, default=1.0)
    parser.add_argument("--latent-weight", type=float, default=0.015)
    parser.add_argument("--porosity-weight", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg, project_root = load_config(args.config)
    checkpoint_path = resolve_code_path(cfg["generation"]["checkpoint"])
    normalizer_path = resolve_code_path(cfg["generation"]["normalizer"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    if not normalizer_path.exists():
        raise FileNotFoundError(normalizer_path)

    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    normalizer = FeatureNormalizer.load(normalizer_path)
    model = build_model(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if not hasattr(model, "encode_prior") or not hasattr(model, "decode"):
        raise RuntimeError("response_guided_generate currently requires slot_query_v2-style encode_prior/decode APIs")

    regularizer = FrozenPredictNetRegularizer(cfg, project_root, normalizer.state, device).to(device)
    strain, response = read_curve(
        Path(args.target_curve), cfg["data"]["curve_columns"], cfg["data"]["curve_points"]
    )
    curve = torch.from_numpy(normalizer.transform_curve(strain, response)).unsqueeze(0).to(device)
    predictnet_points = int(cfg.get("screening", {}).get("porenet_curve_points", 50))
    _, response_predictnet = read_curve(
        Path(args.target_curve),
        cfg["data"]["curve_columns"],
        predictnet_points,
    )
    target_label = torch.from_numpy(
        predictnet_target_label(response_predictnet, cfg, project_root)
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        curve_features = model.curve_encoder(curve)
        prior_mu, prior_logvar = model.encode_prior(curve_features)
        prior_std = torch.exp(0.5 * prior_logvar.clamp(-12, 8)).clamp_min(1e-4)
        z0 = prior_mu.repeat(args.samples, 1) + float(args.temperature) * torch.randn(
            args.samples, prior_mu.shape[-1], device=device
        ) * prior_std.repeat(args.samples, 1)
    z = torch.nn.Parameter(z0)
    optimizer = torch.optim.Adam([z], lr=float(args.learning_rate))
    curve_features = curve_features.repeat(args.samples, 1)
    target_label = target_label.repeat(args.samples, 1)

    loss_history: list[dict[str, float]] = []
    for step in range(int(args.iterations)):
        outputs = model.decode(curve_features, z)
        predicted = regularizer(outputs)
        response_loss = F.smooth_l1_loss(predicted, target_label)
        latent_loss = (((z - prior_mu) / prior_std) ** 2).mean()
        porosity_loss = z.sum() * 0.0
        if args.target_porosity is not None:
            macro_physical = regularizer.inverse_macro(outputs["macro"])
            target_porosity = macro_physical.new_full(
                macro_physical[:, 4].shape, float(args.target_porosity)
            )
            porosity_loss = F.smooth_l1_loss(macro_physical[:, 4], target_porosity)
        total = (
            float(args.response_weight) * response_loss
            + float(args.latent_weight) * latent_loss
            + float(args.porosity_weight) * porosity_loss
        )
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_([z], 1.0)
        optimizer.step()
        if step == 0 or step == int(args.iterations) - 1 or (step + 1) % 25 == 0:
            loss_history.append({
                "step": step + 1,
                "total": float(total.detach().cpu()),
                "response": float(response_loss.detach().cpu()),
                "latent": float(latent_loss.detach().cpu()),
                "porosity": float(porosity_loss.detach().cpu()),
            })

    with torch.no_grad():
        outputs = model.decode(curve_features, z)
        final_prediction = regularizer(outputs)
        per_candidate_response = F.smooth_l1_loss(
            final_prediction, target_label, reduction="none"
        ).mean(dim=1).detach().cpu().numpy()
        pores_norm = outputs["pores"].detach().cpu().numpy()
        presence = torch.sigmoid(outputs["presence_logits"]).detach().cpu().numpy()
        macros = normalizer.inverse_macro(outputs["macro"].detach().cpu().numpy())

    records = []
    cube_size = float(cfg["data"]["cube_size_mm"])
    for index in np.argsort(per_candidate_response):
        all_physical = normalizer.inverse_pores(pores_norm[index])
        selected = select_pore_indices(presence[index], all_physical, macros[index], cfg, cube_size)
        physical = all_physical[selected]
        records.append(make_record(
            f"rg{len(records):05d}",
            physical,
            macros[index],
            {
                "type": "response_guided_latent_search",
                "model_name": cfg["model_name"],
                "release_version": cfg["release_version"],
                "target_curve": str(Path(args.target_curve).resolve()),
                "checkpoint": str(checkpoint_path.resolve()),
                "response_smooth_l1": float(per_candidate_response[index]),
                "iterations": int(args.iterations),
                "temperature": float(args.temperature),
            },
            cube_size,
        ))

    output = Path(args.output)
    if output.suffix.lower() != ".jsonl":
        output = output / "candidates.jsonl"
    write_jsonl(records, output)
    print(f"Wrote {len(records)} response-guided candidates: {output}")
    for item in loss_history:
        print(
            "step={step} total={total:.6f} response={response:.6f} "
            "latent={latent:.6f} porosity={porosity:.6f}".format(**item)
        )


if __name__ == "__main__":
    main()
