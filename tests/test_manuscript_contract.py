"""Verify the manuscript-facing architecture and data contract without raw data."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    config = yaml.safe_load((ROOT / "configs" / "manuscript.yaml").read_text(encoding="utf-8"))

    assert config["data"]["curve_points"] == 50
    assert config["data"]["strain_min"] == 0.0
    assert config["data"]["strain_max"] == 0.6
    assert config["screening"]["porenet_curve_points"] == 50
    assert config["training"]["selection_mode"] == "fixed_final"
    assert config["model"]["architecture"] == "slot_query_v2"
    assert config["model"]["latent_dim"] == 128
    assert config["model"]["curve_hidden"] == 160
    assert config["model"]["set_hidden"] == 192
    assert config["model"]["slot_dim"] == 192

    expected_weights = {
        "pore": 1.00,
        "presence": 0.35,
        "macro": 0.25,
        "porosity_consistency": 1.25,
        "spatial_moments": 0.75,
        "marginal_distribution": 0.20,
        "response_predictnet": 0.75,
        "kl": 0.01,
    }
    assert config["training"]["loss_weights"] == expected_weights

    generator_split_path = ROOT / config["data"]["split_file"]
    generator_split = json.loads(generator_split_path.read_text(encoding="utf-8"))
    assert {key: len(generator_split[key]) for key in ("train", "val", "test")} == {
        "train": 675,
        "val": 84,
        "test": 85,
    }
    generator_names = (
        generator_split["train"] + generator_split["val"] + generator_split["test"]
    )
    assert len(generator_names) == len(set(generator_names)) == 844

    forward_split_path = ROOT / config["forward_predictor"]["split_file"]
    forward_split = json.loads(forward_split_path.read_text(encoding="utf-8"))
    assert {key: len(forward_split[key]) for key in ("train", "val", "test")} == {
        "train": 727,
        "val": 0,
        "test": 117,
    }
    forward_names = forward_split["train"] + forward_split["val"] + forward_split["test"]
    assert len(forward_names) == len(set(forward_names)) == 844
    assert set(generator_names) == set(forward_names)

    from external_predictor.model import PredictNet

    predictor = PredictNet(num_classes=50).eval()
    state = torch.load(
        ROOT / "checkpoints" / "predictnet_manuscript_8_16" / "best_model.pth",
        map_location="cpu",
    )
    predictor.load_state_dict(state, strict=True)
    with torch.no_grad():
        predicted_curve = predictor(torch.randn(2, 3), torch.randn(2, 32, 9))
    assert tuple(predicted_curve.shape) == (2, 50)

    from src.models.poregen_slot_cvae import GeoLoopPoreGenV2

    generator = GeoLoopPoreGenV2(
        max_pores=16,
        pore_feature_dim=config["model"]["pore_feature_dim"],
        macro_dim=config["model"]["macro_dim"],
        curve_hidden=config["model"]["curve_hidden"],
        set_hidden=config["model"]["set_hidden"],
        latent_dim=config["model"]["latent_dim"],
        decoder_hidden=config["model"]["decoder_hidden"],
        slot_dim=config["model"]["slot_dim"],
        slot_attention_layers=config["model"]["slot_attention_layers"],
        slot_attention_heads=config["model"]["slot_attention_heads"],
        dropout=config["model"]["dropout"],
    ).eval()
    with torch.no_grad():
        generated = generator.generate(torch.randn(3, 50, 2), temperature=0.8)
    assert tuple(generated["pores"].shape) == (3, 16, 10)
    assert tuple(generated["presence_logits"].shape) == (3, 16)
    assert tuple(generated["macro"].shape) == (3, 5)
    assert generator.latent_dim == 128

    forbidden_phrases = ("fixed pore count", "exact pore count", "constant pore count")
    for suffix in ("*.py", "*.md", "*.yaml"):
        for path in ROOT.rglob(suffix):
            if path.resolve() == Path(__file__).resolve():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            assert not any(phrase in text for phrase in forbidden_phrases), path

    print(
        "PASS: manuscript contract, task-specific splits, fixed curve grid, "
        "forward checkpoint and variable-cardinality decoder"
    )


if __name__ == "__main__":
    main()
