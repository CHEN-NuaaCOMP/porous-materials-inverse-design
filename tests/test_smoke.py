from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io_utils import export_reconstruction_csv, pore_array_to_records
from src.utils.normalization import FeatureNormalizer, NormalizerState


def test_normalization_roundtrip() -> None:
    state = NormalizerState(
        pore_min=[0, 0, 0, 0, 0, 0, 0, 0, -1.8, 0],
        pore_max=[14, 14, 14, 2, 4, 3, 1, 0.9, 1.8, 1],
        macro_min=[0, 0, 0, 0, 0], macro_max=[1, 1, 1, 1, 1],
        response_min=0, response_max=10, strain_max=0.6,
        cube_size_mm=14, curve_kind="stress_strain",
    )
    norm = FeatureNormalizer(state)
    raw = np.array([[1, 2, 3, np.expm1(1), np.expm1(2), np.expm1(1.5),
                     0.8, 0.4, -0.2, 0.7]], dtype=np.float32)
    restored = norm.inverse_pores(norm.transform_pores(raw))
    np.testing.assert_allclose(restored, raw, rtol=1e-5, atol=1e-5)
    stress_state = NormalizerState(
        pore_min=state.pore_min, pore_max=state.pore_max,
        macro_min=state.macro_min, macro_max=state.macro_max,
        response_min=0.0, response_max=np.arcsinh(10.0 / 2.0), strain_max=0.6,
        cube_size_mm=14, curve_kind="stress_strain",
        response_transform="asinh", response_scale=2.0,
    )
    stress_norm = FeatureNormalizer(stress_state)
    stress = np.array([0.0, 1.0, 5.0, 10.0])
    normalized_stress = stress_norm.transform_curve(
        np.linspace(0, 0.6, len(stress)), stress
    )[:, 1]
    np.testing.assert_allclose(stress_norm.inverse_response(normalized_stress), stress,
                               rtol=1e-5, atol=1e-5)


def test_model_mask_and_shapes() -> None:
    try:
        import torch
        from src.models.poregen_cvae import GeoLoopPoreGen, poregen_loss
    except ModuleNotFoundError:
        print("SKIP model smoke test: PyTorch is not installed in this runtime")
        return
    model = GeoLoopPoreGen(max_pores=16, pore_feature_dim=10, macro_dim=5,
                        curve_hidden=16, set_hidden=16, latent_dim=8, decoder_hidden=32)
    curve = torch.rand(2, 50, 2)
    pores = torch.rand(2, 16, 10)
    mask = torch.zeros(2, 16, dtype=torch.bool); mask[:, :7] = True
    macro = torch.rand(2, 5)
    out = model(curve, pores, mask, macro)
    assert out["pores"].shape == (2, 16, 10)
    assert out["presence_logits"].shape == (2, 16)
    losses = poregen_loss(
        out, pores, mask, macro,
        {"pore": 1, "presence": 1, "macro": 1, "porosity_consistency": 1,
         "spatial_moments": 1, "kl": 0.01},
        geometry_context={"set_loss_points": 8, "volume_log_min": 0.0,
                          "volume_log_max": 2.0, "cube_size_mm": 14.0,
                          "macro_porosity_min": 0.4, "macro_porosity_max": 0.9},
    )
    assert torch.isfinite(losses["total"])
    from src.models.poregen_slot_cvae import GeoLoopPoreGenV2, poregen_loss_v2
    model_v2 = GeoLoopPoreGenV2(max_pores=16, pore_feature_dim=10, macro_dim=5,
                                curve_hidden=16, set_hidden=16, latent_dim=8,
                                decoder_hidden=64, slot_dim=24)
    out_v2 = model_v2(curve, pores, mask, macro)
    assert out_v2["pores"].shape == (2, 16, 10)
    assert out_v2["presence_logits"].shape == (2, 16)
    losses_v2 = poregen_loss_v2(
        out_v2, pores, mask, macro,
        {"pore": 1, "presence": 0.5, "macro": 1, "porosity_consistency": 1,
         "spatial_moments": 1, "kl": 0.01},
        geometry_context={"set_loss_points": 8, "volume_log_min": 0.0,
                          "volume_log_max": 2.0, "cube_size_mm": 14.0,
                          "macro_porosity_min": 0.4, "macro_porosity_max": 0.9,
                          "kl_free_bits": 0.02},
    )
    assert torch.isfinite(losses_v2["total"])


def test_reconstruction_export() -> None:
    pores = np.ones((3, 10), dtype=np.float32)
    candidate = {"candidate_id": "x", "pores": pore_array_to_records(pores)}
    output_root = Path(
        os.environ.get("POROUS_TEST_OUTPUT", str(Path.cwd() / "outputs" / "test_smoke"))
    )
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "candidate.csv"
    export_reconstruction_csv(candidate, path)
    header = path.read_text(encoding="utf-8-sig").splitlines()[0]
    assert header == "X,Y,Z,Area,Volume,Phi,Theta"


if __name__ == "__main__":
    test_normalization_roundtrip()
    test_model_mask_and_shapes()
    test_reconstruction_export()
    print("smoke tests passed")
