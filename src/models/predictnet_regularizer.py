"""Frozen validated PredictNet response regularizer for generator fine-tuning."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

from src.data.dataset import GLOBAL_PREDICTNET_LABEL_MODE, LEGACY_PREDICTNET_LABEL_MODE


def resolve_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (project_root / path).resolve()


class FrozenPredictNetRegularizer(nn.Module):
    """Differentiable adapter from GeoLoop-PoreGen outputs to frozen PredictNet."""

    def __init__(self, cfg: dict, project_root: Path, normalizer_state, device: torch.device):
        super().__init__()
        rules = cfg["screening"]
        checkpoint_path = resolve_project_path(project_root, rules["porenet_checkpoint"])
        scaler_path = resolve_project_path(project_root, rules["porenet_scalers"])
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        if not scaler_path.exists():
            raise FileNotFoundError(scaler_path)

        predictor_code_dir = resolve_project_path(
            project_root,
            rules.get("porenet_code_dir", checkpoint_path.parents[2]),
        )
        if not (predictor_code_dir / "model.py").exists():
            raise FileNotFoundError(
                f"PredictNet code directory has no model.py: {predictor_code_dir}"
            )
        sys.path.insert(0, str(predictor_code_dir))
        try:
            from joblib import load
            from model import PredictNet
        finally:
            if sys.path[0] == str(predictor_code_dir):
                sys.path.pop(0)

        self.curve_points = int(rules.get("porenet_curve_points", 50))
        self.predictor = PredictNet(
            feature_channels=9,
            global_channels=3,
            num_classes=self.curve_points,
        ).to(device)
        self.predictor.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=True)
        self.predictor.eval()
        for parameter in self.predictor.parameters():
            parameter.requires_grad_(False)

        scalers = load(scaler_path)
        self.label_mode = str(
            rules.get("porenet_label_mode", scalers.get("label_mode", LEGACY_PREDICTNET_LABEL_MODE))
        )
        if self.label_mode == GLOBAL_PREDICTNET_LABEL_MODE and "scaler_label" not in scalers:
            raise RuntimeError("global_log1p_standard PredictNet regularizer requires scaler_label")
        if self.label_mode == GLOBAL_PREDICTNET_LABEL_MODE:
            scaler_points = int(getattr(scalers["scaler_label"], "n_features_in_", self.curve_points))
            if scaler_points != self.curve_points:
                raise RuntimeError(
                    f"PredictNet scaler has {scaler_points} response nodes, "
                    f"but screening.porenet_curve_points={self.curve_points}"
                )
        if self.label_mode not in {LEGACY_PREDICTNET_LABEL_MODE, GLOBAL_PREDICTNET_LABEL_MODE}:
            raise ValueError(f"Unknown PredictNet label mode: {self.label_mode}")
        coarse_scaler = scalers["scaler_coarse"]
        self.register_buffer(
            "coarse_scale",
            torch.as_tensor(coarse_scaler.scale_, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "coarse_offset",
            torch.as_tensor(coarse_scaler.min_, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "pore_min",
            torch.as_tensor(normalizer_state.pore_min, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "pore_max",
            torch.as_tensor(normalizer_state.pore_max, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "macro_min",
            torch.as_tensor(normalizer_state.macro_min, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "macro_max",
            torch.as_tensor(normalizer_state.macro_max, dtype=torch.float32, device=device),
        )
        self.cube_size = float(cfg["data"]["cube_size_mm"])
        self.npoints = int(cfg["training"].get("response_predictnet_points", 1024))
        self.presence_gate_power = float(cfg["training"].get("response_predictnet_presence_power", 0.0))
        self.selection_mode = str(cfg["training"].get("response_predictnet_selection", "topk_presence"))
        self.min_pores = int(cfg.get("screening", {}).get("min_pores", 1))
        self.max_pores = int(cfg.get("screening", {}).get("max_pores", self.npoints))
        self.presence_count_scale = float(cfg.get("generation", {}).get("presence_count_scale", 1.0))

    def inverse_pores(self, normalized: torch.Tensor) -> torch.Tensor:
        pre = normalized * (self.pore_max - self.pore_min) + self.pore_min
        return torch.cat(
            (
                pre[..., 0:3].clamp(0.0, self.cube_size),
                torch.expm1(pre[..., 3:6]).clamp_min(0.0),
                pre[..., 6:7].clamp(0.0, 1.0),
                pre[..., 7:8].clamp(0.0, 0.9),
                pre[..., 8:9].clamp(-1.8, 1.8),
                pre[..., 9:10].clamp(0.0, 1.0),
            ),
            dim=-1,
        )

    def inverse_macro(self, normalized: torch.Tensor) -> torch.Tensor:
        return normalized * (self.macro_max - self.macro_min) + self.macro_min

    def _select_topk_presence(
        self,
        physical: torch.Tensor,
        presence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, slots, _ = physical.shape
        k = min(self.npoints, slots)
        indices = presence.topk(k=k, dim=1, largest=True).indices
        gather_index = indices.unsqueeze(-1).expand(batch, k, physical.shape[-1])
        selected = torch.gather(physical, dim=1, index=gather_index)
        selected_presence = torch.gather(presence, dim=1, index=indices).unsqueeze(-1)
        return selected, selected_presence

    def _select_generation_topk_porosity(
        self,
        physical: torch.Tensor,
        presence: torch.Tensor,
        macro: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mirror generation.select_pore_indices(topk_porosity) for PredictNet loss."""

        selected_sets: list[torch.Tensor] = []
        selected_presence_sets: list[torch.Tensor] = []
        batch, slots, _ = physical.shape
        max_allowed = min(self.npoints, self.max_pores, slots)
        min_allowed = min(max(self.min_pores, 1), max_allowed)
        for batch_index in range(batch):
            scores = presence[batch_index]
            order = torch.argsort(scores, descending=True)
            expected_count = int(round(float(scores.detach().sum().cpu()) * self.presence_count_scale))
            expected_count = int(max(min_allowed, min(expected_count, max_allowed)))

            target_volume = float(macro[batch_index, 4].detach().cpu()) * self.cube_size ** 3
            ordered_volume = physical[batch_index, order, 5].detach().clamp_min(0.0)
            cumulative = torch.cumsum(ordered_volume, dim=0).cpu()
            porosity_count = int(torch.searchsorted(
                cumulative,
                torch.tensor(max(target_volume, 0.0), dtype=cumulative.dtype),
                right=False,
            ).item() + 1)
            count = int(max(expected_count, porosity_count))
            count = int(max(min_allowed, min(count, max_allowed)))
            selected_indices = order[:count]
            selected_sets.append(physical[batch_index, selected_indices])
            selected_presence_sets.append(scores[selected_indices].unsqueeze(-1))

        padded = physical.new_zeros(batch, self.npoints, physical.shape[-1])
        padded_presence = presence.new_zeros(batch, self.npoints, 1)
        for batch_index, selected in enumerate(selected_sets):
            count = min(selected.shape[0], self.npoints)
            padded[batch_index, :count] = selected[:count]
            padded_presence[batch_index, :count] = selected_presence_sets[batch_index][:count]
        return padded, padded_presence

    def forward(self, outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        pores_norm = outputs["pores"]
        presence = torch.sigmoid(outputs["presence_logits"])
        physical_all = self.inverse_pores(pores_norm)
        macro = self.inverse_macro(outputs["macro"])
        if self.selection_mode == "generation_topk_porosity":
            physical, selected_presence = self._select_generation_topk_porosity(
                physical_all, presence, macro
            )
        elif self.selection_mode == "topk_presence":
            physical, selected_presence = self._select_topk_presence(physical_all, presence)
        else:
            raise ValueError(f"Unknown response_predictnet_selection: {self.selection_mode}")

        local = torch.stack(
            [
                physical[..., 0] / self.cube_size,
                physical[..., 1] / self.cube_size,
                physical[..., 2] / self.cube_size,
                torch.log1p(physical[..., 4].clamp_min(0.0)) / torch.log(physical.new_tensor(200.0)),
                torch.log1p(physical[..., 5].clamp_min(0.0)) / torch.log(physical.new_tensor(200.0)),
                physical[..., 6],
                physical[..., 7] / 0.9,
                torch.remainder(physical[..., 8] + 3.6, 3.6) / 3.6,
                physical[..., 9],
            ],
            dim=-1,
        )
        if self.presence_gate_power > 0.0 and self.presence_gate_power != 1.0:
            selected_presence = selected_presence.clamp(0.0, 1.0).pow(self.presence_gate_power)
        if self.presence_gate_power > 0.0:
            local = local * selected_presence.clamp(0.0, 1.0)

        coarse = torch.stack((macro[:, 0], macro[:, 3], macro[:, 4]), dim=1)
        coarse = coarse * self.coarse_scale + self.coarse_offset
        return self.predictor(coarse, local)
