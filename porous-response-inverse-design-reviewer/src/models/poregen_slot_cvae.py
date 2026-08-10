"""Slot-query conditional VAE for variable-count aluminum-foam pore clouds.

This module is a non-destructive V2 architecture for manuscript-scale runs.  The
original ``poregen_cvae.py`` is kept for checkpoint compatibility.  The key
change is the decoder: instead of linearly expanding one global hidden vector
into ``max_pores * feature_dim`` outputs, each pore slot has a learnable query
embedding that is conditioned by the stress-strain curve and latent variable.
This makes the decoder closer to a set generator and usually gives smoother
presence/count convergence.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .poregen_cvae import MaskedSetEncoder


class ResidualMLP(nn.Module):
    def __init__(self, dim: int, hidden: int | None = None, dropout: float = 0.05):
        super().__init__()
        hidden = hidden or dim
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class CurveEncoderV2(nn.Module):
    """Shape-preserving curve encoder for 50-point stress-strain sequences."""

    def __init__(self, input_dim: int = 2, hidden: int = 160, dropout: float = 0.05):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(64, hidden, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.gru = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.out = nn.Sequential(
            nn.LayerNorm(2 * hidden),
            nn.Dropout(dropout),
        )

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        x = self.conv(curve.transpose(1, 2)).transpose(1, 2)
        _, h = self.gru(x)
        return self.out(torch.cat((h[-2], h[-1]), dim=-1))


class GeoLoopPoreGenV2(nn.Module):
    """GeoLoop-PoreGen V2 with slot-query decoding.

    Parameters intentionally mirror the V1 model where possible.  ``slot_dim``
    controls the per-pore query representation and can be reduced for memory
    constrained runs.
    """

    def __init__(
        self,
        max_pores: int = 1200,
        pore_feature_dim: int = 10,
        macro_dim: int = 5,
        curve_hidden: int = 160,
        set_hidden: int = 192,
        latent_dim: int = 128,
        decoder_hidden: int = 512,
        slot_dim: int = 192,
        slot_attention_layers: int = 0,
        slot_attention_heads: int = 4,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.max_pores = max_pores
        self.pore_feature_dim = pore_feature_dim
        self.latent_dim = latent_dim
        self.curve_encoder = CurveEncoderV2(2, curve_hidden, dropout)
        curve_dim = 2 * curve_hidden
        self.set_encoder = MaskedSetEncoder(pore_feature_dim, set_hidden)
        set_dim = 2 * set_hidden
        self.macro_encoder = nn.Sequential(
            nn.Linear(macro_dim, 96), nn.LayerNorm(96), nn.GELU()
        )

        posterior_dim = curve_dim + set_dim + 96
        self.posterior = nn.Sequential(
            nn.Linear(posterior_dim, decoder_hidden),
            nn.LayerNorm(decoder_hidden),
            nn.GELU(),
            ResidualMLP(decoder_hidden, dropout=dropout),
        )
        self.posterior_mu = nn.Linear(decoder_hidden, latent_dim)
        self.posterior_logvar = nn.Linear(decoder_hidden, latent_dim)

        self.prior = nn.Sequential(
            nn.Linear(curve_dim, decoder_hidden // 2),
            nn.LayerNorm(decoder_hidden // 2),
            nn.GELU(),
            ResidualMLP(decoder_hidden // 2, dropout=dropout),
        )
        self.prior_mu = nn.Linear(decoder_hidden // 2, latent_dim)
        self.prior_logvar = nn.Linear(decoder_hidden // 2, latent_dim)

        self.context = nn.Sequential(
            nn.Linear(curve_dim + latent_dim, decoder_hidden),
            nn.LayerNorm(decoder_hidden),
            nn.GELU(),
            ResidualMLP(decoder_hidden, dropout=dropout),
        )
        self.context_to_slot = nn.Linear(decoder_hidden, slot_dim)
        self.slot_embedding = nn.Parameter(torch.randn(max_pores, slot_dim) * 0.02)
        if slot_attention_layers > 0:
            if slot_dim % slot_attention_heads != 0:
                raise ValueError("slot_dim must be divisible by slot_attention_heads")
            attention_layer = nn.TransformerEncoderLayer(
                d_model=slot_dim,
                nhead=slot_attention_heads,
                dim_feedforward=slot_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.slot_interaction = nn.TransformerEncoder(
                attention_layer, num_layers=slot_attention_layers
            )
        else:
            self.slot_interaction = nn.Identity()
        self.slot_decoder = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, slot_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(slot_dim, slot_dim),
            nn.GELU(),
        )
        self.pore_head = nn.Linear(slot_dim, pore_feature_dim)
        self.presence_head = nn.Linear(slot_dim, 1)
        self.macro_head = nn.Sequential(
            nn.Linear(decoder_hidden, decoder_hidden // 2),
            nn.GELU(),
            nn.Linear(decoder_hidden // 2, macro_dim),
            nn.Sigmoid(),
        )

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        logvar = logvar.clamp(-12, 8)
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def encode_prior(self, curve_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.prior(curve_features)
        return self.prior_mu(h), self.prior_logvar(h)

    def decode(self, curve_features: torch.Tensor, z: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.context(torch.cat((curve_features, z), dim=-1))
        slot_context = self.context_to_slot(h).unsqueeze(1)
        slot_state = self.slot_embedding.unsqueeze(0) + slot_context
        slot_state = self.slot_interaction(slot_state)
        slot_state = self.slot_decoder(slot_state)
        return {
            "macro": self.macro_head(h),
            "pores": torch.sigmoid(self.pore_head(slot_state)),
            "presence_logits": self.presence_head(slot_state).squeeze(-1),
        }

    def forward(
        self,
        curve: torch.Tensor,
        pores: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        macro: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        c = self.curve_encoder(curve)
        prior_mu, prior_logvar = self.encode_prior(c)
        if pores is not None and mask is not None and macro is not None:
            s = self.set_encoder(pores, mask)
            m = self.macro_encoder(macro)
            q = self.posterior(torch.cat((c, s, m), dim=-1))
            mu, logvar = self.posterior_mu(q), self.posterior_logvar(q)
        else:
            mu, logvar = prior_mu, prior_logvar
        out = self.decode(c, self.reparameterize(mu, logvar))
        out.update({
            "posterior_mu": mu,
            "posterior_logvar": logvar,
            "prior_mu": prior_mu,
            "prior_logvar": prior_logvar,
        })
        return out

    @torch.no_grad()
    def generate(
        self, curve: torch.Tensor, n_samples: int = 1, temperature: float = 1.0
    ) -> dict[str, torch.Tensor]:
        if curve.shape[0] == 1 and n_samples > 1:
            curve = curve.repeat(n_samples, 1, 1)
        c = self.curve_encoder(curve)
        mu, logvar = self.encode_prior(c)
        std = torch.exp(0.5 * logvar.clamp(-12, 8))
        z = mu + float(temperature) * torch.randn_like(mu) * std
        return self.decode(c, z)


def _kl_between_diagonal_gaussians_free_bits(
    q_mu: torch.Tensor,
    q_logvar: torch.Tensor,
    p_mu: torch.Tensor,
    p_logvar: torch.Tensor,
    free_bits: float = 0.02,
) -> torch.Tensor:
    q_logvar = q_logvar.clamp(-12, 8)
    p_logvar = p_logvar.clamp(-12, 8)
    q_var, p_var = q_logvar.exp(), p_logvar.exp()
    kl_dim = 0.5 * (
        p_logvar - q_logvar + (q_var + (q_mu - p_mu).pow(2)) / p_var - 1
    )
    if free_bits > 0:
        kl_dim = torch.clamp(kl_dim, min=float(free_bits))
    return kl_dim.mean()


def _marginal_distribution_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    feature_weights: torch.Tensor,
) -> torch.Tensor:
    """Match low-order marginal statistics for one unordered pore set."""

    if predicted.numel() == 0 or target.numel() == 0:
        return predicted.sum() * 0.0
    predicted = predicted * feature_weights
    target = target * feature_weights
    loss = F.smooth_l1_loss(predicted.mean(0), target.mean(0))
    loss = loss + F.smooth_l1_loss(
        predicted.std(0, unbiased=False),
        target.std(0, unbiased=False),
    )
    if predicted.shape[0] > 1 and target.shape[0] > 1:
        q = predicted.new_tensor([0.1, 0.5, 0.9])
        loss = loss + 0.5 * F.smooth_l1_loss(
            torch.quantile(predicted, q, dim=0),
            torch.quantile(target, q, dim=0),
        )
    return loss


def poregen_loss_v2(
    outputs: dict[str, torch.Tensor],
    target_pores: torch.Tensor,
    target_mask: torch.Tensor,
    target_macro: torch.Tensor,
    weights: dict[str, float],
    kl_scale: float = 1.0,
    geometry_context: dict[str, object] | None = None,
) -> dict[str, torch.Tensor]:
    """Set loss using top-k presence-selected slots plus stabilized KL."""

    max_points = int((geometry_context or {}).get("set_loss_points", 256))
    free_bits = float((geometry_context or {}).get("kl_free_bits", 0.02))
    set_selection = str((geometry_context or {}).get("set_selection", "topk_presence"))
    feature_weights = outputs["pores"].new_tensor(
        [1.0, 1.0, 1.0, 0.45, 0.20, 0.45, 0.10, 0.10, 0.10, 0.10]
    )
    marginal_feature_weights_value = (geometry_context or {}).get("marginal_feature_weights")
    if marginal_feature_weights_value is None:
        marginal_feature_weights = feature_weights
    else:
        marginal_feature_weights = outputs["pores"].new_tensor(marginal_feature_weights_value)
        if marginal_feature_weights.numel() != outputs["pores"].shape[-1]:
            raise ValueError(
                "training.marginal_feature_weights must match pore_feature_dim "
                f"({outputs['pores'].shape[-1]} values expected)"
            )
    set_terms, moment_terms, marginal_terms = [], [], []
    presence_prob = torch.sigmoid(outputs["presence_logits"])
    for batch_index in range(target_pores.shape[0]):
        count = int(target_mask[batch_index].sum().item())
        if count == 0:
            continue
        scores = presence_prob[batch_index]
        if set_selection == "first_count":
            indices = torch.arange(min(count, scores.numel()), device=scores.device)
        elif set_selection == "topk_presence":
            indices = scores.topk(k=min(count, scores.numel()), largest=True).indices
        else:
            raise ValueError(f"Unknown set_selection: {set_selection}")
        predicted = outputs["pores"][batch_index, indices]
        target = target_pores[batch_index, target_mask[batch_index]]
        if count > max_points:
            indices = torch.linspace(
                0, count - 1, max_points, device=predicted.device
            ).long()
            predicted, target = predicted[indices], target[indices]
        distances = torch.cdist(predicted * feature_weights, target * feature_weights, p=2)
        set_terms.append(
            0.5 * (distances.min(1).values.mean() + distances.min(0).values.mean())
        )
        moment_terms.append(
            F.smooth_l1_loss(predicted[:, :3].mean(0), target[:, :3].mean(0))
            + F.smooth_l1_loss(
                predicted[:, :3].std(0, unbiased=False),
                target[:, :3].std(0, unbiased=False),
            )
        )
        marginal_terms.append(
            _marginal_distribution_loss(predicted, target, marginal_feature_weights)
        )

    zero = outputs["pores"].sum() * 0.0
    pore = torch.stack(set_terms).mean() if set_terms else zero
    spatial = torch.stack(moment_terms).mean() if moment_terms else zero
    marginal = torch.stack(marginal_terms).mean() if marginal_terms else zero

    target_presence = target_mask.to(target_pores.dtype)
    active = target_presence.sum().clamp_min(1.0)
    inactive = (target_presence.numel() - active).clamp_min(1.0)
    pos_weight = (inactive / active).clamp(1.0, 12.0)
    presence = F.binary_cross_entropy_with_logits(
        outputs["presence_logits"], target_presence, pos_weight=pos_weight
    )
    macro = F.smooth_l1_loss(outputs["macro"], target_macro)
    porosity = zero
    if geometry_context:
        volume_log_min = float(geometry_context["volume_log_min"])
        volume_log_max = float(geometry_context["volume_log_max"])
        cube_size = float(geometry_context["cube_size_mm"])
        macro_min = float(geometry_context["macro_porosity_min"])
        macro_max = float(geometry_context["macro_porosity_max"])
        log_volume = outputs["pores"][..., 5] * (volume_log_max - volume_log_min) + volume_log_min
        physical_volume = torch.expm1(log_volume).clamp_min(0.0)
        predicted_porosity = (physical_volume * presence_prob).sum(1) / cube_size ** 3
        target_porosity = target_macro[:, 4] * (macro_max - macro_min) + macro_min
        porosity = F.smooth_l1_loss(predicted_porosity, target_porosity)

    kl = _kl_between_diagonal_gaussians_free_bits(
        outputs["posterior_mu"], outputs["posterior_logvar"],
        outputs["prior_mu"], outputs["prior_logvar"], free_bits=free_bits,
    )
    total = (
        weights["pore"] * pore
        + weights["presence"] * presence
        + weights["macro"] * macro
        + weights.get("porosity_consistency", 0.0) * porosity
        + weights.get("spatial_moments", 0.0) * spatial
        + weights.get("marginal_distribution", 0.0) * marginal
        + weights["kl"] * kl_scale * kl
    )
    return {
        "total": total, "pore": pore, "presence": presence, "macro": macro,
        "porosity": porosity, "spatial": spatial, "marginal": marginal, "kl": kl,
    }
