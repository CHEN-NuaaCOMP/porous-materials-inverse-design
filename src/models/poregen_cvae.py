"""Mask-aware conditional VAE for variable-size aluminum-foam pore clouds."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class CurveEncoder(nn.Module):
    def __init__(self, input_dim: int = 2, hidden: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 64, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, hidden, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
        )
        self.gru = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        x = self.conv(curve.transpose(1, 2)).transpose(1, 2)
        _, h = self.gru(x)
        return torch.cat((h[-2], h[-1]), dim=-1)


class MaskedSetEncoder(nn.Module):
    def __init__(self, feature_dim: int = 10, hidden: int = 192):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )

    def forward(self, pores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.point_mlp(pores)
        m = mask.unsqueeze(-1).to(h.dtype)
        mean = (h * m).sum(1) / m.sum(1).clamp_min(1.0)
        masked = h.masked_fill(~mask.unsqueeze(-1), torch.finfo(h.dtype).min)
        max_pool = masked.max(1).values
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
        return torch.cat((mean, max_pool), dim=-1)


class GeoLoopPoreGen(nn.Module):
    """GeoLoop-PoreGen: a new closed-loop, curve-conditioned pore generator.

    The output capacity is fixed for batching, but the number of physical pores
    is variable and represented by ``presence_logits``. Padding rows therefore
    never participate in pore reconstruction loss.
    """

    def __init__(
        self,
        max_pores: int = 1200,
        pore_feature_dim: int = 10,
        macro_dim: int = 5,
        curve_hidden: int = 128,
        set_hidden: int = 192,
        latent_dim: int = 128,
        decoder_hidden: int = 512,
    ):
        super().__init__()
        self.max_pores = max_pores
        self.pore_feature_dim = pore_feature_dim
        self.latent_dim = latent_dim
        self.curve_encoder = CurveEncoder(2, curve_hidden)
        curve_dim = 2 * curve_hidden
        self.set_encoder = MaskedSetEncoder(pore_feature_dim, set_hidden)
        set_dim = 2 * set_hidden
        self.macro_encoder = nn.Sequential(nn.Linear(macro_dim, 64), nn.ReLU())

        self.posterior = nn.Sequential(
            nn.Linear(curve_dim + set_dim + 64, decoder_hidden), nn.ReLU()
        )
        self.posterior_mu = nn.Linear(decoder_hidden, latent_dim)
        self.posterior_logvar = nn.Linear(decoder_hidden, latent_dim)
        self.prior = nn.Sequential(nn.Linear(curve_dim, 256), nn.ReLU())
        self.prior_mu = nn.Linear(256, latent_dim)
        self.prior_logvar = nn.Linear(256, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(curve_dim + latent_dim, decoder_hidden),
            nn.LayerNorm(decoder_hidden), nn.ReLU(),
            nn.Linear(decoder_hidden, decoder_hidden), nn.ReLU(),
        )
        self.macro_head = nn.Sequential(nn.Linear(decoder_hidden, macro_dim), nn.Sigmoid())
        self.pore_head = nn.Linear(decoder_hidden, max_pores * pore_feature_dim)
        self.presence_head = nn.Linear(decoder_hidden, max_pores)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar.clamp(-20, 10))

    def encode_prior(self, curve_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.prior(curve_features)
        return self.prior_mu(h), self.prior_logvar(h)

    def decode(self, curve_features: torch.Tensor, z: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.decoder(torch.cat((curve_features, z), dim=-1))
        pores = torch.sigmoid(self.pore_head(h)).view(
            -1, self.max_pores, self.pore_feature_dim
        )
        return {
            "macro": self.macro_head(h),
            "pores": pores,
            "presence_logits": self.presence_head(h),
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
        std = torch.exp(0.5 * logvar.clamp(-20, 10))
        z = mu + float(temperature) * torch.randn_like(mu) * std
        return self.decode(c, z)


# Compatibility alias for early local scripts. New publications and checkpoints
# must use the public model name ``GeoLoopPoreGen`` / ``GeoLoop-PoreGen``.
PoreGenCVAE = GeoLoopPoreGen


def _kl_between_diagonal_gaussians(
    q_mu: torch.Tensor, q_logvar: torch.Tensor, p_mu: torch.Tensor, p_logvar: torch.Tensor
) -> torch.Tensor:
    q_var, p_var = q_logvar.exp(), p_logvar.exp()
    kl = 0.5 * (p_logvar - q_logvar + (q_var + (q_mu - p_mu).pow(2)) / p_var - 1)
    return kl.mean()


def poregen_loss(
    outputs: dict[str, torch.Tensor],
    target_pores: torch.Tensor,
    target_mask: torch.Tensor,
    target_macro: torch.Tensor,
    weights: dict[str, float],
    kl_scale: float = 1.0,
    geometry_context: dict[str, float | int] | None = None,
) -> dict[str, torch.Tensor]:
    # A pore cloud is an unordered set. Direct slot-wise regression creates a
    # conditional-average cloud even after deterministic sorting. The bounded
    # symmetric set distance below removes that false point correspondence.
    max_points = int((geometry_context or {}).get("set_loss_points", 256))
    feature_weights = outputs["pores"].new_tensor(
        [1.0, 1.0, 1.0, 0.45, 0.20, 0.45, 0.10, 0.10, 0.10, 0.10]
    )
    set_terms, moment_terms = [], []
    for batch_index in range(target_pores.shape[0]):
        count = int(target_mask[batch_index].sum().item())
        if count == 0:
            continue
        predicted = outputs["pores"][batch_index, :count]
        target = target_pores[batch_index, target_mask[batch_index]]
        if count > max_points:
            indices = torch.linspace(
                0, count - 1, max_points, device=predicted.device
            ).long()
            predicted, target = predicted[indices], target[indices]
        distances = torch.cdist(
            predicted * feature_weights, target * feature_weights, p=2
        )
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
    zero = outputs["pores"].sum() * 0.0
    pore = torch.stack(set_terms).mean() if set_terms else zero
    spatial = torch.stack(moment_terms).mean() if moment_terms else zero
    marginal = zero
    presence = F.binary_cross_entropy_with_logits(
        outputs["presence_logits"], target_mask.to(target_pores.dtype)
    )
    macro = F.smooth_l1_loss(outputs["macro"], target_macro)
    porosity = zero
    if geometry_context:
        volume_log_min = float(geometry_context["volume_log_min"])
        volume_log_max = float(geometry_context["volume_log_max"])
        cube_size = float(geometry_context["cube_size_mm"])
        macro_min = float(geometry_context["macro_porosity_min"])
        macro_max = float(geometry_context["macro_porosity_max"])
        log_volume = (
            outputs["pores"][..., 5] * (volume_log_max - volume_log_min)
            + volume_log_min
        )
        physical_volume = torch.expm1(log_volume).clamp_min(0.0)
        predicted_porosity = (
            physical_volume * target_mask.to(physical_volume.dtype)
        ).sum(1) / cube_size ** 3
        target_porosity = target_macro[:, 4] * (macro_max - macro_min) + macro_min
        porosity = F.smooth_l1_loss(predicted_porosity, target_porosity)
    kl = _kl_between_diagonal_gaussians(
        outputs["posterior_mu"], outputs["posterior_logvar"],
        outputs["prior_mu"], outputs["prior_logvar"],
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
