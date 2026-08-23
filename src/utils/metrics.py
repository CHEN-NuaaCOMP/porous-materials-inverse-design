"""Geometry and response metrics saved for every candidate."""

from __future__ import annotations

import numpy as np


def equivalent_radius(volume: np.ndarray) -> np.ndarray:
    return np.cbrt(np.maximum(3.0 * np.asarray(volume) / (4.0 * np.pi), 0.0))


def geometry_metrics(pores: np.ndarray, cube_size_mm: float = 14.0, grid_bins: int = 5) -> dict[str, float]:
    pores = np.asarray(pores, dtype=float)
    if len(pores) == 0:
        return {"pore_count": 0, "porosity_sum": 0.0, "boundary_violation_fraction": 1.0,
                "overlap_pair_fraction": 1.0, "overlap_pair_count": 0,
                "coverage_fraction": 0.0}
    xyz, volume = pores[:, :3], pores[:, 5]
    r = equivalent_radius(volume)
    boundary = np.any((xyz - r[:, None] < 0) | (xyz + r[:, None] > cube_size_mm), axis=1)
    # Candidate pairs are limited to nearby centers; this is a conservative sphere-equivalent proxy.
    diff = xyz[:, None, :] - xyz[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    upper = np.triu(np.ones_like(dist, dtype=bool), 1)
    near = upper & (dist < (r[:, None] + r[None, :]))
    possible = max(len(pores) * (len(pores) - 1) // 2, 1)
    bins = np.clip((xyz / cube_size_mm * grid_bins).astype(int), 0, grid_bins - 1)
    occupied = np.unique(bins, axis=0).shape[0]
    return {
        "pore_count": int(len(pores)),
        "porosity_sum": float(volume.sum() / cube_size_mm ** 3),
        "boundary_violation_fraction": float(boundary.mean()),
        "overlap_pair_fraction": float(near.sum() / possible),
        "overlap_pair_count": int(near.sum()),
        "coverage_fraction": float(occupied / grid_bins ** 3),
    }


def _phase_rmse(
    target: np.ndarray,
    prediction: np.ndarray,
    strain: np.ndarray,
    start_strain: float,
    stop_strain: float,
    include_stop: bool = False,
) -> float:
    if include_stop:
        mask = (strain >= start_strain) & (strain <= stop_strain)
    else:
        mask = (strain >= start_strain) & (strain < stop_strain)
    if not np.any(mask):
        raise ValueError(
            f"No response nodes in phase [{start_strain}, {stop_strain}]"
        )
    err = prediction[mask] - target[mask]
    return float(np.sqrt(np.mean(err ** 2)))


def curve_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    strain: np.ndarray | None = None,
) -> dict[str, float]:
    target, prediction = np.asarray(target, float), np.asarray(prediction, float)
    if strain is None:
        strain = np.linspace(0.0, 0.6, len(target))
    strain = np.asarray(strain, float)
    if target.shape != prediction.shape or target.shape != strain.shape:
        raise ValueError("target, prediction, and strain must have the same one-dimensional shape")
    err = prediction - target
    nz = np.abs(target) > 1e-8
    rmse = float(np.sqrt(np.mean(err ** 2)))
    response_range = float(np.max(target) - np.min(target))
    scale = max(response_range, 1e-12)
    peak_index = int(np.argmax(target))
    terminal_abs_error = float(abs(prediction[-1] - target[-1]))
    terminal_relative_error = float(terminal_abs_error / max(abs(target[-1]), 1e-12))
    peak_abs_error = float(abs(prediction[peak_index] - target[peak_index]))
    return {
        "curve_mae": float(np.mean(np.abs(err))),
        "curve_rmse": rmse,
        "curve_nrmse": rmse / scale,
        "curve_mape_nonzero": float(np.mean(np.abs(err[nz] / target[nz]))) if np.any(nz) else float("nan"),
        "curve_elastic_rmse": _phase_rmse(target, prediction, strain, 0.0, 0.1),
        "curve_elastic_nrmse": _phase_rmse(target, prediction, strain, 0.0, 0.1) / scale,
        "curve_plateau_rmse": _phase_rmse(target, prediction, strain, 0.1, 0.4),
        "curve_plateau_nrmse": _phase_rmse(target, prediction, strain, 0.1, 0.4) / scale,
        "curve_densification_rmse": _phase_rmse(
            target, prediction, strain, 0.4, 0.6, include_stop=True
        ),
        "curve_densification_nrmse": _phase_rmse(
            target, prediction, strain, 0.4, 0.6, include_stop=True
        ) / scale,
        "curve_peak_abs_error": peak_abs_error,
        "curve_peak_relative_error": float(peak_abs_error / max(abs(target[peak_index]), 1e-12)),
        "curve_terminal_abs_error": terminal_abs_error,
        "curve_terminal_relative_error": terminal_relative_error,
    }
