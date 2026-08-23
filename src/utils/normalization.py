"""Single-source normalization and exact inverse transforms."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


PORE_COLUMNS = [
    "x", "y", "z", "eq_diameter", "area", "volume",
    "anisotropy", "phi", "theta", "sphericity",
]
MACRO_COLUMNS = ["mean", "std", "max", "fractal_dimension", "volume_fraction"]


def _pretransform_pores(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64).copy()
    x[..., 3:6] = np.log1p(np.clip(x[..., 3:6], 0.0, None))
    return x


def _inverse_pretransform_pores(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64).copy()
    x[..., 3:6] = np.expm1(x[..., 3:6])
    return x


@dataclass
class NormalizerState:
    pore_min: list[float]
    pore_max: list[float]
    macro_min: list[float]
    macro_max: list[float]
    response_min: float
    response_max: float
    strain_max: float
    cube_size_mm: float
    curve_kind: str
    response_transform: str = "linear"
    response_scale: float = 1.0


class FeatureNormalizer:
    def __init__(self, state: NormalizerState):
        self.state = state

    @classmethod
    def fit(
        cls,
        pore_arrays: list[np.ndarray],
        macros: np.ndarray,
        responses: list[np.ndarray],
        strains: list[np.ndarray],
        cube_size_mm: float,
        curve_kind: str,
        response_transform: str = "linear",
        response_scale_quantile: float = 0.75,
    ) -> "FeatureNormalizer":
        pores = _pretransform_pores(np.concatenate(pore_arrays, axis=0))
        pmin, pmax = pores.min(0), pores.max(0)
        pmin[:3], pmax[:3] = 0.0, cube_size_mm
        pmin[6], pmax[6] = 0.0, 1.0
        pmin[7], pmax[7] = 0.0, max(0.9, pmax[7])
        pmin[8], pmax[8] = -1.8, 1.8
        pmin[9], pmax[9] = 0.0, 1.0
        macros = np.asarray(macros, dtype=np.float64)
        response = np.concatenate(responses)
        response_scale = 1.0
        if response_transform == "asinh":
            nonzero = np.abs(response[np.abs(response) > 1e-12])
            if len(nonzero):
                response_scale = float(np.quantile(nonzero, response_scale_quantile))
            response = np.arcsinh(response / max(response_scale, 1e-12))
        elif response_transform != "linear":
            raise ValueError(f"Unsupported response_transform: {response_transform}")
        state = NormalizerState(
            pore_min=pmin.tolist(), pore_max=pmax.tolist(),
            macro_min=macros.min(0).tolist(), macro_max=macros.max(0).tolist(),
            response_min=float(response.min()), response_max=float(response.max()),
            strain_max=float(max(np.max(s) for s in strains)),
            cube_size_mm=float(cube_size_mm), curve_kind=curve_kind,
            response_transform=response_transform,
            response_scale=response_scale,
        )
        return cls(state)

    @staticmethod
    def _scale(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        return np.clip((x - lo) / np.maximum(hi - lo, 1e-12), 0.0, 1.0)

    @staticmethod
    def _unscale(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        return x * (hi - lo) + lo

    def transform_pores(self, values: np.ndarray) -> np.ndarray:
        x = _pretransform_pores(values)
        return self._scale(x, np.array(self.state.pore_min), np.array(self.state.pore_max)).astype(np.float32)

    def inverse_pores(self, values: np.ndarray) -> np.ndarray:
        x = self._unscale(np.asarray(values), np.array(self.state.pore_min), np.array(self.state.pore_max))
        x = _inverse_pretransform_pores(x)
        x[..., :3] = np.clip(x[..., :3], 0.0, self.state.cube_size_mm)
        x[..., 3:6] = np.clip(x[..., 3:6], 0.0, None)
        return x.astype(np.float32)

    def transform_macro(self, values: np.ndarray) -> np.ndarray:
        return self._scale(values, np.array(self.state.macro_min), np.array(self.state.macro_max)).astype(np.float32)

    def inverse_macro(self, values: np.ndarray) -> np.ndarray:
        return self._unscale(values, np.array(self.state.macro_min), np.array(self.state.macro_max)).astype(np.float32)

    def transform_curve(self, strain: np.ndarray, response: np.ndarray) -> np.ndarray:
        s = np.asarray(strain) / max(self.state.strain_max, 1e-12)
        response = np.asarray(response)
        if self.state.response_transform == "asinh":
            response = np.arcsinh(response / max(self.state.response_scale, 1e-12))
        r = (response - self.state.response_min) / max(
            self.state.response_max - self.state.response_min, 1e-12
        )
        return np.stack((np.clip(s, 0, 1), np.clip(r, 0, 1)), axis=-1).astype(np.float32)

    def inverse_response(self, normalized: np.ndarray) -> np.ndarray:
        response = normalized * (self.state.response_max - self.state.response_min) + self.state.response_min
        if self.state.response_transform == "asinh":
            response = np.sinh(response) * self.state.response_scale
        return response

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self.state), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FeatureNormalizer":
        return cls(NormalizerState(**json.loads(Path(path).read_text(encoding="utf-8"))))
