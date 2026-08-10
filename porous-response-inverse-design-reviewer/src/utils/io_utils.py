"""Canonical candidate schema and non-destructive I/O."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


CANONICAL_COLUMNS = [
    "x", "y", "z", "eq_diameter", "area", "volume",
    "anisotropy", "phi", "theta", "sphericity",
]
RECONSTRUCTION_COLUMNS = ["X", "Y", "Z", "Area", "Volume", "Phi", "Theta"]


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(records: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")


def pore_array_to_records(values: np.ndarray) -> list[dict]:
    array = np.asarray(values)
    if array.size == 0:
        return []
    return [dict(zip(CANONICAL_COLUMNS, map(float, row))) for row in array.reshape(-1, len(CANONICAL_COLUMNS))]


def pore_records_to_array(records: list[dict]) -> np.ndarray:
    if not records:
        return np.empty((0, len(CANONICAL_COLUMNS)), dtype=np.float32)
    return np.asarray([[r[c] for c in CANONICAL_COLUMNS] for r in records], dtype=np.float32)


def export_reconstruction_csv(candidate: dict, path: str | Path) -> None:
    pores = pore_records_to_array(candidate["pores"])
    frame = pd.DataFrame({
        "X": pores[:, 0], "Y": pores[:, 1], "Z": pores[:, 2],
        "Area": pores[:, 4], "Volume": pores[:, 5],
        "Phi": pores[:, 7], "Theta": pores[:, 8],
    })
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
