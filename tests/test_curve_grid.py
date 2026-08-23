"""Regression tests for the manuscript's common 0--0.6, 50-node grid."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import read_curve


def main() -> None:
    valid_frame = pd.DataFrame(
        {
            "Strain": [0.6, 0.3, 0.0, 0.3],
            "Stress": [12.0, 5.0, 0.0, 6.0],
        }
    )
    with patch("src.data.dataset.pd.read_csv", return_value=valid_frame):
        grid, response = read_curve(Path("valid.csv"), ["Strain", "Stress"], 50, 0.0, 0.6)
    assert len(grid) == len(response) == 50
    assert np.isclose(grid[0], 0.0)
    assert np.isclose(grid[-1], 0.6)
    assert np.allclose(grid, np.linspace(0.0, 0.6, 50))

    short_frame = pd.DataFrame(
        {
            "Strain": [0.0, 0.2, 0.59],
            "Stress": [0.0, 2.0, 11.0],
        }
    )
    try:
        with patch("src.data.dataset.pd.read_csv", return_value=short_frame):
            read_curve(Path("short.csv"), ["Strain", "Stress"], 50, 0.0, 0.6)
    except ValueError as error:
        assert "requires [0, 0.6]" in str(error)
    else:
        raise AssertionError("A curve that does not cover 0--0.6 must be rejected")

    print("PASS: common 0--0.6 strain grid and coverage validation")


if __name__ == "__main__":
    main()
