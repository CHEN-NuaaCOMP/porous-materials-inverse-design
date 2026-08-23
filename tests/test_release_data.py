"""Check that the released processed dataset and split manifests are aligned."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    data_root = ROOT / "data" / "full"
    global_frame = pd.read_csv(data_root / "global.csv")
    assert "name" in global_frame.columns
    global_names = {str(value) for value in global_frame["name"]}
    pore_names = {path.stem for path in (data_root / "pore_cloud").glob("*.csv")}
    curve_names = {
        path.stem for path in (data_root / "stress_strain_responses").glob("*.csv")
    }
    assert len(global_names) == len(pore_names) == len(curve_names) == 844
    assert global_names == pore_names == curve_names

    for filename in ("predictnet_727_117.json", "generator_675_84_85.json"):
        split = json.loads((ROOT / "splits" / filename).read_text(encoding="utf-8"))
        names = set(split["train"]) | set(split["val"]) | set(split["test"])
        assert names == global_names

    print("PASS: 844 aligned processed samples and both fixed split manifests")


if __name__ == "__main__":
    main()
