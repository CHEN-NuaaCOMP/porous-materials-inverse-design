"""Exercise the no-validation fixed-final training branch on synthetic data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def make_fixture(output: Path) -> Path:
    data_root = output / "data" / "full"
    pore_dir = data_root / "pore_cloud"
    curve_dir = data_root / "stress_strain_responses"
    pore_dir.mkdir(parents=True, exist_ok=True)
    curve_dir.mkdir(parents=True, exist_ok=True)

    names = ["sample_a", "sample_b"]
    global_rows = []
    rng = np.random.default_rng(7)
    pore_columns = [
        "CenterX(mm)", "CenterY(mm)", "CenterZ(mm)", "EqDiameter(mm)",
        "Area(mm^2)", "Volume(mm^3)", "Anisotropy", "OrientationPhi",
        "OrientationTheta", "Sphericity",
    ]
    for index, name in enumerate(names):
        count = 12 + index
        diameter = rng.uniform(0.5, 1.8, count)
        pore_frame = pd.DataFrame(
            np.column_stack(
                [
                    rng.uniform(0.2, 13.8, (count, 3)),
                    diameter,
                    np.pi * diameter**2,
                    np.pi * diameter**3 / 6.0,
                    rng.uniform(0.2, 0.9, count),
                    rng.uniform(0.0, 0.9, count),
                    rng.uniform(-1.8, 1.8, count),
                    rng.uniform(0.4, 1.0, count),
                ]
            ),
            columns=pore_columns,
        )
        pore_frame.to_csv(pore_dir / f"{name}.csv", index=False)
        strain = np.linspace(0.0, 0.6, 50)
        stress = (0.8 + 0.2 * index) * strain + (2.0 + index) * strain**3
        pd.DataFrame({"Strain": strain, "Stress": stress}).to_csv(
            curve_dir / f"{name}.csv", index=False
        )
        global_rows.append(
            {
                "name": name,
                "mean": 1.0 + 0.1 * index,
                "std": 0.2 + 0.02 * index,
                "max": 1.8 + 0.1 * index,
                "fractal_dimension": 2.2 + 0.05 * index,
                "volume_fraction": 0.55 + 0.04 * index,
            }
        )
    pd.DataFrame(global_rows).to_csv(data_root / "global.csv", index=False)

    split_path = output / "split.json"
    split_path.write_text(
        json.dumps({"train": names, "val": [], "test": []}, indent=2),
        encoding="utf-8",
    )

    config = yaml.safe_load((ROOT / "configs" / "manuscript.yaml").read_text(encoding="utf-8"))
    config["project_root"] = str(output)
    config["data"]["split_file"] = str(split_path)
    config["data"]["max_pores"] = 32
    config["training"]["response_predictnet_points"] = 32
    config["screening"]["min_pores"] = 1
    config["screening"]["max_pores"] = 32
    config["screening"]["porenet_checkpoint"] = str(
        ROOT / "checkpoints" / "predictnet_manuscript_8_16" / "best_model.pth"
    )
    config["screening"]["porenet_scalers"] = str(
        ROOT / "checkpoints" / "predictnet_manuscript_8_16" / "scalers.bin"
    )
    config["screening"]["porenet_code_dir"] = str(ROOT / "external_predictor")
    config["generation"]["checkpoint"] = str(output / "checkpoints" / "generator" / "best_model.pt")
    config["generation"]["normalizer"] = str(output / "checkpoints" / "generator" / "normalizer.json")
    config_path = output / "synthetic_smoke.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    config_path = make_fixture(output)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.train",
            "--config",
            str(config_path),
            "--device",
            "cpu",
            "--smoke",
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    (output / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (output / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    checkpoint = output / "checkpoints" / "generator" / "best_model.pt"
    if not checkpoint.exists():
        raise RuntimeError("fixed-final smoke run did not save the final checkpoint")
    state = __import__("torch").load(checkpoint, map_location="cpu")
    if state.get("selection_mode") != "fixed_final" or state.get("epoch") != 1:
        raise RuntimeError("checkpoint metadata does not record the fixed-final protocol")
    print("PASS: fixed-final generator training smoke test")


if __name__ == "__main__":
    main()
