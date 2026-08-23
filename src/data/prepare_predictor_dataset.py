"""Prepare an aligned PredictNet retraining dataset from the project raw data.

The original PredictNet code expects this layout:

dataset_root/
  train/global.csv
  train/local/<name>.csv
  train/energy/<name>.csv
  val/...
  test/...

Here ``energy`` keeps the historical folder name used by PredictNet, but the
files are the stress-strain response CSVs requested for the current project.
The source feature folder may use either ``local`` or the manuscript release
name ``pore_cloud``.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def load_split_file(path: Path) -> dict[str, list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, list[str]] = {}
    for key in ("train", "val", "test"):
        values = data.get(key, [])
        result[key] = [str(Path(item).stem) for item in values]
    return result


def load_split_root(path: Path) -> dict[str, list[str]]:
    """Read frozen train/val/test names from PredictNet-style global.csv files."""

    result: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for key in result:
        global_csv = path / key / "global.csv"
        if not global_csv.exists():
            continue
        frame = pd.read_csv(global_csv)
        if "name" not in frame.columns:
            raise ValueError(f"{global_csv} must contain a 'name' column")
        result[key] = [str(value) for value in frame["name"].tolist()]
    if not result["train"] or not result["test"]:
        raise ValueError(f"{path} must contain non-empty train/test global.csv files")
    return result


def make_random_split(
    names: list[str],
    test_fraction: float,
    val_fraction: float,
    seed: int,
) -> dict[str, list[str]]:
    """Split independent sample identifiers without prefix-based grouping."""

    if not 0.0 <= test_fraction < 1.0 or not 0.0 <= val_fraction < 1.0:
        raise ValueError("test_fraction and val_fraction must be in [0, 1)")
    if test_fraction + val_fraction >= 1.0:
        raise ValueError("test_fraction + val_fraction must be less than 1")
    rng = random.Random(seed)
    shuffled = sorted(set(names))
    rng.shuffle(shuffled)
    n_test = int(round(len(shuffled) * test_fraction))
    n_val = int(round(len(shuffled) * val_fraction))
    return {
        "train": sorted(shuffled[n_test + n_val:]),
        "val": sorted(shuffled[n_test:n_test + n_val]),
        "test": sorted(shuffled[:n_test]),
    }


def copy_part(
    part: str,
    names: list[str],
    global_frame: pd.DataFrame,
    local_dir: Path,
    curve_dir: Path,
    output_root: Path,
    curve_points: int,
    curve_columns: tuple[str, str],
    strain_min: float,
    strain_max: float,
) -> int:
    part_root = output_root / part
    local_out = part_root / "local"
    curve_out = part_root / "energy"
    local_out.mkdir(parents=True, exist_ok=True)
    curve_out.mkdir(parents=True, exist_ok=True)

    selected = global_frame[global_frame["name"].isin(names)].copy()
    selected = selected.set_index("name").loc[names].reset_index()
    selected.to_csv(part_root / "global.csv", index=False, encoding="utf-8-sig")

    copied = 0
    for name in names:
        shutil.copy2(local_dir / f"{name}.csv", local_out / f"{name}.csv")
        source_curve = curve_dir / f"{name}.csv"
        destination_curve = curve_out / f"{name}.csv"
        if curve_points > 0:
            frame = pd.read_csv(source_curve)
            if not set(curve_columns).issubset(frame.columns):
                raise ValueError(
                    f"{source_curve} must contain {list(curve_columns)}, "
                    f"found {list(frame.columns)}"
                )
            clean = frame.loc[:, list(curve_columns)].apply(pd.to_numeric, errors="coerce").dropna()
            clean = clean.sort_values(curve_columns[0]).drop_duplicates(curve_columns[0], keep="last")
            strain = clean[curve_columns[0]].to_numpy(float)
            response = clean[curve_columns[1]].to_numpy(float)
            if len(strain) < 2:
                raise ValueError(f"{source_curve} must contain at least two distinct strain nodes")
            tolerance = 1e-8
            if strain[0] > strain_min + tolerance or strain[-1] < strain_max - tolerance:
                raise ValueError(
                    f"{source_curve} covers strain [{strain[0]:.8g}, {strain[-1]:.8g}], "
                    f"but the manuscript grid requires [{strain_min:.8g}, {strain_max:.8g}]"
                )
            grid = np.linspace(float(strain_min), float(strain_max), curve_points)
            pd.DataFrame(
                {
                    curve_columns[0]: grid,
                    curve_columns[1]: np.interp(grid, strain, response),
                }
            ).to_csv(destination_curve, index=False, encoding="utf-8-sig")
        else:
            shutil.copy2(source_curve, destination_curve)
        copied += 1
    return copied


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-root",
        required=True,
        help="folder containing global.csv and local/",
    )
    parser.add_argument(
        "--curve-dir",
        required=True,
        help="folder containing stress-strain response CSVs",
    )
    parser.add_argument(
        "--output-root",
        default="dataset_retrain_stress_strain",
        help="PredictNet-format output dataset folder",
    )
    parser.add_argument("--split-file", help="optional JSON with train/val/test sample-name lists")
    parser.add_argument(
        "--split-root",
        help="optional PredictNet dataset root whose train/val/test global.csv files define the split",
    )
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument(
        "--curve-points",
        type=int,
        default=0,
        help="resample every curve to this many equally spaced nodes; 0 preserves source rows",
    )
    parser.add_argument("--strain-min", type=float, default=0.0)
    parser.add_argument("--strain-max", type=float, default=0.6)
    parser.add_argument("--curve-columns", nargs=2, default=["Strain", "Stress"])
    args = parser.parse_args()
    if args.strain_max <= args.strain_min:
        raise ValueError("--strain-max must be greater than --strain-min")

    feature_root = resolve(args.feature_root)
    local_dir = feature_root / "local"
    if not local_dir.exists():
        local_dir = feature_root / "pore_cloud"
    if not local_dir.exists():
        raise FileNotFoundError(
            f"{feature_root} must contain either a 'local' or 'pore_cloud' directory"
        )
    curve_dir = resolve(args.curve_dir)
    output_root = resolve(args.output_root)

    global_frame = pd.read_csv(feature_root / "global.csv")
    if "name" not in global_frame.columns:
        raise ValueError("global.csv must contain a 'name' column")

    local_names = {path.stem for path in local_dir.glob("*.csv")}
    curve_names = {path.stem for path in curve_dir.glob("*.csv")}
    global_names = [str(name) for name in global_frame["name"].tolist()]
    aligned_names = [name for name in global_names if name in local_names and name in curve_names]
    if not aligned_names:
        raise RuntimeError("No aligned local/global/curve samples found")

    if args.split_file and args.split_root:
        raise ValueError("Use only one of --split-file and --split-root")
    if args.split_file:
        raw_split = load_split_file(resolve(args.split_file))
        split_source = str(resolve(args.split_file))
    elif args.split_root:
        raw_split = load_split_root(resolve(args.split_root))
        split_source = str(resolve(args.split_root))
    else:
        raw_split = None
        split_source = "deterministic_random_split"

    if raw_split is not None:
        aligned_set = set(aligned_names)
        split = {
            key: sorted([name for name in raw_split.get(key, []) if name in aligned_set])
            for key in ("train", "val", "test")
        }
        used = set(split["train"]) | set(split["val"]) | set(split["test"])
        split["train"].extend(sorted(set(aligned_names) - used))
        split["train"] = sorted(set(split["train"]))
    else:
        split = make_random_split(
            aligned_names,
            test_fraction=args.test_fraction,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "feature_root": str(feature_root),
        "curve_dir": str(curve_dir),
        "output_root": str(output_root),
        "aligned_samples": len(aligned_names),
        "split_counts": {key: len(values) for key, values in split.items()},
        "split_source": split_source,
        "curve_points": int(args.curve_points) if args.curve_points > 0 else "source",
        "strain_min": float(args.strain_min),
        "strain_max": float(args.strain_max),
        "curve_columns": list(args.curve_columns),
        "splits": split,
    }
    for part, names in split.items():
        if names:
            copy_part(
                part,
                names,
                global_frame,
                local_dir,
                curve_dir,
                output_root,
                int(args.curve_points),
                (str(args.curve_columns[0]), str(args.curve_columns[1])),
                float(args.strain_min),
                float(args.strain_max),
            )

    (output_root / "split_names.json").write_text(
        json.dumps(split, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    (output_root / "prepare_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    print(
        "Prepared PredictNet dataset at "
        f"{output_root} with train={len(split['train'])}, "
        f"val={len(split['val'])}, test={len(split['test'])}."
    )


if __name__ == "__main__":
    main()
