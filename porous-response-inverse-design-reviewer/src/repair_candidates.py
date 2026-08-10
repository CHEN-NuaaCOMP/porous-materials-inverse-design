"""Batch repair generated candidate centers before response validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.optimize_geometry import optimize_centers
from src.utils.io_utils import pore_array_to_records, pore_records_to_array, read_jsonl, write_jsonl
from src.utils.metrics import geometry_metrics


def rescale_pore_volume(pores: np.ndarray, target_porosity: float, cube_size: float) -> np.ndarray:
    result = np.asarray(pores, dtype=float).copy()
    if result.size == 0:
        return result.reshape(0, 10)
    current = result[:, 5].sum() / cube_size ** 3
    if current <= 0:
        return result
    factor = float(target_porosity) / current
    result[:, 5] *= factor
    result[:, 3] *= factor ** (1.0 / 3.0)
    result[:, 4] *= factor ** (2.0 / 3.0)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cube-size", type=float, default=14.0)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--boundary-policy", choices=["strict", "allow_truncated"],
                        default="allow_truncated")
    parser.add_argument("--redistribute", action="store_true")
    parser.add_argument("--target-porosity", type=float,
                        help="known target porosity; use only when independently available")
    parser.add_argument("--match-macro-porosity", action="store_true",
                        help="rescale pore descriptors to each candidate's predicted macro porosity")
    args = parser.parse_args()
    repaired, rows = [], []
    for record in read_jsonl(args.input):
        before = pore_records_to_array(record["pores"])
        target_porosity = args.target_porosity
        porosity_source = "explicit" if target_porosity is not None else None
        if target_porosity is None and args.match_macro_porosity and record.get("macro"):
            target_porosity = float(record["macro"][4])
            porosity_source = "generated_macro"
        prepared = before
        if target_porosity is not None:
            prepared = rescale_pore_volume(prepared, target_porosity, args.cube_size)
        if len(prepared) == 0:
            after = prepared.reshape(0, 10).astype(np.float32)
            history = []
        else:
            after, history = optimize_centers(
                prepared, args.cube_size, args.iterations,
                boundary_policy=args.boundary_policy, redistribute=args.redistribute,
            )
        result = dict(record)
        result["candidate_id"] = f"{record['candidate_id']}_repaired"
        result["pores"] = pore_array_to_records(after)
        result["repair"] = {
            "method": "rank-spread plus sphere-equivalent projection" if args.redistribute
            else "sphere-equivalent projection",
            "boundary_policy": args.boundary_policy,
            "porosity_source": porosity_source,
            "target_porosity": target_porosity,
            "before": geometry_metrics(before, args.cube_size),
            "after": geometry_metrics(after, args.cube_size),
            "iterations_recorded": len(history),
            "skipped_empty_candidate": bool(len(prepared) == 0),
        }
        repaired.append(result)
        rows.append({"candidate_id": result["candidate_id"], **result["repair"]["after"]})
    output = Path(args.output)
    write_jsonl(repaired, output)
    pd.DataFrame(rows).to_csv(output.with_suffix(".metrics.csv"), index=False, encoding="utf-8-sig")
    print(f"Wrote {len(repaired)} repaired candidates to {output}")


if __name__ == "__main__":
    main()
