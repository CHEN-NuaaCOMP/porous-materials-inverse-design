"""Deterministic post-generation center optimization with explicit diagnostics.

This lightweight stage reduces sphere-equivalent overlap and boundary overflow
without changing pore volume or morphology descriptors. The subsequent legacy
ellipsoid optimizer may still refine axes and orientations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io_utils import pore_array_to_records, pore_records_to_array, read_jsonl, write_jsonl
from src.utils.metrics import equivalent_radius, geometry_metrics


def optimize_centers(
    pores: np.ndarray, cube_size: float, iterations: int = 100,
    step: float = 0.35, clearance: float = 0.02,
    boundary_policy: str = "allow_truncated", redistribute: bool = False,
    log_every: int = 1,
) -> tuple[np.ndarray, list[dict]]:
    out = np.asarray(pores, dtype=float).copy()
    xyz = out[:, :3]
    radii = equivalent_radius(out[:, 5])
    if redistribute and len(xyz) > 1:
        # Preserve each axis' generated rank while expanding a collapsed cloud
        # over the physical domain. This is deterministic and explicitly logged
        # as post-generation repair rather than raw model output.
        quantiles = (np.arange(len(xyz), dtype=float) + 0.5) / len(xyz) * cube_size
        for axis in range(3):
            order = np.argsort(xyz[:, axis], kind="stable")
            xyz[order, axis] = quantiles
    history = []
    initial_metrics = geometry_metrics(out, cube_size)
    history.append(
        {
            "iteration": -1,
            "stage": "before",
            "active_overlap_pair_count": np.nan,
            "active_overlap_depth_sum": np.nan,
            "mean_step_norm": 0.0,
            "max_step_norm": 0.0,
            **initial_metrics,
        }
    )
    log_every = max(1, int(log_every))
    for iteration in range(iterations):
        disp = np.zeros_like(xyz)
        delta_all = xyz[:, None, :] - xyz[None, :, :]
        distance_all = np.sqrt(np.sum(delta_all * delta_all, axis=-1))
        target_all = radii[:, None] + radii[None, :] + clearance
        active = np.triu(distance_all < target_all, 1)
        pair_i, pair_j = np.where(active)
        overlap_sum = 0.0
        if len(pair_i):
            pair_delta = xyz[pair_i] - xyz[pair_j]
            pair_distance = distance_all[pair_i, pair_j]
            pair_target = target_all[pair_i, pair_j]
            pair_overlap = np.maximum(pair_target - pair_distance, 0.0)
            directions = pair_delta / np.maximum(pair_distance[:, None], 1e-12)
            coincident = pair_distance <= 1e-12
            if np.any(coincident):
                directions[coincident] = np.eye(3)[
                    (pair_i[coincident] + pair_j[coincident]) % 3
                ]
            moves = 0.5 * pair_overlap[:, None] * directions
            np.add.at(disp, pair_i, moves)
            np.add.at(disp, pair_j, -moves)
            overlap_sum = float(pair_overlap.sum())
        step_vectors = step * disp
        step_norm = np.sqrt(np.sum(step_vectors * step_vectors, axis=1))
        xyz += step * disp
        if boundary_policy == "strict":
            xyz[:] = np.minimum(np.maximum(xyz, radii[:, None]), cube_size - radii[:, None])
        elif boundary_policy == "allow_truncated":
            xyz[:] = np.clip(xyz, 0.0, cube_size)
        else:
            raise ValueError(f"Unknown boundary_policy: {boundary_policy}")
        if iteration % log_every == 0 or iteration == iterations - 1:
            out[:, :3] = xyz
            metric = geometry_metrics(out, cube_size)
            history.append(
                {
                    "iteration": iteration,
                    "stage": "after_step",
                    "active_overlap_pair_count": int(len(pair_i)),
                    "active_overlap_depth_sum": overlap_sum,
                    "mean_step_norm": float(step_norm.mean()) if len(step_norm) else 0.0,
                    "max_step_norm": float(step_norm.max()) if len(step_norm) else 0.0,
                    "overlap_depth_sum": overlap_sum,
                    **metric,
                }
            )
            if metric["overlap_pair_count"] == 0 and metric["boundary_violation_fraction"] == 0:
                break
    out[:, :3] = xyz
    return out.astype(np.float32), history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--output-dir", default="outputs/optimized")
    parser.add_argument("--cube-size", type=float, default=14.0)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--boundary-policy", choices=["strict", "allow_truncated"],
                        default="allow_truncated")
    parser.add_argument("--redistribute", action="store_true")
    parser.add_argument("--log-every", type=int, default=1,
                        help="Record optimization diagnostics every N iterations.")
    args = parser.parse_args()
    records = read_jsonl(args.input)
    candidate = next((x for x in records if x["candidate_id"] == args.candidate_id), None)
    if candidate is None:
        raise KeyError(args.candidate_id)
    before = pore_records_to_array(candidate["pores"])
    after, history = optimize_centers(
        before, args.cube_size, args.iterations,
        boundary_policy=args.boundary_policy, redistribute=args.redistribute,
        log_every=args.log_every,
    )
    result = dict(candidate)
    result["candidate_id"] = f"{candidate['candidate_id']}_optimized"
    result["pores"] = pore_array_to_records(after)
    result["optimization"] = {
        "method": "rank-spread plus sphere-equivalent center projection" if args.redistribute
        else "sphere-equivalent center projection",
        "boundary_policy": args.boundary_policy,
        "before": geometry_metrics(before, args.cube_size),
        "after": geometry_metrics(after, args.cube_size),
        "note": "Pore descriptors and volumes are unchanged; ellipsoid-axis refinement remains downstream.",
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl([result], out / "optimized_candidate.jsonl")
    pd.DataFrame(history).to_csv(out / "optimization_history.csv", index=False, encoding="utf-8-sig")
    (out / "optimization_summary.json").write_text(
        json.dumps(result["optimization"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result["optimization"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
