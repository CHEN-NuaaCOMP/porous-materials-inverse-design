"""Screen candidates by geometry and optional externally computed PoreNet curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

from src.data.dataset import load_config, read_curve
from src.utils.io_utils import pore_records_to_array, read_jsonl, write_jsonl
from src.utils.metrics import curve_metrics, geometry_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="outputs/screened")
    parser.add_argument("--target-curve", help="CSV matching config curve columns")
    parser.add_argument("--target-pore-file", help="optional real target pore CSV for evaluation-only pore-count checks")
    parser.add_argument("--target-pore-count", type=int, help="optional real target pore count for evaluation-only checks")
    parser.add_argument("--prediction-dir",
                        help="optional candidate_id.csv files with the same response column")
    parser.add_argument("--target-porosity", type=float,
                        help="optional known porosity; overrides the generated macro head")
    parser.add_argument("--geometry-only", action="store_true",
                        help="diagnostic mode: do not require stress-response predictions")
    args = parser.parse_args()
    cfg, _ = load_config(args.config)
    rules = cfg["screening"]
    cube = cfg["data"]["cube_size_mm"]
    target = None
    if args.target_curve:
        _, target = read_curve(
            Path(args.target_curve), cfg["data"]["curve_columns"],
            int(cfg["data"]["curve_points"]),
        )
    target_pore_count = args.target_pore_count
    if target_pore_count is None and args.target_pore_file:
        target_pore_count = len(pd.read_csv(args.target_pore_file))
    all_records, accepted = read_jsonl(args.input), []
    report = []
    for record in all_records:
        pores = pore_records_to_array(record["pores"])
        metrics = geometry_metrics(pores, cube)
        geometry_reasons, response_reasons = [], []
        pore_count_relative_error = None
        if not rules["min_pores"] <= metrics["pore_count"] <= rules["max_pores"]:
            geometry_reasons.append("pore_count")
        if target_pore_count is not None and rules.get("target_pore_count_relative_tolerance") is not None:
            pore_count_relative_error = (
                abs(metrics["pore_count"] - int(target_pore_count)) / max(int(target_pore_count), 1)
            )
            if pore_count_relative_error > float(rules["target_pore_count_relative_tolerance"]):
                geometry_reasons.append("target_pore_count")
        target_porosity = args.target_porosity
        if target_porosity is None and record.get("macro") and len(record["macro"]) >= 5:
            target_porosity = float(record["macro"][4])
        porosity_error = None
        if target_porosity is not None:
            porosity_error = abs(metrics["porosity_sum"] - target_porosity)
            if abs(metrics["porosity_sum"] - target_porosity) > rules["target_porosity_tolerance"]:
                geometry_reasons.append("porosity")
        boundary_policy = rules.get("boundary_policy", "strict")
        if (boundary_policy == "strict" and
                metrics["boundary_violation_fraction"] > rules["max_boundary_violation_fraction"]):
            geometry_reasons.append("boundary")
        if metrics["overlap_pair_fraction"] > rules["max_overlap_pair_fraction"]:
            geometry_reasons.append("overlap")
        if metrics["coverage_fraction"] < rules["min_coverage_fraction"]:
            geometry_reasons.append("coverage")
        curve_result = None
        if not args.geometry_only:
            if target is None:
                if rules.get("require_response_prediction", False):
                    response_reasons.append("target_curve_missing")
            elif args.prediction_dir:
                prediction_file = Path(args.prediction_dir) / f"{record['candidate_id']}.csv"
                if prediction_file.exists():
                    _, prediction = read_curve(
                        prediction_file, cfg["data"]["curve_columns"],
                        int(cfg["data"]["curve_points"]),
                    )
                    curve_result = curve_metrics(target, prediction)
                    if curve_result["curve_nrmse"] > rules.get("max_curve_nrmse", float("inf")):
                        response_reasons.append("response")
                    optional_curve_limits = {
                        "curve_elastic_nrmse": "max_elastic_nrmse",
                        "curve_plateau_nrmse": "max_plateau_nrmse",
                        "curve_densification_nrmse": "max_densification_nrmse",
                        "curve_peak_relative_error": "max_peak_relative_error",
                        "curve_terminal_relative_error": "max_terminal_relative_error",
                    }
                    for metric_name, rule_name in optional_curve_limits.items():
                        limit = rules.get(rule_name)
                        value = curve_result.get(metric_name)
                        if limit is not None and value is not None and value > float(limit):
                            response_reasons.append(rule_name.removeprefix("max_"))
                else:
                    response_reasons.append("response_prediction_missing")
            elif rules.get("require_response_prediction", False):
                response_reasons.append("response_prediction_missing")
        reasons = geometry_reasons + response_reasons
        response_evaluated = curve_result is not None
        response_accepted = None if args.geometry_only else not response_reasons
        record["screening"] = {"accepted": not reasons, "reasons": reasons,
                               "geometry_accepted": not geometry_reasons,
                               "response_evaluated": response_evaluated,
                               "response_accepted": response_accepted,
                               "boundary_policy": boundary_policy,
                               "target_porosity": target_porosity,
                               "target_pore_count": target_pore_count,
                               "geometry": metrics, "curve": curve_result}
        curve_columns = {
            "curve_mae": None,
            "curve_rmse": None,
            "curve_nrmse": None,
            "curve_mape_nonzero": None,
            "curve_elastic_rmse": None,
            "curve_elastic_nrmse": None,
            "curve_plateau_rmse": None,
            "curve_plateau_nrmse": None,
            "curve_densification_rmse": None,
            "curve_densification_nrmse": None,
            "curve_peak_abs_error": None,
            "curve_peak_relative_error": None,
            "curve_terminal_abs_error": None,
            "curve_terminal_relative_error": None,
        }
        if curve_result is not None:
            curve_columns.update(curve_result)
        report.append({"candidate_id": record["candidate_id"], "accepted": not reasons,
                       "geometry_accepted": not geometry_reasons,
                       "response_evaluated": response_evaluated,
                       "response_accepted": response_accepted,
                       "reasons": ";".join(reasons), **metrics,
                       "target_porosity": target_porosity,
                       "porosity_abs_error": porosity_error,
                       "target_pore_count": target_pore_count,
                       "pore_count_relative_error": pore_count_relative_error,
                       **curve_columns})
        if not reasons:
            accepted.append(record)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(all_records, out / "all_candidates.jsonl")
    write_jsonl(accepted, out / "accepted_candidates.jsonl")
    pd.DataFrame(report).to_csv(out / "screening_metrics.csv", index=False, encoding="utf-8-sig")
    reason_counts = Counter(reason for item in report for reason in item["reasons"].split(";") if reason)
    (out / "screening_summary.json").write_text(json.dumps({
        "total": len(all_records),
        "geometry_accepted": sum(bool(item["geometry_accepted"]) for item in report),
        "response_evaluated": sum(bool(item["response_evaluated"]) for item in report),
        "response_accepted": sum(item["response_accepted"] is True for item in report),
        "accepted": len(accepted),
        "reason_counts": dict(reason_counts),
        "boundary_policy": rules.get("boundary_policy", "strict"),
        "response_status": "external predictions used" if args.prediction_dir else
        ("geometry-only diagnostic" if args.geometry_only else "response predictions required but absent"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"accepted={len(accepted)}/{len(all_records)} -> {out}")


if __name__ == "__main__":
    main()
