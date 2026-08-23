"""Run generation/repair/StressNet screening for every target in a data split.

This module is intentionally an orchestration layer around the already audited
single-target CLIs. It keeps one directory per target curve, then writes a
target-level summary that can be used directly for manuscript figures.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataset import load_config
from src.utils.normalization import MACRO_COLUMNS


CODE_ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def run_command(command: list[str], cwd: Path, dry_run: bool = False) -> None:
    printable = " ".join(f'"{x}"' if " " in str(x) else str(x) for x in command)
    print(printable, flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=str(cwd), check=True)


def split_target_names(
    split_file: Path, splits: list[str], max_targets: int | None, target_names: list[str] | None
) -> list[tuple[str, str]]:
    split_data = read_json(split_file)
    selected: list[tuple[str, str]] = []
    explicit = set(target_names or [])
    for split in splits:
        names = list(split_data[split])
        if explicit:
            names = [name for name in names if name in explicit]
        if max_targets is not None:
            names = names[:max_targets]
        selected.extend((split, name) for name in names)
    return selected


def summarize_target(target_dir: Path, split: str, name: str, target_curve: Path, known_porosity: float | None) -> dict:
    summary_path = target_dir / "screened" / "screening_summary.json"
    metrics_path = target_dir / "screened" / "screening_metrics.csv"
    if not summary_path.exists() or not metrics_path.exists():
        return {
            "split": split,
            "target_name": name,
            "target_curve": str(target_curve),
            "known_porosity": known_porosity,
            "status": "missing",
        }
    summary = read_json(summary_path)
    metrics = pd.read_csv(metrics_path)
    accepted = metrics[as_bool(metrics["accepted"])].sort_values("curve_nrmse")
    curve_nrmse = metrics["curve_nrmse"].dropna().to_numpy(float)
    row = {
        "split": split,
        "target_name": name,
        "target_curve": str(target_curve),
        "known_porosity": known_porosity,
        "status": "complete",
        "generated": int(summary["total"]),
        "geometry_accepted": int(summary["geometry_accepted"]),
        "response_evaluated": int(summary["response_evaluated"]),
        "response_accepted": int(summary["response_accepted"]),
        "accepted": int(summary["accepted"]),
        "acceptance_rate": float(summary["accepted"] / max(summary["total"], 1)),
        "reason_counts": json.dumps(summary.get("reason_counts", {}), ensure_ascii=False),
        "median_nrmse": float(np.median(curve_nrmse)) if len(curve_nrmse) else np.nan,
        "p95_nrmse": float(np.quantile(curve_nrmse, 0.95)) if len(curve_nrmse) else np.nan,
    }
    if len(accepted):
        best = accepted.iloc[0]
        row.update({
            "best_candidate": best["candidate_id"],
            "best_curve_rmse": float(best["curve_rmse"]),
            "best_curve_nrmse": float(best["curve_nrmse"]),
            "best_elastic_nrmse": float(best["curve_elastic_nrmse"]),
            "best_plateau_nrmse": float(best["curve_plateau_nrmse"]),
            "best_densification_nrmse": float(best["curve_densification_nrmse"]),
            "best_terminal_relative_error": float(best["curve_terminal_relative_error"]),
            "best_pore_count": int(best["pore_count"]),
            "best_porosity_sum": float(best["porosity_sum"]),
            "best_overlap_pair_fraction": float(best["overlap_pair_fraction"]),
            "best_coverage_fraction": float(best["coverage_fraction"]),
        })
    else:
        row.update({
            "best_candidate": None,
            "best_curve_rmse": np.nan,
            "best_curve_nrmse": np.nan,
            "best_elastic_nrmse": np.nan,
            "best_plateau_nrmse": np.nan,
            "best_densification_nrmse": np.nan,
            "best_terminal_relative_error": np.nan,
            "best_pore_count": np.nan,
            "best_porosity_sum": np.nan,
            "best_overlap_pair_fraction": np.nan,
            "best_coverage_fraction": np.nan,
        })
    return row


def write_summary(rows: list[dict], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "target_set_summary.csv", index=False, encoding="utf-8-sig")
    (output_root / "target_set_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--splits", nargs="+", default=["val", "test"], choices=["train", "val", "test"])
    parser.add_argument("--target-names", nargs="*", help="optional subset by exact sample name")
    parser.add_argument("--max-targets", type=int, help="debug limit applied per split")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--repair-iterations", type=int, default=80)
    parser.add_argument("--repair-mode", choices=["curve_only", "known_porosity"], default="known_porosity",
                        help="known_porosity uses dataset metadata for manuscript target-set evaluation; curve_only matches each decoded macro porosity")
    parser.add_argument("--output-root", default="target_set_runs/stress_v2_val_test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--predictor-mode",
        choices=["external_predictnet", "internal_stressnet"],
        default="external_predictnet",
        help="external_predictnet uses the bundled validated PredictNet checkpoint",
    )
    parser.add_argument("--fallback-response-guided", action="store_true",
                        help="for targets with no accepted candidate, run response-guided latent search")
    parser.add_argument("--fallback-min-accepted", type=int, default=1)
    parser.add_argument("--fallback-samples", type=int, default=16)
    parser.add_argument("--fallback-iterations", type=int, default=100)
    parser.add_argument("--fallback-temperature", type=float, default=0.9)
    parser.add_argument("--fallback-learning-rate", type=float, default=0.04)
    parser.add_argument("--fallback-response-weight", type=float, default=1.0)
    parser.add_argument("--fallback-latent-weight", type=float, default=0.015)
    parser.add_argument("--fallback-porosity-weight", type=float, default=0.35)
    parser.add_argument("--force", action="store_true", help="rerun targets even if screened summary exists")
    parser.add_argument("--dry-run", action="store_true", help="print commands without executing them")
    args = parser.parse_args()

    cfg, project_root = load_config(args.config)
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = CODE_ROOT / output_root
    checkpoint_dir = CODE_ROOT / Path(cfg["generation"]["checkpoint"]).parent
    split_file = checkpoint_dir / "split_names.json"
    global_df = pd.read_csv(project_root / cfg["data"]["global_csv"]).set_index("name")
    target_items = split_target_names(split_file, args.splits, args.max_targets, args.target_names)
    if not target_items:
        raise SystemExit("No target curves selected.")

    manifest_rows = []
    summary_rows: list[dict] = []
    for index, (split, name) in enumerate(target_items, start=1):
        target_curve = project_root / cfg["data"]["curve_dir"] / f"{name}.csv"
        if not target_curve.exists():
            raise FileNotFoundError(target_curve)
        target_pore_file = project_root / cfg["data"]["pore_dir"] / f"{name}.csv"
        target_dir = output_root / split / name
        screened_summary = target_dir / "screened" / "screening_summary.json"
        known_porosity = float(global_df.loc[name, MACRO_COLUMNS[4]]) if name in global_df.index else None
        repair_porosity_args = (
            ["--target-porosity", f"{known_porosity:.10f}"]
            if args.repair_mode == "known_porosity" and known_porosity is not None
            else ["--match-macro-porosity"]
        )
        screen_porosity_args = (
            ["--target-porosity", f"{known_porosity:.10f}"]
            if args.repair_mode == "known_porosity" and known_porosity is not None
            else []
        )
        manifest_rows.append({
            "split": split,
            "target_name": name,
            "target_curve": str(target_curve),
            "known_porosity": known_porosity,
        })
        print(f"\n[{index}/{len(target_items)}] {split}/{name}", flush=True)
        if args.force or not screened_summary.exists():
            candidate_dir = target_dir / "candidates"
            repaired_dir = target_dir / "repaired"
            prediction_dir = target_dir / "predictions"
            screened_dir = target_dir / "screened"
            repaired_jsonl = repaired_dir / "candidates_repaired.jsonl"
            commands = [
                [sys.executable, "-m", "src.generate_candidates",
                 "--config", str(Path(args.config)),
                 "--target-curve", str(target_curve),
                 "--samples", str(args.samples),
                 "--temperature", str(args.temperature),
                 "--output", str(candidate_dir)],
                [sys.executable, "-m", "src.repair_candidates",
                 "--input", str(candidate_dir / "candidates.jsonl"),
                 "--output", str(repaired_jsonl),
                 "--redistribute",
                 *repair_porosity_args,
                 "--boundary-policy", cfg["screening"].get("boundary_policy", "allow_truncated"),
                 "--iterations", str(args.repair_iterations)],
                [sys.executable, "-m", "src.predict_candidate_responses",
                 "--config", str(Path(args.config)),
                 "--input", str(repaired_jsonl),
                 "--output-dir", str(prediction_dir),
                 "--target-curve", str(target_curve),
                 "--predictor-mode", args.predictor_mode,
                 "--device", args.device],
                [sys.executable, "-m", "src.screen_candidates",
                 "--config", str(Path(args.config)),
                 "--input", str(repaired_jsonl),
                 "--output-dir", str(screened_dir),
                 "--target-curve", str(target_curve),
                 "--prediction-dir", str(prediction_dir),
                 *(["--target-pore-file", str(target_pore_file)] if target_pore_file.exists() else []),
                 *screen_porosity_args],
            ]
            for command in commands:
                run_command(command, CODE_ROOT, dry_run=args.dry_run)
        else:
            print("existing screened summary found; use --force to rerun", flush=True)
        if not args.dry_run:
            row = summarize_target(target_dir, split, name, target_curve, known_porosity)
            if (
                args.fallback_response_guided
                and row.get("status") == "complete"
                and int(row.get("accepted", 0)) < int(args.fallback_min_accepted)
            ):
                print(
                    "no accepted candidate after standard sampling; "
                    "running response-guided latent fallback",
                    flush=True,
                )
                guided_candidate_dir = target_dir / "rg_cand"
                guided_repaired_dir = target_dir / "rg_repair"
                guided_prediction_dir = target_dir / "rg_pred"
                guided_repaired_jsonl = guided_repaired_dir / "candidates_repaired.jsonl"
                guided_commands = [
                    [sys.executable, "-m", "src.response_guided_generate",
                     "--config", str(Path(args.config)),
                     "--target-curve", str(target_curve),
                     "--output", str(guided_candidate_dir),
                     "--samples", str(args.fallback_samples),
                     "--iterations", str(args.fallback_iterations),
                     "--learning-rate", str(args.fallback_learning_rate),
                     "--temperature", str(args.fallback_temperature),
                     "--response-weight", str(args.fallback_response_weight),
                     "--latent-weight", str(args.fallback_latent_weight),
                     "--porosity-weight", str(args.fallback_porosity_weight),
                     "--device", args.device,
                     *(["--target-porosity", f"{known_porosity:.10f}"]
                       if known_porosity is not None else [])],
                    [sys.executable, "-m", "src.repair_candidates",
                     "--input", str(guided_candidate_dir / "candidates.jsonl"),
                     "--output", str(guided_repaired_jsonl),
                     "--redistribute",
                     *repair_porosity_args,
                     "--boundary-policy", cfg["screening"].get("boundary_policy", "allow_truncated"),
                     "--iterations", str(args.repair_iterations)],
                    [sys.executable, "-m", "src.predict_candidate_responses",
                     "--config", str(Path(args.config)),
                     "--input", str(guided_repaired_jsonl),
                     "--output-dir", str(guided_prediction_dir),
                     "--target-curve", str(target_curve),
                     "--predictor-mode", args.predictor_mode,
                     "--device", args.device],
                    [sys.executable, "-m", "src.screen_candidates",
                     "--config", str(Path(args.config)),
                     "--input", str(guided_repaired_jsonl),
                     "--output-dir", str(target_dir / "screened"),
                     "--target-curve", str(target_curve),
                     "--prediction-dir", str(guided_prediction_dir),
                     *(["--target-pore-file", str(target_pore_file)] if target_pore_file.exists() else []),
                     *screen_porosity_args],
                ]
                for command in guided_commands:
                    run_command(command, CODE_ROOT, dry_run=False)
                row = summarize_target(target_dir, split, name, target_curve, known_porosity)
            summary_rows.append(row)
            write_summary(summary_rows, output_root)

    if args.dry_run:
        print(
            f"\nDry run complete. Selected {len(manifest_rows)} targets. "
            "No files were written."
        )
        return
    pd.DataFrame(manifest_rows).to_csv(output_root / "target_set_manifest.csv", index=False, encoding="utf-8-sig")
    write_summary(summary_rows, output_root)
    complete = [row for row in summary_rows if row.get("status") == "complete"]
    if complete:
        accepted_targets = sum(row["accepted"] > 0 for row in complete)
        print(
            f"\nCompleted {len(complete)} targets. "
            f"Targets with at least one accepted candidate: {accepted_targets}/{len(complete)}. "
            f"Summary: {output_root / 'target_set_summary.csv'}"
        )
    else:
        print(f"\nNo completed target summaries yet. Manifest: {output_root / 'target_set_manifest.csv'}")


if __name__ == "__main__":
    main()
