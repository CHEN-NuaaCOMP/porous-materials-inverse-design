"""Single-run PredictNet training for the current stress-strain dataset.

This script keeps the original ``PredictNet`` architecture, but avoids the
large sweep in ``train.py`` and writes validation/test evidence that can be
checked before using a retrained predictor for generator screening.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import traceback
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn
from torch.utils import data

from dataset import FoamAlDataset
from model import PredictNet

LEGACY_LABEL_MODE = "per_curve_log_minmax"
GLOBAL_LABEL_MODE = "global_log1p_standard"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log(message: str, log_file: Path | None = None) -> None:
    print(message, flush=True)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as f:
            f.write(message + "\n")


def make_loader(dataset: FoamAlDataset, batch_size: int, shuffle: bool, drop_last: bool) -> data.DataLoader:
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def unpack_batch(batch):
    if len(batch) == 4:
        fine, coarse, targets, max_stress = batch
        return fine, coarse, targets, max_stress
    fine, coarse, targets = batch
    return fine, coarse, targets, None


def low_stress_weights(max_stress: torch.Tensor | None, device: torch.device) -> torch.Tensor | None:
    if max_stress is None:
        return None
    max_stress = max_stress.to(device=device, dtype=torch.float32)
    weights = torch.ones_like(max_stress)
    weights = torch.where(max_stress < 5.0, torch.full_like(weights, 2.0), weights)
    weights = torch.where(max_stress < 2.0, torch.full_like(weights, 3.0), weights)
    return weights


def phase_point_weights(num_points: int, phase_weights: tuple[float, float, float], device: torch.device) -> torch.Tensor:
    strain = torch.linspace(0.0, 0.6, num_points, device=device)
    weights = torch.ones(num_points, dtype=torch.float32, device=device)
    weights = torch.where(strain <= 0.1, torch.full_like(weights, float(phase_weights[0])), weights)
    weights = torch.where(
        (strain > 0.1) & (strain <= 0.4),
        torch.full_like(weights, float(phase_weights[1])),
        weights,
    )
    weights = torch.where(strain > 0.4, torch.full_like(weights, float(phase_weights[2])), weights)
    return weights / weights.mean().clamp_min(1e-12)


def weighted_mean(
    values: torch.Tensor,
    point_weights: torch.Tensor | None = None,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    weights = torch.ones_like(values)
    if point_weights is not None:
        view_shape = [1] * values.ndim
        view_shape[-1] = point_weights.shape[0]
        weights = weights * point_weights.view(*view_shape)
    if sample_weights is not None:
        sample_weights = sample_weights / sample_weights.mean().clamp_min(1e-12)
        view_shape = [sample_weights.shape[0]] + [1] * (values.ndim - 1)
        weights = weights * sample_weights.view(*view_shape)
    return (values * weights).sum() / weights.sum().clamp_min(1e-12)


def loss_terms(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    lambda_monotonic: float,
    lambda_local: float,
    lambda_smooth: float,
    phase_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    terminal_points: int = 0,
    lambda_terminal: float = 0.0,
    sample_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    point_weights = phase_point_weights(outputs.shape[1], phase_weights, outputs.device)
    base_loss = weighted_mean((outputs - targets) ** 2, point_weights, sample_weights)
    local_loss = weighted_mean(torch.abs(outputs - targets), point_weights, sample_weights)
    delta_violation = torch.relu(outputs[:, :-1] - outputs[:, 1:])
    delta_weights = 0.5 * (point_weights[:-1] + point_weights[1:])
    monotonic_loss = weighted_mean(delta_violation, delta_weights, sample_weights)
    if outputs.shape[1] >= 3:
        second_diff = outputs[:, 2:] - 2 * outputs[:, 1:-1] + outputs[:, :-2]
        smooth_weights = (point_weights[:-2] + point_weights[1:-1] + point_weights[2:]) / 3.0
        smooth_loss = weighted_mean(second_diff ** 2, smooth_weights, sample_weights)
    else:
        smooth_loss = outputs.new_tensor(0.0)
    terminal_points = int(max(0, min(terminal_points, outputs.shape[1])))
    if terminal_points > 0 and lambda_terminal > 0.0:
        terminal_loss = weighted_mean(
            (outputs[:, -terminal_points:] - targets[:, -terminal_points:]) ** 2,
            None,
            sample_weights,
        )
    else:
        terminal_loss = outputs.new_tensor(0.0)
    total_loss = (
        base_loss
        + lambda_monotonic * monotonic_loss
        + lambda_local * local_loss
        + lambda_smooth * smooth_loss
        + lambda_terminal * terminal_loss
    )
    return total_loss, {
        "base_loss": float(base_loss.detach().cpu()),
        "local_loss": float(local_loss.detach().cpu()),
        "monotonic_loss": float(monotonic_loss.detach().cpu()),
        "smooth_loss": float(smooth_loss.detach().cpu()),
        "terminal_loss": float(terminal_loss.detach().cpu()),
    }


def evaluate_normalized(
    model: PredictNet,
    loader: data.DataLoader,
    device: torch.device,
    lambda_monotonic: float,
    lambda_local: float,
    lambda_smooth: float,
    phase_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    terminal_points: int = 0,
    lambda_terminal: float = 0.0,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total = 0.0
    outputs_all = []
    targets_all = []
    with torch.no_grad():
        for batch in loader:
            fine, coarse, targets, _ = unpack_batch(batch)
            fine = fine.to(device)
            coarse = coarse.to(device)
            targets = targets.to(device)
            outputs = model(coarse, fine)
            loss, _ = loss_terms(
                outputs,
                targets,
                lambda_monotonic,
                lambda_local,
                lambda_smooth,
                phase_weights,
                terminal_points,
                lambda_terminal,
            )
            total += float(loss.detach().cpu())
            outputs_all.append(outputs.detach().cpu().numpy())
            targets_all.append(targets.detach().cpu().numpy())
    return total / max(len(loader), 1), np.concatenate(outputs_all), np.concatenate(targets_all)


def inverse_log_minmax(values: np.ndarray, dataset_root: Path) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    min_max = joblib.load(dataset_root / "min_max_vals.bin")
    min_vals = min_max["min_vals"]
    max_vals = min_max["max_vals"]
    return np.exp(values * (np.log(max_vals) - np.log(min_vals)) + np.log(min_vals))


def inverse_global_log1p_standard(values: np.ndarray, dataset_root: Path) -> tuple[np.ndarray, int]:
    scalers = joblib.load(dataset_root / "scalers.bin")
    if scalers.get("label_mode") != GLOBAL_LABEL_MODE or "scaler_label" not in scalers:
        raise RuntimeError(f"{dataset_root / 'scalers.bin'} has no {GLOBAL_LABEL_MODE} scaler_label")
    log_stress = scalers["scaler_label"].inverse_transform(np.asarray(values, dtype=np.float32))
    stress = np.expm1(log_stress)
    negative_count = int(np.sum(stress < 0.0))
    return np.clip(stress, 0.0, None).astype(np.float32), negative_count


def inverse_labels(values: np.ndarray, dataset_root: Path, label_mode: str) -> tuple[np.ndarray, int]:
    if label_mode == GLOBAL_LABEL_MODE:
        return inverse_global_log1p_standard(values, dataset_root)
    return inverse_log_minmax(values, dataset_root).astype(np.float32), 0


def monotonic_stats(curves: np.ndarray) -> dict[str, float | int]:
    deltas = np.diff(np.asarray(curves, dtype=np.float32), axis=1)
    violations = np.clip(-deltas, 0.0, None)
    return {
        "monotonic_violation_count": int(np.sum(violations > 0.0)),
        "monotonic_violation_curve_count": int(np.sum(np.sum(violations, axis=1) > 0.0)),
        "monotonic_violation_area": float(np.sum(violations)),
        "monotonic_violation_mean": float(np.mean(violations)),
    }


def per_sample_metrics(outputs: np.ndarray, targets: np.ndarray, names: list[str]) -> list[dict[str, float | str]]:
    rows = []
    for name, pred, true in zip(names, outputs, targets):
        rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
        mae = float(np.mean(np.abs(pred - true)))
        denom = float(np.max(true) - np.min(true))
        nrmse = rmse / max(denom, 1e-12)
        rows.append({"name": name, "rmse": rmse, "mae": mae, "nrmse": nrmse})
    return rows


def nrmse_by_mask(outputs: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rmse = np.sqrt(np.mean((outputs[:, mask] - targets[:, mask]) ** 2, axis=1))
    denom = np.maximum(np.max(targets, axis=1) - np.min(targets, axis=1), 1e-12)
    return rmse / denom


def write_evaluation_reports(
    result_dir: Path,
    outputs: np.ndarray,
    targets: np.ndarray,
    names: list[str],
    metric_rows: list[dict[str, float | str]],
) -> dict[str, float]:
    strain = np.linspace(0.0, 0.6, outputs.shape[1])
    segments = {
        "elastic_0_0.1": strain <= 0.1,
        "plateau_0.1_0.4": (strain > 0.1) & (strain <= 0.4),
        "densification_0.4_0.6": strain > 0.4,
        "terminal_0.5_0.6": strain >= 0.5,
    }
    segment_summary: dict[str, dict[str, float]] = {}
    flattened: dict[str, float] = {}
    for name, mask in segments.items():
        values = nrmse_by_mask(outputs, targets, mask)
        segment_summary[name] = {
            "mean_nrmse": float(np.mean(values)),
            "median_nrmse": float(np.median(values)),
            "p90_nrmse": float(np.quantile(values, 0.9)),
        }
        flattened[f"{name}_mean_nrmse"] = segment_summary[name]["mean_nrmse"]
        flattened[f"{name}_median_nrmse"] = segment_summary[name]["median_nrmse"]
    (result_dir / "segment_metrics.json").write_text(
        json.dumps(segment_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    nrmse_values = np.asarray([float(row["nrmse"]) for row in metric_rows], dtype=np.float32)
    threshold_summary = {}
    for threshold in (0.02, 0.03, 0.05, 0.08, 0.10):
        mask = nrmse_values <= threshold
        threshold_summary[f"nrmse_le_{threshold:.2f}"] = {
            "count": int(np.sum(mask)),
            "total": int(len(mask)),
            "fraction": float(np.mean(mask)),
        }
    (result_dir / "threshold_summary.json").write_text(
        json.dumps(threshold_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    flattened["nrmse_gt_0.10_count"] = int(np.sum(nrmse_values > 0.10))

    worst_rows = sorted(metric_rows, key=lambda row: float(row["nrmse"]), reverse=True)[:15]
    with (result_dir / "worst_samples.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "rmse", "mae", "nrmse"])
        writer.writeheader()
        writer.writerows(worst_rows)

    max_stress = np.max(targets, axis=1)
    bins = [
        ("0_2", 0.0, 2.0),
        ("2_5", 2.0, 5.0),
        ("5_10", 5.0, 10.0),
        ("10_20", 10.0, 20.0),
        ("20_50", 20.0, 50.0),
        ("50_inf", 50.0, float("inf")),
    ]
    with (result_dir / "max_stress_bin_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["bin", "min_inclusive", "max_exclusive", "count", "mean_nrmse", "median_nrmse", "max_nrmse"],
        )
        writer.writeheader()
        for label, lower, upper in bins:
            in_bin = (max_stress >= lower) & (max_stress < upper)
            if not np.any(in_bin):
                continue
            values = nrmse_values[in_bin]
            row = {
                "bin": label,
                "min_inclusive": lower,
                "max_exclusive": upper,
                "count": int(np.sum(in_bin)),
                "mean_nrmse": float(np.mean(values)),
                "median_nrmse": float(np.median(values)),
                "max_nrmse": float(np.max(values)),
            }
            writer.writerow(row)
            flattened[f"max_stress_{label}_mean_nrmse"] = row["mean_nrmse"]

    return flattened


def write_history(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "learning_rate",
                "base_loss",
                "local_loss",
                "monotonic_loss",
                "smooth_loss",
                "terminal_loss",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_history(history: list[dict[str, float]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(epochs, [row["train_loss"] for row in history], label="train")
    ax.plot(epochs, [row["val_loss"] for row in history], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("PredictNet single-run training curve")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_examples(
    outputs: np.ndarray,
    targets: np.ndarray,
    rows: list[dict[str, float | str]],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(rows) == 0:
        return
    order = np.argsort([float(row["nrmse"]) for row in rows])
    picks = list(order[:2]) + list(order[len(order) // 2: len(order) // 2 + 2]) + list(order[-2:])
    picks = list(dict.fromkeys(int(idx) for idx in picks))[:6]
    strain = np.linspace(0.0, 0.6, outputs.shape[1])
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.5), sharex=True)
    for ax, idx in zip(axes.ravel(), picks):
        ax.plot(strain, targets[idx], label="target", linewidth=1.8)
        ax.plot(strain, outputs[idx], label="predict", linewidth=1.5)
        ax.set_title(f"{rows[idx]['name']} NRMSE={float(rows[idx]['nrmse']):.3f}", fontsize=9)
        ax.grid(alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("strain")
    for ax in axes[:, 0]:
        ax.set_ylabel("stress")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def smoke_test_model(
    model: PredictNet,
    dataset: FoamAlDataset,
    device: torch.device,
    batch_size: int,
    log_file: Path,
) -> None:
    loader = make_loader(dataset, max(2, min(batch_size, 8)), False, drop_last=True)
    fine, coarse, targets, _ = unpack_batch(next(iter(loader)))
    fine = fine.to(device)
    coarse = coarse.to(device)
    targets = targets.to(device)
    model.train()
    outputs = model(coarse, fine)
    loss, _ = loss_terms(outputs, targets, 0.1, 0.5, 0.0)
    loss.backward()
    model.zero_grad(set_to_none=True)
    log(f"[smoke] forward/backward ok on {device}; output={tuple(outputs.shape)}", log_file)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset_retrain_stress_strain")
    parser.add_argument("--ckpt", default="dataset_retrain_stress_strain/ckpt")
    parser.add_argument("--result-path", default="dataset_retrain_stress_strain/results")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--attention-dim", type=int, default=256)
    parser.add_argument("--lambda-monotonic", type=float, default=0.1)
    parser.add_argument("--lambda-local", type=float, default=0.5)
    parser.add_argument("--lambda-smooth", type=float, default=0.0)
    parser.add_argument("--phase-weights", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    parser.add_argument("--terminal-points", type=int, default=0)
    parser.add_argument("--lambda-terminal", type=float, default=0.0)
    parser.add_argument("--low-stress-weighting", action="store_true")
    parser.add_argument(
        "--label-mode",
        choices=[LEGACY_LABEL_MODE, GLOBAL_LABEL_MODE],
        default=LEGACY_LABEL_MODE,
    )
    parser.add_argument("--augment", action="store_true", help="enable original PredictNet data augmentation")
    parser.add_argument("--drop", type=float, default=0.1)
    parser.add_argument("--jitter", type=float, default=0.005)
    parser.add_argument("--outlier-ratio", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument(
        "--selection-mode",
        choices=["validation", "fixed-final"],
        default="validation",
        help=(
            "validation selects the lowest validation-loss checkpoint; "
            "fixed-final trains for the requested epoch count and evaluates the test set only once"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--smoke-only", action="store_true", help="load data and run one small forward/backward only")
    parser.add_argument("--num-threads", type=int, default=1, help="Torch CPU thread count; 1 avoids OpenMP crashes")
    args = parser.parse_args()

    set_seed(args.seed)
    torch.set_num_threads(max(1, int(args.num_threads)))
    dataset_root = Path(args.dataset).resolve()
    ckpt_dir = Path(args.ckpt).resolve()
    result_dir = Path(args.result_path).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    log_file = result_dir / "train_single.log"
    if log_file.exists():
        log_file.unlink()
    log(f"[start] dataset={dataset_root}", log_file)
    log(f"[start] ckpt={ckpt_dir}", log_file)
    log(f"[start] result={result_dir}", log_file)
    log(f"[env] torch={torch.__version__} cuda_available={torch.cuda.is_available()} device_arg={args.device}", log_file)

    log("[data] loading train dataset...", log_file)
    train_dataset = FoamAlDataset(
        root=dataset_root,
        label_norm=True,
        part="train",
        augmentation=args.augment,
        drop=args.drop,
        jitter=args.jitter,
        outlier_ratio=args.outlier_ratio,
        label_mode=args.label_mode,
        return_max_stress=args.low_stress_weighting,
    )
    val_dataset = None
    if args.selection_mode == "validation":
        if not (dataset_root / "val").exists():
            raise RuntimeError(
                "selection-mode=validation requires a separate validation split; "
                "the test split is never used for checkpoint selection"
            )
        log("[data] loading validation dataset...", log_file)
        val_dataset = FoamAlDataset(
            root=dataset_root,
            label_norm=True,
            part="val",
            augmentation=False,
            label_mode=args.label_mode,
        )
    log("[data] loading test dataset...", log_file)
    test_dataset = FoamAlDataset(
        root=dataset_root,
        label_norm=True,
        part="test",
        augmentation=False,
        label_mode=args.label_mode,
    )
    log(
        f"[data] train={len(train_dataset)} "
        f"val={len(val_dataset) if val_dataset is not None else 0} test={len(test_dataset)} "
        f"channels=({train_dataset.f_channel},{train_dataset.c_channel},{train_dataset.label_channel})",
        log_file,
    )
    log(f"[selection] mode={args.selection_mode}", log_file)
    phase_weights = tuple(float(value) for value in args.phase_weights)
    log(
        f"[loss] phase_weights={phase_weights} terminal_points={args.terminal_points} "
        f"lambda_terminal={args.lambda_terminal} low_stress_weighting={args.low_stress_weighting}",
        log_file,
    )

    train_loader = make_loader(train_dataset, args.batch_size, True, drop_last=True)
    val_loader = (
        make_loader(val_dataset, max(args.batch_size, 64), False, drop_last=False)
        if val_dataset is not None
        else None
    )
    test_loader = make_loader(test_dataset, max(args.batch_size, 64), False, drop_last=False)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = PredictNet(
        feature_channels=train_dataset.f_channel,
        global_channels=train_dataset.c_channel,
        num_classes=train_dataset.label_channel,
        dropout=args.dropout,
        attention_dim=args.attention_dim,
    ).to(device)
    log(f"[model] PredictNet initialized on {device}", log_file)
    try:
        smoke_test_model(model, train_dataset, device, args.batch_size, log_file)
    except Exception:
        log("[error] smoke test failed with Python exception:", log_file)
        log(traceback.format_exc(), log_file)
        raise
    if args.smoke_only:
        log("[done] smoke-only run finished.", log_file)
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=15)

    best_val = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        log(f"[epoch {epoch}] training...", log_file)
        model.train()
        train_loss = 0.0
        term_sums = {
            "base_loss": 0.0,
            "local_loss": 0.0,
            "monotonic_loss": 0.0,
            "smooth_loss": 0.0,
            "terminal_loss": 0.0,
        }
        for batch in train_loader:
            fine, coarse, targets, max_stress = unpack_batch(batch)
            fine = fine.to(device)
            coarse = coarse.to(device)
            targets = targets.to(device)
            sample_weights = low_stress_weights(max_stress, device) if args.low_stress_weighting else None
            optimizer.zero_grad()
            outputs = model(coarse, fine)
            loss, terms = loss_terms(
                outputs,
                targets,
                args.lambda_monotonic,
                args.lambda_local,
                args.lambda_smooth,
                phase_weights,
                args.terminal_points,
                args.lambda_terminal,
                sample_weights,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += float(loss.detach().cpu())
            for key in term_sums:
                term_sums[key] += terms[key]

        train_loss /= max(len(train_loader), 1)
        for key in term_sums:
            term_sums[key] /= max(len(train_loader), 1)
        if val_loader is not None:
            val_loss, _, _ = evaluate_normalized(
                model,
                val_loader,
                device,
                args.lambda_monotonic,
                args.lambda_local,
                args.lambda_smooth,
                phase_weights,
                args.terminal_points,
                args.lambda_terminal,
            )
            scheduler_metric = val_loss
        else:
            val_loss = float("nan")
            scheduler_metric = train_loss
        scheduler.step(scheduler_metric)
        lr = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": lr,
                **term_sums,
            }
        )
        val_text = f"{val_loss:.6f}" if np.isfinite(val_loss) else "NA"
        log(f"epoch={epoch} train={train_loss:.6f} val={val_text} lr={lr:.2e}", log_file)

        if args.selection_mode == "fixed-final":
            best_val = train_loss
            best_epoch = epoch
            if epoch == args.epochs:
                torch.save(model.state_dict(), ckpt_dir / "best_model.pth")
        elif val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), ckpt_dir / "best_model.pth")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping at epoch {epoch}; best epoch {best_epoch}.")
                log(f"Early stopping at epoch {epoch}; best epoch {best_epoch}.", log_file)
                break

    write_history(result_dir / "train_history.csv", history)
    plot_history(history, result_dir / "predictnet_training_curves.png")

    model.load_state_dict(torch.load(ckpt_dir / "best_model.pth", map_location=device))
    test_loss, test_outputs_norm, test_targets_norm = evaluate_normalized(
        model,
        test_loader,
        device,
        args.lambda_monotonic,
        args.lambda_local,
        args.lambda_smooth,
        phase_weights,
        args.terminal_points,
        args.lambda_terminal,
    )
    test_outputs, negative_output_count = inverse_labels(test_outputs_norm, dataset_root, args.label_mode)
    test_targets, negative_target_count = inverse_labels(test_targets_norm, dataset_root, args.label_mode)
    metric_rows = per_sample_metrics(test_outputs, test_targets, test_dataset.names)
    extra_metrics = write_evaluation_reports(result_dir, test_outputs, test_targets, test_dataset.names, metric_rows)
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    metrics = {
        "best_epoch": best_epoch,
        "checkpoint_selection_loss": best_val,
        "best_validation_loss": best_val if args.selection_mode == "validation" else None,
        "selection_mode": args.selection_mode,
        "train_count_effective": len(train_dataset),
        "test_count": len(test_dataset),
        "curve_points": int(train_dataset.label_channel),
        "test_normalized_loss": test_loss,
        "test_mse": float(mean_squared_error(test_targets, test_outputs)),
        "test_mae": float(mean_absolute_error(test_targets, test_outputs)),
        "test_r2": float(r2_score(test_targets, test_outputs)),
        "test_nrmse_mean": float(np.mean([row["nrmse"] for row in metric_rows])),
        "test_nrmse_median": float(np.median([row["nrmse"] for row in metric_rows])),
        "label_mode": args.label_mode,
        "lambda_monotonic": args.lambda_monotonic,
        "lambda_local": args.lambda_local,
        "lambda_smooth": args.lambda_smooth,
        "phase_weights": list(phase_weights),
        "terminal_points": int(args.terminal_points),
        "lambda_terminal": args.lambda_terminal,
        "low_stress_weighting": bool(args.low_stress_weighting),
        "negative_output_before_clip_count": negative_output_count,
        "negative_target_before_clip_count": negative_target_count,
        **monotonic_stats(test_outputs),
        **extra_metrics,
        "checkpoint": str(ckpt_dir / "best_model.pth"),
        "scaler": str(dataset_root / "scalers.bin"),
    }
    (result_dir / "test_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (result_dir / "test_per_sample_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "rmse", "mae", "nrmse"])
        writer.writeheader()
        writer.writerows(metric_rows)
    strain = np.linspace(0.0, 0.6, test_outputs.shape[1], dtype=np.float32)
    np.savez_compressed(
        result_dir / "test_predictions_targets.npz",
        names=np.asarray(test_dataset.names, dtype=str),
        strain=strain,
        prediction=test_outputs.astype(np.float32),
        target=test_targets.astype(np.float32),
    )
    with (result_dir / "test_prediction_target_long.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["name", "strain", "target_stress_mpa", "predicted_stress_mpa"],
        )
        writer.writeheader()
        for name, target_curve, predicted_curve in zip(
            test_dataset.names, test_targets, test_outputs
        ):
            writer.writerows(
                {
                    "name": name,
                    "strain": float(epsilon),
                    "target_stress_mpa": float(target_value),
                    "predicted_stress_mpa": float(predicted_value),
                }
                for epsilon, target_value, predicted_value in zip(
                    strain, target_curve, predicted_curve
                )
            )
    plot_examples(test_outputs, test_targets, metric_rows, result_dir / "predictnet_test_examples.png")
    log(json.dumps(metrics, ensure_ascii=False, indent=2), log_file)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
