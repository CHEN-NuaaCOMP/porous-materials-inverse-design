"""Train the independently released GeoLoop-PoreGen model."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.dataset import PoreDataset, available_names, build_normalizer, load_config, split_names
from src.utils.normalization import MACRO_COLUMNS
from src.models.poregen_cvae import GeoLoopPoreGen, poregen_loss
from src.models.poregen_slot_cvae import GeoLoopPoreGenV2, poregen_loss_v2
from src.models.predictnet_regularizer import FrozenPredictNetRegularizer


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(cfg: dict) -> GeoLoopPoreGen:
    model_cfg = dict(cfg["model"])
    architecture = model_cfg.pop("architecture", "slot_linear_v1")
    if architecture == "slot_query_v2":
        return GeoLoopPoreGenV2(max_pores=cfg["data"]["max_pores"], **model_cfg)
    return GeoLoopPoreGen(max_pores=cfg["data"]["max_pores"], **model_cfg)


def load_or_make_splits(cfg: dict, root: Path, config_path: Path) -> dict[str, list[str]]:
    split_file = cfg["data"].get("split_file")
    if split_file:
        candidates = []
        path = Path(split_file)
        if path.is_absolute():
            candidates.append(path)
        else:
            candidates.extend([Path.cwd() / path, config_path.parent / path, root / path])
        for candidate in candidates:
            if candidate.exists():
                return json.loads(candidate.read_text(encoding="utf-8"))
        raise FileNotFoundError(f"Configured split_file was not found: {split_file}")
    return split_names(
        available_names(cfg, root), cfg["seed"],
        cfg["data"]["train_fraction"], cfg["data"]["val_fraction"],
    )


def make_training_sampler(cfg: dict, root: Path, train_names: list[str]):
    spec = cfg["training"].get("balanced_sampling")
    if not spec:
        return None
    global_df = pd.read_csv(root / cfg["data"]["global_csv"]).set_index("name")
    weights = []
    prefix_weights = {
        str(key): float(value) for key, value in spec.get("prefix_weights", {}).items()
    }
    porosity_ranges = spec.get("porosity_ranges", [])
    for name in train_names:
        weight = float(spec.get("base_weight", 1.0))
        prefix = str(name).split("_")[0]
        weight *= prefix_weights.get(prefix, 1.0)
        if name in global_df.index:
            porosity = float(global_df.loc[name, MACRO_COLUMNS[4]])
            for entry in porosity_ranges:
                low = float(entry["min"])
                high = float(entry["max"])
                if low <= porosity < high:
                    weight *= float(entry.get("weight", 1.0))
        weights.append(weight)
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def metric_value(metrics: dict[str, float], name: str) -> float:
    if name not in metrics:
        available = ", ".join(sorted(metrics))
        raise KeyError(f"Unknown training metric '{name}'. Available metrics: {available}")
    return float(metrics[name])


def write_training_history(history: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in history:
        row: dict[str, Any] = {"epoch": item["epoch"]}
        for split in ("train", "val"):
            for key, value in item[split].items():
                row[f"{split}_{key}"] = value
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "training_history.csv", index=False, encoding="utf-8-sig")

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return
    if frame.empty:
        return
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    axes[0].plot(frame["epoch"], frame["train_total"], label="train total", color="#1f77b4")
    axes[0].plot(frame["epoch"], frame["val_total"], label="val total", color="#d62728")
    if "train_base_total" in frame:
        axes[0].plot(frame["epoch"], frame["train_base_total"], "--", label="train base", color="#17becf")
        axes[0].plot(frame["epoch"], frame["val_base_total"], "--", label="val base", color="#ff7f0e")
    if "train_monitor_total" in frame:
        axes[0].plot(frame["epoch"], frame["train_monitor_total"], ":", label="train monitor", color="#2ca02c")
        axes[0].plot(frame["epoch"], frame["val_monitor_total"], ":", label="val monitor", color="#9467bd")
    axes[0].set_ylabel("loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    if "train_response_predictnet" in frame:
        axes[1].plot(
            frame["epoch"], frame["train_response_predictnet"],
            label="train raw response", color="#2ca02c",
        )
        axes[1].plot(
            frame["epoch"], frame["val_response_predictnet"],
            label="val raw response", color="#9467bd",
        )
    if "train_response_predictnet_weighted" in frame:
        axes[1].plot(
            frame["epoch"], frame["train_response_predictnet_weighted"],
            "--", label="train weighted response", color="#8c564b",
        )
        axes[1].plot(
            frame["epoch"], frame["val_response_predictnet_weighted"],
            "--", label="val weighted response", color="#e377c2",
        )
    axes[1].set_ylabel("PredictNet loss")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)

    if "train_response_predictnet_weight" in frame:
        axes[2].plot(
            frame["epoch"], frame["train_response_predictnet_weight"],
            label="response weight", color="#7f7f7f",
        )
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("weight")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_dir / "training_diagnostics.png", dpi=180)
    plt.close(fig)


def run_epoch(
    model, loader, optimizer, device, cfg, epoch, train: bool, geometry_context: dict,
    response_regularizer: FrozenPredictNetRegularizer | None = None,
) -> dict[str, float]:
    model.train(train)
    sums = {k: 0.0 for k in (
        "total", "monitor_total", "response_base_score", "base_total", "pore",
        "presence", "macro", "porosity", "spatial", "marginal",
        "response_predictnet", "response_predictnet_weighted",
        "response_predictnet_weight", "kl",
    )}
    kl_scale = min(1.0, (epoch + 1) / max(1, cfg["training"]["kl_warmup_epochs"]))
    response_base_weight = float(cfg["training"]["loss_weights"].get("response_predictnet", 0.0))
    response_base_score_weight = float(cfg["training"].get("response_base_score_weight", 0.05))
    response_start = int(cfg["training"].get("response_predictnet_start_epoch", 0))
    response_warmup = int(cfg["training"].get("response_predictnet_warmup_epochs", 1))
    response_weight = 0.0
    if response_base_weight > 0.0 and (epoch + 1) >= response_start:
        response_step = (epoch + 1) - response_start + 1
        response_weight = response_base_weight * min(1.0, response_step / max(1, response_warmup))
    measure_response = response_regularizer is not None
    for batch in loader:
        curve = batch["curve"].to(device)
        pores = batch["pores"].to(device)
        mask = batch["mask"].to(device)
        macro = batch["macro"].to(device)
        with torch.set_grad_enabled(train):
            outputs = model(curve, pores, mask, macro)
            loss_fn = poregen_loss_v2 if cfg["model"].get("architecture") == "slot_query_v2" else poregen_loss
            losses = loss_fn(outputs, pores, mask, macro,
                             cfg["training"]["loss_weights"], kl_scale,
                             geometry_context)
            losses["base_total"] = losses["total"]
            if measure_response:
                predictnet_target = batch["predictnet_response"].to(device)
                response_loss = torch.nn.functional.smooth_l1_loss(
                    response_regularizer(outputs), predictnet_target
                )
                losses["response_predictnet"] = response_loss
                losses["response_predictnet_weighted"] = response_weight * response_loss
                losses["total"] = losses["total"] + losses["response_predictnet_weighted"]
            else:
                zero = outputs["pores"].sum() * 0.0
                losses["response_predictnet"] = zero
                losses["response_predictnet_weighted"] = zero
            losses["monitor_total"] = (
                losses["base_total"] + response_base_weight * losses["response_predictnet"]
            )
            losses["response_base_score"] = (
                losses["response_predictnet"] + response_base_score_weight * losses["base_total"]
            )
            losses["response_predictnet_weight"] = outputs["pores"].new_tensor(response_weight)
            if train:
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        for key in sums:
            sums[key] += float(losses[key].detach())
    return {key: value / max(len(loader), 1) for key, value in sums.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke", action="store_true", help="one batch and one epoch")
    parser.add_argument(
        "--init-checkpoint",
        help="initialize model weights from an existing GeoLoop-PoreGen checkpoint before training",
    )
    args = parser.parse_args()
    cfg, root = load_config(args.config)
    seed_everything(int(cfg["seed"]))
    splits = load_or_make_splits(cfg, root, Path(args.config).resolve())
    if not splits["train"]:
        raise RuntimeError("No complete training samples were found.")
    if args.smoke:
        splits["train"] = splits["train"][:2]
        splits["val"] = splits["val"][:2] or splits["train"]
    normalizer = build_normalizer(cfg, root, splits["train"])
    state = normalizer.state
    geometry_context = {
        "set_loss_points": int(cfg["training"].get("set_loss_points", 256)),
        "set_selection": cfg["training"].get("set_selection", "topk_presence"),
        "kl_free_bits": float(cfg["training"].get("kl_free_bits", 0.0)),
        "volume_log_min": state.pore_min[5],
        "volume_log_max": state.pore_max[5],
        "cube_size_mm": state.cube_size_mm,
        "macro_porosity_min": state.macro_min[4],
        "macro_porosity_max": state.macro_max[4],
        "marginal_feature_weights": cfg["training"].get("marginal_feature_weights"),
    }
    checkpoint = Path(cfg["generation"]["checkpoint"])
    normalizer_path = Path(cfg["generation"]["normalizer"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    normalizer.save(normalizer_path)
    (checkpoint.parent / "split_names.json").write_text(
        json.dumps(splits, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    train_set = PoreDataset(cfg, root, splits["train"], normalizer)
    val_set = PoreDataset(cfg, root, splits["val"], normalizer)
    batch_size = 1 if args.smoke else cfg["training"]["batch_size"]
    sampler = None if args.smoke else make_training_sampler(cfg, root, splits["train"])
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=sampler is None, sampler=sampler, num_workers=0
    )
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    device = torch.device(args.device)
    model = make_model(cfg).to(device)
    if args.init_checkpoint:
        init_path = Path(args.init_checkpoint)
        checkpoint_state = torch.load(init_path, map_location=device)
        model_state = checkpoint_state.get("model_state", checkpoint_state)
        model.load_state_dict(model_state, strict=True)
        print(f"initialized model weights from {init_path}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"],
                                  weight_decay=cfg["training"]["weight_decay"])
    response_regularizer = None
    if float(cfg["training"]["loss_weights"].get("response_predictnet", 0.0)) > 0.0:
        response_regularizer = FrozenPredictNetRegularizer(cfg, root, state, device).to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5,
        patience=int(cfg["training"].get("lr_patience", 25)),
        min_lr=float(cfg["training"].get("min_learning_rate", 1e-6)),
    )
    epochs = 1 if args.smoke else (args.epochs or cfg["training"]["epochs"])
    checkpoint_start_epoch = int(cfg["training"].get("checkpoint_start_epoch", 1))
    checkpoint_metric = str(cfg["training"].get("checkpoint_metric", "total"))
    scheduler_metric = str(cfg["training"].get("scheduler_metric", checkpoint_metric))
    early_stopping_patience = cfg["training"].get("early_stopping_patience")
    early_stopping_patience = None if early_stopping_patience is None else int(early_stopping_patience)
    history, best, epochs_without_improvement = [], float("inf"), 0
    for epoch in range(epochs):
        tr = run_epoch(
            model, train_loader, optimizer, device, cfg, epoch, True,
            geometry_context, response_regularizer,
        )
        va = run_epoch(
            model, val_loader, optimizer, device, cfg, epoch, False,
            geometry_context, response_regularizer,
        )
        history.append({"epoch": epoch + 1, "train": tr, "val": va})
        ckpt_value = metric_value(va, checkpoint_metric)
        sched_value = metric_value(va, scheduler_metric)
        print(
            f"epoch={epoch+1} train={tr['total']:.5f} val={va['total']:.5f} "
            f"val_monitor={va['monitor_total']:.5f} "
            f"val_base={va['base_total']:.5f} val_resp={va['response_predictnet']:.5f} "
            f"resp_w={va['response_predictnet_weight']:.3f} "
            f"ckpt_{checkpoint_metric}={ckpt_value:.5f}",
            flush=True,
        )
        scheduler.step(sched_value)
        if (epoch + 1) >= checkpoint_start_epoch and ckpt_value < best:
            best = ckpt_value
            epochs_without_improvement = 0
            torch.save({"model_state": model.state_dict(), "config": cfg,
                        "model_name": cfg["model_name"], "release_version": cfg["release_version"],
                        "normalizer": str(normalizer_path), "epoch": epoch + 1,
                        "best_metric": checkpoint_metric, "best_metric_value": best}, checkpoint)
        elif (epoch + 1) >= checkpoint_start_epoch:
            epochs_without_improvement += 1
            if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
                print(
                    f"early stopping at epoch={epoch+1}; no {checkpoint_metric} improvement "
                    f"for {epochs_without_improvement} epochs",
                    flush=True,
                )
                break
    if not checkpoint.exists():
        torch.save({"model_state": model.state_dict(), "config": cfg,
                    "model_name": cfg["model_name"], "release_version": cfg["release_version"],
                    "normalizer": str(normalizer_path), "epoch": epochs}, checkpoint)
    (checkpoint.parent / "training_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_training_history(history, checkpoint.parent)


if __name__ == "__main__":
    main()
