"""Deterministic loading, interpolation, ordering, padding, and masks."""

from __future__ import annotations

from pathlib import Path
import ast

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:  # permits schema/round-trip tools in data-only environments
    torch = None

    class Dataset:  # type: ignore[no-redef]
        pass

from src.utils.normalization import FeatureNormalizer, MACRO_COLUMNS


RAW_PORE_COLUMNS = [
    "CenterX(mm)", "CenterY(mm)", "CenterZ(mm)", "EqDiameter(mm)",
    "Area(mm^2)", "Volume(mm^3)", "Anisotropy",
    "OrientationPhi", "OrientationTheta", "Sphericity",
]

LEGACY_PREDICTNET_LABEL_MODE = "per_curve_log_minmax"
GLOBAL_PREDICTNET_LABEL_MODE = "global_log1p_standard"


def _scalar(value: str):
    value = value.strip()
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.startswith("[") and value.endswith("]"):
        return [item.strip().strip("'\"") for item in value[1:-1].split(",") if item.strip()]
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _load_yaml_compatible(text: str) -> dict:
    try:
        import yaml
        return yaml.safe_load(text)
    except ModuleNotFoundError:
        root: dict = {}
        stack: list[tuple[int, dict]] = [(-1, root)]
        for raw in text.splitlines():
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            indent = len(raw) - len(raw.lstrip(" "))
            key, value = raw.strip().split(":", 1)
            while stack[-1][0] >= indent:
                stack.pop()
            parent = stack[-1][1]
            if value.strip():
                parent[key] = _scalar(value)
            else:
                parent[key] = {}
                stack.append((indent, parent[key]))
        return root


def load_config(path: str | Path) -> tuple[dict, Path]:
    path = Path(path).resolve()
    cfg = _load_yaml_compatible(path.read_text(encoding="utf-8"))
    project_root = (path.parent / cfg["project_root"]).resolve()
    return cfg, project_root


def resolve_project_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _paths(cfg: dict, root: Path) -> tuple[Path, Path, Path]:
    return (
        root / cfg["data"]["global_csv"],
        root / cfg["data"]["pore_dir"],
        root / cfg["data"]["curve_dir"],
    )


def available_names(cfg: dict, root: Path) -> list[str]:
    global_csv, pore_dir, curve_dir = _paths(cfg, root)
    global_names = set(pd.read_csv(global_csv)["name"].astype(str))
    pore_names = {p.stem for p in pore_dir.glob("*.csv")}
    curve_names = {p.stem for p in curve_dir.glob("*.csv")}
    return sorted(global_names & pore_names & curve_names)


def split_names(names: list[str], seed: int, train_fraction: float, val_fraction: float) -> dict[str, list[str]]:
    rng = np.random.default_rng(seed)
    names = np.array(sorted(names), dtype=object)
    rng.shuffle(names)
    n_train = int(len(names) * train_fraction)
    n_val = int(len(names) * val_fraction)
    return {"train": names[:n_train].tolist(), "val": names[n_train:n_train+n_val].tolist(),
            "test": names[n_train+n_val:].tolist(), "all": names.tolist()}


def read_pores(path: Path) -> np.ndarray:
    frame = pd.read_csv(path)
    missing = [c for c in RAW_PORE_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    values = frame[RAW_PORE_COLUMNS].to_numpy(np.float32)
    # Deterministic spatial ordering: lexicographic proxy for a Morton ordering.
    order = np.lexsort((values[:, 2], values[:, 1], values[:, 0]))
    return values[order]


def read_curve(
    path: Path,
    columns: list[str],
    points: int,
    strain_min: float = 0.0,
    strain_max: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:
    """Read a response curve on the fixed manuscript strain grid.

    Every learning sample is evaluated at the same 50 physical strain nodes
    over 0--0.6.  Curves that do not cover that interval are rejected instead
    of being silently stretched to their individual endpoints.
    """

    frame = pd.read_csv(path)
    if not set(columns).issubset(frame.columns):
        raise ValueError(f"{path} must contain {columns}, found {list(frame.columns)}")
    clean = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce").dropna()
    clean = clean.sort_values(columns[0]).drop_duplicates(columns[0], keep="last")
    if len(clean) < 2:
        raise ValueError(f"{path} must contain at least two distinct strain nodes")
    strain = clean[columns[0]].to_numpy(float)
    response = clean[columns[1]].to_numpy(float)
    tolerance = 1e-8
    if strain[0] > strain_min + tolerance or strain[-1] < strain_max - tolerance:
        raise ValueError(
            f"{path} covers strain [{strain[0]:.8g}, {strain[-1]:.8g}], "
            f"but the manuscript grid requires [{strain_min:.8g}, {strain_max:.8g}]"
        )
    grid = np.linspace(float(strain_min), float(strain_max), int(points))
    return grid, np.interp(grid, strain, response)


def predictnet_label(response: np.ndarray) -> np.ndarray:
    """Per-curve log min-max label used by the validated external PredictNet."""

    eps = 1e-8
    safe = np.clip(np.asarray(response, dtype=np.float32), eps, None)
    log_min = np.log(np.min(safe))
    log_max = np.log(np.max(safe))
    denom = log_max - log_min
    if abs(float(denom)) < eps:
        denom = 1.0
    return ((np.log(safe) - log_min) / denom).astype(np.float32)


def predictnet_label_mode(cfg: dict, scalers: dict | None = None) -> str:
    return str(
        cfg.get("screening", {}).get(
            "porenet_label_mode",
            (scalers or {}).get("label_mode", LEGACY_PREDICTNET_LABEL_MODE),
        )
    )


def load_predictnet_scalers(cfg: dict, root: Path) -> dict:
    scaler_value = cfg.get("screening", {}).get("porenet_scalers")
    if not scaler_value:
        raise KeyError("screening.porenet_scalers is required for global PredictNet labels")
    scaler_path = resolve_project_path(root, scaler_value)
    if not scaler_path.exists():
        raise FileNotFoundError(scaler_path)
    from joblib import load

    return load(scaler_path)


def predictnet_global_log1p_standard_label(response: np.ndarray, scalers: dict) -> np.ndarray:
    if "scaler_label" not in scalers:
        raise RuntimeError("scalers.bin does not contain scaler_label for global_log1p_standard")
    if scalers.get("label_mode") not in {None, GLOBAL_PREDICTNET_LABEL_MODE}:
        raise RuntimeError(f"Unexpected PredictNet label_mode in scalers.bin: {scalers.get('label_mode')}")
    safe = np.clip(np.asarray(response, dtype=np.float32), 0.0, None)
    log_stress = np.log1p(safe).reshape(1, -1)
    return scalers["scaler_label"].transform(log_stress)[0].astype(np.float32)


def predictnet_target_label(
    response: np.ndarray,
    cfg: dict,
    root: Path,
    scalers: dict | None = None,
) -> np.ndarray:
    explicit_mode = cfg.get("screening", {}).get("porenet_label_mode")
    active_scalers = scalers
    if active_scalers is None and cfg.get("screening", {}).get("porenet_scalers"):
        try:
            active_scalers = load_predictnet_scalers(cfg, root)
        except (FileNotFoundError, KeyError):
            if explicit_mode == GLOBAL_PREDICTNET_LABEL_MODE:
                raise
    mode = predictnet_label_mode(cfg, active_scalers)
    if mode == GLOBAL_PREDICTNET_LABEL_MODE:
        if active_scalers is None:
            active_scalers = load_predictnet_scalers(cfg, root)
        return predictnet_global_log1p_standard_label(response, active_scalers)
    if mode == LEGACY_PREDICTNET_LABEL_MODE:
        return predictnet_label(response)
    raise ValueError(f"Unknown PredictNet label mode: {mode}")


def build_normalizer(cfg: dict, root: Path, names: list[str]) -> FeatureNormalizer:
    global_csv, pore_dir, curve_dir = _paths(cfg, root)
    global_df = pd.read_csv(global_csv).set_index("name")
    pores, macros, responses, strains = [], [], [], []
    for name in names:
        pores.append(read_pores(pore_dir / f"{name}.csv"))
        macros.append(global_df.loc[name, MACRO_COLUMNS].to_numpy(float))
        strain, response = read_curve(
            curve_dir / f"{name}.csv",
            cfg["data"]["curve_columns"],
            cfg["data"]["curve_points"],
            cfg["data"].get("strain_min", 0.0),
            cfg["data"].get("strain_max", 0.6),
        )
        strains.append(strain); responses.append(response)
    return FeatureNormalizer.fit(
        pores, np.asarray(macros), responses, strains,
        cfg["data"]["cube_size_mm"], cfg["data"]["curve_kind"],
        cfg.get("normalization", {}).get("response_transform", "linear"),
        cfg.get("normalization", {}).get("response_scale_quantile", 0.75),
    )


class PoreDataset(Dataset):
    def __init__(self, cfg: dict, root: Path, names: list[str], normalizer: FeatureNormalizer):
        self.cfg, self.root, self.names, self.normalizer = cfg, root, names, normalizer
        self.global_csv, self.pore_dir, self.curve_dir = _paths(cfg, root)
        self.global_df = pd.read_csv(self.global_csv).set_index("name")
        self.max_pores = int(cfg["data"]["max_pores"])
        self.predictnet_scalers = None
        if cfg.get("screening", {}).get("porenet_scalers"):
            self.predictnet_scalers = load_predictnet_scalers(cfg, root)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        if torch is None:
            raise RuntimeError("PyTorch is required for PoreDataset tensor loading")
        name = self.names[index]
        raw = read_pores(self.pore_dir / f"{name}.csv")
        if len(raw) > self.max_pores:
            # Preserve the largest-volume pores deterministically if capacity is exceeded.
            keep = np.argsort(raw[:, 5])[-self.max_pores:]
            raw = raw[np.sort(keep)]
        normalized = self.normalizer.transform_pores(raw)
        padded = np.zeros((self.max_pores, normalized.shape[1]), dtype=np.float32)
        mask = np.zeros(self.max_pores, dtype=bool)
        padded[:len(normalized)] = normalized
        mask[:len(normalized)] = True
        strain, response = read_curve(
            self.curve_dir / f"{name}.csv", self.cfg["data"]["curve_columns"],
            self.cfg["data"]["curve_points"],
            self.cfg["data"].get("strain_min", 0.0),
            self.cfg["data"].get("strain_max", 0.6),
        )
        predictnet_points = int(
            self.cfg.get("screening", {}).get("porenet_curve_points", 50)
        )
        _, predictnet_response = read_curve(
            self.curve_dir / f"{name}.csv",
            self.cfg["data"]["curve_columns"],
            predictnet_points,
            self.cfg["data"].get("strain_min", 0.0),
            self.cfg["data"].get("strain_max", 0.6),
        )
        curve = self.normalizer.transform_curve(strain, response)
        macro = self.global_df.loc[name, MACRO_COLUMNS].to_numpy(np.float32)
        return {
            "name": name,
            "curve": torch.from_numpy(curve),
            "pores": torch.from_numpy(padded),
            "mask": torch.from_numpy(mask),
            "macro": torch.from_numpy(self.normalizer.transform_macro(macro)),
            "predictnet_response": torch.from_numpy(
                predictnet_target_label(predictnet_response, self.cfg, self.root, self.predictnet_scalers)
            ),
        }
