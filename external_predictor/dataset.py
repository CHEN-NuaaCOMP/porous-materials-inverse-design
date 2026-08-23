import os
import copy
import torch.utils.data as data
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib
import torch
import numpy as np
import pandas as pd


def insert_outliers(fine, coarse, labels, outlier_ratio=0.1, outlier_range=(1, 4)):
    # Insert synthetic outliers.
    length = len(fine)
    for i in range(length):
        outlier_num = int(fine[i].shape[0] * outlier_ratio)
        outlier_index = np.random.choice(fine[i].shape[0], outlier_num, replace=False)
        outlier_value = np.random.uniform(outlier_range[0], outlier_range[1], (outlier_num, fine[i].shape[1]))
        fine[i][outlier_index] += outlier_value

    outlier_num = int(coarse.shape[0] * outlier_ratio)
    outlier_index = np.random.choice(coarse.shape[0], outlier_num, replace=False)
    outlier_value = np.random.uniform(outlier_range[0], outlier_range[1], (outlier_num, coarse.shape[1]))
    coarse[outlier_index] += outlier_value

    for i in range(len(fine)):
        sigma = outlier_ratio * 0.2
        clip = sigma * 2
        labels[i] += np.clip(sigma * np.random.randn(*labels[i].shape), -1 * clip, clip)
    return fine, coarse, labels


def data_jitter(fine, coarse, label, sigma=0.01):
    # for i in range(int(len(fine)/3), len(fine)):
    clip = sigma * 2
    for i in range(len(fine)):
        fine[i] += np.clip(sigma * np.random.randn(*fine[i].shape), -1 * clip, clip)
        coarse[i] += np.clip(sigma * np.random.randn(*coarse[i].shape), -1 * clip, clip)
        label[i] += np.clip(sigma * np.random.randn(*label[i].shape), -1 * clip, clip)
    return fine, coarse, label


def data_augmentation(fine, coarse, labels, drop=0.1):
    length = len(fine)
    # Swap x and y as a mirror transformation.
    for i in range(length):
        transformed = copy.deepcopy(fine[i])
        transformed[:, [0, 1]] = transformed[:, [1, 0]]
        fine.append(transformed)
        # Keep phi unchanged and reflect theta on its periodic interval.
        fine[-1][:, 7] = (0.9 - fine[-1][:, 7] + 3.6) % 3.6
        coarse = np.append(coarse, coarse[i].reshape(1, -1), axis=0)
        labels = np.append(labels, labels[i].reshape(1, -1), axis=0)

    # Apply central symmetry about the 14 mm cube center.
    for i in range(length):
        fine.append(copy.deepcopy(fine[i]))
        fine[-1][:, :3] = 14 - fine[-1][:, :3]
        # Keep phi unchanged and rotate theta by half a period.
        fine[-1][:, 7] = (fine[-1][:, 7] + 1.8) % 3.6
        coarse = np.append(coarse, coarse[i].reshape(1, -1), axis=0)
        labels = np.append(labels, labels[i].reshape(1, -1), axis=0)

    # Randomly remove a fraction of pore rows.
    if drop > 0:
        for i in range(len(fine)):
            # Delete selected rows.
            drop_num = int(fine[i].shape[0] * drop)
            drop_index = np.random.choice(fine[i].shape[0], drop_num, replace=False)
            fine[i] = np.delete(fine[i], drop_index, axis=0)

    return fine, coarse, labels


def fill_and_sample(fines, num=1024, fill='zero'):
    for i, fine in enumerate(fines):
        if fine.shape[0] >= num:
            sample = np.random.choice(fine.shape[0], num, replace=False)
            fine = fine[sample]
        else:
            add_dim = num - fine.shape[0]
            if fill == 'zero':
                zero_np = np.zeros((add_dim, fine.shape[1]))
                fine = np.concatenate((fine, zero_np), axis=0)
            elif fill == 'repeat':
                repeat_np = np.repeat(fine, add_dim // fine.shape[0] + 1, axis=0)
                fine = np.concatenate((fine, repeat_np[:add_dim]), axis=0)
        fines[i] = fine
    fine = np.array(fines, dtype=np.float32)
    return fine


LEGACY_LABEL_MODE = "per_curve_log_minmax"
GLOBAL_LABEL_MODE = "global_log1p_standard"


def _normalize_labels_per_curve_log_minmax(labels, part, root):
    labels = np.asarray(labels, dtype=np.float32)
    eps = 1e-8
    labels_safe = np.clip(labels, eps, None)
    min_vals = np.min(labels_safe, axis=1, keepdims=True)
    max_vals = np.max(labels_safe, axis=1, keepdims=True)
    log_min = np.log(min_vals)
    log_max = np.log(max_vals)
    denom = log_max - log_min
    denom[denom == 0] = 1.0
    labels = (np.log(labels_safe) - log_min) / denom
    if part == 'test':
        joblib.dump({'min_vals': min_vals, 'max_vals': max_vals}, Path(root) / 'min_max_vals.bin')
    return labels.astype(np.float32)


def _normalize_labels_global_log1p(labels, part, root, scalers):
    labels = np.asarray(labels, dtype=np.float32)
    log_labels = np.log1p(np.clip(labels, 0.0, None))
    if part == 'train':
        label_scaler = StandardScaler()
        normalized = label_scaler.fit_transform(log_labels)
        scalers['scaler_label'] = label_scaler
        scalers['label_mode'] = GLOBAL_LABEL_MODE
        scalers['label_transform'] = 'log1p'
    else:
        label_mode = scalers.get('label_mode', LEGACY_LABEL_MODE)
        if label_mode != GLOBAL_LABEL_MODE or 'scaler_label' not in scalers:
            raise RuntimeError(
                f"{Path(root) / 'scalers.bin'} does not contain a train-fitted "
                f"'{GLOBAL_LABEL_MODE}' scaler_label"
            )
        normalized = scalers['scaler_label'].transform(log_labels)
    return normalized.astype(np.float32)


def data_normalize(fine, coarse, labels, label_norm, part, root, label_mode=LEGACY_LABEL_MODE):
    root = Path(root)
    if label_mode not in {LEGACY_LABEL_MODE, GLOBAL_LABEL_MODE}:
        raise ValueError(f"Unknown label_mode: {label_mode}")
    for i in range(len(fine)):  # Anisotropy and sphericity remain unscaled.
        # Columns 0-2 are coordinates normalized by the cube width.
        fine[i][:, :3] /= 14
        # Columns 3-4 use log1p scaling with base 200.
        fine[i][:, 3:5] = np.log1p(fine[i][:, 3:5]) / np.log(200)
        # Normalize phi by 0.9 and theta on the interval [0, 3.6).
        fine[i][:, 6] /= 0.9
        fine[i][:, 7] /= 3.6

    # The Euler-number column is removed; global features use MinMax scaling.
    if part == 'train':
        # feature_scaler = MinMaxScaler()
        # all_fine_5 = np.concatenate([x[:, 5].reshape(-1, 1) for x in fine], axis=0)
        # feature_scaler.fit(all_fine_5)
        # for i in range(len(fine)):
        #     fine[i][:, 5] = feature_scaler.transform(fine[i][:, 5].reshape(-1, 1)).flatten()
        global_scaler = MinMaxScaler()
        coarse = global_scaler.fit_transform(coarse)
        scalers = {
            # 'scaler_fine': feature_scaler,
            'scaler_coarse': global_scaler,
            'label_mode': label_mode,
        }
    else:
        scaler_path = root / 'scalers.bin'
        assert scaler_path.exists(), f'{scaler_path} not found, run train dataset first'
        scalers = joblib.load(scaler_path)
        # for i in range(len(fine)):
        #     fine[i][:, 5] = scalers['scaler_fine'].transform(fine[i][:, 5].reshape(-1, 1)).flatten()
        coarse = scalers['scaler_coarse'].transform(coarse)

    if label_norm:
        if label_mode == GLOBAL_LABEL_MODE:
            labels = _normalize_labels_global_log1p(labels, part, root, scalers)
        else:
            labels = _normalize_labels_per_curve_log_minmax(labels, part, root)

    if part == 'train':
        root.mkdir(parents=True, exist_ok=True)
        joblib.dump(scalers, root / 'scalers.bin')

    return fine, coarse, labels


class FoamAlDataset(data.Dataset):
    def __init__(self, root=Path('dataset/'), label_norm=True, part='train', npoints=1024, augmentation=True,
                 drop=0.1, jitter=0.01, outlier_ratio=0.1, label_mode=LEGACY_LABEL_MODE,
                 return_max_stress=False):
        if label_mode not in {LEGACY_LABEL_MODE, GLOBAL_LABEL_MODE}:
            raise ValueError(f"Unknown label_mode: {label_mode}")
        self.label_mode = label_mode
        self.return_max_stress = return_max_stress
        self.root = Path(root) / part
        fine_path = self.root / 'local'
        coarse_path = self.root / "global.csv"
        curve_path = self.root / "energy"

        global_frame = pd.read_csv(str(coarse_path))
        if "name" not in global_frame.columns:
            raise ValueError(f"{coarse_path} must contain a 'name' column")

        def global_feature_values(row):
            """Return wall-thickness, fractal-dimension and porosity features."""

            aliases = (
                ("avg_cell_thickness", "mean", "average_cell_thickness"),
                ("fractal_dimension", "fractal dimension"),
                ("volume_fraction", "porosity", "volume fraction"),
            )
            columns = []
            for candidates in aliases:
                match = next((name for name in candidates if name in row.index), None)
                if match is None:
                    columns = []
                    break
                columns.append(match)
            if columns:
                return row.loc[columns].to_numpy(dtype=np.float32)
            if len(row) > 5:
                return row.iloc[[1, 4, 5]].to_numpy(dtype=np.float32)
            raise ValueError(
                "global.csv must provide wall-thickness/mean, fractal-dimension "
                "and porosity/volume-fraction columns"
            )

        names = []
        coarse_rows = []
        labels = []
        fine = []
        aliases = {
            "x": ("CenterX(mm)", "BaryCenterX(mm)"),
            "y": ("CenterY(mm)", "BaryCenterY(mm)"),
            "z": ("CenterZ(mm)", "BaryCenterZ(mm)"),
            "area": ("Area(mm^2)", "Area3d(mm^2)"),
            "volume": ("Volume(mm^3)", "Volume3d(mm^3)"),
            "anisotropy": ("Anisotropy",),
            "phi": ("OrientationPhi",),
            "theta": ("OrientationTheta",),
            "sphericity": ("Sphericity",),
        }

        def resolve_column(frame, semantic_name):
            for candidate in aliases[semantic_name]:
                if candidate in frame.columns:
                    return candidate
            raise ValueError(
                f"Missing {semantic_name} column in {list(frame.columns)}; "
                f"accepted names are {aliases[semantic_name]}"
            )

        for _, row in global_frame.iterrows():
            name = str(row["name"])
            local_file = fine_path / f"{name}.csv"
            curve_file = curve_path / f"{name}.csv"
            if not local_file.exists() or not curve_file.exists():
                continue
            names.append(name)
            coarse_rows.append(global_feature_values(row))
            labels.append(np.array(pd.read_csv(str(curve_file)).values[:, 1], dtype=np.float32))
            local_frame = pd.read_csv(str(local_file))
            columns = [
                resolve_column(local_frame, semantic_name)
                for semantic_name in (
                    "x", "y", "z", "area", "volume",
                    "anisotropy", "phi", "theta", "sphericity",
                )
            ]
            fine.append(local_frame.loc[:, columns].to_numpy(dtype=np.float32))

        if not names:
            raise RuntimeError(f"No aligned samples found under {self.root}")

        coarse = np.array(coarse_rows, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)

        # Map theta from (-1.8, 1.8) to the periodic interval [0, 3.6).
        for i in range(len(fine)):
            fine[i][:, 7] = (fine[i][:, 7] + 3.6) % 3.6

        # Data augmentation.
        if part == 'train' and augmentation:
            fine, coarse, labels = data_augmentation(fine, coarse, labels, drop=drop)
            names = names + [f"{name}__swap_xy" for name in names] + [f"{name}__center_flip" for name in names]

        raw_labels = np.asarray(labels, dtype=np.float32)
        max_stress = np.max(raw_labels, axis=1).astype(np.float32)

        # Normalization.
        fine, coarse, labels = data_normalize(
            fine, coarse, labels, label_norm, part, root, label_mode=label_mode
        )

        # Optional perturbation.
        if part == 'train' and augmentation and jitter > 0:
            fine, coarse, labels = data_jitter(fine, coarse, labels, sigma=jitter)
        if part == 'train' and augmentation and outlier_ratio > 0:
            fine, coarse, labels = insert_outliers(fine, coarse, labels, outlier_ratio=outlier_ratio)

        # Pad or sample to a fixed row count.
        fine = fill_and_sample(fine, npoints, fill='zero')  # (M, N, 10)

        # Convert arrays to tensors.
        self.fine = torch.from_numpy(fine)
        self.coarse = torch.from_numpy(coarse)
        self.labels = torch.from_numpy(labels)
        self.raw_labels = torch.from_numpy(raw_labels)
        self.max_stress = torch.from_numpy(max_stress)
        self.names = names
        self.f_channel = self.fine.shape[2]
        self.c_channel = self.coarse.shape[1]
        self.label_channel = self.labels.shape[1]

    def __getitem__(self, index):
        # (M, N, 9)
        # (M, N, 3)
        # (M, N, 50)
        if self.return_max_stress:
            return self.fine[index], self.coarse[index], self.labels[index], self.max_stress[index]
        return self.fine[index], self.coarse[index], self.labels[index]

    def __len__(self):
        return len(self.fine)


if __name__ == '__main__':
    train_dataset = FoamAlDataset(part='train')
    print(train_dataset[0])
    test_dataset = FoamAlDataset(part='test')
    print(test_dataset[0])
    print(train_dataset[0][0].shape, train_dataset[0][1].shape, train_dataset[0][2].shape)
