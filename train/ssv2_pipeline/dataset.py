"""
BAS Action Sequence Dataset & Strict Video-ID Splitter
- Strictly splits by VIDEO ID, preventing any temporal data leakage.
- Computes mean and standard deviation from the TRAINING SET ONLY.
- Applies z-score normalization and saves normalization.npz.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from train.ssv2_pipeline import CLASSES, LABEL_TO_IDX


class ActionSequenceDataset(Dataset):
    """
    PyTorch Dataset yielding (features, label_idx, video_id).
    features: (seq_len, 227)
    label_idx: int [0..4]
    """
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        video_ids: List[str]
    ):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.video_ids = video_ids

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx], self.video_ids[idx]


def load_dataset_from_landmarks(
    landmarks_dir: str = "dataset/landmarks",
    val_ratio: float = 0.20,
    test_ratio: float = 0.0,
    random_state: int = 42,
    normalization_save_path: str = "models/normalization.npz"
) -> Tuple[ActionSequenceDataset, ActionSequenceDataset, Optional[ActionSequenceDataset], Dict[str, np.ndarray]]:
    """
    Loads all landmark files from landmarks_dir.
    Splits strictly by VIDEO ID.
    Calculates normalization statistics on TRAINING SET ONLY.
    Saves normalization.npz and returns PyTorch Dataset objects.
    """
    lm_path = Path(landmarks_dir)
    npz_files = list(lm_path.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No landmark .npz files found in: {lm_path.resolve()}")

    video_ids = []
    labels_raw = []
    features_list = []

    for f in npz_files:
        data = np.load(f, allow_pickle=True)
        lbl = str(data["label"])
        if lbl in LABEL_TO_IDX:
            video_ids.append(str(data["video_id"]))
            labels_raw.append(LABEL_TO_IDX[lbl])
            # Ensure shape is (8, 227)
            feat = data["features"]
            features_list.append(feat)

    if not features_list:
        raise ValueError("No matching BAS action samples found in landmark directory.")

    all_features = np.array(features_list, dtype=np.float32)  # (N, seq_len, 227)
    all_labels = np.array(labels_raw, dtype=np.int64)

    # 1. Split strictly by VIDEO ID
    indices = np.arange(len(video_ids))
    try:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_ratio,
            stratify=all_labels,
            random_state=random_state
        )
    except ValueError:
        # Fallback if some classes have very few samples for stratification
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_ratio,
            random_state=random_state
        )

    # Separate train and val sets
    train_features_raw = all_features[train_idx]
    train_labels = all_labels[train_idx]
    train_vids = [video_ids[i] for i in train_idx]

    val_features_raw = all_features[val_idx]
    val_labels = all_labels[val_idx]
    val_vids = [video_ids[i] for i in val_idx]

    # 2. Compute Normalization Statistics on TRAINING SET ONLY
    # Flatten over samples and sequence length: (N_train * seq_len, 227)
    train_flat = train_features_raw.reshape(-1, train_features_raw.shape[-1])
    mean = np.mean(train_flat, axis=0)
    std = np.std(train_flat, axis=0)
    # Prevent division by zero for unobserved or invariant coordinates
    std[std < 1e-6] = 1.0

    # Save normalization stats
    os.makedirs(os.path.dirname(normalization_save_path), exist_ok=True)
    np.savez_compressed(normalization_save_path, mean=mean, std=std)

    # 3. Apply normalization using training stats
    def normalize(data: np.ndarray) -> np.ndarray:
        return (data - mean) / std

    train_norm = normalize(train_features_raw)
    val_norm = normalize(val_features_raw)

    train_ds = ActionSequenceDataset(train_norm, train_labels, train_vids)
    val_ds = ActionSequenceDataset(val_norm, val_labels, val_vids)

    norm_stats = {"mean": mean, "std": std}
    return train_ds, val_ds, None, norm_stats
