#!/usr/bin/env python3
"""
BAS Payload Operations Copilot - Temporal Action Recognition Training
Trains lightweight GRU classifier on MediaPipe Pose + Dual-Hand sequences.

Evaluates and outputs:
- Overall Accuracy
- Macro F1 score
- Confusion Matrix
- Per-class Precision, Recall, and F1

Saves:
- models/bas_action_gru.pt
- models/label_map.json
- models/normalization.npz
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support
)

from train.ssv2_pipeline import CLASSES, LABEL_TO_IDX, IDX_TO_LABEL
from train.ssv2_pipeline.dataset import load_dataset_from_landmarks
from train.ssv2_pipeline.model import BASActionGRU, save_checkpoint


def compute_class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Computes balanced inverse-frequency weights for cross entropy loss."""
    counts = torch.bincount(labels, minlength=num_classes).float()
    total = len(labels)
    # Avoid div by zero if a class has 0 samples in train
    weights = total / (num_classes * torch.clamp(counts, min=1.0))
    # Normalize weights so mean is 1.0
    weights = weights / weights.mean()
    return weights


def evaluate_model(
    model: BASActionGRU,
    loader: DataLoader,
    criterion: nn.Module,
    device: str = "cpu"
) -> Tuple[float, float, float, np.ndarray, Dict]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * len(y)
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(y.cpu().numpy().tolist())

    avg_loss = total_loss / max(1, len(all_targets))
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(len(CLASSES))))
    
    # Per-class metrics
    p, r, f, s = precision_recall_fscore_support(
        all_targets, all_preds, labels=list(range(len(CLASSES))), zero_division=0
    )
    per_class = {
        CLASSES[i]: {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
            "support": int(s[i])
        }
        for i in range(len(CLASSES))
    }

    return avg_loss, acc, macro_f1, cm, per_class


def train(
    landmarks_dir: str = "dataset/landmarks",
    model_save_path: str = "models/bas_action_gru.pt",
    label_map_path: str = "models/label_map.json",
    norm_save_path: str = "models/normalization.npz",
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    patience: int = 12
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print("       BHARATIYA ANTARIKSH STATION (BAS) PAYLOAD COPILOT")
    print("          TEMPORAL ACTION RECOGNITION (SSV2) TRAINING")
    print("=" * 70)
    print(f"[*] Compute Device        : {device}")
    print(f"[*] Input Landmarks Dir   : {landmarks_dir}")
    print(f"[*] Model Checkpoint Path : {model_save_path}")
    print(f"[*] Normalization Path    : {norm_save_path}")
    print(f"[*] Label Map Path        : {label_map_path}")
    print(f"[*] Target Classes ({len(CLASSES)})   : {CLASSES}")
    print("-" * 70)

    # 1. Load data with strict Video-ID split and training-set-only normalization
    train_ds, val_ds, _, norm_stats = load_dataset_from_landmarks(
        landmarks_dir=landmarks_dir,
        val_ratio=0.20,
        random_state=42,
        normalization_save_path=norm_save_path
    )

    print(f"[OK] Video-ID level split completed:")
    print(f"     - Training samples   : {len(train_ds)} video sequences")
    print(f"     - Validation samples : {len(val_ds)} video sequences")
    print(f"[OK] Computed and saved training set normalization statistics.")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 2. Compute Class Weights for Imbalance
    train_labels = train_ds.labels
    class_weights = compute_class_weights(train_labels, len(CLASSES)).to(device)
    print(f"[*] Class Weights for Balanced Loss: {[round(w.item(), 3) for w in class_weights]}")

    # 3. Initialize Model, Criterion, Optimizer
    input_dim = train_ds.features.shape[-1]
    model = BASActionGRU(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=len(CLASSES),
        dropout=dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-5
    )

    best_val_f1 = -1.0
    best_cm = None
    best_per_class = None
    epochs_no_improve = 0

    print("-" * 70)
    print(f"{'Epoch':<6} | {'Train Loss':<10} | {'Val Loss':<10} | {'Val Acc':<9} | {'Val Macro F1':<12} | Status")
    print("-" * 70)

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0

        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            total_train_loss += loss.item() * len(y)

        train_loss = total_train_loss / max(1, len(train_ds))
        val_loss, val_acc, val_f1, val_cm, per_class = evaluate_model(
            model, val_loader, criterion, device=device
        )
        scheduler.step(val_f1)

        improved = val_f1 > best_val_f1
        if improved:
            best_val_f1 = val_f1
            best_cm = val_cm
            best_per_class = per_class
            epochs_no_improve = 0
            # Save best checkpoint
            save_checkpoint(model, save_path=model_save_path, label_map_path=label_map_path)
            status = f"* BEST (Saved)"
        else:
            epochs_no_improve += 1
            status = f"Patience {epochs_no_improve}/{patience}"

        print(f"{epoch:<6} | {train_loss:<10.4f} | {val_loss:<10.4f} | {val_acc:<9.4f} | {val_f1:<12.4f} | {status}")

        if epochs_no_improve >= patience:
            print(f"[*] Early stopping triggered after {epoch} epochs (Patience: {patience}).")
            break

    # 4. Final Verification and Comprehensive Report
    print("=" * 70)
    print("                  FINAL VALIDATION REPORT")
    print("=" * 70)
    print(f"[*] Top Validation Macro F1 Score : {best_val_f1:.4f}")
    print("\n--- PER-CLASS PRECISION / RECALL / F1 ---")
    header = f"{'Class Name':<20} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}"
    print(header)
    print("-" * len(header))
    if best_per_class:
        for cname, m in best_per_class.items():
            print(f"{cname:<20} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['f1']:<10.4f} | {m['support']:<8d}")

    print("\n--- CONFUSION MATRIX ---")
    # Pretty print confusion matrix
    cm_header = "Actual \\ Pred    " + "".join([f"{c[:7]:>9}" for c in CLASSES])
    print(cm_header)
    print("-" * len(cm_header))
    if best_cm is not None:
        for idx, row in enumerate(best_cm):
            row_str = f"{CLASSES[idx][:15]:<16} " + "".join([f"{val:>9d}" for val in row])
            print(row_str)

    print("=" * 70)
    print("[OK] Artifacts Saved Successfully:")
    print(f"    - Model Weights    : {os.path.abspath(model_save_path)}")
    print(f"    - Label Map        : {os.path.abspath(label_map_path)}")
    print(f"    - Normalization    : {os.path.abspath(norm_save_path)}")
    print("=" * 70)
    print("[DISCLAIMER] 'inspect_proxy' is a temporary proxy from SSV2 'Holding something'.")
    print("             Decoupled architecture: Perception -> Action Rec -> BAS Procedure State Machine.")
    print("=" * 70)


def main():
    ap = argparse.ArgumentParser(description="Train BAS Temporal Action Recognition GRU.")
    ap.add_argument("--landmarks_dir", default="dataset/landmarks", help="Path to landmark .npz directory")
    ap.add_argument("--model_out", default="models/bas_action_gru.pt", help="Path to output model file")
    ap.add_argument("--label_map_out", default="models/label_map.json", help="Path to label map output")
    ap.add_argument("--norm_out", default="models/normalization.npz", help="Path to normalization output")
    ap.add_argument("--epochs", type=int, default=50, help="Max training epochs")
    ap.add_argument("--batch_size", type=int, default=16, help="Batch size")
    ap.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    ap.add_argument("--hidden_dim", type=int, default=128, help="GRU hidden dimension")
    ap.add_argument("--num_layers", type=int, default=2, help="Number of GRU layers")
    ap.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")
    ap.add_argument("--patience", type=int, default=12, help="Early stopping patience")
    args = ap.parse_args()

    train(
        landmarks_dir=args.landmarks_dir,
        model_save_path=args.model_out,
        label_map_path=args.label_map_out,
        norm_save_path=args.norm_out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        patience=args.patience
    )


if __name__ == "__main__":
    main()
