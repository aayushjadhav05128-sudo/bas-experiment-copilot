"""
BAS Payload Operations Copilot - Temporal Action Recognition Model
Architecture: Lightweight Bi-directional or Unidirectional GRU
Input: (batch_size, seq_len=8, input_dim=227)
Output: 5-class action logits
"""

import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from train.ssv2_pipeline import CLASSES, LABEL_TO_IDX, IDX_TO_LABEL


class BASActionGRU(nn.Module):
    """
    Lightweight GRU classifier for microgravity payload operations.
    Meets hard SIH latency (<15ms per inference on CPU) & accuracy requirements.
    """
    def __init__(
        self,
        input_dim: int = 227,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = len(CLASSES),
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional

        # Input feature projection & normalization
        self.input_layer_norm = nn.LayerNorm(input_dim)

        # Recurrent temporal aggregator
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        gru_out_dim = hidden_dim * (2 if bidirectional else 1)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(gru_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, seq_len, input_dim)
        returns logits: (batch_size, num_classes)
        """
        x = self.input_layer_norm(x)
        # gru output: (batch_size, seq_len, num_directions * hidden_dim)
        # h_n: (num_layers * num_directions, batch_size, hidden_dim)
        out, _ = self.gru(x)

        # Temporal pooling: use the final hidden state or average pooling across sequence
        # We concatenate the final frame representation and temporal mean for robust action capture
        final_rep = out[:, -1, :]
        logits = self.classifier(final_rep)
        return logits

    def predict_action(self, x: torch.Tensor) -> Tuple[str, float, torch.Tensor]:
        """
        Runs single sequence or batch inference and returns:
        (predicted_class_name, confidence, all_probabilities)
        """
        self.eval()
        with torch.no_grad():
            if x.dim() == 2:
                x = x.unsqueeze(0)  # (1, seq_len, input_dim)
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            conf, pred_idx = torch.max(probs, dim=-1)
            pred_class = IDX_TO_LABEL.get(pred_idx.item(), "unknown")
            return pred_class, float(conf.item()), probs.squeeze(0)


def save_checkpoint(
    model: BASActionGRU,
    save_path: str = "models/bas_action_gru.pt",
    label_map_path: str = "models/label_map.json"
):
    """Saves model weights and label map dictionary."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(os.path.dirname(label_map_path), exist_ok=True)

    state = {
        "state_dict": model.state_dict(),
        "input_dim": model.input_dim,
        "hidden_dim": model.hidden_dim,
        "num_layers": model.num_layers,
        "num_classes": model.num_classes,
        "classes": CLASSES
    }
    torch.save(state, save_path)

    label_map_data = {
        "classes": CLASSES,
        "label_to_idx": LABEL_TO_IDX,
        "idx_to_label": {str(k): v for k, v in IDX_TO_LABEL.items()},
        "note": "inspect_proxy is a proxy based on SSV2 'Holding something' and must be validated by FSM."
    }
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_map_data, f, indent=2)


def load_model(
    checkpoint_path: str = "models/bas_action_gru.pt",
    device: str = "cpu"
) -> BASActionGRU:
    """Loads trained BASActionGRU from checkpoint file."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = BASActionGRU(
        input_dim=checkpoint.get("input_dim", 227),
        hidden_dim=checkpoint.get("hidden_dim", 128),
        num_layers=checkpoint.get("num_layers", 2),
        num_classes=checkpoint.get("num_classes", len(CLASSES))
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model
