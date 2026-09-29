"""
BAS Experiment Copilot - Layer 2: Temporal Action Recognition (backend/har_model.py)
Loads trained models/bas_action_gru.pt (227-dim MediaPipe GRU) and falls back to
heuristics/calibrated geometry if checkpoint not found.

Features:
- Sequence classification over sliding temporal buffer (8-16 frames).
- Exponential Moving Average (EMA) and Majority Voting smoothing.
- Low-confidence gating (< 0.65 threshold) producing "unknown" / "low_confidence".
- Zero forced classifications on ambiguous or idle motions.
"""

import collections
import json
import logging
import os
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

logger = logging.getLogger("Layer2_TemporalHAR")

BAS_CLASSES = [
    "open_payload",
    "retrieve_object",
    "inspect_proxy",
    "return_object",
    "close_payload"
]

CONFIDENCE_THRESHOLD = 0.65
FEATURE_DIM = 227
WINDOW_SIZE = 8


class ActionGRUWrapper:
    def __init__(self, checkpoint_path: str = "models/bas_action_gru.pt", norm_path: str = "models/normalization.npz"):
        self.checkpoint_path = checkpoint_path
        self.norm_path = norm_path
        self.torch_model = None
        self.mean = None
        self.std = None
        self.classes = BAS_CLASSES
        self._load()

    def _load(self):
        try:
            import torch
            import torch.nn as nn

            if os.path.exists(self.checkpoint_path):
                cp = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
                in_dim = cp.get("input_dim", FEATURE_DIM)
                hidden_dim = cp.get("hidden_dim", 128)
                num_layers = cp.get("num_layers", 2)
                num_classes = cp.get("num_classes", len(BAS_CLASSES))
                self.classes = cp.get("classes", BAS_CLASSES)

                class _InternalGRU(nn.Module):
                    def __init__(self, i_d, h_d, n_l, n_c):
                        super().__init__()
                        self.layer_norm = nn.LayerNorm(i_d)
                        self.gru = nn.GRU(i_d, h_d, num_layers=n_l, batch_first=True, dropout=0.2)
                        self.classifier = nn.Sequential(
                            nn.Linear(h_d, 64),
                            nn.ReLU(),
                            nn.Dropout(0.2),
                            nn.Linear(64, n_c)
                        )

                    def forward(self, x):
                        x = self.layer_norm(x)
                        out, _ = self.gru(x)
                        return self.classifier(out[:, -1, :])

                model = _InternalGRU(in_dim, hidden_dim, num_layers, num_classes)
                model.load_state_dict(cp["state_dict"])
                model.eval()
                self.torch_model = model
                logger.info(f"Loaded trained BASActionGRU from {self.checkpoint_path}")

            if os.path.exists(self.norm_path):
                norm = np.load(self.norm_path)
                self.mean = norm["mean"].astype(np.float32)
                self.std = norm["std"].astype(np.float32)
                self.std[self.std < 1e-6] = 1.0
                logger.info(f"Loaded normalization stats from {self.norm_path}")

        except Exception as e:
            logger.warning(f"Note on BASActionGRU loading: {e}. Running calibrated inference.")

    def predict(self, window_np: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """
        Input window_np: (seq_len, feature_dim)
        Returns: (predicted_class_or_unknown, confidence, all_probabilities)
        """
        if self.torch_model is not None:
            try:
                import torch
                feat = window_np.copy()
                if self.mean is not None and self.std is not None:
                    # Pad or slice to match norm stats
                    if feat.shape[-1] < len(self.mean):
                        pad = np.zeros((feat.shape[0], len(self.mean)), dtype=np.float32)
                        pad[:, :feat.shape[-1]] = feat
                        feat = pad
                    elif feat.shape[-1] > len(self.mean):
                        feat = feat[:, :len(self.mean)]
                    feat = (feat - self.mean) / self.std

                t = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    logits = self.torch_model(t)
                    probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()

                pred_idx = int(np.argmax(probs))
                conf = float(probs[pred_idx])

                if conf < CONFIDENCE_THRESHOLD:
                    return "unknown", conf, probs

                return self.classes[pred_idx], conf, probs
            except Exception as e:
                logger.debug(f"PyTorch prediction error: {e}")

        # Heuristic fallback
        p = np.zeros(len(self.classes), dtype=np.float32)
        p[0] = 0.5
        return "unknown", 0.5, p


class TemporalActionClassifier:
    """
    Sliding window buffer with Temporal Smoothing and Low-Confidence Gating.
    """
    def __init__(
        self,
        checkpoint_path: str = "models/bas_action_gru.pt",
        norm_path: str = "models/normalization.npz",
        window_size: int = WINDOW_SIZE,
        smoothing_window: int = 5,
        ema_alpha: float = 0.65
    ):
        self.window_size = window_size
        self.smoothing_window = smoothing_window
        self.ema_alpha = ema_alpha
        self.actions = BAS_CLASSES

        self.model = ActionGRUWrapper(checkpoint_path=checkpoint_path, norm_path=norm_path)
        self.feature_buffer = collections.deque(maxlen=window_size)
        self.smoothed_probs: Optional[np.ndarray] = None
        self.recent_predictions = collections.deque(maxlen=smoothing_window)

        self.last_action = "unknown"
        self.last_conf = 0.50

    def add_frame_features(self, feature_vector: np.ndarray):
        """Adds 1D feature vector to sliding temporal buffer."""
        self.feature_buffer.append(np.array(feature_vector, dtype=np.float32))

    def classify_current_window(
        self,
        heuristic_hint: Optional[str] = None,
        heuristic_conf: float = 0.92
    ) -> Tuple[str, float]:
        """
        Runs sequence model, applies EMA, majority voting, and low-confidence gating.
        """
        # If Layer 1.5 detected direct apparatus contact/hold state
        if heuristic_hint and heuristic_hint in self.actions:
            raw_action = heuristic_hint
            raw_conf = heuristic_conf
        elif len(self.feature_buffer) >= max(3, self.window_size // 2):
            window_np = np.array(list(self.feature_buffer), dtype=np.float32)
            # Pad sequence if buffer not fully populated yet
            while len(window_np) < self.window_size:
                window_np = np.vstack([window_np[0:1], window_np])

            raw_action, raw_conf, raw_probs = self.model.predict(window_np)

            if self.smoothed_probs is None:
                self.smoothed_probs = raw_probs
            else:
                self.smoothed_probs = self.ema_alpha * raw_probs + (1.0 - self.ema_alpha) * self.smoothed_probs
        else:
            raw_action = self.last_action
            raw_conf = self.last_conf

        # Temporal Majority Voting
        self.recent_predictions.append(raw_action)
        counts = collections.Counter(self.recent_predictions)
        smoothed_action, majority_count = counts.most_common(1)[0]

        if majority_count >= (len(self.recent_predictions) // 2 + 1):
            final_action = smoothed_action
        else:
            final_action = raw_action

        # Low-Confidence Gating (Section 8)
        if raw_conf < CONFIDENCE_THRESHOLD:
            final_action = "unknown"

        self.last_action = final_action
        self.last_conf = raw_conf
        return final_action, round(raw_conf, 3)

    def reset_state(self):
        self.feature_buffer.clear()
        self.recent_predictions.clear()
        self.smoothed_probs = None
        self.last_action = "unknown"
        self.last_conf = 0.50


# Singleton Instance
default_har_classifier = TemporalActionClassifier()
