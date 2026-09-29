#!/usr/bin/env python3
"""
BAS Payload Operations Copilot - Temporal Action Inference Engine
Runs sliding-window temporal action recognition on webcam or video files.

Outputs:
- predicted action
- confidence
- recent temporal predictions

Decoupled Design:
Perception/Landmarks -> Action Recognition -> BAS Procedure State Machine.
"""

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from train.ssv2_pipeline import CLASSES, IDX_TO_LABEL
from train.ssv2_pipeline.extract_landmarks import LandmarkFeatureExtractor
from train.ssv2_pipeline.model import BASActionGRU, load_model


class ActionPredictor:
    """
    Real-time sliding-window action predictor.
    Accepts incoming BGR frames, extracts MediaPipe landmarks,
    normalizes using training set statistics, and predicts the 5 BAS actions.
    """
    def __init__(
        self,
        model_path: str = "models/bas_action_gru.pt",
        norm_path: str = "models/normalization.npz",
        window_size: int = 8,
        history_len: int = 10,
        device: str = "cpu"
    ):
        self.window_size = window_size
        self.device = device
        self.history_len = history_len

        # 1. Load trained model
        self.model = load_model(checkpoint_path=model_path, device=device)
        self.model.eval()

        # 2. Load normalization stats
        if not os.path.exists(norm_path):
            raise FileNotFoundError(f"Normalization file not found: {norm_path}")
        norm_data = np.load(norm_path)
        self.mean = norm_data["mean"].astype(np.float32)
        self.std = norm_data["std"].astype(np.float32)
        self.std[self.std < 1e-6] = 1.0

        # 3. Feature Extractor
        self.feature_extractor = LandmarkFeatureExtractor()

        # 4. Sliding Window Buffers
        self.feature_buffer: Deque[np.ndarray] = collections.deque(maxlen=window_size)
        self.prediction_history: Deque[Dict] = collections.deque(maxlen=history_len)

    def process_frame(self, frame_bgr: np.ndarray) -> Dict:
        """
        Processes a single frame, updates temporal buffer, and returns action prediction.
        """
        # Extract 227-dim feature vector
        feat = self.feature_extractor.extract_frame_landmarks(frame_bgr)
        self.feature_buffer.append(feat)

        # Pad with copies of first feature if buffer not yet full
        seq = list(self.feature_buffer)
        while len(seq) < self.window_size:
            seq.insert(0, seq[0] if seq else feat)

        seq_arr = np.array(seq, dtype=np.float32)  # (window_size, 227)

        # Apply training-set normalization
        seq_norm = (seq_arr - self.mean) / self.std

        # Run inference
        tensor_in = torch.tensor(seq_norm, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(tensor_in)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        pred_idx = int(np.argmax(probs))
        pred_action = IDX_TO_LABEL.get(pred_idx, "unknown")
        confidence = float(probs[pred_idx])

        entry = {
            "timestamp": time.time(),
            "predicted_action": pred_action,
            "confidence": confidence,
            "class_probabilities": {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
        }
        self.prediction_history.append(entry)

        return {
            "predicted_action": pred_action,
            "confidence": confidence,
            "recent_temporal_predictions": list(self.prediction_history),
            "class_probabilities": entry["class_probabilities"],
            "ready": len(self.feature_buffer) >= self.window_size
        }

    def close(self):
        self.feature_extractor.close()


def run_live_inference(
    video_source: str = "0",
    model_path: str = "models/bas_action_gru.pt",
    norm_path: str = "models/normalization.npz",
    show_ui: bool = True
):
    print("=" * 65)
    print("      BAS PAYLOAD OPERATIONS - ACTION INFERENCE ENGINE")
    print("=" * 65)
    print(f"[*] Input Source       : {video_source}")
    print(f"[*] Model Checkpoint   : {model_path}")
    print(f"[*] Normalization File : {norm_path}")
    print("-" * 65)

    source = int(video_source) if video_source.isdigit() else video_source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[!] Error: Could not open video source: {video_source}")
        return

    predictor = ActionPredictor(model_path=model_path, norm_path=norm_path)
    print("[OK] Real-time inference initialized. Press 'Q' to exit.")

    fps_t0 = time.time()
    frames = 0
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frames += 1
            if time.time() - fps_t0 >= 1.0:
                fps = frames / (time.time() - fps_t0)
                frames = 0
                fps_t0 = time.time()

            result = predictor.process_frame(frame)
            action = result["predicted_action"]
            conf = result["confidence"]
            recent = result["recent_temporal_predictions"]

            # Terminal printout
            sys.stdout.write(
                f"\r[ACTION: {action:<18s}] | Conf: {conf:5.1%} | FPS: {fps:4.1f} | Buffer: {len(predictor.feature_buffer)}/8"
            )
            sys.stdout.flush()

            if show_ui:
                # Render HUD overlay on video frame
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (10, 10), (440, 150), (15, 15, 25), -1)
                cv2.rectangle(frame, (10, 10), (440, 150), (0, 200, 255), 1)

                cv2.putText(frame, "BAS PAYLOAD COPILOT - ACTION HUD", (20, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 240, 255), 2)
                cv2.putText(frame, f"Action: {action.upper()}", (20, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 120), 2)
                cv2.putText(frame, f"Confidence: {conf:.1%} | FPS: {fps:.1f}", (20, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

                # Recent prediction trail
                recent_str = " -> ".join([r["predicted_action"][:6] for r in recent[-4:]])
                cv2.putText(frame, f"Recent: {recent_str}", (20, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

                cv2.imshow("BAS Action Recognition HUD", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        cap.release()
        predictor.close()
        cv2.destroyAllWindows()
        print("\n[OK] Inference session terminated.")


def main():
    ap = argparse.ArgumentParser(description="Run BAS action recognition inference.")
    ap.add_argument("--source", default="0", help="Webcam index (0) or path to video file")
    ap.add_argument("--model", default="models/bas_action_gru.pt", help="Model weights path")
    ap.add_argument("--norm", default="models/normalization.npz", help="Normalization stats path")
    ap.add_argument("--no-ui", action="store_true", help="Disable OpenCV GUI window")
    args = ap.parse_args()

    run_live_inference(
        video_source=args.source,
        model_path=args.model,
        norm_path=args.norm,
        show_ui=not args.no_ui
    )


if __name__ == "__main__":
    main()
