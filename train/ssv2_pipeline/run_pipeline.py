#!/usr/bin/env python3
"""
BAS Payload Operations Copilot - Master SSV2 Pipeline Runner
Executes the complete pipeline end-to-end:
1. Filter SSV2 annotations -> bas_manifest.csv
2. Uniform frame extraction (8 frames/video) -> frames/
3. MediaPipe Pose + Hands landmark extraction -> dataset/landmarks/
4. Split by Video ID + Train-only normalization
5. Train BAS Action GRU with early stopping on Macro F1
6. Save: models/bas_action_gru.pt, models/label_map.json, models/normalization.npz
7. Verify inference engine output
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np

from train.ssv2_pipeline.filter_ssv2 import filter_annotations
from train.ssv2_pipeline.extract_frames import run_frame_extraction
from train.ssv2_pipeline.extract_landmarks import process_frames_directory
from train.ssv2_pipeline.train_action_gru import train
from train.ssv2_pipeline.generate_synthetic_benchmark import generate_benchmark_dataset
from train.ssv2_pipeline.inference import ActionPredictor


def run_full_pipeline(
    train_json: str,
    val_json: str,
    video_dir: str,
    manifest_out: str = "bas_manifest.csv",
    frames_dir: str = "frames",
    landmarks_dir: str = "dataset/landmarks",
    model_out: str = "models/bas_action_gru.pt",
    label_map_out: str = "models/label_map.json",
    norm_out: str = "models/normalization.npz",
    epochs: int = 40,
    frames_per_video: int = 8
):
    print("=" * 70)
    print("   BHARATIYA ANTARIKSH STATION (BAS) PAYLOAD OPERATIONS COPILOT")
    print("       END-TO-END TEMPORAL ACTION RECOGNITION PIPELINE")
    print("=" * 70)

    # STEP 1: Filter Annotations
    print("\n[PHASE 1/5] Filtering SSV2 annotations to 5 BAS Action Classes...")
    filter_annotations(train_json, val_json, manifest_out)

    # STEP 2: Uniform Frame Extraction
    print(f"\n[PHASE 2/5] Extracting {frames_per_video} uniform frames per video clip...")
    run_frame_extraction(manifest_out, video_dir, frames_dir, frames_per_video=frames_per_video)

    # STEP 3: MediaPipe Pose & Hands Feature Extraction
    print("\n[PHASE 3/5] Extracting MediaPipe Pose & Hands landmarks (227 dims/frame)...")
    process_frames_directory(frames_dir, landmarks_dir, target_frames=frames_per_video)

    # STEP 4: Training & Model Serialization
    print("\n[PHASE 4/5] Training BAS Action GRU & Evaluating Metrics...")
    train(
        landmarks_dir=landmarks_dir,
        model_save_path=model_out,
        label_map_path=label_map_out,
        norm_save_path=norm_out,
        epochs=epochs,
        batch_size=16,
        lr=1e-3,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2,
        patience=10
    )

    # STEP 5: Verification of Inference Engine
    print("\n[PHASE 5/5] Verifying Inference Pipeline...")
    predictor = ActionPredictor(model_path=model_out, norm_path=norm_out, window_size=frames_per_video)
    dummy_frame = np.ones((300, 400, 3), dtype=np.uint8) * 128
    result = predictor.process_frame(dummy_frame)
    predictor.close()

    print("[OK] Inference verification test passed successfully:")
    print(f"    - Predicted Action   : {result['predicted_action']}")
    print(f"    - Confidence Score   : {result['confidence']:.2%}")
    print(f"    - Recent Predictions : {len(result['recent_temporal_predictions'])} entry logged")
    print("=" * 70)
    print("ALL PIPELINE ARTIFACTS READY FOR BAS PAYLOAD COPILOT!")
    print("=" * 70)


def main():
    ap = argparse.ArgumentParser(description="Run complete BAS SSV2 dataset and training pipeline.")
    ap.add_argument("--train_json", help="Path to SSV2 train.json")
    ap.add_argument("--val_json", help="Path to SSV2 validation.json")
    ap.add_argument("--video_dir", help="Path to SSV2 videos directory")
    ap.add_argument("--synthetic", action="store_true", help="Generate benchmark dataset first if no raw data provided")
    ap.add_argument("--samples_per_class", type=int, default=14, help="Samples per class for benchmark")
    ap.add_argument("--epochs", type=int, default=35, help="Training epochs")
    args = ap.parse_args()

    train_json = args.train_json
    val_json = args.val_json
    video_dir = args.video_dir

    if not train_json or not val_json or not video_dir or args.synthetic:
        if not args.synthetic and (not train_json or not os.path.exists(train_json or "")):
            print("[*] No existing SSV2 dataset path supplied or found. Initializing turnkey benchmark dataset...")
        train_json, val_json, video_dir = generate_benchmark_dataset(
            output_dir="dataset/ssv2_benchmark",
            samples_per_class=args.samples_per_class
        )

    run_full_pipeline(
        train_json=train_json,
        val_json=val_json,
        video_dir=video_dir,
        epochs=args.epochs
    )


if __name__ == "__main__":
    main()
