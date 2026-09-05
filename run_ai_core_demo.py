"""
BAS Experiment Copilot - 4-Layer AI Core End-to-End Loop Runner (run_ai_core_demo.py)

Wires together:
- Layer 1: Perception (YOLOv8n + MediaPipe)
- Layer 1.5: Hand-Object Interaction Features
- Layer 2: Temporal Action Recognition (PyTorch GRU with EMA temporal smoothing)
- Layer 3: Protocol FSM (Deterministic Safety Engine)
- Layer 4: Local LLM Guidance Explainer (Ollama + High-Reliability Flight Fallback)

Evaluates on recorded video benchmarks in dataset/raw_videos/ and prints each
standardized contract event to the console:
{
  "timestamp": ISO8601,
  "action_detected": string,
  "confidence": float,
  "expected_step": string,
  "result": "CORRECT|SKIPPED|OUT_OF_ORDER|UNKNOWN|LOW_CONFIDENCE_HOLD",
  "guidance_text": string
}
"""

import os
import sys
import time
import json
import argparse
import cv2

# Import the 4 isolated layers
from backend.perception import PerceptionPipeline
from backend.har_model import TemporalActionClassifier
from backend.protocol_fsm import ProtocolFSM
from backend.explain import GuidanceExplainer


def run_pipeline_on_video(video_path: str, protocol_path: str = "protocols/fluid_physics.json"):
    print("=" * 80)
    print("  BHARATIYA ANTARIKSH STATION (BAS) — 4-LAYER AI CORE PIPELINE")
    print(f"  Target Video: {video_path}")
    print(f"  Protocol:     {protocol_path}")
    print("=" * 80)

    if not os.path.exists(video_path):
        print(f"ERROR: Video file not found at '{video_path}'.")
        print("Run 'python train/generate_demo_dataset.py' to generate benchmark videos.")
        return

    # Initialize 4 Layers
    print("[INIT] Initializing Layer 1 & 1.5: Perception & Interaction Feature Engine...")
    perception = PerceptionPipeline(use_camera=False)

    print("[INIT] Initializing Layer 2: Temporal Action Recognition (PyTorch GRU)...")
    har = TemporalActionClassifier(window_size=16, smoothing_window=5)

    print("[INIT] Initializing Layer 4: Local LLM Guidance Explainer...")
    explainer = GuidanceExplainer()

    print("[INIT] Initializing Layer 3: Deterministic Protocol FSM...")
    fsm = ProtocolFSM(protocol_path=protocol_path, confidence_threshold=0.65, explainer=explainer)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    print(f"[READY] Video loaded: {total_frames} frames @ {fps:.1f} FPS. Beginning inference loop...")
    print("-" * 80)

    frame_idx = 0
    last_eval_time = 0.0
    emitted_events_count = 0
    last_emitted_step = -1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_idx += 1
        video_time = frame_idx / fps
        simulated_now = time.time()

        # ---------------- LAYER 1: PERCEPTION ----------------
        layer1_data = perception.extract_layer1_perception(frame)

        # ---------------- LAYER 1.5: INTERACTION FEATURES ----------------
        layer1_5_data = perception.compute_layer1_5_features(layer1_data, simulated_now)

        # ---------------- LAYER 2: TEMPORAL BUFFER ----------------
        har.add_frame_features(layer1_5_data["feature_vector"])

        # Check for pre-error nudge (Tier 1.1)
        # Periodic evaluation every ~15 frames (0.75s)
        if (frame_idx - last_eval_time) >= 15:
            action_detected, confidence = har.classify_current_window(
                heuristic_hint=layer1_5_data["contact_action"],
                heuristic_conf=layer1_5_data["action_confidence"]
            )

            # Evaluate against FSM if hand is active or interacting
            if action_detected != "idle":
                # ---------------- LAYER 3 & 4: FSM + EXPLANATION ----------------
                event = fsm.evaluate_action(
                    action=action_detected,
                    confidence=confidence,
                    hesitation=layer1_5_data["hesitation_score"]
                )

                # Format strictly to the Output Contract:
                contract_event = {
                    "timestamp": event["timestamp"],
                    "action_detected": event["action_detected"],
                    "confidence": event["confidence"],
                    "expected_step": event["expected_step"],
                    "result": event["result"],
                    "guidance_text": event["guidance_text"]
                }

                # Print contract event as JSON
                print(f"[EVENT #{emitted_events_count + 1} | Frame {frame_idx} | T={video_time:.2f}s]")
                print(json.dumps(contract_event, indent=2))
                print("-" * 80)

                emitted_events_count += 1
                last_emitted_step = fsm.current_step_index

            last_eval_time = frame_idx

    cap.release()

    print("=" * 80)
    print("  AI CORE DEMO EXECUTION COMPLETE")
    print(f"  Total Processed Frames:  {frame_idx}")
    print(f"  Emitted Contract Events: {emitted_events_count}")
    print(f"  Final FSM Status:        {fsm.current_step_index}/{len(fsm.steps)} Steps Verified")
    print(f"  Protocol Finished:       {fsm.is_completed}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BAS Experiment Copilot 4-Layer AI Core Runner")
    parser.add_argument(
        "--video",
        type=str,
        default="dataset/raw_videos/correct_run_01.mp4",
        help="Path to test video file"
    )
    parser.add_argument(
        "--protocol",
        type=str,
        default="protocols/fluid_physics.json",
        help="Path to protocol checklist JSON"
    )
    args = parser.parse_args()

    run_pipeline_on_video(args.video, args.protocol)
