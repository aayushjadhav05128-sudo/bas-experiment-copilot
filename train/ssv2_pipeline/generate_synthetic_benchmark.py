#!/usr/bin/env python3
"""
Turnkey Benchmark & Synthetic SSV2 Dataset Generator for BAS Action Recognition.
Generates video clips and SSV2 JSON annotations for the 5 target actions:
- open_payload (Opening something / Uncovering something)
- retrieve_object (Taking something out of something)
- inspect_proxy (Holding something)
- return_object (Putting something into something)
- close_payload (Closing something)

Used for instant turnkey testing, continuous integration, and SIH jury benchmarking.
"""

import argparse
import json
import math
import os
import random
from pathlib import Path

import cv2
import numpy as np

ACTIONS_DEF = [
    {
        "bas_label": "open_payload",
        "ssv2_templates": ["Opening something", "Uncovering something"],
        "color": (0, 240, 255),
        "motion": "open"
    },
    {
        "bas_label": "retrieve_object",
        "ssv2_templates": ["Taking something out of something", "Taking something from somewhere"],
        "color": (0, 255, 136),
        "motion": "retrieve"
    },
    {
        "bas_label": "inspect_proxy",
        "ssv2_templates": ["Holding something"],
        "color": (255, 170, 0),
        "motion": "hold"
    },
    {
        "bas_label": "return_object",
        "ssv2_templates": ["Putting something into something"],
        "color": (255, 70, 150),
        "motion": "return"
    },
    {
        "bas_label": "close_payload",
        "ssv2_templates": ["Closing something"],
        "color": (220, 100, 255),
        "motion": "close"
    }
]


def generate_kinematic_sequence(motion: str, num_frames: int = 8, noise_std: float = 0.015) -> np.ndarray:
    """Generates realistic human kinematic landmark trajectories (8, 227) for the given action motion."""
    seq = np.zeros((num_frames, 227), dtype=np.float32)
    for f in range(num_frames):
        p = f / max(1, num_frames - 1)
        # Base pose: nose, shoulders
        seq[f, 0:3] = [0.5, 0.2, 0.0]
        seq[f, 33:36] = [0.42, 0.35, 0.0]  # L shoulder (11)
        seq[f, 36:39] = [0.58, 0.35, 0.0]  # R shoulder (12)
        
        # Right Wrist (pose landmark 16 => index 48) & Right Hand (offset 163)
        if motion == "open":
            wx = 0.68 - 0.18 * p
            wy = 0.62 - 0.35 * p
        elif motion == "retrieve":
            wx = 0.50 - 0.25 * p
            wy = 0.50 + 0.15 * p
        elif motion == "hold":
            tremor = math.sin(p * 4 * math.pi) * 0.02
            wx = 0.40 + tremor
            wy = 0.35 + tremor
        elif motion == "return":
            wx = 0.25 + 0.25 * p
            wy = 0.65 - 0.15 * p
        elif motion == "close":
            wx = 0.50
            wy = 0.27 + 0.25 * p
        else:
            wx, wy = 0.5, 0.5

        # Add minor motor noise
        wx += float(np.random.normal(0, noise_std))
        wy += float(np.random.normal(0, noise_std))

        seq[f, 48:51] = [wx, wy, 0.0]
        # Right hand presence flag & joints
        seq[f, 163] = 1.0
        seq[f, 164:167] = [wx, wy, 0.0]
        for j in range(1, 21):
            base = 164 + (j * 3)
            fx = wx + 0.02 * math.cos(j * 0.3)
            fy = wy + 0.02 * math.sin(j * 0.3)
            seq[f, base:base+3] = [fx, fy, 0.0]
    return seq


def render_action_frame(motion: str, progress: float, color: tuple, width: int = 400, height: int = 300) -> np.ndarray:
    """
    Renders an astronaut arm, hand, and apparatus interacting according to the action kinematics.
    Provides clear visual contrast for MediaPipe Pose and Hands.
    """
    frame = np.ones((height, width, 3), dtype=np.uint8) * 35  # dark aerospace console background

    # Grid lines
    for y in range(0, height, 40):
        cv2.line(frame, (0, y), (width, y), (45, 45, 45), 1)
    for x in range(0, width, 40):
        cv2.line(frame, (x, 0), (x, height), (45, 45, 45), 1)

    # Apparatus / payload chamber at center
    cx, cy = width // 2, height // 2
    cv2.rectangle(frame, (cx - 70, cy - 50), (cx + 70, cy + 50), (70, 70, 80), -1)
    cv2.rectangle(frame, (cx - 70, cy - 50), (cx + 70, cy + 50), color, 2)
    cv2.putText(frame, "PAYLOAD BAY", (cx - 55, cy - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # Determine hand and object coordinates based on action type & progress
    hand_x, hand_y = cx, cy
    lid_open_angle = 0.0
    obj_present = False
    obj_x, obj_y = cx, cy

    if motion == "open":
        # Hand approaches, grasps handle, and swings open
        hand_x = int(cx + (1.0 - progress) * 100 - 30)
        hand_y = int(cy + 20 - progress * 40)
        lid_open_angle = progress * 80.0
    elif motion == "retrieve":
        # Hand reaches into payload and extracts object towards left
        hand_x = int(cx - progress * 110)
        hand_y = int(cy + 10 - math.sin(progress * math.pi) * 30)
        obj_present = True
        obj_x, obj_y = hand_x + 10, hand_y
    elif motion == "hold":
        # Hand holds object steadily in front of inspection sensor
        jitter = math.sin(progress * 6 * math.pi) * 4
        hand_x = int(cx - 30 + jitter)
        hand_y = int(cy - 20 + jitter)
        obj_present = True
        obj_x, obj_y = hand_x + 15, hand_y
    elif motion == "return":
        # Hand moves object from side back into payload chamber
        hand_x = int(cx - 120 + progress * 120)
        hand_y = int(cy - 30 + progress * 40)
        obj_present = True
        obj_x, obj_y = hand_x + 10, hand_y
    elif motion == "close":
        # Hand pushes lid back down to closed state
        hand_x = int(cx + progress * 70 - 20)
        hand_y = int(cy - 40 + progress * 50)
        lid_open_angle = (1.0 - progress) * 80.0

    # Draw lid
    lid_len = 80
    rad = math.radians(lid_open_angle)
    lx2 = int(cx - 70 + lid_len * math.cos(rad))
    ly2 = int(cy - 50 - lid_len * math.sin(rad))
    cv2.line(frame, (cx - 70, cy - 50), (lx2, ly2), (0, 240, 255), 3)

    # Draw payload sample object
    if obj_present:
        cv2.circle(frame, (int(obj_x), int(obj_y)), 14, (0, 255, 136), -1)
        cv2.circle(frame, (int(obj_x), int(obj_y)), 14, (255, 255, 255), 1)

    # Draw operator torso, arm, and hand (human silhouette for MediaPipe pose/hands)
    # Shoulder
    sx, sy = width - 40, height - 40
    # Elbow
    ex = int(sx - 70 - (sx - hand_x) * 0.4)
    ey = int(sy - 60)
    cv2.line(frame, (sx, sy), (ex, ey), (180, 180, 190), 12)  # Upper arm
    cv2.line(frame, (ex, ey), (hand_x, hand_y), (200, 200, 210), 10)  # Forearm
    cv2.circle(frame, (hand_x, hand_y), 15, (230, 230, 240), -1)  # Hand

    # Fingers
    for f_ang in [-25, -10, 5, 20]:
        frad = math.radians(f_ang)
        fx = int(hand_x - 18 * math.cos(frad))
        fy = int(hand_y + 18 * math.sin(frad))
        cv2.line(frame, (hand_x, hand_y), (fx, fy), (240, 240, 250), 3)

    return frame


def generate_benchmark_dataset(
    output_dir: str = "dataset/ssv2_benchmark",
    samples_per_class: int = 12,
    fps: int = 15,
    duration_sec: float = 1.6
):
    out_path = Path(output_dir)
    video_dir = out_path / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    train_records = []
    val_records = []

    total_frames = int(fps * duration_sec)
    vid_counter = 1000

    print("=" * 65)
    print("      BAS SSV2 SYNTHETIC BENCHMARK DATASET GENERATOR")
    print("=" * 65)
    print(f"[*] Target Directory    : {out_path.resolve()}")
    print(f"[*] Samples per Action  : {samples_per_class}")
    print(f"[*] Total Actions       : {len(ACTIONS_DEF)}")
    print(f"[*] Frames per Clip     : {total_frames} (@ {fps} FPS)")
    print("-" * 65)

    for act in ACTIONS_DEF:
        bas_label = act["bas_label"]
        templates = act["ssv2_templates"]
        color = act["color"]
        motion = act["motion"]

        for i in range(samples_per_class):
            vid_counter += 1
            vid_id = str(vid_counter)
            template = random.choice(templates)
            filename = f"{vid_id}.webm"
            filepath = video_dir / filename

            # Video writer
            fourcc = cv2.VideoWriter_fourcc(*"VP80") if hasattr(cv2, "VideoWriter_fourcc") else 0
            writer = cv2.VideoWriter(str(filepath), fourcc, fps, (400, 300))
            if not writer.isOpened():
                # Fallback to mp4/avi codec if VP80 not registered in OpenCV
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                filename = f"{vid_id}.mp4"
                filepath = video_dir / filename
                writer = cv2.VideoWriter(str(filepath), fourcc, fps, (400, 300))

            for f_idx in range(total_frames):
                prog = f_idx / max(1, total_frames - 1)
                frame = render_action_frame(motion, prog, color)
                writer.write(frame)
            writer.release()

            record = {
                "id": vid_id,
                "label": template,
                "template": template,
                "video_id": vid_id
            }

            # 80/20 train/val split
            if i < int(samples_per_class * 0.8):
                train_records.append(record)
            else:
                val_records.append(record)

    # Save JSON files
    train_json = out_path / "something-something-v2-train.json"
    val_json = out_path / "something-something-v2-validation.json"

    with open(train_json, "w", encoding="utf-8") as f:
        json.dump(train_records, f, indent=2)

    with open(val_json, "w", encoding="utf-8") as f:
        json.dump(val_records, f, indent=2)

    print(f"[OK] Generated {len(train_records)} training clips & {len(val_records)} validation clips.")
    print(f"     - Train Annotations : {train_json.resolve()}")
    print(f"     - Val Annotations   : {val_json.resolve()}")
    print(f"     - Video Directory   : {video_dir.resolve()}")
    print("=" * 65)

    return str(train_json), str(val_json), str(video_dir)


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic SSV2 benchmark clips and annotations.")
    ap.add_argument("--out_dir", default="dataset/ssv2_benchmark", help="Output directory")
    ap.add_argument("--samples_per_class", type=int, default=12, help="Videos per class")
    args = ap.parse_args()

    generate_benchmark_dataset(output_dir=args.out_dir, samples_per_class=args.samples_per_class)


if __name__ == "__main__":
    main()
