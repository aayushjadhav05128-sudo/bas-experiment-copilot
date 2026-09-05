"""
BAS Experiment Copilot - Feature Extraction Pipeline (Phase 3)
Reads every video in dataset/raw_videos and its matching label in dataset/labels.
Runs the perception layer on each frame, extracting landmark coordinates,
nearest-object distance, and the labelled action into CSVs in dataset/processed/.
"""

import os
import glob
import json
import csv
import logging
import cv2
import numpy as np

from backend.perception import PerceptionPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FeatureExtractor")


def extract_all_features():
    raw_dir = "dataset/raw_videos"
    label_dir = "dataset/labels"
    processed_dir = "dataset/processed"
    os.makedirs(processed_dir, exist_ok=True)

    pipeline = PerceptionPipeline(use_camera=False)

    video_files = glob.glob(os.path.join(raw_dir, "*.mp4"))
    if not video_files:
        logger.warning(f"No video files found in {raw_dir}. Run train/generate_demo_dataset.py first.")
        return

    for v_path in video_files:
        base_name = os.path.splitext(os.path.basename(v_path))[0]
        l_path = os.path.join(label_dir, f"{base_name}.json")
        out_csv = os.path.join(processed_dir, f"{base_name}_features.csv")

        if not os.path.exists(l_path):
            logger.warning(f"Label file {l_path} missing; skipping {v_path}")
            continue

        with open(l_path, "r", encoding="utf-8") as f:
            label_info = json.load(f)

        segments = label_info.get("segments", [])
        cap = cv2.VideoCapture(v_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        frame_idx = 0

        # Prepare CSV writer
        # Header: frame, time_s, hand_x, hand_y, vel_x, vel_y, min_dist, closest_obj, hesitation, action_label, [feat0..feat71]
        fieldnames = [
            "frame", "time_s", "hand_x", "hand_y", "vel_x", "vel_y",
            "min_dist", "closest_obj", "hesitation", "action_label"
        ] + [f"feat_{i}" for i in range(72)]

        rows = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            cur_time = frame_idx / fps
            frame_idx += 1

            # Match active action segment
            current_action = "idle"
            for seg in segments:
                if seg["start"] <= cur_time <= seg["end"]:
                    current_action = seg["action"]
                    break

            # Run perception pipeline on frame
            perc = pipeline.process_frame(frame)
            hx, hy = perc["hand_center"]
            vx, vy = perc["velocity"]
            min_d = min([obj["distance"] for obj in perc["detected_objects"]]) if perc["detected_objects"] else 999.0
            c_obj = perc["closest_object"] or "none"
            hesit = perc["hesitation_score"]
            feat_vec = perc["feature_vector"]

            row = {
                "frame": frame_idx,
                "time_s": round(cur_time, 3),
                "hand_x": hx,
                "hand_y": hy,
                "vel_x": round(vx, 2),
                "vel_y": round(vy, 2),
                "min_dist": round(min_d, 2),
                "closest_obj": c_obj,
                "hesitation": round(hesit, 3),
                "action_label": current_action
            }
            for i, val in enumerate(feat_vec):
                row[f"feat_{i}"] = round(float(val), 5)

            rows.append(row)

        cap.release()

        with open(out_csv, "w", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        logger.info(f"Extracted {len(rows)} frames of features -> {out_csv}")


if __name__ == "__main__":
    extract_all_features()
