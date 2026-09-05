"""
BAS Experiment Copilot - Demo Dataset Generator
Creates synthetic benchmark videos and label JSONs matching Section 5 of the build specification:
- correct_run_01.mp4
- skip_step3_run_01.mp4
- wrong_order_run_01.mp4
Matching labels in dataset/labels/
"""

import os
import json
import math
import numpy as np

def create_demo_videos():
    import cv2
    
    os.makedirs("dataset/raw_videos", exist_ok=True)
    os.makedirs("dataset/labels", exist_ok=True)

    apparatus = {
        "sample_chamber": (80, 260, 180, 400),
        "cartridge": (200, 280, 290, 410),
        "sensor_probe": (320, 270, 410, 400),
        "chamber_seal": (430, 280, 520, 410),
        "agitator_switch": (535, 290, 615, 390)
    }

    scenarios = [
        {
            "name": "correct_run_01",
            "steps": [
                ("open_chamber", "sample_chamber", 0.0, 3.0),
                ("insert_cartridge", "cartridge", 3.0, 6.5),
                ("attach_probe", "sensor_probe", 6.5, 10.0),
                ("verify_seal", "chamber_seal", 10.0, 13.5),
                ("activate", "agitator_switch", 13.5, 17.0)
            ]
        },
        {
            "name": "skip_step3_run_01",
            "steps": [
                ("open_chamber", "sample_chamber", 0.0, 3.0),
                ("insert_cartridge", "cartridge", 3.0, 6.5),
                # Deliberately skips attach_probe (step 3 in 1-based or 2 in 0-based)
                ("verify_seal", "chamber_seal", 6.5, 10.5),
                ("activate", "agitator_switch", 10.5, 14.0)
            ]
        },
        {
            "name": "wrong_order_run_01",
            "steps": [
                ("open_chamber", "sample_chamber", 0.0, 3.0),
                ("attach_probe", "sensor_probe", 3.0, 6.5),
                ("insert_cartridge", "cartridge", 6.5, 10.0),
                ("verify_seal", "chamber_seal", 10.0, 13.5)
            ]
        }
    ]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 20
    w, h = 640, 480

    for sc in scenarios:
        video_name = f"{sc['name']}.mp4"
        video_path = os.path.join("dataset/raw_videos", video_name)
        label_path = os.path.join("dataset/labels", f"{sc['name']}.json")

        out = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
        segments = []

        total_duration = sc["steps"][-1][3]
        total_frames = int(total_duration * fps)

        for step_idx, (action, target_obj, start_t, end_t) in enumerate(sc["steps"]):
            segments.append({
                "step": step_idx + 1,
                "action": action,
                "start": start_t,
                "end": end_t
            })

        for f_idx in range(total_frames):
            cur_time = f_idx / fps
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            frame[:, :] = (18, 24, 32)  # Dark workbench

            # Grid
            for x in range(0, w, 40):
                frame[:, x:x+1] = (28, 36, 48)
            for y in range(0, h, 40):
                frame[y:y+1, :] = (28, 36, 48)

            # Draw objects
            for obj_key, (x1, y1, x2, y2) in apparatus.items():
                cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 70, 100), 2)
                cv2.putText(frame, obj_key, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 160, 200), 1)

            # Find active action
            active_target = "sample_chamber"
            for action, target_obj, start_t, end_t in sc["steps"]:
                if start_t <= cur_time <= end_t:
                    active_target = target_obj
                    break

            # Move simulated hand towards active target
            tx1, ty1, tx2, ty2 = apparatus[active_target]
            target_cx, target_cy = (tx1 + tx2) / 2, (ty1 + ty2) / 2
            jitter = math.sin(cur_time * 6) * 5
            hand_x = int(target_cx + jitter)
            hand_y = int(target_cy - 10 + abs(math.cos(cur_time * 4) * 10))

            # Draw hand
            cv2.circle(frame, (hand_x, hand_y), 16, (0, 255, 255), -1)
            cv2.circle(frame, (hand_x, hand_y), 22, (0, 200, 200), 2)

            cv2.putText(frame, f"SCENARIO: {sc['name']}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 240, 255), 1)
            cv2.putText(frame, f"TIME: {cur_time:.1f}s", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            out.write(frame)

        out.release()

        # Write label JSON matching Section 5
        label_data = {
            "video": video_name,
            "segments": segments
        }
        with open(label_path, "w", encoding="utf-8") as lf:
            json.dump(label_data, lf, indent=2)

        print(f"Generated {video_path} and {label_path}")

if __name__ == "__main__":
    create_demo_videos()
