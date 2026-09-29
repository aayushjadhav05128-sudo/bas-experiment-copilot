#!/usr/bin/env python3
"""
Extract a uniform number of frames per action video for BAS temporal recognition.

Example:
python extract_frames.py --manifest bas_manifest.csv \
    --video_dir /path/to/20bn-something-something-v2 \
    --out frames --frames_per_video 8
"""

import argparse
import csv
import os
import sys
from pathlib import Path
import cv2
from tqdm import tqdm


def extract_uniform_frames(video_path: Path, out_dir: Path, n: int = 8) -> int:
    """Uniformly extracts n frames from video_path and writes them to out_dir."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return 0

    # Ensure n indices evenly spaced across the video duration
    indices = [round(i * (total - 1) / max(n - 1, 1)) for i in range(n)]
    written = 0

    for j, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            # Fallback: if seek fails, try sequential read
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        frame_filename = out_dir / f"frame_{j:02d}.jpg"
        cv2.imwrite(str(frame_filename), frame)
        written += 1

    cap.release()
    return written


def run_frame_extraction(manifest_path: str, video_dir_path: str, out_root_path: str = "frames", frames_per_video: int = 8):
    video_dir = Path(video_dir_path)
    out_root = Path(out_root_path)

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    with open(manifest_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print("=" * 65)
    print("      BAS SSV2 UNIFORM FRAME EXTRACTION")
    print("=" * 65)
    print(f"[*] Manifest entries: {len(rows)}")
    print(f"[*] Video search dir: {video_dir.resolve()}")
    print(f"[*] Target frames/clip: {frames_per_video}")
    print(f"[*] Output root dir: {out_root.resolve()}")
    print("-" * 65)

    done = 0
    missing = 0
    skipped_short = 0

    for r in tqdm(rows, desc="Extracting frames", unit="video"):
        fname = r.get("video_filename") or f"{r['video_id']}.webm"
        video_path = video_dir / fname

        # Check alternative common extensions if not found directly
        if not video_path.exists():
            stem = Path(fname).stem
            found = False
            for ext in [".webm", ".mp4", ".avi", ".mkv"]:
                alt = video_dir / f"{stem}{ext}"
                if alt.exists():
                    video_path = alt
                    found = True
                    break
            if not found:
                missing += 1
                continue

        dest = out_root / r["bas_label"] / r["video_id"]
        count = extract_uniform_frames(video_path, dest, frames_per_video)
        if count >= max(1, frames_per_video // 2):
            done += 1
        else:
            skipped_short += 1

    print("-" * 65)
    print(f"[OK] Completed frame extraction:")
    print(f"     - Successfully processed : {done} videos")
    print(f"     - Missing source files   : {missing} videos")
    print(f"     - Truncated / empty      : {skipped_short} videos")
    print("=" * 65)
    return done, missing


def main():
    ap = argparse.ArgumentParser(description="Extract uniform frames per video from manifest.")
    ap.add_argument("--manifest", default="bas_manifest.csv", help="Path to manifest CSV")
    ap.add_argument("--video_dir", required=True, help="Directory containing source SSV2 videos")
    ap.add_argument("--out", default="frames", help="Output directory for extracted frames")
    ap.add_argument("--frames_per_video", type=int, default=8, help="Number of frames per video")
    args = ap.parse_args()

    run_frame_extraction(args.manifest, args.video_dir, args.out, args.frames_per_video)


if __name__ == "__main__":
    main()
