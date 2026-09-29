#!/usr/bin/env python3
"""
Extract a small, uniform number of frames per video.
Wrapper for train.ssv2_pipeline.extract_frames.

Example:
python extract_frames.py --manifest bas_manifest.csv \
    --video_dir /path/to/20bn-something-something-v2 \
    --out frames --frames_per_video 8
"""
import sys
from train.ssv2_pipeline.extract_frames import main

if __name__ == "__main__":
    main()
