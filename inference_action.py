#!/usr/bin/env python3
"""
BAS Payload Operations Copilot - Temporal Action Recognition Inference
Accepts webcam or video sequence and outputs:
- predicted action
- confidence
- recent temporal predictions

Usage:
  python inference_action.py --source 0
  python inference_action.py --source path/to/video.mp4
"""
import sys
from train.ssv2_pipeline.inference import main

if __name__ == "__main__":
    main()
