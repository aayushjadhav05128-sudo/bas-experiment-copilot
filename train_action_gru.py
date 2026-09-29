#!/usr/bin/env python3
"""
BAS Payload Operations Copilot - Temporal Action Recognition Training
Wrapper for train.ssv2_pipeline.train_action_gru.

Usage:
  python train_action_gru.py --landmarks_dir dataset/landmarks
"""
import sys
from train.ssv2_pipeline.train_action_gru import main

if __name__ == "__main__":
    main()
