#!/usr/bin/env python3
"""
Filter Something-Something V2 annotations into BAS classes.
Wrapper for train.ssv2_pipeline.filter_ssv2.

Usage:
  python filter_ssv2.py \
      --train something-something-v2-train.json \
      --val something-something-v2-validation.json \
      --out bas_manifest.csv
"""
import sys
from train.ssv2_pipeline.filter_ssv2 import main

if __name__ == "__main__":
    main()
