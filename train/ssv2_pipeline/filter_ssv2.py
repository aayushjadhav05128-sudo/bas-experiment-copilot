#!/usr/bin/env python3
"""
Filter Something-Something V2 annotations into BAS classes.

ATTRIBUTION & RESEARCH LICENSE:
Dataset: 20BN-SOMETHING-SOMETHING-V2
Authors: Raghav Goyal, Samira Ebrahimi Kahou, Vincent Michalski, Joanna Materzynska,
         Susanne Westphal, Heuna Kim, Valentin Haenel, Ingo Fruend, Peter Yianilos,
         Moritz Mueller-Freitag, Florian Hoppe, Christian Thurau, Ingo Bax, Roland Memisevic
License: Research and academic non-commercial use only. Source videos must not be redistributed.

IMPORTANT NOTE ON 'inspect_proxy':
`inspect_proxy` is NOT genuine microgravity payload inspection. It is a temporary
proxy based on SSV2 "Holding something". In the flight operations domain, formal inspection
requires instrument-verified defect assessment, which is handled at the state machine level.

Usage:
  python filter_ssv2.py \
      --train something-something-v2-train.json \
      --val something-something-v2-validation.json \
      --out bas_manifest.csv
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

MAP = {
    "Opening something": "open_payload",
    "Uncovering something": "open_payload",
    "Taking something out of something": "retrieve_object",
    "Taking something from somewhere": "retrieve_object",
    "Holding something": "inspect_proxy",
    "Putting something into something": "return_object",
    "Closing something": "close_payload",
}


def load_json(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Annotation file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_label(row: dict) -> str:
    return str(row.get("label") or row.get("text") or row.get("template") or "")


def get_id(row: dict) -> str:
    return str(row.get("id") or row.get("video_id") or "")


def filter_annotations(train_path: str, val_path: str, out_path: str = "bas_manifest.csv"):
    rows = []
    splits = []
    if train_path and os.path.exists(train_path):
        splits.append(("train", train_path))
    if val_path and os.path.exists(val_path):
        splits.append(("val", val_path))

    if not splits:
        raise ValueError("At least one valid annotation file (--train or --val) must be provided.")

    for split, path in splits:
        data = load_json(path)
        for r in data:
            label = get_label(r)
            bas = MAP.get(label)
            if bas:
                v_id = get_id(r)
                if not v_id:
                    continue
                rows.append({
                    "video_id": v_id,
                    "split_source": split,
                    "source_label": label,
                    "bas_label": bas,
                    "video_filename": f"{v_id}.webm"
                })

    # Deterministic deduplication by video_id
    seen = set()
    clean = []
    for r in rows:
        if r["video_id"] not in seen:
            seen.add(r["video_id"])
            clean.append(r)

    # Sort deterministically
    clean.sort(key=lambda x: (x["bas_label"], x["video_id"]))

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["video_id", "split_source", "source_label", "bas_label", "video_filename"]
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(clean)

    print("=" * 65)
    print("      BAS SSV2 ANNOTATION FILTER & MANIFEST GENERATOR")
    print("=" * 65)
    print(f"[OK] Filtered {len(clean)} qualifying action videos into: {out_file.resolve()}")
    print("[*] Distribution across BAS action classes:")
    counts = Counter(r["bas_label"] for r in clean)
    for cls_name, count in sorted(counts.items()):
        print(f"    - {cls_name:18s}: {count:6d} samples")
    print("=" * 65)
    print("[NOTE] 'inspect_proxy' is a proxy based on 'Holding something'.")
    print("       Formal BAS payload inspection requires downstream deterministic FSM verification.")
    print("=" * 65)
    return clean


def main():
    ap = argparse.ArgumentParser(description="Filter SSV2 annotations to BAS Payload Operations classes.")
    ap.add_argument("--labels", help="Labels JSON (optional metadata)")
    ap.add_argument("--train", required=True, help="Path to something-something-v2-train.json")
    ap.add_argument("--val", required=True, help="Path to something-something-v2-validation.json")
    ap.add_argument("--out", default="bas_manifest.csv", help="Path to output manifest CSV")
    args = ap.parse_args()

    filter_annotations(args.train, args.val, args.out)


if __name__ == "__main__":
    main()
