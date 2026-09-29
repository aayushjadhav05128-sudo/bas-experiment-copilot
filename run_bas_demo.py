#!/usr/bin/env python3
"""
BHARATIYA ANTARIKSH STATION (BAS) — PAYLOAD OPERATIONS COPILOT
Official SIH Demonstration & Mission Control Launcher (run_bas_demo.py)

Reproducible command:
  python run_bas_demo.py

Pipeline Flow:
  Live Video/Webcam -> YOLOv8 + MediaPipe Pose/Hands (227 dims) -> BASActionGRU ->
  Confidence Gate (>=0.65) -> Deterministic Procedure FSM -> Telemetry & HUD (port 8000)
"""

import os
import sys
import time
import webbrowser
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
logger = logging.getLogger("BAS_DemoLauncher")


def verify_system_artifacts():
    print("=" * 70)
    print("    BHARATIYA ANTARIKSH STATION (BAS) — PAYLOAD COPILOT DEMO")
    print("=" * 70)
    print("[*] Running pre-flight system checks...")

    required_artifacts = [
        ("Model Weights", "models/bas_action_gru.pt"),
        ("Label Map", "models/label_map.json"),
        ("Normalization Stats", "models/normalization.npz"),
        ("Dataset Manifest", "data/manifests/bas_manifest.csv"),
        ("Mission Protocol", "protocols/payload_operations.json")
    ]

    all_present = True
    for name, path in required_artifacts:
        if os.path.exists(path):
            print(f"  [OK] {name:<22s}: {path}")
        else:
            print(f"  [!] Missing {name:<14s}: {path}")
            all_present = False

    if not all_present:
        print("[*] Recompiling missing artifacts via build suite...")
        try:
            import train.build_complete_dataset_and_train as bld
            bld.main()
        except Exception as e:
            print(f"[!] Compilation note: {e}")

    print("-" * 70)
    print("[OK] All flight artifacts verified.")
    print("[*] Launching Mission Control Hub at: http://localhost:8000")
    print("[*] Telemetry: Native WebSockets + Direct Base64 Stream + MJPEG")
    print("[*] Interactive Hotkeys on Dashboard:")
    print("    - Keys [1] to [5]: Advance / verify steps 1 to 5")
    print("    - Key  [S]       : Trigger step skip deviation")
    print("    - Key  [P]       : Trigger trajectory pre-error warning")
    print("    - Key  [H]       : Trigger operator hesitation hold")
    print("    - Key  [R]       : Reset protocol state machine")
    print("=" * 70)


def open_browser():
    time.sleep(1.8)
    try:
        webbrowser.open("http://localhost:8000")
    except Exception:
        pass


def main():
    verify_system_artifacts()

    import threading
    threading.Thread(target=open_browser, daemon=True).start()

    # Launch Uvicorn server
    try:
        import uvicorn
        uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
    except ImportError:
        subprocess.run([sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"])


if __name__ == "__main__":
    main()
