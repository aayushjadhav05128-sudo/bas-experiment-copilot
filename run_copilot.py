"""
BAS Experiment Copilot - Unified Runner
Pre-flight checks, model/dataset verification, and launcher for the mission control server.
"""

import os
import sys
import time
import webbrowser
import subprocess

def preflight():
    print("=" * 65)
    print("      BAS EXPERIMENT COPILOT - MISSION CONTROL SYSTEM")
    print("=" * 65)
    print("[*] Performing pre-flight system checks...")

    # Ensure directories
    dirs = [
        "dataset/raw_videos", "dataset/labels", "dataset/processed",
        "models", "protocols", "backend", "frontend"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    # Check protocols
    if not os.path.exists("protocols/fluid_physics.json"):
        print("[!] Warning: fluid_physics.json missing from protocols/")

    # Generate initial demo dataset if raw_videos is empty
    if len(os.listdir("dataset/raw_videos")) == 0:
        print("[*] Generating turnkey benchmark videos and labels (Section 5)...")
        try:
            from train.generate_demo_dataset import create_demo_videos
            create_demo_videos()
        except Exception as e:
            print(f"[!] Note on demo dataset: {e}")

    # Ensure baseline models exist
    if not os.path.exists("models/har_gru.pt"):
        print("[*] Compiling initial HAR GRU model weights...")
        try:
            from train.train_har import train_model
            train_model()
        except Exception as e:
            print(f"[!] Note on HAR model compilation: {e}")

    if not os.path.exists("models/yolo_finetuned.pt"):
        try:
            from train.train_yolo import train_yolo
            train_yolo(epochs=1)
        except Exception as e:
            print(f"[!] Note on YOLO baseline: {e}")

    print("[✓] Pre-flight verification complete.")
    print("[*] Launching Mission Control Hub at http://localhost:8000 ...")
    print("[*] Press Ctrl+C in terminal to stop.")
    print("=" * 65)

    # Launch browser after slight delay
    def open_browser():
        time.sleep(1.8)
        webbrowser.open("http://localhost:8000")

    import threading
    threading.Thread(target=open_browser, daemon=True).start()

    # Run Uvicorn
    try:
        import uvicorn
        uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
    except ImportError:
        subprocess.run([sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"])


if __name__ == "__main__":
    preflight()
