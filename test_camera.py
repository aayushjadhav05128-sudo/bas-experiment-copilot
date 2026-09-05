"""
CP Plus & USB Camera Diagnostic Tool (test_camera.py)
Scans connected cameras on Windows, tests live video frames,
and provides setup guidance for both USB and Wi-Fi RTSP CP Plus cameras.
"""

import sys
import os
import time
import cv2


def scan_cameras(max_indices: int = 4):
    print("=" * 70)
    print("  CP PLUS & USB CAMERA DIAGNOSTIC SCAN")
    print("=" * 70)

    found_cameras = []

    for idx in range(max_indices):
        print(f"\n[SCAN] Probing Camera Index {idx} (DirectShow)...", end="", flush=True)
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx)

        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w, _ = frame.shape
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                print(f" -> FOUND! Resolution: {w}x{h} @ {fps:.0f} FPS")
                found_cameras.append((idx, w, h, frame))
            else:
                print(" -> Device opened, but frame capture timed out.")
            cap.release()
        else:
            print(" -> Not connected / No device.")

    print("\n" + "-" * 70)
    if not found_cameras:
        print("[!] NO VIDEO CAPTURE DEVICES FOUND.")
    else:
        print(f"[OK] FOUND {len(found_cameras)} ACTIVE CAMERA(S):")
        for idx, w, h, frame in found_cameras:
            filename = f"camera_{idx}_preview.jpg"
            cv2.imwrite(filename, frame)
            print(f"   * Index {idx}: {w}x{h}  (Saved preview to '{filename}')")

    print("-" * 70)
    return found_cameras


def test_rtsp_stream(rtsp_url: str):
    print(f"\n[RTSP] Probing Wi-Fi Stream: {rtsp_url} ...")
    cap = cv2.VideoCapture(rtsp_url)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            h, w, _ = frame.shape
            filename = "rtsp_preview.jpg"
            cv2.imwrite(filename, frame)
            print(f"[OK] RTSP STREAM ACTIVE! Resolution: {w}x{h} (Saved to '{filename}')")
            cap.release()
            return True
        cap.release()
    print("[ERROR] Could not connect to RTSP stream.")
    return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test CP Plus and USB cameras")
    parser.add_argument("--rtsp", type=str, default=None, help="Test a Wi-Fi RTSP stream URL")
    args = parser.parse_args()

    if args.rtsp:
        test_rtsp_stream(args.rtsp)
    else:
        scan_cameras()
