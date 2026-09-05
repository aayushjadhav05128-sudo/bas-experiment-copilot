# Bharatiya Antariksh Station (BAS) — Aerospace Payload Operations Copilot

[![ISRO Standard](https://img.shields.io/badge/Standard-ISRO--AV--STD--2026-00E5FF.svg)](#)
[![Station](https://img.shields.io/badge/Station-Bharatiya%20Antariksh%20Station%20(BAS)-FFB300.svg)](#)
[![Safety Core](https://img.shields.io/badge/Safety%20Core-Rust%20PyO3%20%2B%20Python%20Fallback-00E676.svg)](#)
[![License](https://img.shields.io/badge/Security-Air--Gapped%20Local%20Avionics-64748B.svg)](#)

A production-ready aerospace payload operations copilot engineered for the **Bharatiya Antariksh Station (BAS)**, adhering strictly to ISRO flight-deck ergonomics and safety-critical software reliability standards.

---

## 🛰️ System Architecture

```
                                 [LOCAL WEBCAM / SENSOR FEED]
                                               │
                                               ▼
                                 ┌───────────────────────────┐
                                 │    Perception Pipeline    │
                                 │   • OpenCV Camera Ingest  │
                                 │   • MediaPipe Hands/Pose  │
                                 │   • YOLOv8n Lab Apparatus │
                                 │   • Trajectory Vector     │
                                 └─────────────┬─────────────┘
                                               │ Feature Vectors & Hand Trajectory
                                               ▼
                                 ┌───────────────────────────┐
                                 │   Decision Layer & HAR    │
                                 │   • 2-Layer PyTorch GRU   │
                                 │   • Euclidean Proximity   │
                                 └─────────────┬─────────────┘
                                               │ Identified Action + Confidence + Dwell
                                               ▼
                     ┌──────────────────────────────────────────────────┐
                     │      Safety-Critical State Engine (Dual-Core)     │
                     │  ┌────────────────────┐   ┌───────────────────┐  │
                     │  │ bas_core (Rust/PyO3│◄-►│ protocol_fsm (Py) │  │
                     │  └────────────────────┘   └───────────────────┘  │
                     │   • Deterministic 5-Step Protocol FSM             │
                     │   • 65% Confidence Safety Gate (Hold State)       │
                     │   • Cognitive Hesitation / Dwell Tracker          │
                     │   • Deviation Classifier (OK/SKIP/ORDER/UNEXP)    │
                     └─────────────────────────┬────────────────────────┘
                                               │
                                 ┌─────────────┴─────────────┐
                                 ▼                           ▼
                     ┌───────────────────────┐   ┌───────────────────────┐
                     │ FastAPI Mission Hub   │   │ Threaded Annunciator  │
                     │ • MJPEG /video_feed   │   │ • pyttsx3 Non-Block   │
                     │ • WS /ws/telemetry    │   │ • Atomic JSONL Audit  │
                     └───────────┬───────────┘   └───────────────────────┘
                                 │
                                 ▼
                     ┌───────────────────────────────────────┐
                     │    ISRO Flight-Deck HUD (Frontend)    │
                     │   3-Column Cockpit Telemetry Display  │
                     └───────────────────────────────────────┘
```

---

## 📁 Directory Tree

```
bas-copilot/
├── bas_core/                         # Native Rust safety-critical core
│   ├── Cargo.toml                    # cdylib with pyo3 0.20, serde, chrono
│   └── src/
│       └── lib.rs                    # ProtocolFSM, safety gate, hesitation tracker
├── dataset/                          # Offline mission audit logs and dataset
│   ├── raw_videos/
│   ├── labels/
│   ├── processed/
│   └── events.jsonl                  # Section 8 standardized JSONL event audit log
├── models/                           # Neural network checkpoints
│   ├── yolo_finetuned.pt             # YOLOv8n laboratory apparatus detector
│   └── har_gru.pt                    # PyTorch 2-layer GRU temporal action model
├── protocols/
│   └── fluid_physics.json            # 5-step microgravity protein crystallization protocol
├── backend/                          # FastAPI mission hub and perception engines
│   ├── main.py                       # Server, MJPEG stream, WebSocket telemetry, hotkeys
│   ├── perception.py                 # MediaPipe landmarks, YOLO objects, trajectory extrapolation
│   ├── har_model.py                  # Temporal GRU + geometric Euclidean fallback
│   ├── protocol_fsm.py               # Pure-Python identical FSM fallback
│   ├── voice.py                      # Offline non-blocking pyttsx3 annunciator with queue
│   └── logger.py                     # Thread-safe atomic ISO 8601 JSONL logger
├── frontend/                         # ISRO Flight-Deck Cockpit HUD
│   ├── index.html                    # 3-column flight-deck layout
│   ├── styles.css                    # Deep Obsidian, Mission Cyan, Solar Amber, Alert Crimson
│   └── app.js                        # WebSocket telemetry, ladder renderer, presentation hotkeys
├── requirements.txt
└── README.md
```

---

## ⌨️ Presentation Hotkeys (Live Jury Simulation)

During live evaluation or mission simulations, use the keyboard hotkeys directly in your browser:

| Key | Action | Flight-Deck Effect |
| :---: | :--- | :--- |
| **`1`** | **Trigger Step 0** | Confirms *Open sample chamber* (`STEP_OK`), updates ladder `[✓]` |
| **`2`** | **Trigger Step 1** | Confirms *Insert sample cartridge* (`STEP_OK`), updates ladder `[✓]` |
| **`3`** | **Trigger Step 2** | Confirms *Attach sensor probe* (`STEP_OK`), updates ladder `[✓]` |
| **`4`** | **Trigger Step 3** | Confirms *Verify seal integrity* (`STEP_OK`), updates ladder `[✓]` |
| **`5`** | **Trigger Step 4** | Confirms *Activate agitator* (`STEP_OK`), protocol completion! |
| **`S`** | **Simulate Step Skip** | Triggers `SKIPPED` protocol deviation, crimson banner, audio warning |
| **`H`** | **Simulate Hold** | Triggers `LOW_CONFIDENCE_HOLD` (<65% gate), cyan hold badge |
| **`R`** | **Reset Protocol** | Resets the state machine and checklist ladder back to Step 0 |

---

## 🚀 Execution Instructions

### Prerequisites
Install Python dependencies:
```bash
python -m pip install -r requirements.txt
```

---

### Mode A: Pure-Python Mode (Runs Immediately Out of the Box)

No Rust toolchain required. The system dynamically detects that `bas_core` is not yet compiled, activates the pure-Python `ProtocolFSM` fallback, and flags `CORE: PY_FALLBACK` on the flight-deck HUD:

```bash
# Terminal 1: Launch FastAPI Mission Hub
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Open your browser to:
👉 **`http://localhost:8000`**

---

### Mode B: Native Rust Mode (`bas_core` Compiled)

For compiled memory-safe native execution:

1. Ensure Rust and `maturin` are installed:
   ```bash
   # If Rust is not yet installed:
   winget install Rustlang.Rustup
   # or download rustup-init from https://rustup.rs
   ```

2. Compile `bas_core` native extension using `maturin`:
   ```bash
   cd bas_core
   maturin develop --release
   cd ..
   ```

3. Launch the Mission Hub:
   ```bash
   python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
   ```

The dashboard will display **`CORE: RUST_NATIVE_SAFE`** with a green pulse dot in the top header.

---

## 📊 Standardized JSON Event Contract (Section 8)

Every safety event, transition, and deviation is logged atomically to `dataset/events.jsonl` with nanosecond-precision UTC timestamps:

```json
{
  "timestamp": "2026-09-05T04:45:50.499Z",
  "type": "STEP_OK",
  "step_id": 0,
  "step_label": "Open sample chamber",
  "confidence": 0.95,
  "hesitation": 0.12,
  "message": "STEP OK: Step 1 confirmed — Open sample chamber"
}
```

---

## 🎨 ISRO Flight-Deck Ergonomics
- **Deep Obsidian (`#080A0E`)**: Minimizes ocular fatigue in low microgravity ambient lighting.
- **Mission Cyan (`#00E5FF`)**: Signifies confirmed telemetry locks and nominal state machines.
- **Solar Amber (`#FFB300`)**: Indicates cognitive dwell, pre-error trajectory cautions, and safe holds.
- **Alert Crimson (`#FF3B30`)**: Highlights deviations (step skips, out-of-order execution, unexpected actions).
- **JetBrains Mono**: High legibility for avionics telemetry, coordinates, and audit stamps.
