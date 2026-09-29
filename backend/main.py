"""
BHARATIYA ANTARIKSH STATION (BAS) — PAYLOAD OPERATIONS COPILOT
Mission Hub Server: FastAPI + Native WebSockets + MJPEG Video Streaming

Architecture:
- Safety-Critical State Engine: Dynamic Rust Native (`bas_core`) with Pure-Python fallback (`protocol_fsm`)
- Perception Pipeline: Threaded OpenCV + MediaPipe + YOLOv8n object detection + Trajectory extrapolation
- Decision Layer: Temporal HAR GRU with geometric Euclidean proximity fallback
- Voice Annunciator: Non-blocking threaded offline pyttsx3
- Audit Logging: Atomic JSONL event logging (ISO 8601 UTC)
"""

import os
import json
import base64
import asyncio
import logging
from typing import Dict, Any, List, Optional

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from backend.logger import default_logger
from backend.voice import default_voice
from backend.perception import PerceptionPipeline
from backend.har_model import default_har_classifier

# Configure Logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
logger = logging.getLogger("BAS_MissionHub")

# ---------------- DYNAMIC STATE ENGINE LOADER (RUST vs PYTHON) ----------------
RUST_ACTIVE = False
ProtocolFSMClass = None

try:
    import bas_core
    ProtocolFSMClass = bas_core.ProtocolFSM
    RUST_ACTIVE = True
    logger.info("⚡ Native Rust Safety-Critical Engine (bas_core) loaded successfully. RUST_ACTIVE = True")
except (ImportError, AttributeError) as e:
    from backend.protocol_fsm import ProtocolFSM
    ProtocolFSMClass = ProtocolFSM
    RUST_ACTIVE = False
    logger.info(f"🐍 Native bas_core not found ({e}). Activating Pure-Python ProtocolFSM fallback. RUST_ACTIVE = False")


class UnifiedFSMWrapper:
    """
    Unified adapter ensuring both native Rust ProtocolFSM and Python ProtocolFSM
    expose an identical, dictionary-based Python interface to FastAPI.
    """
    def __init__(self, protocol_path: str = "protocols/fluid_physics.json"):
        self.protocol_path = protocol_path
        self.raw_protocol_json = "{}"
        if os.path.exists(protocol_path):
            with open(protocol_path, "r", encoding="utf-8") as f:
                self.raw_protocol_json = f.read()

        if RUST_ACTIVE:
            self._engine = ProtocolFSMClass(self.raw_protocol_json)
        else:
            self._engine = ProtocolFSMClass(protocol_path=protocol_path)

    def evaluate(self, action: str, confidence: float, is_dwelling: bool = False) -> Dict[str, Any]:
        result = self._engine.evaluate(action, float(confidence), bool(is_dwelling))
        if isinstance(result, str):
            return json.loads(result)
        return result

    def trigger_pre_error(self, converging_target: str, confidence: float = 0.85) -> Dict[str, Any]:
        if hasattr(self._engine, "trigger_pre_error"):
            result = self._engine.trigger_pre_error(converging_target, float(confidence))
        elif hasattr(self._engine, "trigger_pre_error_nudge"):
            result = self._engine.trigger_pre_error_nudge(converging_target, float(confidence))
        else:
            result = {"type": "PRE_ERROR_NUDGE", "message": f"Pre-error on {converging_target}", "confidence": confidence}
        if isinstance(result, str):
            return json.loads(result)
        return result

    def get_state(self) -> Dict[str, Any]:
        state = self._engine.get_state()
        if isinstance(state, str):
            state = json.loads(state)
        # Ensure core engine indicator is present
        state["core_engine"] = "RUST_NATIVE_SAFE" if RUST_ACTIVE else "PY_FALLBACK"
        return state

    def reset(self):
        self._engine.reset()

    def dismiss_alert(self):
        self._engine.dismiss_alert()

    @property
    def is_completed(self) -> bool:
        if hasattr(self._engine, "is_finished"):
            return self._engine.is_finished()
        return getattr(self._engine, "is_completed", False)

    @property
    def current_step_index(self) -> int:
        if hasattr(self._engine, "get_current_step_index"):
            return self._engine.get_current_step_index()
        return getattr(self._engine, "current_step_index", 0)

    def get_current_expected_step(self) -> Optional[Dict[str, Any]]:
        if hasattr(self._engine, "get_current_expected_step"):
            return self._engine.get_current_expected_step()
        state = self.get_state()
        return state.get("expected_step")


# Instantiate Global State
fsm = UnifiedFSMWrapper("protocols/fluid_physics.json")
perception = PerceptionPipeline()

# Latest frame cache for MJPEG streaming
latest_annotated_jpeg = None
latest_frame_lock = asyncio.Lock()

# Active WebSocket Clients
connected_websockets: List[WebSocket] = []

app = FastAPI(
    title="BAS Experiment Copilot",
    description="ISRO Bharatiya Antariksh Station Aerospace Payload Operations Copilot",
    version="1.0.0"
)

frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
os.makedirs(frontend_dir, exist_ok=True)


# ---------------- REQUEST MODELS ---------------- #
class ActionSimRequest(BaseModel):
    action: str
    confidence: float = 0.95
    hesitation: float = 0.10


class MuteRequest(BaseModel):
    muted: bool


# ---------------- LIFECYCLE ---------------- #
background_loop_task = None


@app.on_event("startup")
async def startup_event():
    global background_loop_task
    logger.info("Initializing BAS Flight-Deck Mission Hub...")
    background_loop_task = asyncio.create_task(perception_and_telemetry_loop())


@app.on_event("shutdown")
async def shutdown_event():
    global background_loop_task
    if background_loop_task:
        background_loop_task.cancel()
    perception.release()


# ---------------- CORE PERCEPTION & TELEMETRY LOOP ---------------- #
async def perception_and_telemetry_loop():
    """Continuous perception, feature extraction, action classification, and telemetry broadcast loop."""
    global latest_annotated_jpeg
    logger.info("Perception & Telemetry background loop activated.")
    last_action_eval_time = 0.0

    while True:
        try:
            # 1. Capture raw frame
            raw_frame = perception.read_raw_frame()
            if raw_frame is None:
                await asyncio.sleep(0.04)
                continue

            current_exp_step = fsm.get_current_expected_step()

            # 2. Run Computer Vision Pipeline (MediaPipe landmarks, YOLO apparatus, velocity extrapolation)
            perc_result = perception.process_frame(raw_frame, current_expected_step=current_exp_step)

            # 3. Cache latest frame as JPEG for MJPEG stream and WebSocket direct canvas
            ret, jpeg_buf = cv2.imencode(".jpg", perc_result["annotated_frame"], [cv2.IMWRITE_JPEG_QUALITY, 75])
            latest_b64_frame = None
            if ret:
                latest_annotated_jpeg = jpeg_buf.tobytes()
                latest_b64_frame = base64.b64encode(latest_annotated_jpeg).decode("ascii")

            # 4. Add frame features to HAR buffer
            default_har_classifier.add_frame_features(perc_result["feature_vector"])

            # 5. Pre-Error Trajectory Anticipation (Tier 1.1)
            if perc_result.get("pre_error_warning"):
                nudge_event = fsm.trigger_pre_error(
                    converging_target=perc_result["pre_error_warning"],
                    confidence=0.88
                )
                # Log to atomic JSONL
                default_logger.log_event(
                    event_type=nudge_event["type"],
                    step_id=nudge_event["step_id"],
                    step_label=nudge_event["step_label"],
                    confidence=nudge_event["confidence"],
                    hesitation=nudge_event["hesitation"],
                    message=nudge_event["message"],
                    extra=nudge_event.get("extra")
                )
                # Voice warning
                default_voice.speak(
                    f"Caution: Hand converging on {perc_result['pre_error_warning']}. Expected: {nudge_event['step_label']}",
                    is_deviation_or_warning=True
                )
                await broadcast_ws({
                    "type": "EVENT",
                    "data": nudge_event,
                    "state": fsm.get_state()
                })

            # 6. Periodic Temporal Action Recognition (~every 0.75s)
            now = asyncio.get_event_loop().time()
            if now - last_action_eval_time > 0.75:
                heuristic_act = perc_result.get("contact_action")
                heuristic_conf = perc_result.get("action_confidence", 0.5)

                pred_action, pred_conf = default_har_classifier.classify_current_window(
                    heuristic_hint=heuristic_act,
                    heuristic_conf=heuristic_conf
                )

                if pred_action != "idle" and not fsm.is_completed:
                    is_dwelling = bool(perc_result["hesitation_score"] > 0.40)
                    fsm_event = fsm.evaluate(
                        action=pred_action,
                        confidence=pred_conf,
                        is_dwelling=is_dwelling
                    )
                    
                    # Log event to JSONL
                    default_logger.log_event(
                        event_type=fsm_event["type"],
                        step_id=fsm_event["step_id"],
                        step_label=fsm_event["step_label"],
                        confidence=fsm_event["confidence"],
                        hesitation=fsm_event["hesitation"],
                        message=fsm_event["message"],
                        extra=fsm_event.get("extra")
                    )

                    # Voice Annunciation
                    is_dev = fsm_event["type"] in ["SKIPPED", "OUT_OF_ORDER", "UNEXPECTED"]
                    if fsm_event["type"] == "STEP_OK":
                        default_voice.update_streak(is_nominal=True, hesitation=fsm_event["hesitation"])
                        if fsm.is_completed:
                            default_voice.speak("Protocol complete. All steps verified.", force=True)
                        else:
                            default_voice.speak(f"Step {fsm_event['step_id'] + 1} confirmed: {fsm_event['step_label']}")
                    elif is_dev:
                        default_voice.update_streak(is_nominal=False)
                        default_voice.speak(fsm_event["message"], is_deviation_or_warning=True)
                    elif fsm_event["type"] == "LOW_CONFIDENCE_HOLD":
                        default_voice.speak("Confidence hold. Please hold position.", is_deviation_or_warning=False)

                    await broadcast_ws({
                        "type": "EVENT",
                        "data": fsm_event,
                        "state": fsm.get_state()
                    })
                    last_action_eval_time = now

            # 7. Broadcast Telemetry Frame Packet over WebSocket
            telemetry_packet = {
                "type": "TELEMETRY",
                "engine": "RUST_NATIVE_SAFE" if RUST_ACTIVE else "PY_FALLBACK",
                "state": fsm.get_state(),
                "confidence": perc_result.get("action_confidence", 0.90),
                "hesitation": perc_result["hesitation_score"],
                "closest_object": perc_result.get("closest_object"),
                "contact_action": perc_result.get("contact_action"),
                "frame_b64": latest_b64_frame
            }
            await broadcast_ws(telemetry_packet)

            # Target ~20 FPS
            await asyncio.sleep(0.05)

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Perception loop error: {e}", exc_info=False)
            await asyncio.sleep(0.1)


async def broadcast_ws(message: Dict[str, Any]):
    """Broadcasts a JSON message to all connected WebSocket clients."""
    disconnected = []
    for ws in connected_websockets:
        try:
            await ws.send_json(message)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        if ws in connected_websockets:
            connected_websockets.remove(ws)


# ---------------- MJPEG VIDEO STREAMING ENDPOINT ---------------- #
async def mjpeg_frame_generator():
    """Generator yielding multipart MJPEG frames for /video_feed."""
    while True:
        if latest_annotated_jpeg is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                latest_annotated_jpeg + b"\r\n"
            )
        await asyncio.sleep(0.04)


@app.get("/video_feed")
async def video_feed():
    """MJPEG Video Feed consumed directly by <img> or HUD canvas."""
    return StreamingResponse(
        mjpeg_frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ---------------- WEBSOCKET TELEMETRY ENDPOINT ---------------- #
@app.websocket("/ws/telemetry")
@app.websocket("/ws")
async def websocket_telemetry_endpoint(websocket: WebSocket):
    """Native WebSocket endpoint for flight-deck HUD telemetry."""
    await websocket.accept()
    connected_websockets.append(websocket)
    # Send initial state synchronization packet
    await websocket.send_json({
        "type": "INIT",
        "engine": "RUST_NATIVE_SAFE" if RUST_ACTIVE else "PY_FALLBACK",
        "state": fsm.get_state(),
        "recent_logs": default_logger.get_recent_events(limit=30),
        "voice_status": default_voice.get_status()
    })

    try:
        while True:
            raw_text = await websocket.receive_text()
            try:
                cmd = json.loads(raw_text)
                if cmd.get("action") == "ping":
                    await websocket.send_json({"type": "pong"})
                elif cmd.get("action") == "dismiss_alert":
                    fsm.dismiss_alert()
                    await broadcast_ws({"type": "ALERT_DISMISSED", "state": fsm.get_state()})
                elif cmd.get("action") == "reset":
                    fsm.reset()
                    await broadcast_ws({"type": "STATE_RESET", "state": fsm.get_state()})
            except Exception:
                pass
    except WebSocketDisconnect:
        if websocket in connected_websockets:
            connected_websockets.remove(websocket)


# ---------------- REST & SIMULATION API ---------------- #
@app.get("/")
async def get_index():
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({"status": "BAS Copilot Online", "engine": "RUST" if RUST_ACTIVE else "PYTHON"})


@app.get("/api/state")
async def get_state():
    return fsm.get_state()


@app.post("/api/fsm/reset")
@app.post("/api/simulation/reset")
async def reset_fsm():
    fsm.reset()
    state = fsm.get_state()
    await broadcast_ws({"type": "STATE_RESET", "state": state})
    default_voice.speak("Protocol state reset to initial checklist.", force=True)
    return {"status": "reset", "state": state}


@app.post("/api/fsm/dismiss")
async def dismiss_alert():
    fsm.dismiss_alert()
    state = fsm.get_state()
    await broadcast_ws({"type": "ALERT_DISMISSED", "state": state})
    return {"status": "dismissed"}


@app.post("/api/voice/mute")
async def set_voice_mute(req: MuteRequest):
    default_voice.set_mute(req.muted)
    return default_voice.get_status()


@app.get("/api/logs")
async def get_logs(limit: int = 50):
    return default_logger.get_recent_events(limit=limit)


# Presentation Hotkey Simulation Endpoints (Keys 1-5, S, H, R)
@app.post("/api/simulation/step/{step_id}")
async def simulate_step(step_id: int):
    """Triggers step 0 to 4 (Hotkeys 1 to 5)."""
    actions = ["open_chamber", "insert_cartridge", "attach_probe", "verify_seal", "activate"]
    if 0 <= step_id < len(actions):
        action = actions[step_id]
        event = fsm.evaluate(action, confidence=0.95, is_dwelling=False)
        default_logger.log_event(
            event_type=event["type"],
            step_id=event["step_id"],
            step_label=event["step_label"],
            confidence=event["confidence"],
            hesitation=event["hesitation"],
            message=event["message"]
        )
        if event["type"] == "STEP_OK":
            default_voice.update_streak(is_nominal=True, hesitation=event["hesitation"])
            if fsm.is_completed:
                default_voice.speak("Protocol complete. All steps verified.", force=True)
            else:
                default_voice.speak(f"Step {step_id + 1} confirmed: {event['step_label']}")
        await broadcast_ws({"type": "EVENT", "data": event, "state": fsm.get_state()})
        return event
    raise HTTPException(status_code=400, detail="Invalid step ID (must be 0-4)")


@app.post("/api/simulation/skip")
async def simulate_skip():
    """Simulates a step skip deviation (Hotkey S)."""
    # Attempt an action ahead of current step
    curr = fsm.current_step_index
    actions = ["open_chamber", "insert_cartridge", "attach_probe", "verify_seal", "activate"]
    target_idx = min(len(actions) - 1, curr + 2)
    action = actions[target_idx]
    event = fsm.evaluate(action, confidence=0.92, is_dwelling=False)
    default_logger.log_event(
        event_type=event["type"],
        step_id=event["step_id"],
        step_label=event["step_label"],
        confidence=event["confidence"],
        hesitation=event["hesitation"],
        message=event["message"],
        extra=event.get("extra")
    )
    default_voice.speak(event["message"], is_deviation_or_warning=True)
    await broadcast_ws({"type": "EVENT", "data": event, "state": fsm.get_state()})
    return event


@app.post("/api/simulation/hold")
async def simulate_hold():
    """Simulates a low-confidence hold state (Hotkey H)."""
    actions = ["open_chamber", "insert_cartridge", "attach_probe", "verify_seal", "activate"]
    curr = min(len(actions) - 1, fsm.current_step_index)
    action = actions[curr]
    # Confidence below gate (< 0.65)
    event = fsm.evaluate(action, confidence=0.48, is_dwelling=True)
    default_logger.log_event(
        event_type=event["type"],
        step_id=event["step_id"],
        step_label=event["step_label"],
        confidence=event["confidence"],
        hesitation=event["hesitation"],
        message=event["message"]
    )
    default_voice.speak("Hold state. Confidence below safe gate.", is_deviation_or_warning=False)
    await broadcast_ws({"type": "EVENT", "data": event, "state": fsm.get_state()})
    return event


@app.post("/api/simulation/pre_error")
async def simulate_pre_error():
    """Simulates pre-error trajectory anticipation (Hotkey P)."""
    curr = fsm.current_step_index
    wrong_targets = ["agitator_switch", "chamber_seal", "sensor_probe"]
    target = wrong_targets[curr % len(wrong_targets)]
    event = fsm.trigger_pre_error(converging_target=target, confidence=0.88)
    default_logger.log_event(
        event_type=event["type"],
        step_id=event["step_id"],
        step_label=event["step_label"],
        confidence=event["confidence"],
        hesitation=event["hesitation"],
        message=event["message"],
        extra=event.get("extra")
    )
    default_voice.speak(f"Caution: Hand converging on {target}. Check expected step.", is_deviation_or_warning=True)
    await broadcast_ws({"type": "EVENT", "data": event, "state": fsm.get_state()})
    return event


@app.post("/api/simulation/hesitation")
async def simulate_hesitation():
    """Simulates high astronaut hesitation / spatial jitter hover."""
    curr = fsm.current_step_index
    actions = ["open_chamber", "insert_cartridge", "attach_probe", "verify_seal", "activate"]
    action = actions[curr] if curr < len(actions) else "open_chamber"
    event = fsm.evaluate(action, confidence=0.72, is_dwelling=True)
    default_logger.log_event(
        event_type="CAUTION",
        step_id=event["step_id"],
        step_label=event["step_label"],
        confidence=0.72,
        hesitation=0.88,
        message="High cognitive dwell & spatial jitter detected near apparatus."
    )
    default_voice.speak("Cognitive load alert: Operator dwell time elevated.", is_deviation_or_warning=True)
    state = fsm.get_state()
    state["hesitation"] = 0.88
    await broadcast_ws({"type": "EVENT", "data": {
        "timestamp": default_logger.current_iso_timestamp(),
        "type": "PRE_ERROR_NUDGE",
        "step_id": curr,
        "step_label": event["step_label"],
        "confidence": 0.72,
        "hesitation": 0.88,
        "message": "High cognitive hesitation detected. Operator please verify procedure checklist."
    }, "state": state})
    return {"status": "hesitation_triggered", "hesitation": 0.88}


@app.post("/api/simulation/scenario/{scenario_name}")
async def set_simulation_scenario(scenario_name: str):
    """Sets scenario mode on perception pipeline ('normal', 'skip_step', 'wrong_order', 'hesitation', 'pre_error')."""
    if hasattr(perception, "set_simulation_scenario"):
        perception.set_simulation_scenario(scenario_name)
    return {"status": "scenario_set", "scenario": scenario_name}


@app.post("/api/action/simulate")
async def simulate_action_direct(req: ActionSimRequest):
    """Direct arbitrary action simulation."""
    is_dwelling = bool(req.hesitation > 0.40)
    event = fsm.evaluate(req.action, confidence=req.confidence, is_dwelling=is_dwelling)
    default_logger.log_event(
        event_type=event["type"],
        step_id=event["step_id"],
        step_label=event["step_label"],
        confidence=event["confidence"],
        hesitation=event["hesitation"],
        message=event["message"]
    )
    await broadcast_ws({"type": "EVENT", "data": event, "state": fsm.get_state()})
    return event


class CameraSwitchRequest(BaseModel):
    source: Any  # e.g. 0, 1, "rtsp://...", "simulation"


@app.get("/api/camera/status")
async def get_camera_status():
    """Returns current camera feed status."""
    return {
        "simulated_mode": perception.simulated_mode,
        "camera_index": perception.camera_index,
        "camera_active": bool(perception.cap and perception.cap.isOpened())
    }


@app.post("/api/camera/switch")
async def switch_camera_source(req: CameraSwitchRequest):
    """Dynamically switches live video source (USB webcam index, RTSP URL, or simulation)."""
    success = perception.switch_source(req.source)
    return {
        "success": success,
        "current_source": req.source,
        "simulated_mode": perception.simulated_mode
    }


# Mount static files
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
