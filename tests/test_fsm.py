"""
BAS Experiment Copilot - Protocol FSM & Deviation Tests
Validates:
- Nominal progression (STEP_OK)
- Skipped step detection (SKIPPED)
- Out of order action (OUT_OF_ORDER)
- Unexpected action (UNEXPECTED)
- Safe State Gate threshold (LOW_CONFIDENCE_HOLD)
- Pre-Error Anticipation (PRE_ERROR_NUDGE)
- Event contract schema conformance
"""

import os
import json
import pytest
from backend.protocol_fsm import ProtocolFSM
from backend.logger import EventLogger
from backend.voice import VoiceEngine
from backend.analytics import DriftAnalytics


@pytest.fixture
def mock_fsm(tmp_path):
    proto_data = {
        "name": "Test Protocol",
        "steps": [
            { "id": 0, "action": "open_chamber", "label": "Open chamber", "target_object": "sample_chamber" },
            { "id": 1, "action": "insert_cartridge", "label": "Insert cartridge", "target_object": "cartridge" },
            { "id": 2, "action": "attach_probe", "label": "Attach probe", "target_object": "sensor_probe" }
        ]
    }
    proto_file = tmp_path / "test_proto.json"
    proto_file.write_text(json.dumps(proto_data), encoding="utf-8")

    logger = EventLogger(log_dir=str(tmp_path / "logs"))
    voice = VoiceEngine()
    analytics = DriftAnalytics(data_dir=str(tmp_path / "analytics"))

    fsm = ProtocolFSM(
        protocol_path=str(proto_file),
        confidence_threshold=0.60,
        voice_engine=voice,
        event_logger=logger,
        analytics_engine=analytics
    )
    return fsm


def test_nominal_step_progression(mock_fsm):
    # Step 0
    event = mock_fsm.evaluate_action("open_chamber", confidence=0.92)
    assert event["type"] == "STEP_OK"
    assert event["step_id"] == 0
    assert mock_fsm.current_step_index == 1

    # Step 1
    event = mock_fsm.evaluate_action("insert_cartridge", confidence=0.95)
    assert event["type"] == "STEP_OK"
    assert event["step_id"] == 1
    assert mock_fsm.current_step_index == 2

    # Step 2 (Final)
    event = mock_fsm.evaluate_action("attach_probe", confidence=0.88)
    assert event["type"] == "STEP_OK"
    assert event["step_id"] == 2
    assert mock_fsm.is_completed is True


def test_skipped_step_detection(mock_fsm):
    # Operator attempts Step 1 (insert_cartridge) without performing Step 0 (open_chamber)
    event = mock_fsm.evaluate_action("insert_cartridge", confidence=0.91)
    assert event["type"] == "SKIPPED"
    assert event["step_id"] == 0
    assert "skipped" in event["message"].lower()
    # FSM must NOT advance
    assert mock_fsm.current_step_index == 0


def test_out_of_order_step_detection(mock_fsm):
    # Complete Step 0
    mock_fsm.evaluate_action("open_chamber", confidence=0.90)
    # Operator repeats Step 0 instead of Step 1
    event = mock_fsm.evaluate_action("open_chamber", confidence=0.85)
    assert event["type"] == "OUT_OF_ORDER"
    assert mock_fsm.current_step_index == 1


def test_low_confidence_safe_state_hold(mock_fsm):
    # Confidence 0.45 < threshold 0.60
    event = mock_fsm.evaluate_action("open_chamber", confidence=0.45)
    assert event["type"] == "LOW_CONFIDENCE_HOLD"
    # Must hold state, not advance
    assert mock_fsm.current_step_index == 0


def test_unexpected_action(mock_fsm):
    event = mock_fsm.evaluate_action("random_switch_press", confidence=0.90)
    assert event["type"] == "UNEXPECTED"
    assert mock_fsm.current_step_index == 0


def test_pre_error_nudge(mock_fsm):
    event = mock_fsm.trigger_pre_error_nudge(converging_target="cartridge", confidence=0.88, hesitation=0.15)
    assert event["type"] == "PRE_ERROR_NUDGE"
    assert event["step_id"] == 0
    assert "Nudge" in event["message"]


def test_event_json_contract_schema(mock_fsm):
    event = mock_fsm.evaluate_action("open_chamber", confidence=0.91234, hesitation=0.221)
    # Schema check per Section 8:
    assert "timestamp" in event
    assert "type" in event
    assert "step_id" in event
    assert "step_label" in event
    assert "confidence" in event
    assert "hesitation" in event
    assert "message" in event
    assert isinstance(event["confidence"], float)
    assert isinstance(event["hesitation"], float)
