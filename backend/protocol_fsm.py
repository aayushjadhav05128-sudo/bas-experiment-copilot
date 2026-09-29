"""
BAS Experiment Copilot - Protocol Finite State Machine (backend/protocol_fsm.py)

Pure deterministic logic — zero AI/ML involved.
Controls procedure state and safety gating for the experiment checklist.

Decisions strictly returned:
- CORRECT: The detected action matches the current expected step.
- OUT_OF_ORDER: The detected action matches a future step (a step was skipped/out of order).
- SKIPPED: The detected action matches a past (already completed) step.
- UNKNOWN: The detected action is not part of this protocol.
- LOW_CONFIDENCE_HOLD: Detection confidence is below threshold; state machine holds.
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

logger = logging.getLogger("ProtocolFSM")

# FSM Decision Constants
CORRECT = "CORRECT"
OUT_OF_ORDER = "OUT_OF_ORDER"
SKIPPED = "SKIPPED"
UNKNOWN = "UNKNOWN"
LOW_CONFIDENCE_HOLD = "LOW_CONFIDENCE_HOLD"

DEFAULT_PROTOCOL_DATA = {
    "name": "Fluid Physics & Protein Crystallization",
    "steps": [
        {"id": 0, "action": "open_chamber",     "label": "Open sample chamber"},
        {"id": 1, "action": "insert_cartridge", "label": "Insert sample cartridge"},
        {"id": 2, "action": "attach_probe",     "label": "Attach sensor probe"},
        {"id": 3, "action": "verify_seal",      "label": "Verify seal integrity"},
        {"id": 4, "action": "activate",         "label": "Activate agitator"}
    ]
}

ACTION_ALIASES = {
    "open_payload": ["open_chamber", "open_payload"],
    "open_chamber": ["open_payload", "open_chamber"],
    "retrieve_object": ["retrieve_object", "insert_cartridge"],
    "insert_cartridge": ["retrieve_object", "insert_cartridge"],
    "inspect_proxy": ["inspect_proxy", "attach_probe"],
    "attach_probe": ["inspect_proxy", "attach_probe"],
    "return_object": ["return_object", "verify_seal"],
    "verify_seal": ["return_object", "verify_seal"],
    "close_payload": ["close_payload", "activate"],
    "activate": ["close_payload", "activate"]
}

def actions_match(action1: str, action2: str) -> bool:
    if action1 == action2:
        return True
    aliases = ACTION_ALIASES.get(action1, [action1])
    return action2 in aliases


class ProtocolFSM:
    """
    Pure deterministic protocol state machine.
    Tracks current_step_index and evaluates operator actions without any AI/ML guessing.
    """
    def __init__(
        self,
        protocol_path: Optional[str] = "protocols/fluid_physics.json",
        confidence_threshold: float = 0.6,
        explainer: Any = None,
        *args,
        **kwargs
    ):
        self.confidence_threshold = confidence_threshold
        self.explainer = explainer
        self.protocol_name = ""
        self.steps: List[Dict[str, Any]] = []
        self.action_to_step: Dict[str, Dict[str, Any]] = {}

        self.current_step_index = 0
        self.completed_step_ids: List[int] = []
        self.is_completed = False
        self.hesitation_score: float = 0.0
        self.active_alert: Optional[Dict[str, Any]] = None

        if protocol_path and os.path.exists(protocol_path):
            self.load_protocol_from_file(protocol_path)
        else:
            self.load_default_protocol()

    @staticmethod
    def current_iso_timestamp() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    def load_protocol_from_file(self, protocol_path: str):
        """Loads protocol definition from a JSON file."""
        with open(protocol_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._apply_protocol_data(data)

    def load_protocol(self, json_content: str):
        """Loads protocol definition from JSON string or file path."""
        if os.path.exists(json_content):
            self.load_protocol_from_file(json_content)
        else:
            data = json.loads(json_content)
            self._apply_protocol_data(data)

    def load_default_protocol(self):
        """Loads built-in default protocol data."""
        self._apply_protocol_data(DEFAULT_PROTOCOL_DATA)

    def _apply_protocol_data(self, data: Dict[str, Any]):
        self.protocol_name = data.get("name", "Fluid Physics & Protein Crystallization")
        self.steps = data.get("steps", [])
        self.action_to_step = {s["action"]: s for s in self.steps}
        self.reset()
        logger.info(f"Loaded protocol: '{self.protocol_name}' with {len(self.steps)} steps.")

    def reset(self):
        """Resets the state machine back to Step 0."""
        self.current_step_index = 0
        self.completed_step_ids = []
        self.is_completed = False
        self.hesitation_score = 0.0
        self.active_alert = None

    def dismiss_alert(self):
        """Dismisses the active HUD alert."""
        self.active_alert = None

    def get_current_expected_step(self) -> Optional[Dict[str, Any]]:
        """Returns the dictionary of the step currently expected, or None if complete."""
        if self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None

    def get_next_expected_step(self) -> Optional[Dict[str, Any]]:
        """Returns the subsequent step after current, or None."""
        if self.current_step_index + 1 < len(self.steps):
            return self.steps[self.current_step_index + 1]
        return None

    def check(
        self,
        detected_action: str,
        confidence: float,
        threshold: Optional[float] = None
    ) -> str:
        """
        Pure deterministic decision logic — no AI/ML involved.
        - if confidence < threshold: return LOW_CONFIDENCE_HOLD (do not move)
        - if detected_action == steps[current_step_index].action:
            advance current_step_index by 1, return CORRECT
        - if detected_action matches a FUTURE step's action:
            return OUT_OF_ORDER
        - if detected_action matches a PAST (already completed) step:
            return SKIPPED (they went back / a required step was missed)
        - else: return UNKNOWN
        """
        if threshold is None:
            threshold = self.confidence_threshold

        # 1. Low-confidence safety gate
        if confidence < threshold:
            return LOW_CONFIDENCE_HOLD

        # 2. If protocol already complete
        if self.current_step_index >= len(self.steps):
            past_actions = [s["action"] for s in self.steps]
            if detected_action in past_actions:
                return SKIPPED
            return UNKNOWN

        # 3. Matching current step -> advance by 1
        current_step = self.steps[self.current_step_index]
        if actions_match(detected_action, current_step["action"]):
            self.completed_step_ids.append(current_step["id"])
            self.current_step_index += 1
            if self.current_step_index >= len(self.steps):
                self.is_completed = True
            return CORRECT

        # 4. Matches a future step
        future_actions = [s["action"] for s in self.steps[self.current_step_index + 1:]]
        if any(actions_match(detected_action, fa) for fa in future_actions):
            return OUT_OF_ORDER

        # 5. Matches a past (already completed) step
        past_actions = [s["action"] for s in self.steps[:self.current_step_index]]
        if any(actions_match(detected_action, pa) for pa in past_actions):
            return SKIPPED

        # 6. Unrelated / unknown action
        return UNKNOWN

    def evaluate(
        self,
        detected_action: str,
        confidence: float,
        is_dwelling: bool = False
    ) -> str:
        """Evaluates detected action and returns JSON string event."""
        event_dict = self.evaluate_action(detected_action, confidence, is_dwelling=is_dwelling)
        return json.dumps(event_dict)

    def evaluate_action(
        self,
        action: str,
        confidence: float,
        hesitation: Optional[float] = None,
        is_dwelling: bool = False
    ) -> Dict[str, Any]:
        """
        Integrates check() with aerospace telemetry formatting, Layer 4 explanations,
        and backward-compatible test hooks.
        """
        if hesitation is not None:
            self.hesitation_score = hesitation
        elif is_dwelling:
            self.hesitation_score = min(1.0, self.hesitation_score + 0.08)
        else:
            self.hesitation_score = max(0.0, self.hesitation_score - 0.04)

        prev_step_idx = self.current_step_index
        expected_step = self.get_current_expected_step()
        expected_id = expected_step["id"] if expected_step else len(self.steps)
        expected_action = expected_step["action"] if expected_step else "none"
        expected_label = expected_step["label"] if expected_step else "Protocol Complete"

        # Run pure deterministic decision logic
        result = self.check(action, confidence, threshold=self.confidence_threshold)

        # Generate guidance text (using explainer or deterministic fallback)
        guidance = ""
        if self.explainer and hasattr(self.explainer, "explain"):
            try:
                guidance = self.explainer.explain(
                    expected_step=expected_action,
                    detected_action=action,
                    result=result,
                    step_label=expected_label,
                    confidence=confidence
                )
            except Exception:
                guidance = ""

        if not guidance:
            if result == CORRECT:
                step_done = self.steps[prev_step_idx] if prev_step_idx < len(self.steps) else None
                lbl = step_done["label"] if step_done else expected_label
                guidance = f"{lbl} completed nominally. Proceed to the next checklist step."
            elif result == OUT_OF_ORDER:
                guidance = f"Protocol deviation: Future step '{action}' attempted. Expected step is {expected_label}."
            elif result == SKIPPED:
                guidance = f"Protocol notice: Past step '{action}' repeated. Expected step is {expected_label}."
            elif result == LOW_CONFIDENCE_HOLD:
                guidance = f"Observation confidence below {self.confidence_threshold:.0%}. Maintaining safe hold."
            else:
                guidance = f"Unexpected action '{action}'. Please resume {expected_label}."

        # Map to compatibility event type
        if result == CORRECT:
            event_type = "STEP_OK"
            message = f"STEP OK: Step {prev_step_idx + 1} confirmed — {self.steps[prev_step_idx]['label']}"
        elif result == LOW_CONFIDENCE_HOLD:
            event_type = "LOW_CONFIDENCE_HOLD"
            message = guidance
        elif result == UNKNOWN:
            event_type = "UNEXPECTED"
            message = guidance
        elif result == OUT_OF_ORDER:
            event_type = "SKIPPED"
            message = f"Step {expected_id + 1} skipped — expected: {expected_label}"
        elif result == SKIPPED:
            event_type = "OUT_OF_ORDER"
            message = f"Out-of-order execution: Detected past action '{action}'. Current expected procedure is {expected_label}."
        else:
            event_type = result
            message = guidance

        event = {
            "timestamp": self.current_iso_timestamp(),
            "action_detected": action,
            "confidence": round(float(confidence), 3),
            "expected_step": expected_action,
            "result": result,
            "guidance_text": guidance,
            # Telemetry & test compatibility fields
            "type": event_type,
            "step_id": expected_id,
            "step_label": expected_label,
            "hesitation": round(float(self.hesitation_score), 3),
            "message": message,
            "current_step_index": self.current_step_index,
            "is_completed": self.is_completed
        }

        if result in [LOW_CONFIDENCE_HOLD, OUT_OF_ORDER, SKIPPED, UNKNOWN]:
            self.active_alert = event
        else:
            self.active_alert = None

        return event

    def trigger_pre_error_nudge(
        self,
        converging_target: str,
        confidence: float = 0.85,
        hesitation: float = 0.0
    ) -> Dict[str, Any]:
        """
        Anticipatory caution triggered when operator's hand vector is converging
        toward the wrong apparatus zone before physical contact.
        """
        expected_step = self.get_current_expected_step()
        expected_id = expected_step["id"] if expected_step else 0
        expected_label = expected_step["label"] if expected_step else "None"

        message = f"Nudge Caution: Trajectory converging toward {converging_target}. Expected step is {expected_label}."
        event = {
            "timestamp": self.current_iso_timestamp(),
            "action_detected": f"approach_{converging_target}",
            "confidence": round(float(confidence), 3),
            "expected_step": expected_step["action"] if expected_step else "none",
            "result": "PRE_ERROR_NUDGE",
            "guidance_text": message,
            "type": "PRE_ERROR_NUDGE",
            "step_id": expected_id,
            "step_label": expected_label,
            "hesitation": round(float(hesitation), 3),
            "message": message,
            "extra": {"converging_target": converging_target}
        }
        self.active_alert = event
        return event

    trigger_pre_error = trigger_pre_error_nudge

    def get_state(self) -> Dict[str, Any]:
        """Returns snapshot of current FSM state for API and frontend HUD."""
        return {
            "protocol_name": self.protocol_name,
            "current_step_index": self.current_step_index,
            "total_steps": len(self.steps),
            "is_completed": self.is_completed,
            "expected_step": self.get_current_expected_step(),
            "completed_step_ids": self.completed_step_ids,
            "hesitation_score": round(self.hesitation_score, 2),
            "active_alert": self.active_alert
        }
