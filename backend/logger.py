"""
BAS Experiment Copilot - Event Logger
Appends every state transition and deviation event to a local .jsonl file with ISO timestamps.
Strictly adheres to Section 8 schema of the build specification.
"""

import os
import json
import threading
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional


class EventLogger:
    def __init__(self, log_dir: str = "dataset"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, "events.jsonl")
        self._lock = threading.Lock()
        
        # Ensure log file exists
        if not os.path.exists(self.log_file):
            with open(self.log_file, "a", encoding="utf-8") as f:
                pass

    @staticmethod
    def current_iso_timestamp() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    def log_event(
        self,
        event_type: str,
        step_id: int,
        step_label: str,
        confidence: float,
        hesitation: float,
        message: str,
        extra: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Logs an event adhering to the Section 8 JSON contract:
        {
          "timestamp": "2026-09-04T10:15:32.120Z",
          "type": "STEP_OK | SKIPPED | OUT_OF_ORDER | UNEXPECTED | LOW_CONFIDENCE_HOLD | PRE_ERROR_NUDGE",
          "step_id": 3,
          "step_label": "Attach sensor probe",
          "confidence": 0.91,
          "hesitation": 0.22,
          "message": "Step 3 confirmed - Attach sensor probe"
        }
        """
        record: Dict[str, Any] = {
            "timestamp": self.current_iso_timestamp(),
            "type": str(event_type),
            "step_id": int(step_id),
            "step_label": str(step_label),
            "confidence": round(float(confidence), 3),
            "hesitation": round(float(hesitation), 3),
            "message": str(message)
        }
        if extra:
            record["extra"] = extra

        serialized = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(serialized + "\n")

        return record

    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Returns the most recent events, newest first."""
        with self._lock:
            if not os.path.exists(self.log_file):
                return []
            with open(self.log_file, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
        
        events = []
        for line in reversed(lines[-limit:]):
            try:
                events.append(json.loads(line))
            except Exception:
                continue
        return events

    def clear_logs(self):
        """Clears the event log file."""
        with self._lock:
            with open(self.log_file, "w", encoding="utf-8") as f:
                pass


# Singleton instance
default_logger = EventLogger()
