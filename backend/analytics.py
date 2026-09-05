"""
BAS Experiment Copilot - Cross-Session Drift Analytics (Tier 2.2)
Persists a small local file counting which steps get flagged most across runs,
shown as an analytics overview on the mission control dashboard.
"""

import os
import json
import threading
from typing import Dict, Any


class DriftAnalytics:
    def __init__(self, data_dir: str = "dataset"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.analytics_file = os.path.join(self.data_dir, "analytics.json")
        self._lock = threading.Lock()
        self._init_file()

    def _init_file(self):
        with self._lock:
            if not os.path.exists(self.analytics_file):
                initial_data = {
                    "total_runs": 0,
                    "clean_runs": 0,
                    "total_deviations": 0,
                    "step_flags": {},        # "step_id": count
                    "deviation_types": {     # SKIPPED, OUT_OF_ORDER, etc.
                        "SKIPPED": 0,
                        "OUT_OF_ORDER": 0,
                        "UNEXPECTED": 0,
                        "LOW_CONFIDENCE_HOLD": 0,
                        "PRE_ERROR_NUDGE": 0
                    }
                }
                with open(self.analytics_file, "w", encoding="utf-8") as f:
                    json.dump(initial_data, f, indent=2)

    def record_deviation(self, step_id: int, deviation_type: str):
        with self._lock:
            data = self._read_data()
            step_key = str(step_id)
            data["step_flags"][step_key] = data["step_flags"].get(step_key, 0) + 1
            data["total_deviations"] = data.get("total_deviations", 0) + 1
            
            dev_types = data.setdefault("deviation_types", {})
            dev_types[deviation_type] = dev_types.get(deviation_type, 0) + 1
            
            with open(self.analytics_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

    def record_run_completion(self, clean: bool = True):
        with self._lock:
            data = self._read_data()
            data["total_runs"] = data.get("total_runs", 0) + 1
            if clean:
                data["clean_runs"] = data.get("clean_runs", 0) + 1
            with open(self.analytics_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

    def _read_data(self) -> Dict[str, Any]:
        try:
            with open(self.analytics_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {
                "total_runs": 0,
                "clean_runs": 0,
                "total_deviations": 0,
                "step_flags": {},
                "deviation_types": {}
            }

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            return self._read_data()


default_analytics = DriftAnalytics()
