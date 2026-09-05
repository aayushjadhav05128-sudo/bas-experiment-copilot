"""
BAS Experiment Copilot - Event Logger & Analytics Tests
"""

import os
import json
import pytest
from backend.logger import EventLogger
from backend.analytics import DriftAnalytics


def test_event_logger_writes_and_reads(tmp_path):
    log_dir = tmp_path / "logs"
    logger = EventLogger(log_dir=str(log_dir))

    rec = logger.log_event(
        event_type="STEP_OK",
        step_id=1,
        step_label="Insert cartridge",
        confidence=0.94,
        hesitation=0.12,
        message="Step 1 confirmed"
    )

    recent = logger.get_recent_events(limit=10)
    assert len(recent) == 1
    assert recent[0]["type"] == "STEP_OK"
    assert recent[0]["step_id"] == 1
    assert recent[0]["confidence"] == 0.94


def test_drift_analytics_aggregation(tmp_path):
    data_dir = tmp_path / "analytics"
    analytics = DriftAnalytics(data_dir=str(data_dir))

    analytics.record_deviation(step_id=2, deviation_type="SKIPPED")
    analytics.record_deviation(step_id=2, deviation_type="SKIPPED")
    analytics.record_deviation(step_id=3, deviation_type="OUT_OF_ORDER")
    analytics.record_run_completion(clean=False)

    summary = analytics.get_summary()
    assert summary["total_runs"] == 1
    assert summary["clean_runs"] == 0
    assert summary["total_deviations"] == 3
    assert summary["step_flags"]["2"] == 2
    assert summary["step_flags"]["3"] == 1
    assert summary["deviation_types"]["SKIPPED"] == 2
