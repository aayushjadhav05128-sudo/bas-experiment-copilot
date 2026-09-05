"""
BAS Experiment Copilot - Offline Voice Engine (pyttsx3)
Features:
- Dedicated background worker thread with speech queue (non-blocking for CV loop)
- Confidence-Calibrated Silent Mode (Tier 1.4):
  After N consecutive correct, low-hesitation steps, suppresses routine voice narration;
  re-engages immediately if risk rises, pre-error occurs, or deviation happens.
- Manual Quiet / Mute override.
"""

import sys
import queue
import logging
import threading
from typing import Optional

logger = logging.getLogger("VoiceEngine")


class VoiceEngine:
    def __init__(self, silent_threshold_steps: int = 3, hesitation_limit: float = 0.45):
        self.silent_threshold_steps = silent_threshold_steps
        self.hesitation_limit = hesitation_limit
        
        self.is_muted = False
        self.consecutive_clean_steps = 0
        self.auto_silent_mode = False
        
        self._queue = queue.Queue(maxsize=20)
        self._worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self._worker_thread.start()

    def _speech_worker(self):
        engine = None
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 165)  # crisp, intelligible cadence
            engine.setProperty("volume", 0.9)
        except Exception as e:
            logger.warning(f"Offline TTS initialization note: {e}")

        while True:
            try:
                item = self._queue.get()
                if item is None:
                    break
                text, priority = item
                if engine and not self.is_muted:
                    try:
                        engine.say(text)
                        engine.runAndWait()
                    except Exception as err:
                        logger.debug(f"TTS run error: {err}")
                self._queue.task_done()
            except Exception as outer_err:
                logger.debug(f"Speech worker error: {outer_err}")

    def update_streak(self, is_nominal: bool, hesitation: float = 0.0):
        """Updates the consecutive clean step streak for Tier 1.4 Silent Mode."""
        if is_nominal and hesitation < self.hesitation_limit:
            self.consecutive_clean_steps += 1
            if self.consecutive_clean_steps >= self.silent_threshold_steps:
                self.auto_silent_mode = True
        else:
            self.consecutive_clean_steps = 0
            self.auto_silent_mode = False

    def speak(self, text: str, is_deviation_or_warning: bool = False, force: bool = False):
        """
        Speaks text through offline TTS.
        If auto_silent_mode is active, routine confirmations are skipped,
        but deviations and warnings always break silence and speak.
        """
        if self.is_muted and not force:
            return

        if not is_deviation_or_warning and self.auto_silent_mode and not force:
            # Silent mode active: suppress routine narration
            return

        # Deviation / warning: immediately break auto-silent mode
        if is_deviation_or_warning:
            self.consecutive_clean_steps = 0
            self.auto_silent_mode = False

        try:
            # Clear old queued speech if high priority deviation comes in
            if is_deviation_or_warning and not self._queue.empty():
                try:
                    while not self._queue.empty():
                        self._queue.get_nowait()
                        self._queue.task_done()
                except Exception:
                    pass
            self._queue.put_nowait((text, 1 if is_deviation_or_warning else 0))
        except queue.Full:
            pass

    def set_mute(self, muted: bool):
        self.is_muted = muted

    def get_status(self):
        return {
            "muted": self.is_muted,
            "auto_silent_active": self.auto_silent_mode,
            "consecutive_clean_steps": self.consecutive_clean_steps,
            "silent_threshold": self.silent_threshold_steps
        }


# Singleton instance
default_voice = VoiceEngine()
