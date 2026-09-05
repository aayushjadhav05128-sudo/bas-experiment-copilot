"""
BAS Experiment Copilot - Layer 4: Local LLM Guidance Explainer (backend/explain.py)

Responsibilities:
- Strictly Layer 4 of the 4-layer AI Core.
- Never decides correctness — only explains the deterministic decision made by Layer 3 (Protocol FSM).
- Queries local Ollama instance (e.g. llama3.2:3b, phi3) without internet or external API keys.
- Instant, robust fallback to rule-based flight-deck template strings if Ollama is unavailable.
"""

import json
import logging
import urllib.request
import urllib.error
from typing import Dict, Any, Optional

logger = logging.getLogger("Layer4_Explainer")


class GuidanceExplainer:
    """
    Translates structured FSM decision outcomes into concise, natural-language
    guidance for the astronaut's voice annunciator and on-screen HUD.
    """
    def __init__(
        self,
        ollama_url: str = "http://localhost:11434/api/generate",
        model: str = "llama3.2:3b",
        timeout: float = 0.75
    ):
        self.ollama_url = ollama_url
        self.model = model
        self.timeout = timeout
        self.ollama_online: Optional[bool] = None

    def explain(
        self,
        expected_step: str,
        detected_action: str,
        result: str,
        step_label: Optional[str] = None,
        confidence: float = 0.90,
        extra: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generates one natural-language sentence.
        Tries local Ollama first; falls back immediately to flight-deck templates.
        """
        # 1. Attempt local Ollama generation if available
        llm_text = self._query_ollama(expected_step, detected_action, result, step_label)
        if llm_text:
            return llm_text

        # 2. High-reliability aerospace template fallback
        return self._template_fallback(expected_step, detected_action, result, step_label, confidence)

    def _query_ollama(
        self,
        expected_step: str,
        detected_action: str,
        result: str,
        step_label: Optional[str] = None
    ) -> Optional[str]:
        """Queries local Ollama REST API with a tight timeout."""
        prompt = (
            "You are an ISRO flight-deck copilot on the Bharatiya Antariksh Station (BAS). "
            "Turn this structured safety decision into ONE brief, clear, professional flight-deck spoken sentence (under 18 words). "
            f"Expected Step: {step_label or expected_step}. "
            f"Action Detected: {detected_action}. "
            f"FSM Result: {result}. "
            "Respond with ONLY the single spoken guidance sentence, no markdown or preamble."
        )

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 35
            }
        }

        try:
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.ollama_url,
                data=req_data,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status == 200:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                    text = resp_data.get("response", "").strip()
                    # Clean any accidental quotes
                    text = text.strip('"\'')
                    if text:
                        if not self.ollama_online:
                            logger.info(f"Ollama connected ({self.model}). Generating local AI explanations.")
                            self.ollama_online = True
                        return text
        except Exception:
            # Expected when Ollama is offline or not installed
            if self.ollama_online is not False:
                logger.debug("Ollama unavailable on localhost:11434; operating with flight-deck template explainer.")
                self.ollama_online = False

        return None

    def _template_fallback(
        self,
        expected_step: str,
        detected_action: str,
        result: str,
        step_label: Optional[str] = None,
        confidence: float = 0.90
    ) -> str:
        """
        Deterministic, natural-language flight-deck guidance templates
        strictly mapped to FSM results.
        """
        label = step_label or expected_step.replace("_", " ").title()

        if result in ("CORRECT", "STEP_OK"):
            return f"{label} completed nominally. Proceed to the next checklist step."

        elif result == "SKIPPED":
            return (
                f"Protocol violation: Step '{label}' was skipped. "
                f"Detected '{detected_action.replace('_', ' ')}'. Please return and complete {label}."
            )

        elif result == "OUT_OF_ORDER":
            return (
                f"Out-of-order execution: Detected past action '{detected_action.replace('_', ' ')}'. "
                f"Current expected procedure is {label}."
            )

        elif result in ("UNKNOWN", "UNEXPECTED"):
            return (
                f"Unrecognized action '{detected_action}'. "
                f"Operator please verify procedure checklist at step: {label}."
            )

        elif result == "LOW_CONFIDENCE_HOLD":
            return (
                f"Low detection confidence ({confidence:.0%}). "
                f"Operator please hold position and stabilize near {label}."
            )

        elif result == "PRE_ERROR_NUDGE":
            return (
                f"Caution: Hand trajectory converging toward wrong apparatus. "
                f"Expected target: {label}."
            )

        return f"Current expected step: {label}. Detected: {detected_action}."


# Singleton Instance
default_explainer = GuidanceExplainer()
