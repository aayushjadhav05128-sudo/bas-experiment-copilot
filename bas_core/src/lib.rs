//! Bharatiya Antariksh Station (BAS) Avionics & Payload Operations
//! Native Safety-Critical Protocol State Machine Engine (bas_core)
//! 
//! Deterministic memory-safe implementation of protocol state tracking,
//! confidence safety gating, cognitive hesitation tracking, and deviation classification.

use chrono::{SecondsFormat, Utc};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Representation of a discrete protocol step in the flight-deck checklist
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolStep {
    pub id: usize,
    pub action: String,
    pub label: String,
    #[serde(default)]
    pub target_object: Option<String>,
    #[serde(default)]
    pub expected_duration_s: Option<u32>,
}

/// Incoming JSON protocol payload definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolDefinition {
    #[serde(default = "default_protocol_name")]
    pub name: String,
    pub steps: Vec<ProtocolStep>,
}

fn default_protocol_name() -> String {
    "Fluid Physics & Protein Crystallization".to_string()
}

/// Standardized JSON telemetry event adhering strictly to Section 8 schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolEvent {
    pub timestamp: String,
    #[serde(rename = "type")]
    pub event_type: String,
    pub step_id: usize,
    pub step_label: String,
    pub confidence: f32,
    pub hesitation: f32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra: Option<serde_json::Value>,
}

/// Full state snapshot returned to flight-deck dashboard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FsmStateSnapshot {
    pub protocol_name: String,
    pub current_step_index: usize,
    pub total_steps: usize,
    pub completed_steps: Vec<usize>,
    pub is_completed: bool,
    pub expected_step: Option<ProtocolStep>,
    pub next_step: Option<ProtocolStep>,
    pub steps: Vec<ProtocolStep>,
    pub hesitation: f32,
    pub active_alert: Option<ProtocolEvent>,
    pub core_engine: String,
}

/// Protocol Finite State Machine (FSM) compiled native PyO3 class
#[pyclass]
#[derive(Debug)]
pub struct ProtocolFSM {
    protocol_name: String,
    steps: Vec<ProtocolStep>,
    action_to_step: HashMap<String, ProtocolStep>,
    current_step_index: usize,
    completed_step_ids: Vec<usize>,
    is_completed: bool,
    confidence_threshold: f32,
    hesitation_score: f32,
    active_alert: Option<ProtocolEvent>,
}

#[pymethods]
impl ProtocolFSM {
    /// Instantiates a new ProtocolFSM with optional raw JSON protocol string
    #[new]
    #[pyo3(signature = (protocol_json=None))]
    pub fn new(protocol_json: Option<&str>) -> PyResult<Self> {
        let mut fsm = ProtocolFSM {
            protocol_name: "Fluid Physics & Protein Crystallization".to_string(),
            steps: Vec::new(),
            action_to_step: HashMap::new(),
            current_step_index: 0,
            completed_step_ids: Vec::new(),
            is_completed: false,
            confidence_threshold: 0.65,
            hesitation_score: 0.0,
            active_alert: None,
        };

        if let Some(json_content) = protocol_json {
            fsm.load_protocol_from_str(json_content)?;
        } else {
            // Load default 5-step BAS Fluid Physics protocol
            fsm.load_default_protocol();
        }

        Ok(fsm)
    }

    /// Loads protocol definitions from a JSON string
    pub fn load_protocol(&mut self, json_str: &str) -> PyResult<()> {
        self.load_protocol_from_str(json_str)
    }

    /// Resets the state machine back to Step 0
    pub fn reset(&mut self) {
        self.current_step_index = 0;
        self.completed_step_ids.clear();
        self.is_completed = false;
        self.hesitation_score = 0.0;
        self.active_alert = None;
    }

    /// Acknowledges / dismisses any currently active alert banner
    pub fn dismiss_alert(&mut self) {
        self.active_alert = None;
    }

    /// Triggers a Tier 1.1 Pre-Error Anticipation Nudge (e.g. hand vector heading to wrong object)
    pub fn trigger_pre_error(&mut self, converging_target: &str, confidence: f32) -> PyResult<String> {
        let expected = self.get_current_expected();
        let expected_label = expected.as_ref().map(|s| s.label.clone()).unwrap_or_else(|| "Next Step".to_string());
        let expected_target = expected.as_ref().and_then(|s| s.target_object.clone()).unwrap_or_else(|| "target".to_string());
        let expected_id = expected.as_ref().map(|s| s.id).unwrap_or(self.current_step_index);

        let msg = format!(
            "PRE-ERROR NUDGE: Hand converging on {}. Expected: {}",
            converging_target, expected_label
        );

        let event = ProtocolEvent {
            timestamp: Self::current_iso_timestamp(),
            event_type: "PRE_ERROR_NUDGE".to_string(),
            step_id: expected_id,
            step_label: expected_label,
            confidence,
            hesitation: self.hesitation_score,
            message: msg,
            extra: Some(serde_json::json!({
                "converging_target": converging_target,
                "expected_target": expected_target
            })),
        };

        self.active_alert = Some(event.clone());
        serde_json::to_string(&event).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Evaluates per-frame input:
    /// - `detected_action`: Action string identified by perception/HAR
    /// - `confidence`: Model certainty score (0.0 to 1.0)
    /// - `is_dwelling`: True if spatial jitter or linger detected near apparatus
    /// 
    /// Emits a standardized JSON event string.
    pub fn evaluate(
        &mut self,
        detected_action: &str,
        confidence: f32,
        is_dwelling: bool,
    ) -> PyResult<String> {
        // 1. Update cognitive hesitation tracker
        if is_dwelling {
            self.hesitation_score = (self.hesitation_score + 0.08).min(1.0);
        } else {
            self.hesitation_score = (self.hesitation_score - 0.04).max(0.0);
        }

        // Check if protocol is already complete
        if self.is_completed {
            let event = ProtocolEvent {
                timestamp: Self::current_iso_timestamp(),
                event_type: "COMPLETED".to_string(),
                step_id: self.steps.len(),
                step_label: "Protocol Complete".to_string(),
                confidence,
                hesitation: self.hesitation_score,
                message: format!("Protocol '{}' is already fully verified and completed.", self.protocol_name),
                extra: None,
            };
            return serde_json::to_string(&event)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()));
        }

        let expected_step = self.get_current_expected();
        let expected_id = expected_step.as_ref().map(|s| s.id).unwrap_or(0);
        let expected_label = expected_step.as_ref().map(|s| s.label.clone()).unwrap_or_else(|| "None".to_string());

        // 2. Confidence Safety Gate (Tier 1.2): If confidence < 0.65, emit LOW_CONFIDENCE_HOLD
        // Never guess or advance on low confidence.
        if confidence < self.confidence_threshold {
            let msg = format!(
                "HOLD STATE: Low confidence ({:.0}%). Operator please hold position or repeat: Step {} ({})",
                confidence * 100.0,
                expected_id + 1,
                expected_label
            );

            let event = ProtocolEvent {
                timestamp: Self::current_iso_timestamp(),
                event_type: "LOW_CONFIDENCE_HOLD".to_string(),
                step_id: expected_id,
                step_label: expected_label,
                confidence,
                hesitation: self.hesitation_score,
                message: msg,
                extra: Some(serde_json::json!({
                    "gate_threshold": self.confidence_threshold
                })),
            };

            self.active_alert = Some(event.clone());
            return serde_json::to_string(&event)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()));
        }

        // Check if action matches any known step
        let matched_step = match self.action_to_step.get(detected_action) {
            Some(step) => step.clone(),
            None => {
                // Action is unknown / unexpected in this protocol
                let msg = format!(
                    "UNEXPECTED ACTION '{}'. Expected Step {}: {}",
                    detected_action,
                    expected_id + 1,
                    expected_label
                );

                let event = ProtocolEvent {
                    timestamp: Self::current_iso_timestamp(),
                    event_type: "UNEXPECTED".to_string(),
                    step_id: expected_id,
                    step_label: expected_label,
                    confidence,
                    hesitation: self.hesitation_score,
                    message: msg,
                    extra: Some(serde_json::json!({
                        "detected_action": detected_action
                    })),
                };

                self.active_alert = Some(event.clone());
                return serde_json::to_string(&event)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()));
            }
        };

        let step_id = matched_step.id;

        // 3. Deviation Classifier:
        // Case A: Nominal Step (matching expected sequence)
        if step_id == expected_id {
            self.completed_step_ids.push(step_id);
            self.current_step_index += 1;

            if self.current_step_index >= self.steps.len() {
                self.is_completed = true;
            }

            let msg = format!(
                "STEP OK: Step {} confirmed — {}",
                step_id + 1,
                matched_step.label
            );

            let event = ProtocolEvent {
                timestamp: Self::current_iso_timestamp(),
                event_type: "STEP_OK".to_string(),
                step_id,
                step_label: matched_step.label,
                confidence,
                hesitation: self.hesitation_score,
                message: msg,
                extra: None,
            };

            self.active_alert = None;
            serde_json::to_string(&event)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
        }
        // Case B: Step Skipped (Action is ahead of expected step)
        else if step_id > expected_id {
            let msg = format!(
                "VIOLATION: Step {} skipped. Detected Step {} ({}). Expected: Step {} ({})",
                expected_id + 1,
                step_id + 1,
                matched_step.label,
                expected_id + 1,
                expected_label
            );

            let event = ProtocolEvent {
                timestamp: Self::current_iso_timestamp(),
                event_type: "SKIPPED".to_string(),
                step_id: expected_id,
                step_label: expected_label,
                confidence,
                hesitation: self.hesitation_score,
                message: msg,
                extra: Some(serde_json::json!({
                    "attempted_step_id": step_id,
                    "attempted_action": detected_action
                })),
            };

            self.active_alert = Some(event.clone());
            serde_json::to_string(&event)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
        }
        // Case C: Out of Order (Action is behind expected step or repeated)
        else {
            let msg = format!(
                "VIOLATION: Out of order action. Detected past Step {} ({}). Expected: Step {} ({})",
                step_id + 1,
                matched_step.label,
                expected_id + 1,
                expected_label
            );

            let event = ProtocolEvent {
                timestamp: Self::current_iso_timestamp(),
                event_type: "OUT_OF_ORDER".to_string(),
                step_id: expected_id,
                step_label: expected_label,
                confidence,
                hesitation: self.hesitation_score,
                message: msg,
                extra: Some(serde_json::json!({
                    "attempted_step_id": step_id,
                    "attempted_action": detected_action
                })),
            };

            self.active_alert = Some(event.clone());
            serde_json::to_string(&event)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
        }
    }

    /// Returns full telemetry state as a JSON string
    pub fn get_state(&self) -> PyResult<String> {
        let snapshot = FsmStateSnapshot {
            protocol_name: self.protocol_name.clone(),
            current_step_index: self.current_step_index,
            total_steps: self.steps.len(),
            completed_steps: self.completed_step_ids.clone(),
            is_completed: self.is_completed,
            expected_step: self.get_current_expected(),
            next_step: self.get_next_expected(),
            steps: self.steps.clone(),
            hesitation: (self.hesitation_score * 100.0).round() / 100.0,
            active_alert: self.active_alert.clone(),
            core_engine: "RUST_NATIVE_SAFE".to_string(),
        };

        serde_json::to_string(&snapshot)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Returns current hesitation score (0.0 to 1.0)
    pub fn get_hesitation(&self) -> f32 {
        self.hesitation_score
    }

    /// Returns current step index (0-based)
    pub fn get_current_step_index(&self) -> usize {
        self.current_step_index
    }

    /// Returns whether the full protocol is completed
    pub fn is_finished(&self) -> bool {
        self.is_completed
    }
}

impl ProtocolFSM {
    fn current_iso_timestamp() -> String {
        Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true)
    }

    fn get_current_expected(&self) -> Option<ProtocolStep> {
        if self.current_step_index < self.steps.len() {
            Some(self.steps[self.current_step_index].clone())
        } else {
            None
        }
    }

    fn get_next_expected(&self) -> Option<ProtocolStep> {
        if self.current_step_index + 1 < self.steps.len() {
            Some(self.steps[self.current_step_index + 1].clone())
        } else {
            None
        }
    }

    fn load_protocol_from_str(&mut self, json_str: &str) -> PyResult<()> {
        let def: ProtocolDefinition = serde_json::from_str(json_str)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid protocol JSON: {}", e)))?;

        self.protocol_name = def.name;
        self.steps = def.steps;
        self.action_to_step.clear();
        for s in &self.steps {
            self.action_to_step.insert(s.action.clone(), s.clone());
        }
        self.reset();
        Ok(())
    }

    fn load_default_protocol(&mut self) {
        let default_steps = vec![
            ProtocolStep { id: 0, action: "open_chamber".into(), label: "Open sample chamber".into(), target_object: Some("sample_chamber".into()), expected_duration_s: Some(5) },
            ProtocolStep { id: 1, action: "insert_cartridge".into(), label: "Insert sample cartridge".into(), target_object: Some("cartridge".into()), expected_duration_s: Some(7) },
            ProtocolStep { id: 2, action: "attach_probe".into(), label: "Attach sensor probe".into(), target_object: Some("sensor_probe".into()), expected_duration_s: Some(6) },
            ProtocolStep { id: 3, action: "verify_seal".into(), label: "Verify seal integrity".into(), target_object: Some("chamber_seal".into()), expected_duration_s: Some(4) },
            ProtocolStep { id: 4, action: "activate".into(), label: "Activate agitator".into(), target_object: Some("agitator_switch".into()), expected_duration_s: Some(3) },
        ];

        self.protocol_name = "Fluid Physics & Protein Crystallization".to_string();
        self.steps = default_steps;
        self.action_to_step.clear();
        for s in &self.steps {
            self.action_to_step.insert(s.action.clone(), s.clone());
        }
        self.reset();
    }
}

/// PyO3 Module definition for bas_core
#[pymodule]
fn bas_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<ProtocolFSM>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("CORE_TYPE", "RUST_NATIVE_SAFE")?;
    Ok(())
}
