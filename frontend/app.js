/**
 * BHARATIYA ANTARIKSH STATION (BAS) — PAYLOAD OPERATIONS COPILOT
 * Mission Control Flight-Deck HUD Client (app.js)
 * 
 * Features:
 * - Native WebSocket telemetry streaming (/ws/telemetry)
 * - Clickable & Animated Protocol Step Ladder ([✓], [● ACTIVE], [ ], [! SKIPPED])
 * - Live confidence gauge bar (with 65% gate indicator)
 * - Cognitive hesitation / dwell meter
 * - Rolling 30s Sparkline Trend Canvas (Confidence vs. Hesitation)
 * - Built-in Web Audio API Avionics Sound Synthesis (Chirps, Alarms, Cautions)
 * - Auto-Run Mission Simulator (automatic 5-step nominal execution)
 * - Priority Alert Banner with operator acknowledgment
 * - Presentation Hotkeys for Jury Demonstration:
 *     1-5: Trigger Steps 0 to 4
 *     S  : Simulate Step Skip Violation
 *     H  : Simulate Low-Confidence Hold
 *     P  : Simulate Pre-Error Trajectory Nudge
 *     A  : Toggle Auto-Run Mission
 *     R  : Reset Protocol State Machine
 */

// Global App State
let ws = null;
let reconnectTimer = null;
let missionStartTime = Date.now();
let isVoiceMuted = false;
let currentProtocolState = null;
let alertAutoDismissTimer = null;
let frameCount = 0;
let lastFpsCalculation = performance.now();

// Auto-run simulation state
let isAutoRunning = false;
let autoRunTimer = null;
let autoRunCurrentStep = 0;

// Rolling Sparkline History (last 30 samples)
const sparklineHistory = [];
const MAX_SPARKLINE_POINTS = 30;

// Web Audio Synthesizer Context
let audioCtx = null;

// Protocol Steps Map (Default fallback)
const DEFAULT_STEPS = [
  { id: 0, action: "open_chamber", label: "Open sample chamber", target_object: "sample_chamber" },
  { id: 1, action: "insert_cartridge", label: "Insert sample cartridge", target_object: "cartridge" },
  { id: 2, action: "attach_probe", label: "Attach sensor probe", target_object: "sensor_probe" },
  { id: 3, action: "verify_seal", label: "Verify seal integrity", target_object: "chamber_seal" },
  { id: 4, action: "activate", label: "Activate agitator", target_object: "agitator_switch" }
];

// Initialize on DOM Ready
document.addEventListener("DOMContentLoaded", () => {
  initMissionClock();
  initVideoStreamWatchdog();
  initWebSocket();
  initPresentationHotkeys();
  initUIControls();
  fetchInitialState();
  initFpsMonitor();
  initSparklineCanvas();
});

// ==================== VIDEO STREAM WATCHDOG ====================
function initVideoStreamWatchdog() {
  const streamImg = document.getElementById("hud-video-stream");
  if (streamImg) {
    streamImg.onerror = () => {
      console.warn("Video stream stalled or disconnected. Reconnecting...");
      setTimeout(reloadVideoStream, 1200);
    };
  }
  // Check camera status on startup
  updateCameraStatusDisplay();
  setInterval(updateCameraStatusDisplay, 4000);
}

function reloadVideoStream() {
  const streamImg = document.getElementById("hud-video-stream");
  if (streamImg) {
    streamImg.src = "/video_feed?t=" + Date.now();
  }
}

async function updateCameraStatusDisplay() {
  try {
    const res = await fetch("/api/camera/status");
    if (res.ok) {
      const data = await res.json();
      const badge = document.getElementById("feed-status-badge");
      const chip = document.getElementById("mode-header-chip");
      if (data.simulated_mode) {
        if (badge) {
          badge.textContent = "● SIM FEED ACTIVE";
          badge.style.color = "var(--color-solar-amber)";
        }
        if (chip) chip.textContent = "[MODE: SIMULATION]";
      } else {
        if (badge) {
          badge.textContent = `● LIVE WEBCAM ACTIVE (CAM ${data.camera_index})`;
          badge.style.color = "var(--color-nominal-green)";
        }
        if (chip) chip.textContent = "[MODE: LIVE WEBCAM]";
      }
    }
  } catch (err) {
    console.debug("Camera status poll notice:", err);
  }
}

// ==================== 1. MISSION CLOCK (MET) ====================
function initMissionClock() {
  const metEl = document.getElementById("met-clock");
  setInterval(() => {
    const elapsedSec = Math.floor((Date.now() - missionStartTime) / 1000);
    const hrs = String(Math.floor(elapsedSec / 3600)).padStart(2, "0");
    const mins = String(Math.floor((elapsedSec % 3600) / 60)).padStart(2, "0");
    const secs = String(elapsedSec % 60).padStart(2, "0");
    if (metEl) metEl.textContent = `${hrs}:${mins}:${secs}`;
  }, 1000);
}

// ==================== 2. FPS MONITOR ====================
function initFpsMonitor() {
  const fpsEl = document.getElementById("fps-display");
  setInterval(() => {
    const now = performance.now();
    const elapsed = (now - lastFpsCalculation) / 1000;
    if (elapsed > 0 && fpsEl) {
      const fps = Math.round(frameCount / elapsed);
      fpsEl.textContent = `${fps || 20} FPS`;
    }
    frameCount = 0;
    lastFpsCalculation = now;
  }, 1000);
}

// ==================== 3. WEB AUDIO SYNTHESIZER ====================
function playAvionicsTone(type) {
  if (isVoiceMuted) return;

  try {
    if (!audioCtx) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioCtx.state === "suspended") {
      audioCtx.resume();
    }

    const now = audioCtx.currentTime;

    if (type === "confirm") {
      // Crisp rising aerospace double-chirp
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = "sine";
      osc.frequency.setValueAtTime(880, now);
      osc.frequency.exponentialRampToValueAtTime(1175, now + 0.1);
      gain.gain.setValueAtTime(0.15, now);
      gain.gain.exponentialRampToValueAtTime(0.01, now + 0.15);
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start(now);
      osc.stop(now + 0.16);
    } else if (type === "deviation") {
      // Dual-tone alert alarm
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = "sawtooth";
      osc.frequency.setValueAtTime(440, now);
      osc.frequency.setValueAtTime(330, now + 0.12);
      gain.gain.setValueAtTime(0.2, now);
      gain.gain.exponentialRampToValueAtTime(0.01, now + 0.3);
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start(now);
      osc.stop(now + 0.31);
    } else if (type === "pre_error") {
      // Soft amber caution ping
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = "triangle";
      osc.frequency.setValueAtTime(587, now);
      gain.gain.setValueAtTime(0.15, now);
      gain.gain.exponentialRampToValueAtTime(0.01, now + 0.2);
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start(now);
      osc.stop(now + 0.21);
    } else if (type === "hold") {
      // Low steady safe-state hold tone
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = "sine";
      osc.frequency.setValueAtTime(392, now);
      gain.gain.setValueAtTime(0.12, now);
      gain.gain.exponentialRampToValueAtTime(0.01, now + 0.25);
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start(now);
      osc.stop(now + 0.26);
    }
  } catch (err) {
    console.debug("Audio synthesis skipped:", err);
  }
}

// ==================== 4. WEBSOCKET TELEMETRY ====================
function initWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;

  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    console.log("[ISRO HUD] WebSocket telemetry connected to BAS hub.");
    clearTimeout(reconnectTimer);
    const statusEl = document.getElementById("ws-status-indicator");
    if (statusEl) {
      statusEl.textContent = "CONNECTED";
      statusEl.className = "status-active-green";
    }
  };

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      handleTelemetryMessage(msg);
    } catch (err) {
      console.error("[ISRO HUD] Telemetry parse error:", err);
    }
  };

  ws.onclose = () => {
    console.warn("[ISRO HUD] WebSocket disconnected. Re-engaging in 2000ms...");
    const statusEl = document.getElementById("ws-status-indicator");
    if (statusEl) {
      statusEl.textContent = "OFFLINE (RETRY)";
      statusEl.className = "telemetry-code";
      statusEl.style.color = "var(--color-alert-crimson)";
    }
    clearTimeout(reconnectTimer);
    reconnectTimer = setTimeout(initWebSocket, 2000);
  };

  ws.onerror = (err) => {
    console.error("[ISRO HUD] WebSocket error:", err);
  };
}

// ==================== 5. TELEMETRY DISPATCHER ====================
function handleTelemetryMessage(msg) {
  frameCount++;

  if (msg.type === "INIT") {
    if (msg.state) updateProtocolState(msg.state);
    if (msg.recent_logs) renderBatchLogs(msg.recent_logs);
    if (msg.engine) updateEnginePill(msg.engine);
  } else if (msg.type === "TELEMETRY" || msg.type === "FRAME_TELEMETRY") {
    if (msg.state) updateProtocolState(msg.state);
    if (msg.engine) updateEnginePill(msg.engine);

    const conf = msg.confidence ?? msg.action_confidence ?? 0.88;
    const hesit = msg.hesitation ?? msg.hesitation_score ?? 0.08;

    updateMeters(conf, hesit);
    pushSparklineData(conf, hesit);

    if (msg.closest_object) {
      const targetEl = document.getElementById("hud-target-name");
      if (targetEl) targetEl.textContent = msg.closest_object.toUpperCase();
    }
  } else if (msg.type === "EVENT") {
    appendLogCard(msg.data);
    handleEventAlert(msg.data);
    if (msg.state) updateProtocolState(msg.state);
  } else if (msg.type === "STATE_RESET" || msg.type === "PROTOCOL_SWITCHED") {
    if (msg.state) updateProtocolState(msg.state);
    hideAlertBanner();
    setSystemState("NOMINAL");
  } else if (msg.type === "ALERT_DISMISSED") {
    hideAlertBanner();
  }
}

// ==================== 6. PROTOCOL STEP LADDER RENDERER ====================
function updateProtocolState(state) {
  if (!state) return;
  currentProtocolState = state;

  if (state.core_engine) {
    updateEnginePill(state.core_engine);
  }

  const nameEl = document.getElementById("protocol-name-display");
  const counterEl = document.getElementById("ladder-progress-counter");
  const progressFill = document.getElementById("protocol-progress-bar");
  const expectedEl = document.getElementById("expected-action-display");

  if (nameEl) nameEl.textContent = state.protocol_name || "Fluid Physics & Protein Crystallization";

  const total = state.total_steps || (state.steps ? state.steps.length : 5);
  const completedCount = state.completed_steps ? state.completed_steps.length : 0;
  if (counterEl) counterEl.textContent = `${completedCount} / ${total}`;

  const pct = total > 0 ? (completedCount / total) * 100 : 0;
  if (progressFill) progressFill.style.width = `${pct}%`;

  if (state.is_completed) {
    if (expectedEl) expectedEl.textContent = "✓ ALL STEPS VERIFIED & COMPLETE";
    setSystemState("NOMINAL");
    if (isAutoRunning) {
      stopAutoMission();
    }
  } else if (state.expected_step) {
    if (expectedEl) expectedEl.textContent = `Step ${state.expected_step.id + 1} — ${state.expected_step.label}`;
  }

  // Render 5-Step Ladder List (Clickable!)
  const ladderContainer = document.getElementById("ladder-steps-list");
  const steps = state.steps || DEFAULT_STEPS;

  if (ladderContainer && steps) {
    ladderContainer.innerHTML = "";
    steps.forEach((step) => {
      const isCompleted = state.completed_steps && state.completed_steps.includes(step.id);
      const isActive = state.current_step_index === step.id && !state.is_completed;
      const isSkipped = state.active_alert && 
                        state.active_alert.type === "SKIPPED" && 
                        state.active_alert.step_id === step.id;

      let statusClass = "pending";
      let statusIndicator = `[ ]`;
      let badgeLabel = "QUEUED";

      if (isSkipped) {
        statusClass = "skipped";
        statusIndicator = `[! SKIPPED]`;
        badgeLabel = "SKIPPED";
      } else if (isCompleted) {
        statusClass = "completed";
        statusIndicator = `[✓]`;
        badgeLabel = "VERIFIED";
      } else if (isActive) {
        statusClass = "active";
        statusIndicator = `[● ACTIVE]`;
        badgeLabel = "ACTIVE NOW";
      }

      const stepCard = document.createElement("div");
      stepCard.className = `step-item ${statusClass}`;
      stepCard.title = `Click to simulate Step ${step.id + 1}: ${step.label}`;
      stepCard.onclick = () => triggerStep(step.id);

      stepCard.innerHTML = `
        <div class="step-left">
          <span class="step-indicator">${step.id + 1}</span>
          <div class="step-details">
            <span class="step-text-title">${step.label}</span>
            <span class="step-target-tag">TARGET: ${step.target_object || 'APPARATUS'} &bull; ${statusIndicator}</span>
          </div>
        </div>
        <span class="step-badge ${statusClass}">${badgeLabel}</span>
      `;
      ladderContainer.appendChild(stepCard);
    });
  }

  if (state.active_alert) {
    handleEventAlert(state.active_alert);
  }
}

// ==================== 7. ENGINE STATUS PILL ====================
function updateEnginePill(engineName) {
  const pill = document.getElementById("engine-pill");
  const text = document.getElementById("engine-pill-text");
  const cornerEngine = document.getElementById("hud-corner-engine");

  const isRust = engineName.includes("RUST") || engineName === "RUST_NATIVE_SAFE";

  if (isRust) {
    if (pill) {
      pill.className = "engine-status-pill engine-rust";
      pill.title = "Native Memory-Safe Rust State Machine Active";
    }
    if (text) text.textContent = "CORE: RUST_NATIVE_SAFE";
    if (cornerEngine) {
      cornerEngine.textContent = "RUST_SAFE";
      cornerEngine.style.color = "var(--color-nominal-green)";
    }
  } else {
    if (pill) {
      pill.className = "engine-status-pill engine-python";
      pill.title = "High-Reliability Pure Python Fallback FSM Active";
    }
    if (text) text.textContent = "CORE: PY_FALLBACK";
    if (cornerEngine) {
      cornerEngine.textContent = "PY_FALLBACK";
      cornerEngine.style.color = "var(--color-solar-amber)";
    }
  }
}

// ==================== 8. TELEMETRY METERS & SYSTEM STATE ====================
function updateMeters(confidence, hesitation) {
  const confPct = Math.min(100, Math.max(0, Math.round(confidence * 100)));
  const confBar = document.getElementById("confidence-gauge-bar");
  const confVal = document.getElementById("confidence-value-display");

  if (confBar) confBar.style.width = `${confPct}%`;
  if (confVal) confVal.textContent = `${confPct}%`;

  const hesitVal = Math.min(1.0, Math.max(0.0, Number(hesitation)));
  const hesitPct = Math.round(hesitVal * 100);
  const hesitBar = document.getElementById("hesitation-gauge-bar");
  const hesitValDisplay = document.getElementById("hesitation-value-display");
  const hesitStatus = document.getElementById("hesitation-eval-status");

  if (hesitBar) hesitBar.style.width = `${hesitPct}%`;
  if (hesitValDisplay) hesitValDisplay.textContent = hesitVal.toFixed(2);

  if (hesitStatus) {
    if (hesitVal > 0.5) {
      hesitStatus.textContent = "HIGH DWELL (CAUTION)";
      hesitStatus.style.color = "var(--color-solar-amber)";
    } else {
      hesitStatus.textContent = "NOMINAL CADENCE";
      hesitStatus.style.color = "var(--color-nominal-green)";
    }
  }

  if (!currentProtocolState?.active_alert) {
    if (hesitVal > 0.55) {
      setSystemState("PRE-ERROR CAUTION");
    } else if (confPct < 65) {
      setSystemState("HOLD");
    } else {
      setSystemState("NOMINAL");
    }
  }
}

function setSystemState(state) {
  const badge = document.getElementById("system-state-badge");
  const cornerStatus = document.getElementById("hud-corner-status");
  if (!badge) return;

  badge.textContent = state;

  switch (state) {
    case "NOMINAL":
      badge.className = "state-badge badge-nominal";
      if (cornerStatus) { cornerStatus.textContent = "NOMINAL"; cornerStatus.style.color = "var(--color-nominal-green)"; }
      break;
    case "PRE-ERROR CAUTION":
      badge.className = "state-badge badge-caution";
      if (cornerStatus) { cornerStatus.textContent = "CAUTION"; cornerStatus.style.color = "var(--color-solar-amber)"; }
      break;
    case "DEVIATION":
      badge.className = "state-badge badge-deviation";
      if (cornerStatus) { cornerStatus.textContent = "DEVIATION"; cornerStatus.style.color = "var(--color-alert-crimson)"; }
      break;
    case "HOLD":
      badge.className = "state-badge badge-hold";
      if (cornerStatus) { cornerStatus.textContent = "SAFE_HOLD"; cornerStatus.style.color = "var(--color-mission-cyan)"; }
      break;
  }
}

// ==================== 9. PRIORITY ALERT BANNER ====================
function handleEventAlert(event) {
  const banner = document.getElementById("hud-alert-banner");
  const title = document.getElementById("banner-title");
  const message = document.getElementById("banner-message");
  const icon = document.getElementById("banner-icon");

  if (!banner) return;
  clearTimeout(alertAutoDismissTimer);

  if (event.type === "SKIPPED" || event.type === "OUT_OF_ORDER" || event.type === "UNEXPECTED") {
    playAvionicsTone("deviation");
    banner.className = "hud-alert-banner deviation";
    if (title) title.textContent = `PROTOCOL VIOLATION DETECTED: [${event.type}]`;
    if (message) message.textContent = event.message;
    if (icon) icon.textContent = "✖";
    banner.classList.remove("hidden");
    setSystemState("DEVIATION");
  } else if (event.type === "PRE_ERROR_NUDGE") {
    playAvionicsTone("pre_error");
    banner.className = "hud-alert-banner warning";
    if (title) title.textContent = "PRE-ERROR TRAJECTORY ANTICIPATION";
    if (message) message.textContent = event.message;
    if (icon) icon.textContent = "⚡";
    banner.classList.remove("hidden");
    setSystemState("PRE-ERROR CAUTION");

    alertAutoDismissTimer = setTimeout(() => {
      hideAlertBanner();
      setSystemState("NOMINAL");
    }, 4500);
  } else if (event.type === "LOW_CONFIDENCE_HOLD") {
    playAvionicsTone("hold");
    banner.className = "hud-alert-banner hold";
    if (title) title.textContent = "SAFE-STATE GATE: CONFIDENCE HOLD (<65%)";
    if (message) message.textContent = event.message;
    if (icon) icon.textContent = "⏸";
    banner.classList.remove("hidden");
    setSystemState("HOLD");
  } else if (event.type === "STEP_OK") {
    playAvionicsTone("confirm");
    hideAlertBanner();
    setSystemState("NOMINAL");
  }
}

function hideAlertBanner() {
  const banner = document.getElementById("hud-alert-banner");
  if (banner) banner.classList.add("hidden");
}

// ==================== 10. ROLLING SPARKLINE TREND CANVAS ====================
function initSparklineCanvas() {
  for (let i = 0; i < MAX_SPARKLINE_POINTS; i++) {
    sparklineHistory.push({ conf: 0.90, hesit: 0.10 });
  }
  drawSparkline();
}

function pushSparklineData(conf, hesit) {
  sparklineHistory.push({ conf: Number(conf), hesit: Number(hesit) });
  if (sparklineHistory.length > MAX_SPARKLINE_POINTS) {
    sparklineHistory.shift();
  }
  drawSparkline();
}

function drawSparkline() {
  const canvas = document.getElementById("telemetry-sparkline");
  if (!canvas || !canvas.getContext) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);

  // Background grid lines
  ctx.strokeStyle = "#141B26";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, h * 0.35); ctx.lineTo(w, h * 0.35); // 65% gate line (inverted)
  ctx.stroke();

  // Dotted 65% safety gate line
  ctx.strokeStyle = "rgba(255, 255, 255, 0.25)";
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  const gateY = h - (0.65 * h);
  ctx.moveTo(0, gateY); ctx.lineTo(w, gateY);
  ctx.stroke();
  ctx.setLineDash([]);

  const n = sparklineHistory.length;
  if (n < 2) return;

  const dx = w / (n - 1);

  // 1. Draw Confidence Line (Mission Cyan)
  ctx.strokeStyle = "#00E5FF";
  ctx.lineWidth = 2;
  ctx.beginPath();
  sparklineHistory.forEach((pt, idx) => {
    const x = idx * dx;
    const y = h - (Math.min(1.0, Math.max(0.0, pt.conf)) * (h - 6)) - 3;
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // 2. Draw Hesitation Line (Solar Amber)
  ctx.strokeStyle = "#FFB300";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  sparklineHistory.forEach((pt, idx) => {
    const x = idx * dx;
    const y = h - (Math.min(1.0, Math.max(0.0, pt.hesit)) * (h - 6)) - 3;
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

// ==================== 11. MONOSPACE AUDIT EVENT LOG ====================
function appendLogCard(record) {
  const container = document.getElementById("event-log-container");
  if (!container) return;

  const card = document.createElement("div");
  card.className = `log-entry`;

  let timeStr = "NOW";
  if (record.timestamp) {
    try {
      timeStr = record.timestamp.split("T")[1].replace("Z", "");
    } catch (e) {}
  }

  const badgeType = record.type || "INFO";

  card.innerHTML = `
    <div class="log-entry-meta">
      <span class="log-timestamp mono-value">[${timeStr}]</span>
      <span class="log-badge ${badgeType}">${badgeType}</span>
    </div>
    <div class="log-message mono-value">${record.message}</div>
  `;

  container.insertBefore(card, container.firstChild);
  if (container.children.length > 50) {
    container.removeChild(container.lastChild);
  }
}

function renderBatchLogs(logs) {
  const container = document.getElementById("event-log-container");
  if (!container || !logs) return;
  container.innerHTML = "";
  logs.forEach(rec => appendLogCard(rec));
}

// ==================== 12. AUTO-RUN MISSION SIMULATOR ====================
window.toggleAutoMission = function() {
  if (isAutoRunning) {
    stopAutoMission();
  } else {
    startAutoMission();
  }
};

function startAutoMission() {
  isAutoRunning = true;
  autoRunCurrentStep = 0;

  const btn = document.getElementById("auto-run-btn");
  const label = document.getElementById("auto-run-label");
  const icon = document.getElementById("auto-run-icon");
  if (btn) btn.className = "hud-btn auto-btn running";
  if (label) label.textContent = "STOP AUTO-RUN";
  if (icon) icon.textContent = "⏸";

  // Reset first
  resetProtocolState().then(() => {
    executeAutoNextStep();
  });
}

function stopAutoMission() {
  isAutoRunning = false;
  clearTimeout(autoRunTimer);

  const btn = document.getElementById("auto-run-btn");
  const label = document.getElementById("auto-run-label");
  const icon = document.getElementById("auto-run-icon");
  if (btn) btn.className = "hud-btn auto-btn";
  if (label) label.textContent = "AUTO-RUN MISSION";
  if (icon) icon.textContent = "▶";
}

function executeAutoNextStep() {
  if (!isAutoRunning) return;

  if (autoRunCurrentStep < 5) {
    triggerStep(autoRunCurrentStep);
    autoRunCurrentStep++;

    // Paced at natural 2.2 seconds per step
    autoRunTimer = setTimeout(() => {
      executeAutoNextStep();
    }, 2200);
  } else {
    stopAutoMission();
  }
}

// ==================== 13. PRESENTATION HOTKEYS ====================
function initPresentationHotkeys() {
  window.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;

    const key = e.key;

    // Keys '1' to '5': Steps 0 to 4
    if (key >= '1' && key <= '5') {
      const stepIdx = parseInt(key, 10) - 1;
      triggerStep(stepIdx);
    } 
    // Key 'S' or 's': Step skip
    else if (key === 's' || key === 'S') {
      triggerSkip();
    }
    // Key 'H' or 'h': Low-confidence hold
    else if (key === 'h' || key === 'H') {
      triggerHold();
    }
    // Key 'P' or 'p': Pre-error trajectory nudge
    else if (key === 'p' || key === 'P') {
      triggerPreError();
    }
    // Key 'A' or 'a': Auto-run full mission
    else if (key === 'a' || key === 'A') {
      toggleAutoMission();
    }
    // Key 'R' or 'r': Reset
    else if (key === 'r' || key === 'R') {
      resetProtocolState();
    }
  });
}

// Global hotkey actions accessible via keyboard and HTML onclick
window.triggerStep = async function(stepIdx) {
  const stepActions = ["open_chamber", "insert_cartridge", "attach_probe", "verify_seal", "activate"];
  const action = stepActions[stepIdx] || "open_chamber";
  console.log(`[HOTKEY] Triggering Step ${stepIdx}: ${action}`);
  try {
    const res = await fetch(`/api/simulation/step/${stepIdx}`, { method: "POST" });
    if (!res.ok) {
      await fetch("/api/action/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action, confidence: 0.95, hesitation: 0.08 })
      });
    }
  } catch (err) {
    console.error("Step trigger failed:", err);
  }
};

window.triggerSkip = async function() {
  console.log("[HOTKEY] Simulating step skip deviation");
  try {
    await fetch("/api/simulation/skip", { method: "POST" });
  } catch (err) {
    console.error("Skip simulation failed:", err);
  }
};

window.triggerHold = async function() {
  console.log("[HOTKEY] Simulating low confidence hold");
  try {
    await fetch("/api/simulation/hold", { method: "POST" });
  } catch (err) {
    console.error("Hold simulation failed:", err);
  }
};

window.triggerPreError = async function() {
  console.log("[HOTKEY] Simulating pre-error trajectory warning");
  try {
    await fetch("/api/simulation/pre_error", { method: "POST" });
  } catch (err) {
    console.error("Pre-error simulation failed:", err);
  }
};

window.resetProtocolState = async function() {
  console.log("[HOTKEY] Resetting protocol state machine");
  if (isAutoRunning) stopAutoMission();
  try {
    await fetch("/api/simulation/reset", { method: "POST" });
    hideAlertBanner();
    setSystemState("NOMINAL");
  } catch (err) {
    console.error("Reset failed:", err);
  }
};

// ==================== 14. UI BUTTON CONTROLS ====================
function initUIControls() {
  const ackBtn = document.getElementById("banner-ack-btn");
  if (ackBtn) {
    ackBtn.addEventListener("click", async () => {
      hideAlertBanner();
      setSystemState("NOMINAL");
      try {
        await fetch("/api/fsm/dismiss", { method: "POST" });
      } catch (e) {}
    });
  }

  const voiceBtn = document.getElementById("voice-toggle-btn");
  const voiceIcon = document.getElementById("voice-icon");
  const voiceLabel = document.getElementById("voice-label");
  const voiceFooter = document.getElementById("voice-footer-status");

  if (voiceBtn) {
    voiceBtn.addEventListener("click", async () => {
      isVoiceMuted = !isVoiceMuted;
      try {
        await fetch("/api/voice/mute", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ muted: isVoiceMuted })
        });
      } catch (e) {}

      if (isVoiceMuted) {
        voiceBtn.className = "hud-btn muted";
        if (voiceIcon) voiceIcon.textContent = "🔇";
        if (voiceLabel) voiceLabel.textContent = "VOICE: MUTED";
        if (voiceFooter) { voiceFooter.textContent = "MUTED"; voiceFooter.style.color = "var(--color-alert-crimson)"; }
      } else {
        voiceBtn.className = "hud-btn active";
        if (voiceIcon) voiceIcon.textContent = "🔊";
        if (voiceLabel) voiceLabel.textContent = "VOICE: ACTIVE";
        if (voiceFooter) { voiceFooter.textContent = "ONLINE"; voiceFooter.style.color = "var(--color-nominal-green)"; }
        playAvionicsTone("confirm");
      }
    });
  }
}

// Initial state fetch via REST
async function fetchInitialState() {
  try {
    const res = await fetch("/api/state");
    if (res.ok) {
      const state = await res.json();
      updateProtocolState(state);
    }
  } catch (e) {
    console.debug("Initial state fetch pending WS connection.");
  }
}
